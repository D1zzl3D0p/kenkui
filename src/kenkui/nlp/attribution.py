"""Stage 4: LLM speaker attribution engine.

The LLM reads each chunk and assigns a speaker + emotion to every
regex-extracted quote.  Crucially, the final few speakers from the previous
chunk are injected into the next prompt so the model can maintain A-B-A-B
conversational momentum across chunk boundaries.

Public API
----------
attribute_all_chunks(chunks, quotes, roster_names, llm,
                     roster_aliases=None) → dict[int, AttributionItem]
"""

from __future__ import annotations

import json
import logging
from pydantic import ValidationError

from .chunker import Chunk
from .llm import LLMClient
from .models import AttributionItem, AttributionResult, Quote

logger = logging.getLogger(__name__)

_ATTRIBUTION_PROMPT = """\
You are analysing dialogue in a novel chapter.
Your only job is to identify who says each pre-extracted quote.

CANONICAL CHARACTER ROSTER
Use ONLY the exact canonical name shown below (the part before the dash).
The aliases in parentheses help you recognise who is speaking — but always
respond with the canonical name, never an alias.

{roster}

Recent speakers (for conversational continuity):
{last_speakers}

Chapter passage:
---
{chunk_text}
---

Quotes to attribute (do NOT add, skip, or reorder any):
{quotes_json}

INSTRUCTIONS:
- "speaker": copy the EXACT canonical name from the roster, or "NARRATOR", or "Unknown".
  Never abbreviate, nickname, or rephrase. If the text says 'said Tiffany' and the
  canonical is 'Tiffany Aching', respond with 'Tiffany Aching'.
- "emotion": one of neutral, happy, sad, angry, fearful, surprised, disgusted.
- A quoted span that is a title, label, scare quote, acronym, or a single word used
  as a term is NOT dialogue — assign it to "NARRATOR".
- Items with "kind": "italic" are italicised inner monologue or thought. Attribute
  them to the character who is thinking, using context clues. If the thinker cannot
  be determined, use "NARRATOR".
- "confidence": integer 1–5. 5 = speaker is unambiguous from the passage.
  3 = plausible but uncertain. 1 = no clear evidence in the passage.

Return ONLY the JSON — no explanation.
"""

# How many trailing speakers to carry into the next chunk prompt.
_LAST_SPEAKERS_N = 4


def _format_roster(
    roster_names: list[str],
    roster_aliases: dict[str, list[str]] | None,
) -> str:
    """Render the roster section of the prompt.

    Each character is shown as:
        - "Canonical Name" (also: alias1, alias2, …)

    NARRATOR and Unknown are appended with brief descriptions so the model
    understands their meaning.
    """
    lines: list[str] = []
    for name in roster_names:
        if name in ("NARRATOR", "Unknown"):
            continue
        if roster_aliases and name in roster_aliases:
            aliases = [a for a in roster_aliases[name] if a != name]
            if aliases:
                lines.append(f'- "{name}"  (also known as: {", ".join(aliases)})')
                continue
        lines.append(f'- "{name}"')

    lines.append(
        '- "NARRATOR"  (narration, description, internal thought, AND any quoted text '
        'that is NOT actual spoken dialogue — e.g. titles like "War and Peace", '
        'words used as labels like the "lazy" one, acronyms like "UNESCO", scare quotes)'
    )
    lines.append('- "Unknown"  (speaker cannot be determined from context)')
    return "\n".join(lines)


def _build_prompt(
    chunk: Chunk,
    chunk_quotes: list[Quote],
    roster_names: list[str],
    last_speakers: list[str],
    roster_aliases: dict[str, list[str]] | None = None,
) -> str:
    roster_str = _format_roster(roster_names, roster_aliases)
    last_str = ", ".join(last_speakers) if last_speakers else "(start of chapter)"
    quotes_payload = [{"quote_id": q.id, "text": q.text, "kind": q.kind} for q in chunk_quotes]
    return _ATTRIBUTION_PROMPT.format(
        roster=roster_str,
        last_speakers=last_str,
        chunk_text=chunk.text,
        quotes_json=json.dumps(quotes_payload, ensure_ascii=False, indent=2),
    )


def _attribute_chunk(
    chunk: Chunk,
    chunk_quotes: list[Quote],
    roster_names: list[str],
    last_speakers: list[str],
    llm: LLMClient,
    roster_aliases: dict[str, list[str]] | None = None,
) -> AttributionResult:
    """Ask the LLM to attribute every quote in *chunk*.

    On validation failure the quotes are attributed as "Unknown"/"neutral"
    so the pipeline never stalls.
    """
    if not chunk_quotes:
        return AttributionResult(attributions=[])

    prompt = _build_prompt(chunk, chunk_quotes, roster_names, last_speakers, roster_aliases)
    try:
        result = llm.generate(prompt, AttributionResult)
        # Ensure every requested quote_id has a response (fill gaps).
        returned_ids = {a.quote_id for a in result.attributions}
        for q in chunk_quotes:
            if q.id not in returned_ids:
                logger.debug("LLM skipped quote %d; defaulting to Unknown", q.id)
                result.attributions.append(
                    AttributionItem(quote_id=q.id, speaker="Unknown", emotion="neutral", confidence=1)
                )
        return result
    except (ValidationError, Exception) as exc:
        logger.warning("Attribution LLM call failed for chunk (%s); defaulting all", exc)
        return AttributionResult(
            attributions=[
                AttributionItem(quote_id=q.id, speaker="Unknown", emotion="neutral", confidence=1)
                for q in chunk_quotes
            ]
        )


def attribute_all_chunks(
    chunks: list[Chunk],
    quotes: list[Quote],
    roster_names: list[str],
    llm: LLMClient,
    roster_aliases: dict[str, list[str]] | None = None,
    confidence_threshold: int = 0,
    review_llm: LLMClient | None = None,
) -> dict[int, AttributionItem]:
    """Attribute every quote across all chunks with maintained conversational state.

    Args:
        chunks:               Overlapping chapter chunks from ``chunk_paragraphs()``.
        quotes:               All regex-extracted quotes for the chapter.
        roster_names:         Canonical character names + "NARRATOR" + "Unknown".
        llm:                  LLM client to use for attribution.
        roster_aliases:       Optional mapping of canonical → all known aliases.
                              When supplied the prompt shows aliases so the LLM can
                              recognise characters by the names used in the text.
        confidence_threshold: Quotes attributed with confidence below this value
                              will be re-attributed in a second pass.  0 disables
                              the second pass entirely (default).
        review_llm:           LLM client for the second pass.  Falls back to
                              *llm* when None.

    Returns:
        Mapping of ``quote_id → AttributionItem`` covering every quote.
    """
    quote_by_id: dict[int, Quote] = {q.id: q for q in quotes}

    result: dict[int, AttributionItem] = {}
    last_speakers: list[str] = []

    for chunk in chunks:
        chunk_quotes = [quote_by_id[qid] for qid in chunk.quote_ids if qid in quote_by_id]
        attribution = _attribute_chunk(
            chunk, chunk_quotes, roster_names, last_speakers, llm, roster_aliases
        )

        for item in attribution.attributions:
            result[item.quote_id] = item

        # Update last_speakers with the tail of this chunk's attributions.
        tail = attribution.attributions[-_LAST_SPEAKERS_N:]
        last_speakers = [a.speaker for a in tail if a.speaker != "Unknown"]

    # Second pass: re-attribute quotes below the confidence threshold.
    if confidence_threshold > 0:
        _retry_llm = review_llm or llm
        low_ids = {qid for qid, item in result.items() if item.confidence < confidence_threshold}
        if low_ids:
            for chunk in chunks:
                low_in_chunk = [
                    quote_by_id[qid] for qid in chunk.quote_ids
                    if qid in low_ids and qid in quote_by_id
                ]
                if not low_in_chunk:
                    continue
                retry = _attribute_chunk(
                    chunk, low_in_chunk, roster_names, [], _retry_llm, roster_aliases
                )
                for item in retry.attributions:
                    if item.quote_id in result and item.confidence > result[item.quote_id].confidence:
                        result[item.quote_id] = item

    return result
