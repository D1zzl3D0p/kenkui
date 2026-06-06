"""Ollama-backed NLP adapters.

OllamaExtractionAdapter   — character roster extraction via Ollama LLM.
OllamaAttributionAdapter  — quote speaker attribution via Ollama LLM.
OllamaProvider            — legacy wrapper kept for backwards compatibility.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from kenkui.models import Chapter
from kenkui.nlp import run_fast_scan
from kenkui.nlp.models import (
    AttributionItem,
    AttributionResult,
    CharacterRecord,
    CharacterRoster,
    slugify,
)

if TYPE_CHECKING:
    from kenkui.models import AppConfig
    from kenkui.nlp_config import NLPConfig

_logger = logging.getLogger(__name__)


class OllamaExtractionAdapter:
    """Extracts character roster + coreference using the Ollama pipeline.

    Wraps run_fast_scan() from kenkui.nlp and converts the FastScanResult
    into a CharacterRoster with CharacterRecord entries.
    Takes NLPConfig — uses extraction_model.

    Note: book_path is required because run_fast_scan uses it as a cache key.
    """

    def __init__(self, config: NLPConfig) -> None:
        self._config = config

    def build_roster(
        self,
        chapters: list[Chapter],
        series_roster: CharacterRoster | None = None,
        progress_callback: Callable[[str], None] | None = None,
        step_callback: Callable[[str], None] | None = None,
        book_path: Path | None = None,
    ) -> CharacterRoster:
        if book_path is None:
            raise ValueError("OllamaExtractionAdapter.build_roster() requires book_path")

        fast_scan_result = run_fast_scan(
            chapters,
            book_path,
            self._config.extraction_model,
            progress_callback=progress_callback,
            step_callback=step_callback,
            method=self._config.discovery_method,
            use_cache=False,  # NLPPipeline.extract() owns the cache layer
        )

        # Build lookup: canonical name → CharacterInfo (has mention/quote counts)
        char_info_by_name = {ci.character_id: ci for ci in fast_scan_result.characters}
        records: list[CharacterRecord] = []

        for ag in fast_scan_result.roster.characters:
            _cn = getattr(ag, "canonical_name", None)
            canonical = _cn if isinstance(_cn, str) else getattr(ag, "canonical", "")
            _slug = getattr(ag, "slug", None)
            slug = _slug if isinstance(_slug, str) else slugify(canonical)
            aliases = list(getattr(ag, "aliases", []))
            gender = getattr(ag, "gender", "")
            ci = char_info_by_name.get(canonical)
            records.append(CharacterRecord(
                slug=slug,
                canonical_name=canonical,
                aliases=aliases,
                gender=gender,
                mention_count=ci.mention_count if ci else 0,
                quote_count=ci.quote_count if ci else 0,
            ))

        return CharacterRoster(characters=records)


class OllamaAttributionAdapter:
    """Attributes quote speakers using the Ollama annotator pipeline.

    Uses the same annotated-chapter format as CloudProvider so both providers
    produce identical prompts. Takes NLPConfig — uses attribution_model.
    """

    def __init__(self, config: NLPConfig) -> None:
        self._config = config

    def attribute_chapter(
        self,
        chapter: Chapter,
        roster: CharacterRoster,
        progress_callback: Callable[[str], None] | None = None,
    ) -> AttributionResult:
        from kenkui.nlp.annotator import (
            _build_alias_to_slug,
            _build_attribution_dynamic_block,
            _build_attribution_static_block,
            annotate_chapter,
        )
        from kenkui.nlp.chunker import chunk_paragraphs
        from kenkui.nlp.llm import LLMClient, _num_ctx, _num_predict
        from kenkui.nlp.models import AttributionResultWire, Quote, attribution_wire_to_full
        from kenkui.nlp.quotes import extract_quotes, strip_scare_quotes

        if progress_callback:
            progress_callback("Attributing chapter via Ollama")

        clean_paragraphs = strip_scare_quotes(chapter.paragraphs)
        quotes = extract_quotes(clean_paragraphs)
        if not quotes:
            return AttributionResult(attributions=[])

        chapter_label = getattr(chapter, 'title', None) or getattr(chapter, 'index', '?')

        # Italic spans (inner monologue, emphasis) are pre-assigned NARRATOR.
        # The LLM only attributes spoken dialogue, which reduces both prompt
        # size (no italic tags in annotated text) and response size (fewer items).
        italic_quotes = [q for q in quotes if q.kind == "italic"]
        dialogue_quotes = [q for q in quotes if q.kind != "italic"]

        preassigned: dict[int, AttributionItem] = {
            q.id: AttributionItem(quote_id=q.id, speaker="NARRATOR", emotion="neutral", confidence=3)
            for q in italic_quotes
        }

        if italic_quotes:
            _logger.debug(
                "OllamaAttributionAdapter: chapter %r — %d italic quote(s) pre-assigned NARRATOR",
                chapter_label, len(italic_quotes),
            )

        if not dialogue_quotes:
            return AttributionResult(attributions=list(preassigned.values()))

        alias_to_slug = _build_alias_to_slug(roster)
        slug_to_pronoun = {c.slug: c.gender for c in roster.characters}
        static_block = _build_attribution_static_block(roster)
        llm = LLMClient(self._config.attribution_model)

        def _attribute_single(paras: list[str], dq: list[Quote]) -> dict[int, AttributionItem]:
            annotated_text = annotate_chapter(paras, dq, alias_to_slug, slug_to_pronoun)
            prompt = f"{static_block}\n\n{_build_attribution_dynamic_block(annotated_text)}"
            result = llm.generate(prompt, AttributionResultWire)
            return {item.quote_id: item for item in attribution_wire_to_full(result).attributions}

        # Estimate whether a single call fits within token limits.
        # Rough heuristic: 4 chars ≈ 1 token; slim format ~30 chars per attribution item ≈ 8 tokens.
        annotated_preview = annotate_chapter(clean_paragraphs, dialogue_quotes, alias_to_slug, slug_to_pronoun)
        full_prompt = f"{static_block}\n\n{_build_attribution_dynamic_block(annotated_preview)}"
        est_prompt_tok = len(full_prompt) // 4
        est_response_tok = len(dialogue_quotes) * 8
        needs_chunking = (
            est_prompt_tok > (_num_ctx - _num_predict) * 0.85
            or est_response_tok > int(_num_predict * 0.85)
        )

        _logger.debug(
            "OllamaAttributionAdapter: chapter %r — %d dialogue quote(s), "
            "est prompt=%d tok, est response=%d tok, chunking=%s",
            chapter_label, len(dialogue_quotes),
            est_prompt_tok, est_response_tok, needs_chunking,
        )

        all_attributions: dict[int, AttributionItem] = dict(preassigned)

        if not needs_chunking:
            try:
                attributed = _attribute_single(clean_paragraphs, dialogue_quotes)
                returned = len(attributed)
                expected = len(dialogue_quotes)
                if returned < expected:
                    _logger.warning(
                        "OllamaAttributionAdapter: chapter %r — LLM returned %d/%d dialogue quotes "
                        "(missing %d)",
                        chapter_label, returned, expected, expected - returned,
                    )
                else:
                    _logger.debug(
                        "OllamaAttributionAdapter: chapter %r — %d/%d dialogue quotes attributed",
                        chapter_label, returned, expected,
                    )
                all_attributions.update(attributed)
            except Exception as exc:
                _logger.warning(
                    "OllamaAttributionAdapter: chapter %r attribution failed (%s); "
                    "defaulting dialogue quotes to Unknown",
                    chapter_label, exc,
                )
                for q in dialogue_quotes:
                    all_attributions[q.id] = AttributionItem(
                        quote_id=q.id, speaker="Unknown", emotion="neutral", confidence=1
                    )
        else:
            # Safety-valve: chapter is too long for a single call — chunk it.
            chunks = chunk_paragraphs(clean_paragraphs, dialogue_quotes)
            _logger.warning(
                "OllamaAttributionAdapter: chapter %r exceeds token estimate "
                "(prompt~%d tok + response~%d tok vs limit %d) — chunking into %d piece(s)",
                chapter_label, est_prompt_tok, est_response_tok, _num_predict, len(chunks),
            )
            for chunk_idx, chunk in enumerate(chunks):
                chunk_start = chunk.para_indices[0]
                chunk_end = chunk.para_indices[-1] + 1
                chunk_paras = clean_paragraphs[chunk_start:chunk_end]
                chunk_char_start = sum(len(p) + 2 for p in clean_paragraphs[:chunk_start])
                chunk_quote_ids = set(chunk.quote_ids)
                chunk_quotes = [
                    Quote(
                        id=q.id,
                        text=q.text,
                        para_index=q.para_index - chunk_start,
                        char_offset=q.char_offset - chunk_char_start,
                        kind=q.kind,
                    )
                    for q in dialogue_quotes
                    if q.id in chunk_quote_ids
                ]
                if not chunk_quotes:
                    continue
                try:
                    attributed = _attribute_single(chunk_paras, chunk_quotes)
                    returned = len(attributed)
                    expected = len(chunk_quotes)
                    if returned < expected:
                        _logger.warning(
                            "OllamaAttributionAdapter: chapter %r chunk %d/%d — "
                            "LLM returned %d/%d quotes (missing %d)",
                            chapter_label, chunk_idx + 1, len(chunks),
                            returned, expected, expected - returned,
                        )
                    for item_id, item in attributed.items():
                        if item_id not in all_attributions:
                            all_attributions[item_id] = item
                except Exception as exc:
                    _logger.warning(
                        "OllamaAttributionAdapter: chapter %r chunk %d/%d failed (%s); "
                        "defaulting chunk quotes to Unknown",
                        chapter_label, chunk_idx + 1, len(chunks), exc,
                    )
                    for q in chunk_quotes:
                        if q.id not in all_attributions:
                            all_attributions[q.id] = AttributionItem(
                                quote_id=q.id, speaker="Unknown", emotion="neutral", confidence=1
                            )

        # Guarantee every extracted quote has an entry
        for q in quotes:
            if q.id not in all_attributions:
                _logger.warning(
                    "OllamaAttributionAdapter: chapter %r quote %d not covered — Unknown",
                    chapter_label, q.id,
                )
                all_attributions[q.id] = AttributionItem(
                    quote_id=q.id, speaker="Unknown", emotion="neutral", confidence=1
                )

        return AttributionResult(attributions=list(all_attributions.values()))


# ---------------------------------------------------------------------------
# Legacy wrapper — keeps nlp_service.py and existing tests working
# ---------------------------------------------------------------------------

class OllamaProvider:
    """Legacy provider: wraps OllamaExtractionAdapter + OllamaAttributionAdapter.

    Kept for backwards compatibility. New code should use the adapters directly
    via NLPPipeline (Phase 9).
    """

    def __init__(self, config: AppConfig) -> None:
        from kenkui.nlp_config import NLPConfig
        self._nlp_config = NLPConfig.from_app_config(config)
        self._extraction = OllamaExtractionAdapter(self._nlp_config)
        self._attribution = OllamaAttributionAdapter(self._nlp_config)

    def build_roster(
        self,
        chapters: list[Chapter],
        series_roster: CharacterRoster | None = None,
        progress_callback: Callable[[str], None] | None = None,
        step_callback: Callable[[str], None] | None = None,
        book_path: Path | None = None,
    ) -> CharacterRoster:
        return self._extraction.build_roster(
            chapters,
            series_roster=series_roster,
            progress_callback=progress_callback,
            step_callback=step_callback,
            book_path=book_path,
        )

    def attribute_chapter(
        self,
        chapter: Chapter,
        roster: CharacterRoster,
        progress_callback: Callable[[str], None] | None = None,
    ) -> AttributionResult:
        return self._attribution.attribute_chapter(chapter, roster, progress_callback=progress_callback)
