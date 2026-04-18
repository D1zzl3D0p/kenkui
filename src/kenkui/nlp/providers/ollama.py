"""OllamaProvider — wraps the existing 4-stage Ollama NLP pipeline.

Converts legacy roster output to the new ``CharacterRecord`` schema
so both providers share a common output contract.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from kenkui.models import AppConfig, Chapter
from kenkui.nlp import run_fast_scan
from kenkui.nlp.models import (
    AttributionItem,
    AttributionResult,
    CharacterRecord,
    CharacterRoster,
    slugify,
)

_PRESERVED_SPEAKERS = {"NARRATOR", "Unknown"}


def _run_attribution_for_chapter(
    chapter: Chapter,
    roster: CharacterRoster,
    nlp_model: str,
    confidence_threshold: int = 0,
) -> AttributionResult:
    """Run the low-level attribution pipeline for a single chapter."""
    from kenkui.nlp.attribution import attribute_all_chunks
    from kenkui.nlp.chunker import chunk_paragraphs
    from kenkui.nlp.llm import LLMClient
    from kenkui.nlp.quotes import extract_quotes

    llm = LLMClient(nlp_model)
    quotes = extract_quotes(chapter.paragraphs)
    chunks = chunk_paragraphs(chapter.paragraphs, quotes)
    roster_names = [c.canonical_name for c in roster.characters]
    roster_aliases = {c.canonical_name: list(c.aliases) for c in roster.characters}

    raw = attribute_all_chunks(
        chunks,
        quotes,
        roster_names,
        llm,
        roster_aliases,
        confidence_threshold=confidence_threshold,
    )

    items = [
        AttributionItem(
            quote_id=qid,
            speaker=item.speaker,
            emotion=item.emotion,
            confidence=item.confidence,
        )
        for qid, item in raw.items()
    ]
    return AttributionResult(attributions=items)


def _speaker_to_slug(speaker: str, roster: CharacterRoster) -> str:
    """Convert a canonical-name speaker to its roster slug, or preserve special values."""
    if speaker in _PRESERVED_SPEAKERS:
        return speaker
    # Check roster by canonical_name first (exact match)
    for c in roster.characters:
        if c.canonical_name == speaker:
            return c.slug
    # Fall back to slugify — handles slight variations
    return slugify(speaker)


class OllamaProvider:
    """Wraps the existing Ollama NLP pipeline behind the NLPProvider protocol."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def build_roster(
        self,
        chapters: list[Chapter],
        series_roster: CharacterRoster | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> CharacterRoster:
        """Run run_fast_scan and convert output to CharacterRecord roster."""
        if progress_callback:
            progress_callback("Building character roster via Ollama")

        book_path = Path(getattr(chapters[0], "_source_path", "/tmp/kenkui_tmp.epub"))

        fast_scan_result = run_fast_scan(
            chapters,
            book_path,
            self.config.nlp_model,
            progress_callback=progress_callback,
        )

        # Build lookup: canonical name → CharacterInfo (has mention/quote counts)
        char_info_by_name = {ci.character_id: ci for ci in fast_scan_result.characters}
        records: list[CharacterRecord] = []

        for ag in fast_scan_result.roster.characters:
            # Handle both CharacterRecord (.canonical_name) and legacy AliasGroup (.canonical)
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

    def attribute_chapter(
        self,
        chapter: Chapter,
        roster: CharacterRoster,
        progress_callback: Callable[[str], None] | None = None,
    ) -> AttributionResult:
        """Run attribution for one chapter, converting speakers to slugs."""
        if progress_callback:
            progress_callback(f"Attributing chapter via Ollama")

        result = _run_attribution_for_chapter(
            chapter,
            roster,
            self.config.nlp_model,
            self.config.nlp_confidence_threshold,
        )

        converted = [
            AttributionItem(
                quote_id=item.quote_id,
                speaker=_speaker_to_slug(item.speaker, roster),
                emotion=item.emotion,
                confidence=item.confidence,
            )
            for item in result.attributions
        ]
        return AttributionResult(attributions=converted)
