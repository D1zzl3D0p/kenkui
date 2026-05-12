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
        book_path: Path | None = None,
    ) -> CharacterRoster:
        """Run run_fast_scan and convert output to CharacterRecord roster."""
        if progress_callback:
            progress_callback("Building character roster via Ollama")

        if book_path is None:
            raise ValueError("OllamaProvider.build_roster() requires book_path")

        discovery_method = getattr(self.config, "nlp_discovery_method", "auto") or "auto"
        fast_scan_result = run_fast_scan(
            chapters,
            book_path,
            self.config.nlp_model,
            progress_callback=progress_callback,
            method=discovery_method,
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
        """Attribute quotes in *chapter* using the annotation-based pipeline.

        Uses the same annotated-chapter format as CloudProvider so both providers
        produce identical prompts (12-factor Factor IV / X parity). Ollama receives
        the static and dynamic blocks concatenated as a single prompt string.
        """
        from kenkui.nlp.quotes import extract_quotes, strip_scare_quotes
        from kenkui.nlp.annotator import (
            annotate_chapter,
            _build_alias_to_slug,
            _build_attribution_static_block,
            _build_attribution_dynamic_block,
        )
        from kenkui.nlp.models import AttributionResultWire, attribution_wire_to_full
        from kenkui.nlp.llm import LLMClient

        if progress_callback:
            progress_callback("Attributing chapter via Ollama")

        clean_paragraphs = strip_scare_quotes(chapter.paragraphs)
        quotes = extract_quotes(clean_paragraphs)
        if not quotes:
            return AttributionResult(attributions=[])

        alias_to_slug = _build_alias_to_slug(roster)
        slug_to_pronoun = {c.slug: c.gender for c in roster.characters}
        annotated_text = annotate_chapter(clean_paragraphs, quotes, alias_to_slug, slug_to_pronoun)

        static_block = _build_attribution_static_block(roster)
        dynamic_block = _build_attribution_dynamic_block(annotated_text)
        prompt = f"{static_block}\n\n{dynamic_block}"

        llm = LLMClient(self.config.nlp_model)
        result = llm.generate(prompt, AttributionResultWire)
        return attribution_wire_to_full(result)
