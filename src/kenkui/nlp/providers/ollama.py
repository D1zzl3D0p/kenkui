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
from kenkui.nlp.models import AttributionItem, AttributionResult, CharacterRecord, CharacterRoster, slugify

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

    def __init__(self, config: "NLPConfig") -> None:
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

    def __init__(self, config: "NLPConfig") -> None:
        self._config = config

    def attribute_chapter(
        self,
        chapter: Chapter,
        roster: CharacterRoster,
        progress_callback: Callable[[str], None] | None = None,
    ) -> AttributionResult:
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

        llm = LLMClient(self._config.attribution_model)
        try:
            result = llm.generate(prompt, AttributionResultWire)
            return attribution_wire_to_full(result)
        except Exception as exc:
            _logger.warning(
                "OllamaAttributionAdapter: attribution failed (%s); defaulting all quotes to Unknown",
                exc,
            )
            return AttributionResult(attributions=[
                AttributionItem(quote_id=q.id, speaker="Unknown", emotion="neutral", confidence=1)
                for q in quotes
            ])


# ---------------------------------------------------------------------------
# Legacy wrapper — keeps nlp_service.py and existing tests working
# ---------------------------------------------------------------------------

class OllamaProvider:
    """Legacy provider: wraps OllamaExtractionAdapter + OllamaAttributionAdapter.

    Kept for backwards compatibility. New code should use the adapters directly
    via NLPPipeline (Phase 9).
    """

    def __init__(self, config: "AppConfig") -> None:
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
