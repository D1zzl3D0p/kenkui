"""LiteLLM-backed NLP adapters.

LiteLLMExtractionAdapter  — character roster extraction (spaCy + heuristics,
                             with LiteLLM model for disambiguation).
LiteLLMAttributionAdapter — quote speaker attribution via LiteLLM + instructor.

Both ``litellm`` and ``instructor`` are **lazy imports** (inside method bodies)
so this module can be imported without those packages installed.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from kenkui.models import Chapter
from kenkui.nlp import run_fast_scan
from kenkui.nlp.models import AttributionResult, CharacterRecord, CharacterRoster, slugify

if TYPE_CHECKING:
    from kenkui.nlp_config import NLPConfig

_logger = logging.getLogger(__name__)


class LiteLLMExtractionAdapter:
    """Extracts character roster + coreference using spaCy + heuristics.

    The extraction phase uses ``run_fast_scan()`` with ``method="auto"``, which
    relies on spaCy and local heuristics.  The configured *extraction_model* is
    forwarded to ``run_fast_scan`` for any LLM-assisted disambiguation steps.

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
            raise ValueError("LiteLLMExtractionAdapter.build_roster() requires book_path")

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


class LiteLLMAttributionAdapter:
    """Attributes quote speakers using LiteLLM + instructor structured output.

    Uses the same annotated-chapter prompt format as OllamaAttributionAdapter.
    ``litellm`` and ``instructor`` are lazy-imported inside ``attribute_chapter``
    so missing packages only raise at call time, not import time.
    """

    def __init__(self, config: "NLPConfig") -> None:
        self._config = config

    def attribute_chapter(
        self,
        chapter: Chapter,
        roster: CharacterRoster,
        progress_callback: Callable[[str], None] | None = None,
    ) -> AttributionResult:
        # Lazy imports — keep at top of method so they are easy to mock.
        import litellm as _litellm  # noqa: PLC0415
        import instructor  # noqa: PLC0415

        from kenkui.nlp.quotes import extract_quotes, strip_scare_quotes
        from kenkui.nlp.annotator import (
            annotate_chapter,
            _build_alias_to_slug,
            _build_attribution_static_block,
            _build_attribution_dynamic_block,
        )
        from kenkui.nlp.models import AttributionResultWire, attribution_wire_to_full

        if progress_callback:
            progress_callback("Attributing chapter via LiteLLM")

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

        # Build kwargs for the LiteLLM call — only pass api_base/api_key when set.
        extra_kwargs: dict = {}
        if self._config.litellm_api_base is not None:
            extra_kwargs["api_base"] = self._config.litellm_api_base
        if self._config.litellm_api_key is not None:
            extra_kwargs["api_key"] = self._config.litellm_api_key

        client = instructor.from_litellm(_litellm.completion)
        result = client.chat.completions.create(
            model=self._config.attribution_model,
            messages=[{"role": "user", "content": prompt}],
            response_model=AttributionResultWire,
            **extra_kwargs,
        )
        return attribution_wire_to_full(result)
