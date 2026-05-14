"""NLP provider protocols — ExtractionProvider and AttributionProvider.

Two-layer architecture:
  Adapter  = tool implementation (BookNLP / Ollama / LiteLLM).
             Knows HOW to call a specific tool.
  Provider = execution context (Local / Modal).
             Knows WHERE to run an adapter.

Both adapters and execution wrappers satisfy the same protocol,
so the pipeline (NLPPipeline) only ever deals with Provider objects.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

from kenkui.models import Chapter
from kenkui.nlp.models import AttributionResult, CharacterRoster


@runtime_checkable
class ExtractionProvider(Protocol):
    """Extracts character roster + coreference from book chapters.

    Satisfied by both tool adapters (BookNLPExtractionAdapter, etc.)
    and execution wrappers (LocalExtractionProvider, ModalExtractionProvider).
    """

    def build_roster(
        self,
        chapters: list[Chapter],
        series_roster: CharacterRoster | None = None,
        progress_callback: Callable[[str], None] | None = None,
        book_path: Path | None = None,
    ) -> CharacterRoster: ...


@runtime_checkable
class AttributionProvider(Protocol):
    """Attributes each extracted quote in a chapter to a speaker.

    Satisfied by both tool adapters and execution wrappers.
    """

    def attribute_chapter(
        self,
        chapter: Chapter,
        roster: CharacterRoster,
        progress_callback: Callable[[str], None] | None = None,
    ) -> AttributionResult: ...


__all__ = ["ExtractionProvider", "AttributionProvider"]
