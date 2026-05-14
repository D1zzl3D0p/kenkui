"""Local execution context for NLP providers.

LocalExtractionProvider and LocalAttributionProvider run their
wrapped adapters in the current process. They are wrappers that
provide a consistent interface regardless of tool choice, and
serve as the 'LOCAL' execution mode counterpart to ModalExtractionProvider.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from kenkui.models import Chapter
from kenkui.nlp.models import AttributionResult, CharacterRoster
from kenkui.nlp.providers._base import AttributionProvider, ExtractionProvider


class LocalExtractionProvider:
    """Runs an ExtractionProvider adapter in the current process."""

    def __init__(self, adapter: ExtractionProvider) -> None:
        self._adapter = adapter

    def build_roster(
        self,
        chapters: list[Chapter],
        series_roster: CharacterRoster | None = None,
        progress_callback: Callable[[str], None] | None = None,
        book_path: Path | None = None,
    ) -> CharacterRoster:
        return self._adapter.build_roster(
            chapters,
            series_roster=series_roster,
            progress_callback=progress_callback,
            book_path=book_path,
        )


class LocalAttributionProvider:
    """Runs an AttributionProvider adapter in the current process."""

    def __init__(self, adapter: AttributionProvider) -> None:
        self._adapter = adapter

    def attribute_chapter(
        self,
        chapter: Chapter,
        roster: CharacterRoster,
        progress_callback: Callable[[str], None] | None = None,
    ) -> AttributionResult:
        return self._adapter.attribute_chapter(
            chapter,
            roster,
            progress_callback=progress_callback,
        )
