"""NLPProvider protocol — the single interface both Ollama and Cloud providers implement."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from kenkui.models import Chapter
from kenkui.nlp.models import AttributionResult, CharacterRoster


class NLPProvider(Protocol):
    """Common interface for NLP backends.

    Both ``OllamaProvider`` and ``CloudProvider`` satisfy this protocol.
    ``NLPService`` only depends on this interface — it never imports a concrete
    provider directly.
    """

    def build_roster(
        self,
        chapters: list[Chapter],
        series_roster: CharacterRoster | None = None,
        progress_callback: Callable[[str], None] | None = None,
        book_path: Path | None = None,
    ) -> CharacterRoster:
        """Extract the full character roster from *chapters*.

        Args:
            chapters:        All parsed chapters of the book.
            series_roster:   Existing series characters to use as seed context.
                             Known characters keep their established slugs.
            progress_callback: Optional ``(message: str) -> None``.

        Returns:
            ``CharacterRoster`` with ``CharacterRecord`` objects.
        """
        ...

    def attribute_chapter(
        self,
        chapter: Chapter,
        roster: CharacterRoster,
        progress_callback: Callable[[str], None] | None = None,
    ) -> AttributionResult:
        """Attribute every extracted quote in *chapter* to a speaker.

        Args:
            chapter:  A single chapter with ``paragraphs`` populated.
            roster:   The full character roster (from ``build_roster``).
            progress_callback: Optional ``(message: str) -> None``.

        Returns:
            ``AttributionResult`` where each ``AttributionItem.speaker`` is
            a character slug, ``'NARRATOR'``, or ``'Unknown'``.
        """
        ...
