from __future__ import annotations
from collections.abc import Callable
from kenkui.models import AppConfig, Chapter
from kenkui.nlp.models import AttributionResult, CharacterRoster


class OllamaProvider:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def build_roster(
        self,
        chapters: list[Chapter],
        series_roster: CharacterRoster | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> CharacterRoster:
        raise NotImplementedError

    def attribute_chapter(
        self,
        chapter: Chapter,
        roster: CharacterRoster,
        progress_callback: Callable[[str], None] | None = None,
    ) -> AttributionResult:
        raise NotImplementedError
