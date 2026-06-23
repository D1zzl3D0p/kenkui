"""Modal-backed NLP provider wrappers."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from kenkui.models import Chapter
from kenkui.nlp.models import AttributionResult, CharacterRoster
from kenkui.nlp.providers._base import AttributionProvider, ExtractionProvider

from .errors import ModalRuntimeUnavailableError
from .settings import ModalRuntimeConfig


class RemoteNLPClient(Protocol):
    def build_roster(
        self,
        *,
        chapters: list[Chapter],
        series_roster: CharacterRoster | None,
        book_path: Path | None,
        config: Any,
        runtime_config: ModalRuntimeConfig,
    ) -> CharacterRoster: ...

    def attribute_chapter(
        self,
        *,
        chapter: Chapter,
        roster: CharacterRoster,
        config: Any,
        runtime_config: ModalRuntimeConfig,
    ) -> AttributionResult: ...


class ModalExtractionProvider(ExtractionProvider):
    """Runs character extraction through a Modal remote client."""

    def __init__(
        self,
        *,
        config: Any,
        app_config: Any | None = None,
        client: RemoteNLPClient | None = None,
    ) -> None:
        self._config = config
        self._runtime_config = ModalRuntimeConfig.from_app_config(app_config)
        self._client = client

    def build_roster(
        self,
        chapters: list[Chapter],
        series_roster: CharacterRoster | None = None,
        progress_callback: Callable[[str], None] | None = None,
        step_callback: Callable[[str], None] | None = None,
        book_path: Path | None = None,
    ) -> CharacterRoster:
        if progress_callback:
            progress_callback("Submitting character discovery to Modal")
        if self._client is None:
            raise ModalRuntimeUnavailableError(
                "Modal NLP runtime is registered but no remote client is configured yet."
            )
        roster = self._client.build_roster(
            chapters=chapters,
            series_roster=series_roster,
            book_path=book_path,
            config=self._config,
            runtime_config=self._runtime_config,
        )
        if step_callback:
            step_callback("Extracted characters")
        if progress_callback:
            progress_callback("Character discovery complete")
        return roster


class ModalAttributionProvider(AttributionProvider):
    """Runs quote attribution through a Modal remote client."""

    def __init__(
        self,
        *,
        config: Any,
        app_config: Any | None = None,
        client: RemoteNLPClient | None = None,
    ) -> None:
        self._config = config
        self._runtime_config = ModalRuntimeConfig.from_app_config(app_config)
        self._client = client

    def attribute_chapter(
        self,
        chapter: Chapter,
        roster: CharacterRoster,
        progress_callback: Callable[[str], None] | None = None,
    ) -> AttributionResult:
        if progress_callback:
            progress_callback("Submitting quote attribution to Modal")
        if self._client is None:
            raise ModalRuntimeUnavailableError(
                "Modal NLP runtime is registered but no remote client is configured yet."
            )
        result = self._client.attribute_chapter(
            chapter=chapter,
            roster=roster,
            config=self._config,
            runtime_config=self._runtime_config,
        )
        if progress_callback:
            progress_callback("Quote attribution complete")
        return result


__all__ = ["ModalAttributionProvider", "ModalExtractionProvider", "RemoteNLPClient"]
