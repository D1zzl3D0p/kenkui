"""Modal execution context for NLP providers.

ModalExtractionProvider and ModalAttributionProvider wrap adapters and
dispatch their calls to Modal serverless compute. Full Modal dispatch
requires app setup (see docs/modal-setup.md). Until that is configured,
execution falls through to local execution with a warning.

Requires the 'modal' package: pip install modal
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from kenkui.models import Chapter
from kenkui.nlp.models import AttributionResult, CharacterRoster
from kenkui.nlp.providers._base import AttributionProvider, ExtractionProvider

_logger = logging.getLogger(__name__)


class ModalExtractionProvider:
    """Runs an ExtractionProvider adapter via Modal serverless compute.

    Currently falls back to local execution if Modal dispatch is not configured.
    Full Modal dispatch requires app setup (see docs/modal-setup.md).
    """

    def __init__(self, adapter: ExtractionProvider) -> None:
        self._adapter = adapter

    def build_roster(
        self,
        chapters: list[Chapter],
        series_roster: CharacterRoster | None = None,
        progress_callback: Callable[[str], None] | None = None,
        step_callback: Callable[[str], None] | None = None,
        book_path: Path | None = None,
    ) -> CharacterRoster:
        try:
            import modal  # noqa: F401 — verify modal is installed
        except ImportError:
            raise NotImplementedError(
                "Modal provider requires the 'modal' package: pip install modal"
            )
        # TODO: serialize adapter call and dispatch to Modal function
        # For now, fall through to local execution as a placeholder.
        _logger.warning(
            "ModalExtractionProvider: Modal dispatch not yet configured; "
            "running locally. See docs/modal-setup.md."
        )
        return self._adapter.build_roster(
            chapters,
            series_roster=series_roster,
            progress_callback=progress_callback,
            step_callback=step_callback,
            book_path=book_path,
        )


class ModalAttributionProvider:
    """Runs an AttributionProvider adapter via Modal serverless compute.

    Currently falls back to local execution if Modal dispatch is not configured.
    Full Modal dispatch requires app setup (see docs/modal-setup.md).
    """

    def __init__(self, adapter: AttributionProvider) -> None:
        self._adapter = adapter

    def attribute_chapter(
        self,
        chapter: Chapter,
        roster: CharacterRoster,
        progress_callback: Callable[[str], None] | None = None,
    ) -> AttributionResult:
        try:
            import modal  # noqa: F401 — verify modal is installed
        except ImportError:
            raise NotImplementedError(
                "Modal provider requires the 'modal' package: pip install modal"
            )
        # TODO: serialize adapter call and dispatch to Modal function
        # For now, fall through to local execution as a placeholder.
        _logger.warning(
            "ModalAttributionProvider: Modal dispatch not yet configured; "
            "running locally. See docs/modal-setup.md."
        )
        return self._adapter.attribute_chapter(
            chapter,
            roster,
            progress_callback=progress_callback,
        )
