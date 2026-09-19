"""One callback sequence across resolution and rendering effects."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .errors import ErrorCode, RenderError
from .events import (
    CastResolved,
    Completed,
    ExecutionEvent,
    StageCompleted,
    StageProgress,
    StageStarted,
    Started,
)
from .events import Warning as WarningEvent
from .observability import get_logger, log_event

if TYPE_CHECKING:
    from collections.abc import Callable

    from ._domain.planning import ExecutionPlan

_LOGGER = get_logger(__name__)


class EventEmitter:
    """Assign monotonic sequence numbers and isolate callback invocation."""

    def __init__(self, callback: Callable[[ExecutionEvent], None] | None) -> None:
        self._callback = callback
        self._sequence = 0

    def emit_started(self) -> None:
        self._emit(Started(self._next()))

    def emit_completed(self) -> None:
        """Finish resolution while keeping callback errors fatal before return."""
        self._emit(Completed(self._next()))

    def emit_model_progress(
        self, stage: str, completed: int, total: int, chapter_id: str | None
    ) -> None:
        """Adapt serial model progress into the shared public stage events."""
        if completed == 0:
            self.emit_stage_started(stage)
        self.emit_progress(stage, completed, total, chapter_id)
        if completed == total:
            self.emit_stage_completed(stage)

    def emit_cast_resolved(self, plan: ExecutionPlan) -> None:
        """Publish the resolved cast before any worker exists."""
        self._emit(
            CastResolved(
                self._next(),
                "planning",
                plan.cast.narrator.id,
                plan.cast.unknown.id,
                tuple(sorted(plan.cast.assignments.items())),
            )
        )

    def emit_warning(
        self, stage: str, code: str, message: str, chapter_id: str | None = None
    ) -> None:
        """Report a condition worth knowing about that stops nothing."""
        self._emit(WarningEvent(self._next(), stage, code, message, chapter_id))

    def emit_stage_started(self, stage: str) -> None:
        log_event(
            _LOGGER,
            "execution_stage_started",
            context={"boundary": _stage_boundary(stage), "stage": stage},
        )
        self._emit(StageStarted(self._next(), stage))

    def emit_progress(
        self, stage: str, completed: int, total: int, chapter_id: str | None = None
    ) -> None:
        self._emit(StageProgress(self._next(), stage, completed, total, chapter_id))

    def emit_stage_completed(self, stage: str) -> None:
        self._emit(StageCompleted(self._next(), stage))
        log_event(
            _LOGGER,
            "execution_stage_completed",
            context={"boundary": _stage_boundary(stage), "stage": stage},
        )

    def emit_stage_completed_best_effort(self, stage: str) -> None:
        """Notify post-commit stage success without allowing commit revocation."""
        try:
            self._emit(StageCompleted(self._next(), stage))
        except RenderError:
            log_event(
                _LOGGER,
                "execution_callback_failed_post_commit",
                level=logging.WARNING,
                context={"boundary": "callback"},
            )
        log_event(
            _LOGGER,
            "execution_stage_completed",
            context={"boundary": _stage_boundary(stage), "stage": stage},
        )

    def emit_completed_best_effort(self) -> None:
        """Notify terminal success without allowing an observer to revoke commit."""
        try:
            self._emit(Completed(self._next()))
        except RenderError:
            log_event(
                _LOGGER,
                "execution_callback_failed_post_commit",
                level=logging.WARNING,
                context={"boundary": "callback"},
            )

    def _next(self) -> int:
        self._sequence += 1
        return self._sequence

    def _emit(self, event: ExecutionEvent) -> None:
        if self._callback is None:
            return
        try:
            self._callback(event)
        except Exception:  # noqa: BLE001 - sanitize arbitrary public callback.
            raise RenderError(ErrorCode.CALLBACK_FAILED) from None


def _stage_boundary(stage: str) -> str:
    """Map internal execution stages to their public logging boundaries."""
    return {
        "planning": "planning",
        "render": "rendering",
        "assembly": "encoding",
        "publication": "publication",
        "characters": "characters",
        "attribution": "attribution",
    }[stage]
