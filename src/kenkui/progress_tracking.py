"""Progress tracking and error collection for chapter synthesis.

Extracted from ``AudioBuilder._process_chapters`` (QP Task 8). This class owns
the mutable orchestration state that accumulates as worker processes report
progress over the multiprocessing queue: per-worker state, running unit/batch
counters, collected worker errors, and a bounded log ring-buffer.

Behaviour is byte-identical to the previous inline implementation: every branch
emits the same ``ProgressEvent`` payloads in the same order. The caller supplies
an ``emit`` callable (``AudioBuilder._emit_progress``) so book-hash/provider/model
context is still resolved on the builder at emit time.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from .progress import ChapterProgress

logger = logging.getLogger(__name__)

EmitFn = Callable[..., None]


class ChapterProgressTracker:
    """Accumulates worker progress facts and drives progress-event emission."""

    def __init__(self, emit: EmitFn, total_chars: float, total_chapters: int = 0) -> None:
        self._emit = emit
        self._total_chars = total_chars
        self._total_chapters = total_chapters
        self._started_indices: set[int] = set()
        self.worker_state: dict = {}
        self.worker_errors: list[dict] = []
        self.worker_logs: list[str] = []
        self.completed_batches = 0
        self.completed_tts_units = 0
        self.current_chapter = ""

    def active_chapter_progress(self) -> tuple[ChapterProgress, ...]:
        return tuple(
            ChapterProgress(
                index=int(state.get("index", 0)),
                title=str(state.get("title", "")),
                completed_units=float(state.get("current", 0)),
                total_units=float(state.get("total", 0)),
                status=str(state.get("status", "advanced")),  # type: ignore[arg-type]
            )
            for state in sorted(
                self.worker_state.values(), key=lambda item: int(item.get("index", 0))
            )
        )

    def _emit_tts(self, status: str, message: str) -> None:
        self._emit(
            "tts_synthesis",
            status,
            message,
            completed_units=self.completed_tts_units,
            total_units=self._total_chars,
            unit="chars",
            active_chapters=self.active_chapter_progress(),
            total_chapters=self._total_chapters,
            chapter_ordinal=len(self._started_indices),
        )

    def process_message(self, msg) -> None:
        """Apply one worker queue message, mutating state and emitting progress.

        Malformed messages raise ``IndexError``/``KeyError``/``ValueError``/
        ``TypeError`` for the caller to catch and abort the drain, exactly as
        the previous inline loop relied upon.
        """
        event, pid = msg[0], msg[1]
        if event == "START":
            index = msg[6] if len(msg) > 6 else 0
            self._started_indices.add(int(index))
            self.worker_state[pid] = {
                "title": msg[2],
                "total": msg[3],
                "current": 0,
                "total_chars": msg[4] if len(msg) > 4 else 0,
                "is_first": msg[5] if len(msg) > 5 else False,
                "index": msg[6] if len(msg) > 6 else 0,
                "status": "started",
            }
            self.current_chapter = self.worker_state[pid].get("title", "")
            self._emit_tts("message", self.current_chapter)
        elif event == "UPDATE":
            chars = msg[5] if len(msg) > 5 else 0
            self.completed_batches += msg[2]
            self.completed_tts_units = min(
                self._total_chars,
                self.completed_tts_units + max(0, chars),
            )
            if pid in self.worker_state:
                self.worker_state[pid]["current"] += msg[2]
                self.worker_state[pid]["status"] = "advanced"
                self.current_chapter = self.worker_state[pid].get("title", "")
            self._emit_tts("advanced", self.current_chapter)
        elif event == "DONE":
            if pid in self.worker_state:
                self.worker_state[pid]["status"] = "completed"
                self.worker_state[pid]["current"] = self.worker_state[pid].get("total", 0)
                self._emit_tts("advanced", self.worker_state[pid].get("title", ""))
                del self.worker_state[pid]
        elif event == "ERROR":
            if pid in self.worker_state:
                self.worker_state[pid]["status"] = "failed"
            self._emit_tts("failed", msg[3])
            self.worker_errors.append(
                {
                    "pid": pid,
                    "chapter": msg[2],
                    "message": msg[3],
                    "traceback": msg[4],
                }
            )
        elif event == "LOG":
            self.worker_logs.append(f"[{pid}] {msg[2]}")
            if len(self.worker_logs) > 20:
                self.worker_logs.pop(0)

    def finalize_completed(self) -> None:
        """Mark all still-active workers completed, emit, and clear state."""
        for state in tuple(self.worker_state.values()):
            state["status"] = "completed"
            state["current"] = state.get("total", 0)
            self._emit_tts("advanced", state.get("title", ""))
        self.worker_state.clear()

    def log_errors(self) -> None:
        """Log any collected worker errors (called from the cleanup path)."""
        if self.worker_errors:
            logger.error("Worker errors encountered:")
            for err in self.worker_errors:
                logger.error("- PID %s %s: %s", err["pid"], err["chapter"], err["message"])
                tb = err.get("traceback", "")
                if tb:
                    logger.error("%s", tb)


__all__ = ["ChapterProgressTracker"]
