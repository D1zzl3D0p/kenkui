"""Execution provider boundary for queued TTS jobs."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from kenkui.models import QueueItem

logger = logging.getLogger(__name__)


@dataclass
class ExecutionOutcome:
    success: bool
    output_path: str = ""
    error_message: str = ""
    paused: bool = False
    remote_job_id: str = ""
    estimated_cost_usd: float | None = None
    actual_cost_usd: float | None = None
    artifact_uri: str = ""
    artifact_source: str = ""
    provider_status: str = ""


class TTSExecutionProvider(Protocol):
    def execute(
        self,
        *,
        item: QueueItem,
        cfg: Any,
        app_config: Any,
        progress_callback: Callable | None,
        metadata_callback: Callable | None,
        pause_check: Callable[[], bool] | None,
    ) -> ExecutionOutcome: ...

    def cancel(self, item: QueueItem) -> None: ...


class LocalTTSProvider:
    """Runs a TTS job in-process through AudioBuilder."""

    def execute(
        self,
        *,
        item: QueueItem,
        cfg: Any,
        app_config: Any,
        progress_callback: Callable | None,
        metadata_callback: Callable | None,
        pause_check: Callable[[], bool] | None,
    ) -> ExecutionOutcome:
        del item, app_config
        from kenkui.parsing import AudioBuilder

        if metadata_callback:
            metadata_callback(provider_status="running")
        try:
            builder = AudioBuilder(cfg, progress_callback=progress_callback)
            builder.pause_check = pause_check
            success = builder.run()
        except Exception as exc:
            logger.exception("Local TTS execution failed: %s", exc)
            return ExecutionOutcome(success=False, error_message=str(exc), provider_status="failed")

        if getattr(builder, "was_paused", False):
            return ExecutionOutcome(success=False, paused=True, provider_status="paused")
        if success:
            return ExecutionOutcome(success=True, output_path=str(cfg.output_path), provider_status="completed")
        return ExecutionOutcome(success=False, error_message="Conversion failed", provider_status="failed")

    def cancel(self, item: QueueItem) -> None:
        del item


_TTS_PROVIDERS: dict[str, Callable[[], TTSExecutionProvider]] = {
    "local": LocalTTSProvider,
}


def register_tts_execution_provider(mode: str, factory: Callable[[], TTSExecutionProvider]) -> None:
    """Register a non-local TTS execution provider such as a Modal backend."""
    if not mode.strip():
        raise ValueError("mode is required")
    _TTS_PROVIDERS[mode] = factory


def get_tts_execution_provider(item: QueueItem) -> TTSExecutionProvider:
    mode = item.job.tts_execution_mode.value
    factory = _TTS_PROVIDERS.get(mode)
    if factory is None:
        raise NotImplementedError(
            f"TTS execution mode {mode!r} is not registered. "
            "Install/register an execution extension before starting this job."
        )
    return factory()


__all__ = [
    "ExecutionOutcome",
    "LocalTTSProvider",
    "TTSExecutionProvider",
    "get_tts_execution_provider",
    "register_tts_execution_provider",
]

