"""Modal-backed TTS execution provider.

The provider is intentionally client-injected so unit tests and local installs do
not require Modal credentials. The default client is implemented later as the
actual Modal SDK adapter.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from kenkui.models import QueueItem
from kenkui.services.execution_service import ExecutionOutcome, TTSExecutionProvider

from .errors import ModalRuntimeUnavailableError
from .settings import ModalRuntimeConfig


class RemoteTTSClient(Protocol):
    def execute(
        self,
        *,
        item: QueueItem,
        cfg: Any,
        app_config: Any,
        progress_callback: Callable | None,
        metadata_callback: Callable | None,
        pause_check: Callable[[], bool] | None,
        cancel_check: Callable[[], bool] | None,
        runtime_config: ModalRuntimeConfig,
    ) -> ExecutionOutcome: ...

    def cancel(self, item: QueueItem, runtime_config: ModalRuntimeConfig) -> None: ...


class ModalTTSProvider(TTSExecutionProvider):
    """Runs TTS jobs through an injected Modal remote client."""

    def __init__(self, *, app_config: Any | None = None, client: RemoteTTSClient | None = None) -> None:
        self._app_config = app_config
        self._runtime_config = ModalRuntimeConfig.from_app_config(app_config)
        self._client = client

    def execute(
        self,
        *,
        item: QueueItem,
        cfg: Any,
        app_config: Any,
        progress_callback: Callable | None,
        metadata_callback: Callable | None,
        pause_check: Callable[[], bool] | None,
        cancel_check: Callable[[], bool] | None,
    ) -> ExecutionOutcome:
        if metadata_callback:
            metadata_callback(provider_status="queued", execution_provider="modal")
        if self._client is None:
            return ExecutionOutcome(
                success=False,
                error_message=(
                    "Modal TTS runtime is registered but no remote client is configured yet."
                ),
                provider_status="failed",
            )
        try:
            return self._client.execute(
                item=item,
                cfg=cfg,
                app_config=app_config or self._app_config,
                progress_callback=progress_callback,
                metadata_callback=metadata_callback,
                pause_check=pause_check,
                cancel_check=cancel_check,
                runtime_config=self._runtime_config,
            )
        except ModalRuntimeUnavailableError as exc:
            return ExecutionOutcome(success=False, error_message=str(exc), provider_status="failed")

    def cancel(self, item: QueueItem) -> None:
        if self._client is not None:
            self._client.cancel(item, self._runtime_config)


__all__ = ["ModalTTSProvider", "RemoteTTSClient"]
