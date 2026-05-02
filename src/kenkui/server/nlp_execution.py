"""Execution-provider boundary for NLP (entity clustering + attribution) jobs."""

from __future__ import annotations

import importlib
import inspect
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..modal.storage import attribution_chapters_key, nlp_roster_key
from .tts_execution import ModalPollResult


class NlpGateway(Protocol):
    def submit_nlp(self, job_id: str, payload: dict[str, Any]) -> str: ...

    def submit_attribution(self, job_id: str, payload: dict[str, Any]) -> str: ...

    def poll_progress(self, job_id: str) -> ModalPollResult: ...

    def cancel(self, call_id: str) -> None: ...


@dataclass
class NlpStageResult:
    success: bool
    roster_local_path: Path | None = None
    chapters_local_path: Path | None = None
    error_message: str = ""


class EnvNlpGateway:
    """Load an NLP transport object from ``KENKUI_MODAL_GATEWAY`` on demand."""

    def __init__(self):
        gateway_spec = os.environ.get("KENKUI_MODAL_GATEWAY", "").strip()
        if not gateway_spec:
            raise RuntimeError(
                "Modal NLP execution is configured, but KENKUI_MODAL_GATEWAY is not set."
            )

        module_name, sep, attr_name = gateway_spec.partition(":")
        if not sep:
            raise RuntimeError(
                "KENKUI_MODAL_GATEWAY must be in 'module:attribute' form."
            )

        module = importlib.import_module(module_name)
        target = getattr(module, attr_name)
        self._gateway = target() if inspect.isclass(target) else target

        for method_name in ("submit_nlp", "submit_attribution", "poll_progress", "cancel"):
            if not hasattr(self._gateway, method_name):
                raise RuntimeError(
                    f"Configured NLP gateway is missing required method '{method_name}'."
                )

    def submit_nlp(self, job_id: str, payload: dict[str, Any]) -> str:
        return self._gateway.submit_nlp(job_id, payload)

    def submit_attribution(self, job_id: str, payload: dict[str, Any]) -> str:
        return self._gateway.submit_attribution(job_id, payload)

    def poll_progress(self, job_id: str) -> ModalPollResult:
        result = self._gateway.poll_progress(job_id)
        if isinstance(result, ModalPollResult):
            return result
        if isinstance(result, dict):
            return ModalPollResult(**result)
        raise RuntimeError("NLP gateway.poll_progress() must return ModalPollResult or dict.")

    def cancel(self, call_id: str) -> None:
        self._gateway.cancel(call_id)


class NlpModalProvider:
    def __init__(self, gateway: NlpGateway | None = None, storage=None, poll_interval: float = 2.0):
        self._gateway = gateway or EnvNlpGateway()
        self._storage = storage
        self._poll_interval = poll_interval

    def run_nlp_stage(
        self,
        job_id: str,
        payload: dict[str, Any],
        progress_callback,
    ) -> NlpStageResult:
        self._gateway.submit_nlp(job_id, payload)

        while True:
            state = self._gateway.poll_progress(job_id)

            if progress_callback and state.progress is not None:
                progress_callback(state.progress, state.current_chapter, state.eta_seconds or 0)

            if state.status == "completed":
                roster_data = self._storage.get_json(nlp_roster_key(job_id))
                roster_path = Path(tempfile.mktemp(suffix="_nlp_roster.json"))
                roster_path.write_text(json.dumps(roster_data), encoding="utf-8")
                return NlpStageResult(success=True, roster_local_path=roster_path)

            if state.status in {"failed", "cancelled"}:
                return NlpStageResult(
                    success=False,
                    error_message=state.error_message or "NLP stage failed",
                )

            time.sleep(self._poll_interval)

    def run_attribution_stage(
        self,
        job_id: str,
        payload: dict[str, Any],
        progress_callback,
        roster_local_path: Path | None = None,
    ) -> NlpStageResult:
        if roster_local_path is not None:
            roster_data = json.loads(roster_local_path.read_text(encoding="utf-8"))
            self._storage.put_json(nlp_roster_key(job_id), roster_data)

        self._gateway.submit_attribution(job_id, payload)

        while True:
            state = self._gateway.poll_progress(job_id)

            if progress_callback and state.progress is not None:
                progress_callback(state.progress, state.current_chapter, state.eta_seconds or 0)

            if state.status == "completed":
                chapters_data = self._storage.get_json(attribution_chapters_key(job_id))
                chapters_path = Path(tempfile.mktemp(suffix="_attribution_chapters.json"))
                chapters_path.write_text(json.dumps(chapters_data), encoding="utf-8")
                return NlpStageResult(success=True, chapters_local_path=chapters_path)

            if state.status in {"failed", "cancelled"}:
                return NlpStageResult(
                    success=False,
                    error_message=state.error_message or "Attribution stage failed",
                )

            time.sleep(self._poll_interval)
