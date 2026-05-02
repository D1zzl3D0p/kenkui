"""KenkuiModalGateway — the only Modal-SDK-coupled code in kenkui.

Loaded at runtime via:
  KENKUI_MODAL_GATEWAY=kenkui.modal.gateway:KenkuiModalGateway

TTS path: implements ModalGateway (estimate/submit/poll/cancel).
NLP path: implements NlpGateway (submit_nlp/submit_attribution/poll_progress/cancel).
Both paths poll R2 progress.json for status updates.
"""

from __future__ import annotations

import os
import time
from typing import Any

import modal  # type: ignore[import]

from ..server.tts_execution import ModalPollResult
from .storage import BotoStorageBackend, output_m4b_key, progress_key


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip() or default


class KenkuiModalGateway:
    """Unified gateway for dispatching kenkui pipeline stages to Modal."""

    def __init__(self):
        self._app_name = _env("KENKUI_MODAL_APP_NAME", "kenkui")
        self._environment = _env("KENKUI_MODAL_ENVIRONMENT", "main")
        self._storage = BotoStorageBackend()

    # ── Shared R2 polling ──────────────────────────────────────────────────

    def _poll_r2(self, job_id: str) -> ModalPollResult:
        try:
            data = self._storage.get_json(progress_key(job_id))
        except Exception:
            return ModalPollResult(status="running")

        status = data.get("status", "running")
        if status == "completed":
            artifact_key = data.get("artifact_key", output_m4b_key(job_id))
            return ModalPollResult(
                status="completed",
                progress=100.0,
                artifact_uri=f"r2://{os.environ.get('KENKUI_MODAL_BUCKET', '')}/{artifact_key}",
                artifact_source="r2",
                provider_status="completed",
            )
        if status == "failed":
            return ModalPollResult(
                status="failed",
                error_message=data.get("error", "Modal job failed"),
                provider_status="failed",
            )
        return ModalPollResult(
            status="running",
            progress=data.get("progress"),
            current_chapter=data.get("current_chapter", ""),
            eta_seconds=data.get("eta_seconds"),
            provider_status="running",
        )

    def _download_artifact(self, job_id: str, artifact_key: str) -> bytes:
        return self._storage.get_bytes(artifact_key)

    # ── ModalGateway protocol (TTS compat) ────────────────────────────────

    def estimate(self, payload: dict[str, Any]) -> float | None:
        return None

    def submit(self, payload: dict[str, Any]) -> str:
        """Submit a TTS job (tts_inference → audio_stitch chain)."""
        from .functions.tts import tts_inference

        job_id: str = payload["job_id"]
        call = tts_inference.spawn(payload)
        call_id = f"tts:{job_id}:{call.object_id}"
        return call_id

    def poll(self, remote_job_id: str) -> ModalPollResult:
        job_id = remote_job_id.split(":")[1] if ":" in remote_job_id else remote_job_id
        result = self._poll_r2(job_id)
        if result.status == "completed":
            artifact_key = output_m4b_key(job_id)
            result.artifact_bytes = self._download_artifact(job_id, artifact_key)
            result.artifact_source = "r2"
        return result

    def cancel(self, remote_job_id: str) -> None:
        try:
            parts = remote_job_id.split(":")
            call_id = parts[-1] if parts else remote_job_id
            fc = modal.functions.FunctionCall.from_id(call_id)
            fc.cancel()
        except Exception:
            pass

    # ── NlpGateway interface ───────────────────────────────────────────────

    def submit_nlp(self, job_id: str, payload: dict) -> str:
        from .functions.nlp import nlp_entity_clustering

        call = nlp_entity_clustering.spawn(payload)
        return f"nlp:{job_id}:{call.object_id}"

    def submit_attribution(self, job_id: str, payload: dict) -> str:
        from .functions.attribution import quote_attribution

        call = quote_attribution.spawn(payload)
        return f"attr:{job_id}:{call.object_id}"

    def poll_progress(self, job_id: str) -> ModalPollResult:
        return self._poll_r2(job_id)
