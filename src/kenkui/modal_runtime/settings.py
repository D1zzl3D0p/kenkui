"""Modal runtime settings derived from AppConfig/env."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModalRuntimeConfig:
    enabled: bool = False
    app_name: str = "kenkui"
    environment: str = ""
    tts_gpu: str = ""
    tts_cpu: int = 1
    tts_memory_mb: int = 4096
    tts_timeout_s: int = 3600
    nlp_cpu: int = 2
    nlp_memory_mb: int = 4096
    nlp_timeout_s: int = 1800
    artifact_backend: str = "modal_volume"
    artifact_volume: str = "kenkui-artifacts"
    keep_artifacts: bool = False

    @classmethod
    def from_app_config(cls, app_config: Any | None) -> ModalRuntimeConfig:
        if app_config is None:
            return cls()
        return cls(
            enabled=bool(getattr(app_config, "modal_enabled", False)),
            app_name=str(getattr(app_config, "modal_app_name", "kenkui") or "kenkui"),
            environment=str(getattr(app_config, "modal_environment", "") or ""),
            tts_gpu=str(getattr(app_config, "modal_tts_gpu", "") or ""),
            tts_cpu=int(getattr(app_config, "modal_tts_cpu", 1) or 1),
            tts_memory_mb=int(getattr(app_config, "modal_tts_memory_mb", 4096) or 4096),
            tts_timeout_s=int(getattr(app_config, "modal_tts_timeout_s", 3600) or 3600),
            nlp_cpu=int(getattr(app_config, "modal_nlp_cpu", 2) or 2),
            nlp_memory_mb=int(getattr(app_config, "modal_nlp_memory_mb", 4096) or 4096),
            nlp_timeout_s=int(getattr(app_config, "modal_nlp_timeout_s", 1800) or 1800),
            artifact_backend=str(
                getattr(app_config, "modal_artifact_backend", "modal_volume") or "modal_volume"
            ),
            artifact_volume=str(
                getattr(app_config, "modal_artifact_volume", "kenkui-artifacts")
                or "kenkui-artifacts"
            ),
            keep_artifacts=bool(getattr(app_config, "modal_keep_artifacts", False)),
        )


__all__ = ["ModalRuntimeConfig"]
