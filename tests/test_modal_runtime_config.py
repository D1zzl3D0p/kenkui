from __future__ import annotations

from kenkui.models import AppConfig


def test_modal_runtime_defaults_are_disabled():
    cfg = AppConfig.from_dict({})
    assert cfg.modal_enabled is False
    assert cfg.modal_app_name == "kenkui"
    assert cfg.modal_artifact_backend == "modal_volume"
    assert cfg.modal_keep_artifacts is False


def test_modal_runtime_config_round_trips_and_clamps():
    cfg = AppConfig.from_dict(
        {
            "modal_enabled": True,
            "modal_app_name": "custom-kenkui",
            "modal_environment": "prod",
            "modal_tts_gpu": "A10G",
            "modal_tts_cpu": 0,
            "modal_tts_memory_mb": -10,
            "modal_tts_timeout_s": 0,
            "modal_nlp_cpu": 3,
            "modal_nlp_memory_mb": 8192,
            "modal_nlp_timeout_s": 1800,
            "modal_artifact_backend": "local_filesystem",
            "modal_artifact_volume": "books",
            "modal_keep_artifacts": True,
        }
    )
    restored = AppConfig.from_dict(cfg.to_dict())

    assert restored.modal_enabled is True
    assert restored.modal_app_name == "custom-kenkui"
    assert restored.modal_environment == "prod"
    assert restored.modal_tts_gpu == "A10G"
    assert restored.modal_tts_cpu == 1
    assert restored.modal_tts_memory_mb == 512
    assert restored.modal_tts_timeout_s == 60
    assert restored.modal_nlp_cpu == 3
    assert restored.modal_nlp_memory_mb == 8192
    assert restored.modal_nlp_timeout_s == 1800
    assert restored.modal_artifact_backend == "local_filesystem"
    assert restored.modal_artifact_volume == "books"
    assert restored.modal_keep_artifacts is True
