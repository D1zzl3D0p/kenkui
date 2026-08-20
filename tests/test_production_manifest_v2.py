"""Schema v2 parsing: engines map, variety, state, and manifest resolution."""
# ruff: noqa: D103, TC003

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kenkui import ErrorCode, ModelError, RenderError, VoiceError
from kenkui._tts import production
from kenkui.voices import manifest as manifest_module


def _payload(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "model"
    root.mkdir(mode=0o700, exist_ok=True)
    config = root / "english.yaml"
    config.write_text("model: local\n", encoding="utf-8")
    asset = tmp_path / "eponine.safetensors"
    asset.write_bytes(b"fixture")
    return {
        "schema_version": "kenkui-pocket-production-v2",
        "engines": {
            "english": {
                "language": "english",
                "model_root": str(root),
                "config_path": str(config),
                "model_revision": "revision-1",
                "package_version": "2.1.0",
                "files": [
                    {
                        "relative_path": "english.yaml",
                        "size": config.stat().st_size,
                        "sha256": "a" * 64,
                    }
                ],
                "sample_rate_hz": 24000,
                "device": "cpu",
                "timeout_seconds": 30.0,
                "cloning_capable": False,
            }
        },
        "voices": {
            "eponine": {
                "variety": "built-in",
                "state": "loaded",
                "name": "Eponine",
                "enabled": True,
                "language": "english",
                "engine_id": "english",
                "provenance": "kyutai catalog",
                "license_id": "CC-BY-4.0",
                "commercial_use_allowed": False,
                "voice_rights": "review required",
                "asset_path": str(asset),
                "asset_sha256": "b" * 64,
                "compatible_model_revisions": ["revision-1"],
            }
        },
    }


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)
    return path


def _activate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: object) -> None:
    monkeypatch.setenv("KENKUI_POCKET_MANIFEST", str(_write(tmp_path, payload)))
    production.production_bindings_from_environment("eponine")


def test_schema_version_is_v2() -> None:
    assert production.MANIFEST_SCHEMA_VERSION == "kenkui-pocket-production-v2"


def test_reader_and_writer_agree_on_schema_version() -> None:
    assert manifest_module.MANIFEST_SCHEMA_VERSION == production.MANIFEST_SCHEMA_VERSION


def test_v1_schema_version_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _payload(tmp_path)
    payload["schema_version"] = "kenkui-pocket-production-v1"
    with pytest.raises(ModelError) as excinfo:
        _activate(tmp_path, monkeypatch, payload)
    assert excinfo.value.code is ErrorCode.POCKET_MODEL_INVALID


def test_unknown_engine_id_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _payload(tmp_path)
    payload["voices"]["eponine"]["engine_id"] = "italian"
    with pytest.raises(VoiceError) as excinfo:
        _activate(tmp_path, monkeypatch, payload)
    assert excinfo.value.code is ErrorCode.VOICE_UNRESOLVED


def test_unknown_variety_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _payload(tmp_path)
    payload["voices"]["eponine"]["variety"] = "magic"
    with pytest.raises(VoiceError) as excinfo:
        _activate(tmp_path, monkeypatch, payload)
    assert excinfo.value.code is ErrorCode.VOICE_VARIETY_INVALID


def test_registered_voice_is_not_renderable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _payload(tmp_path)
    voice = payload["voices"]["eponine"]
    voice["state"] = "registered"
    voice.pop("asset_path")
    voice.pop("asset_sha256")
    voice.pop("compatible_model_revisions")
    voice["source_path"] = str(tmp_path / "eponine.safetensors")
    voice["source_sha256"] = "c" * 64
    with pytest.raises(VoiceError) as excinfo:
        _activate(tmp_path, monkeypatch, payload)
    assert excinfo.value.code is ErrorCode.VOICE_NOT_PROVISIONED


def test_wav_voice_requires_cloning_capable_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _payload(tmp_path)
    payload["voices"]["eponine"]["variety"] = "wav"
    payload["voices"]["eponine"]["source_path"] = str(tmp_path / "v.wav")
    payload["voices"]["eponine"]["source_sha256"] = "d" * 64
    with pytest.raises(VoiceError) as excinfo:
        _activate(tmp_path, monkeypatch, payload)
    assert excinfo.value.code is ErrorCode.ENGINE_NOT_CLONING_CAPABLE


def test_unknown_key_in_voice_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _payload(tmp_path)
    payload["voices"]["eponine"]["surprise"] = True
    with pytest.raises(VoiceError) as excinfo:
        _activate(tmp_path, monkeypatch, payload)
    assert excinfo.value.code is ErrorCode.VOICE_PROVENANCE_REQUIRED


def test_missing_manifest_everywhere_is_renderer_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("KENKUI_POCKET_MANIFEST", raising=False)
    monkeypatch.setattr(
        manifest_module, "default_manifest_path", lambda: tmp_path / "absent.json"
    )
    with pytest.raises(RenderError) as excinfo:
        production.production_bindings_from_environment("eponine")
    assert excinfo.value.code is ErrorCode.RENDERER_UNAVAILABLE


def test_default_path_is_used_when_env_is_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(tmp_path, _payload(tmp_path))
    monkeypatch.delenv("KENKUI_POCKET_MANIFEST", raising=False)
    monkeypatch.setattr(manifest_module, "default_manifest_path", lambda: path)
    assert production.resolve_manifest_path() == path


def test_env_override_wins_over_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = _write(tmp_path, _payload(tmp_path))
    monkeypatch.setenv("KENKUI_POCKET_MANIFEST", str(override))
    monkeypatch.setattr(
        manifest_module, "default_manifest_path", lambda: tmp_path / "other.json"
    )
    assert production.resolve_manifest_path() == override
