"""Managed manifest round-trips, writes atomically, and stays owner-private."""
# ruff: noqa: D103, PLR2004, SLF001, TC003

from __future__ import annotations

import json
import stat
from pathlib import Path

from kenkui._tts import production
from kenkui.voices.manifest import (
    EngineRecord,
    FileRecord,
    ManifestStore,
    VoiceRecord,
    default_manifest_path,
)

_ENGINE_SIZE = 225


def _engine() -> EngineRecord:
    return EngineRecord(
        id="english",
        language="english",
        model_root="/models/english",
        config_path="/models/english/english.yaml",
        model_revision="revision-1",
        package_version="2.1.0",
        files=(FileRecord("model.safetensors", _ENGINE_SIZE, "a" * 64),),
        sample_rate_hz=24000,
        device="cpu",
        timeout_seconds=300.0,
        cloning_capable=False,
    )


def _voice(**overrides: object) -> VoiceRecord:
    base: dict[str, object] = {
        "id": "eponine",
        "variety": "built-in",
        "state": "loaded",
        "name": "Eponine",
        "enabled": True,
        "language": "english",
        "engine_id": "english",
        "provenance": "hf://kyutai/tts-voices/vctk/p262_023_enhanced.wav",
        "license_id": "CC-BY-4.0",
        "commercial_use_allowed": False,
        "voice_rights": "review required",
        "asset_path": "/cache/voices/english/eponine.safetensors",
        "asset_sha256": "b" * 64,
        "compatible_model_revisions": ("revision-1",),
    }
    return VoiceRecord(**(base | overrides))  # type: ignore[arg-type]


def test_round_trip_preserves_records(tmp_path: Path) -> None:
    store = ManifestStore(tmp_path / "manifest.json")
    store.write({"english": _engine()}, {"eponine": _voice()})
    engines, voices = store.read()
    assert engines["english"] == _engine()
    assert voices["eponine"] == _voice()


def test_read_of_absent_manifest_is_empty(tmp_path: Path) -> None:
    engines, voices = ManifestStore(tmp_path / "none.json").read()
    assert engines == {}
    assert voices == {}


def test_write_creates_owner_private_file_and_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "manifest.json"
    ManifestStore(path).write({"english": _engine()}, {"eponine": _voice()})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_written_schema_version_is_v2(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    ManifestStore(path).write({"english": _engine()}, {"eponine": _voice()})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "kenkui-pocket-production-v2"
    assert set(payload) == {"schema_version", "engines", "voices"}


def test_registered_record_omits_asset_keys(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    registered = _voice(
        state="registered",
        variety="wav",
        asset_path=None,
        asset_sha256=None,
        compatible_model_revisions=(),
        source_path="/voices/mine.wav",
        source_sha256="c" * 64,
    )
    ManifestStore(path).write({}, {"eponine": registered})
    entry = json.loads(path.read_text(encoding="utf-8"))["voices"]["eponine"]
    assert "asset_path" not in entry
    assert "asset_sha256" not in entry
    assert "compatible_model_revisions" not in entry
    assert entry["source_path"] == "/voices/mine.wav"


def test_write_leaves_no_temporary_files(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    store = ManifestStore(path)
    store.write({"english": _engine()}, {"eponine": _voice()})
    store.write({"english": _engine()}, {})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest.json"]


def test_default_manifest_path_is_under_the_versioned_cache() -> None:
    path = default_manifest_path()
    assert path.name == "manifest.json"
    assert path.parent.name == "v1"
    assert path.parent.parent.name == "kenkui"


def test_lock_is_reentrant_across_sequential_uses(tmp_path: Path) -> None:
    store = ManifestStore(tmp_path / "manifest.json")
    with store.lock():
        store.write({}, {})
    with store.lock():
        store.write({}, {})
    assert (tmp_path / "manifest.json").exists()


def test_writer_output_is_accepted_by_the_strict_reader(tmp_path: Path) -> None:
    """The convenience writer and the paranoid reader must agree on schema."""
    path = tmp_path / "manifest.json"
    ManifestStore(path).write({"english": _engine()}, {"eponine": _voice()})
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = production._object(payload, {"schema_version", "engines", "voices"})
    voice_data, engine = production._select(root, "eponine")
    assert engine["language"] == "english"
    assert voice_data["state"] == "loaded"
