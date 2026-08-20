"""Unloading reclaims assets and prunes orphaned engines; removing forgets."""
# ruff: noqa: D103

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from kenkui import (
    ErrorCode,
    VoiceError,
    add_voice,
    load_voice,
    remove_voice,
    unload_voice,
)
from kenkui.voices import provision
from kenkui.voices.manifest import ManifestStore

_RIGHTS: dict[str, object] = {
    "name": "My Narrator",
    "language": "english",
    "provenance": "recorded by me",
    "license_id": "proprietary",
    "commercial_use_allowed": True,
    "voice_rights": "owned outright",
}


@pytest.fixture(autouse=True)
def fake_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = tmp_path / "hub"
    hub.mkdir()

    def fetch(url: str) -> Path:
        path = hub / f"{hashlib.sha256(url.encode()).hexdigest()[:16]}.bin"
        if not path.exists():
            path.write_bytes(url.encode() * 8)
        return path

    monkeypatch.setattr(provision, "_fetch", fetch)


def _add_precompiled(tmp_path: Path, manifest: Path) -> Path:
    source = tmp_path / "mine.safetensors"
    source.write_bytes(b"already-compiled")
    add_voice(source, voice_id="mine", manifest=manifest, **_RIGHTS)  # type: ignore[arg-type]
    return source


def test_unload_reverts_to_registered(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)
    voice = unload_voice("eponine", manifest=manifest)
    assert voice.state == "registered"
    assert voice.engine is None


def test_unload_deletes_the_asset(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)
    _, voices = ManifestStore(manifest).read()
    asset = Path(voices["eponine"].asset_path or "")
    unload_voice("eponine", manifest=manifest)
    assert not asset.exists()


def test_unload_prunes_the_orphaned_engine(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)
    engines, _ = ManifestStore(manifest).read()
    model_root = Path(engines["english"].model_root)
    unload_voice("eponine", manifest=manifest)
    engines_after, _ = ManifestStore(manifest).read()
    assert engines_after == {}
    assert not model_root.exists()


def test_unload_keeps_an_engine_with_a_surviving_sibling(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)
    load_voice("alba", manifest=manifest)
    unload_voice("eponine", manifest=manifest)
    engines, voices = ManifestStore(manifest).read()
    assert set(engines) == {"english"}
    assert voices["alba"].state == "loaded"


def test_unload_retains_hand_entered_rights(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    source = _add_precompiled(tmp_path, manifest)
    load_voice("mine", manifest=manifest)
    unload_voice("mine", manifest=manifest)
    _, voices = ManifestStore(manifest).read()
    record = voices["mine"]
    assert record.state == "registered"
    assert record.provenance == _RIGHTS["provenance"]
    assert record.voice_rights == _RIGHTS["voice_rights"]
    assert record.source_path == str(source)


def test_unload_is_idempotent_on_a_registered_voice(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _add_precompiled(tmp_path, manifest)
    assert unload_voice("mine", manifest=manifest).state == "registered"


def test_remove_deletes_the_entry(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _add_precompiled(tmp_path, manifest)
    load_voice("mine", manifest=manifest)
    remove_voice("mine", manifest=manifest)
    _, voices = ManifestStore(manifest).read()
    assert "mine" not in voices


def test_remove_of_a_builtin_leaves_it_loadable(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)
    remove_voice("eponine", manifest=manifest)
    _, voices = ManifestStore(manifest).read()
    assert "eponine" not in voices
    assert load_voice("eponine", manifest=manifest).state == "loaded"


def test_remove_of_unknown_voice_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(VoiceError) as excinfo:
        remove_voice("nobody", manifest=tmp_path / "manifest.json")
    assert excinfo.value.code is ErrorCode.VOICE_UNKNOWN
