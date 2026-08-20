"""Listing unions catalog and manifest and never hashes."""
# ruff: noqa: D103

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from kenkui import add_voice, list_voices, load_voice
from kenkui.voices import provision
from kenkui.voices.manifest import ManifestStore
from kenkui.voices.registry import CATALOG

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


def test_empty_manifest_lists_the_whole_catalog(tmp_path: Path) -> None:
    voices = list_voices(manifest=tmp_path / "manifest.json")
    assert {v.id for v in voices} == set(CATALOG)
    assert all(v.state == "registered" for v in voices)


def test_results_are_sorted_by_id(tmp_path: Path) -> None:
    voices = list_voices(manifest=tmp_path / "manifest.json")
    assert [v.id for v in voices] == sorted(v.id for v in voices)


def test_loaded_voice_reports_engine_and_size(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)
    voice = next(v for v in list_voices(manifest=manifest) if v.id == "eponine")
    assert voice.state == "loaded"
    assert voice.engine is not None
    assert voice.engine.size_bytes > 0
    assert voice.asset_bytes is not None


def test_local_voice_appears_alongside_the_catalog(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    source = tmp_path / "mine.safetensors"
    source.write_bytes(b"already-compiled")
    add_voice(source, voice_id="mine", manifest=manifest, **_RIGHTS)  # type: ignore[arg-type]
    ids = {v.id for v in list_voices(manifest=manifest)}
    assert ids == set(CATALOG) | {"mine"}


def test_deleted_asset_is_reported_missing(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)
    _, voices = ManifestStore(manifest).read()
    Path(voices["eponine"].asset_path or "").unlink()
    voice = next(v for v in list_voices(manifest=manifest) if v.id == "eponine")
    assert voice.state == "missing"


def test_missing_state_is_never_persisted(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)
    _, voices = ManifestStore(manifest).read()
    Path(voices["eponine"].asset_path or "").unlink()
    list_voices(manifest=manifest)
    _, after = ManifestStore(manifest).read()
    assert after["eponine"].state == "loaded"


def test_listing_does_not_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)

    def explode(path: Path) -> str:
        _ = path
        message = "list_voices must not hash"
        raise AssertionError(message)

    monkeypatch.setattr(provision, "_sha256", explode)
    assert list_voices(manifest=manifest)


def test_engine_dedup_across_sibling_voices(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)
    load_voice("alba", manifest=manifest)
    engines = {v.engine for v in list_voices(manifest=manifest) if v.state == "loaded"}
    assert len(engines) == 1
