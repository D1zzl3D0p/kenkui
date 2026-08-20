"""Built-in voices provision an engine and an embedding, idempotently."""
# ruff: noqa: D103

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from kenkui import ErrorCode, VoiceError, load_voice
from kenkui.voices import provision
from kenkui.voices.manifest import ManifestStore


@pytest.fixture
def fake_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Serve every remote URL from local bytes and count the fetches."""
    hub = tmp_path / "hub"
    hub.mkdir()
    calls: dict[str, int] = {}

    def fetch(url: str) -> Path:
        calls[url] = calls.get(url, 0) + 1
        name = hashlib.sha256(url.encode()).hexdigest()[:16]
        suffix = ".safetensors" if ".safetensors" in url else ".bin"
        path = hub / f"{name}{suffix}"
        if not path.exists():
            path.write_bytes(url.encode() * 8)
        return path

    monkeypatch.setattr(provision, "_fetch", fetch)
    return calls


@pytest.mark.usefixtures("fake_hub")
def test_loads_a_builtin_voice(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    voice = load_voice("eponine", manifest=manifest)
    assert voice.state == "loaded"
    assert voice.variety == "built-in"
    assert voice.engine is not None
    assert voice.engine.language == "english"
    assert voice.asset_bytes is not None
    assert voice.asset_bytes > 0


@pytest.mark.usefixtures("fake_hub")
def test_manifest_records_engine_and_voice(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)
    engines, voices = ManifestStore(manifest).read()
    assert set(engines) == {"english"}
    assert engines["english"].cloning_capable is False
    assert engines["english"].files
    record = voices["eponine"]
    assert record.state == "loaded"
    assert record.asset_path is not None
    assert Path(record.asset_path).is_file()
    assert record.compatible_model_revisions


@pytest.mark.usefixtures("fake_hub")
def test_asset_hash_matches_downloaded_bytes(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)
    _, voices = ManifestStore(manifest).read()
    asset = Path(voices["eponine"].asset_path or "")
    expected = hashlib.sha256(asset.read_bytes()).hexdigest()
    assert voices["eponine"].asset_sha256 == expected


def test_second_load_performs_no_fetches(
    tmp_path: Path, fake_hub: dict[str, int]
) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)
    before = sum(fake_hub.values())
    load_voice("eponine", manifest=manifest)
    assert sum(fake_hub.values()) == before


@pytest.mark.usefixtures("fake_hub")
def test_sibling_voice_reuses_the_engine(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)
    load_voice("alba", manifest=manifest)
    engines, voices = ManifestStore(manifest).read()
    assert set(engines) == {"english"}
    assert set(voices) == {"eponine", "alba"}


@pytest.mark.usefixtures("fake_hub")
def test_written_config_has_no_remote_references(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)
    engines, _ = ManifestStore(manifest).read()
    text = Path(engines["english"].config_path).read_text(encoding="utf-8")
    assert "hf://" not in text
    assert "http://" not in text
    assert "https://" not in text


def test_unknown_voice_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(VoiceError) as excinfo:
        load_voice("nobody", manifest=tmp_path / "manifest.json")
    assert excinfo.value.code is ErrorCode.VOICE_UNKNOWN


@pytest.mark.usefixtures("fake_hub")
def test_missing_asset_is_repaired(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    load_voice("eponine", manifest=manifest)
    _, voices = ManifestStore(manifest).read()
    Path(voices["eponine"].asset_path or "").unlink()
    load_voice("eponine", manifest=manifest)
    _, repaired = ManifestStore(manifest).read()
    assert Path(repaired["eponine"].asset_path or "").is_file()
