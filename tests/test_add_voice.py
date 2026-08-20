"""Registering a local voice records rights without touching the network."""
# ruff: noqa: D103, TC003

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from kenkui import ErrorCode, VoiceError, add_voice
from kenkui.voices.manifest import ManifestStore

_RIGHTS: dict[str, object] = {
    "name": "My Narrator",
    "language": "english",
    "provenance": "recorded by me, 2026-08-19",
    "license_id": "proprietary",
    "commercial_use_allowed": True,
    "voice_rights": "owned outright",
}


def _wav(tmp_path: Path) -> Path:
    path = tmp_path / "mine.wav"
    path.write_bytes(b"RIFF0000WAVEfmt ")
    return path


def test_registers_a_wav_with_source_hash(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    source = _wav(tmp_path)
    voice = add_voice(source, voice_id="mine", manifest=manifest, **_RIGHTS)  # type: ignore[arg-type]
    assert voice.variety == "wav"
    assert voice.state == "registered"
    _, voices = ManifestStore(manifest).read()
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    assert voices["mine"].source_sha256 == expected
    assert voices["mine"].asset_path is None


def test_safetensors_registers_as_pre_compiled(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    source = tmp_path / "mine.safetensors"
    source.write_bytes(b'{"__metadata__":{}}')
    voice = add_voice(source, voice_id="mine", manifest=manifest, **_RIGHTS)  # type: ignore[arg-type]
    assert voice.variety == "pre-compiled"


def test_rights_are_preserved_verbatim(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    add_voice(_wav(tmp_path), voice_id="mine", manifest=manifest, **_RIGHTS)  # type: ignore[arg-type]
    _, voices = ManifestStore(manifest).read()
    record = voices["mine"]
    assert record.provenance == _RIGHTS["provenance"]
    assert record.license_id == _RIGHTS["license_id"]
    assert record.voice_rights == _RIGHTS["voice_rights"]
    assert record.commercial_use_allowed is True


def test_catalog_name_collision_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(VoiceError) as excinfo:
        add_voice(
            _wav(tmp_path),
            voice_id="eponine",
            manifest=tmp_path / "manifest.json",
            **_RIGHTS,  # type: ignore[arg-type]
        )
    assert excinfo.value.code is ErrorCode.VOICE_VARIETY_INVALID


def test_unsupported_suffix_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "mine.mp3"
    source.write_bytes(b"nope")
    with pytest.raises(VoiceError) as excinfo:
        add_voice(
            source,
            voice_id="mine",
            manifest=tmp_path / "manifest.json",
            **_RIGHTS,  # type: ignore[arg-type]
        )
    assert excinfo.value.code is ErrorCode.VOICE_VARIETY_INVALID


def test_absent_source_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(VoiceError) as excinfo:
        add_voice(
            tmp_path / "nothing.wav",
            voice_id="mine",
            manifest=tmp_path / "manifest.json",
            **_RIGHTS,  # type: ignore[arg-type]
        )
    assert excinfo.value.code is ErrorCode.POCKET_VOICE_INVALID


def test_existing_entries_are_preserved(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    add_voice(_wav(tmp_path), voice_id="one", manifest=manifest, **_RIGHTS)  # type: ignore[arg-type]
    second = tmp_path / "two.wav"
    second.write_bytes(b"RIFF1111WAVEfmt ")
    add_voice(second, voice_id="two", manifest=manifest, **_RIGHTS)  # type: ignore[arg-type]
    _, voices = ManifestStore(manifest).read()
    assert set(voices) == {"one", "two"}
