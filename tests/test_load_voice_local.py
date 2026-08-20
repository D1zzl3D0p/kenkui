"""Local voices compile at provision time; the render path never encodes audio."""
# ruff: noqa: D103, SLF001

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from kenkui import ErrorCode, VoiceError, add_voice, load_voice
from kenkui.voices import provision
from kenkui.voices.manifest import EngineRecord, FileRecord, ManifestStore

_RIGHTS: dict[str, object] = {
    "name": "My Narrator",
    "language": "english",
    "provenance": "recorded by me",
    "license_id": "proprietary",
    "commercial_use_allowed": True,
    "voice_rights": "owned outright",
}


@pytest.fixture
def stub_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Avoid real downloads and real model loads; record cloning requests."""
    hub = tmp_path / "hub"
    hub.mkdir()
    requested: list[bool] = []

    def fetch(url: str) -> Path:
        path = hub / f"{hashlib.sha256(url.encode()).hexdigest()[:16]}.bin"
        if not path.exists():
            path.write_bytes(url.encode() * 8)
        return path

    real_provision = provision._provision_engine

    def provision_engine(language: str, *, cloning: bool, root: Path) -> EngineRecord:
        requested.append(cloning)
        return real_provision(language, cloning=cloning, root=root)

    def compile_wav(source: Path, engine: EngineRecord, destination: Path) -> None:
        _ = engine
        destination.write_bytes(b"compiled:" + source.read_bytes())

    monkeypatch.setattr(provision, "_fetch", fetch)
    monkeypatch.setattr(provision, "_provision_engine", provision_engine)
    monkeypatch.setattr(provision, "_compile_wav", compile_wav)
    return requested


def _wav(tmp_path: Path) -> Path:
    source = tmp_path / "mine.wav"
    source.write_bytes(b"RIFF0000WAVEfmt ")
    return source


def _register_wav(tmp_path: Path, manifest: Path) -> Path:
    source = _wav(tmp_path)
    add_voice(source, voice_id="mine", manifest=manifest, **_RIGHTS)  # type: ignore[arg-type]
    return source


@pytest.mark.usefixtures("stub_engine")
def test_wav_compiles_to_safetensors(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _register_wav(tmp_path, manifest)
    voice = load_voice("mine", manifest=manifest)
    assert voice.state == "loaded"
    _, voices = ManifestStore(manifest).read()
    asset = Path(voices["mine"].asset_path or "")
    assert asset.suffix == ".safetensors"
    assert asset.read_bytes().startswith(b"compiled:")


def test_wav_requests_a_cloning_capable_engine(
    tmp_path: Path, stub_engine: list[bool]
) -> None:
    manifest = tmp_path / "manifest.json"
    _register_wav(tmp_path, manifest)
    load_voice("mine", manifest=manifest)
    assert stub_engine == [True]


@pytest.mark.usefixtures("stub_engine")
def test_wav_retains_both_hashes(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    source = _register_wav(tmp_path, manifest)
    load_voice("mine", manifest=manifest)
    _, voices = ManifestStore(manifest).read()
    record = voices["mine"]
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    assert record.source_sha256 == expected
    assert record.asset_sha256 != record.source_sha256


def test_pre_compiled_is_copied_without_a_model(
    tmp_path: Path, stub_engine: list[bool]
) -> None:
    manifest = tmp_path / "manifest.json"
    source = tmp_path / "mine.safetensors"
    source.write_bytes(b"already-compiled")
    add_voice(source, voice_id="mine", manifest=manifest, **_RIGHTS)  # type: ignore[arg-type]
    load_voice("mine", manifest=manifest)
    assert stub_engine == [False]
    _, voices = ManifestStore(manifest).read()
    assert Path(voices["mine"].asset_path or "").read_bytes() == b"already-compiled"


@pytest.mark.usefixtures("stub_engine")
def test_source_deleted_before_load_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    source = _register_wav(tmp_path, manifest)
    source.unlink()
    with pytest.raises(VoiceError) as excinfo:
        load_voice("mine", manifest=manifest)
    assert excinfo.value.code is ErrorCode.POCKET_VOICE_INVALID


@pytest.mark.usefixtures("stub_engine")
def test_source_modified_after_registration_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    source = _register_wav(tmp_path, manifest)
    source.write_bytes(b"RIFF9999WAVEfmt ")
    with pytest.raises(VoiceError) as excinfo:
        load_voice("mine", manifest=manifest)
    assert excinfo.value.code is ErrorCode.POCKET_VOICE_INVALID


def test_absolute_config_resolves_every_relative_asset_name(tmp_path: Path) -> None:
    """Regression: compilation cannot use the stored relative-name config.

    The renderer rejects absolute paths and resolves names against its own
    snapshot; compilation runs outside it and would fail to find any asset.
    """
    root = tmp_path / "engine"
    root.mkdir()
    stored = root / "english.yaml"
    stored.write_text(
        yaml.safe_dump(
            {
                "weights_path": "model.safetensors",
                "flow_lm": {"lookup_table": {"tokenizer_path": "tokenizer.model"}},
                "mimi": {"weights_path": "mimi.safetensors"},
            }
        ),
        encoding="utf-8",
    )
    engine = EngineRecord(
        id="english",
        language="english",
        model_root=str(root),
        config_path=str(stored),
        model_revision="rev",
        package_version="2.1.0",
        files=(FileRecord("model.safetensors", 1, "a" * 64),),
        sample_rate_hz=24000,
        device="cpu",
        timeout_seconds=300.0,
        cloning_capable=True,
    )
    work = tmp_path / "work"
    work.mkdir()
    produced = yaml.safe_load(
        provision._absolute_config(engine, work).read_text(encoding="utf-8")
    )
    assert produced["weights_path"] == str(root / "model.safetensors")
    assert produced["flow_lm"]["lookup_table"]["tokenizer_path"] == str(
        root / "tokenizer.model"
    )
    assert produced["mimi"]["weights_path"] == str(root / "mimi.safetensors")
    for value in (
        produced["weights_path"],
        produced["flow_lm"]["lookup_table"]["tokenizer_path"],
        produced["mimi"]["weights_path"],
    ):
        assert Path(value).is_absolute()
