"""The render path loads embeddings only, and derives voice state once."""
# ruff: noqa: D103, SLF001

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import cast

import pytest

from kenkui import ErrorCode, VoiceError
from kenkui._tts import pocket
from kenkui._tts.protocols import SynthesisTask


def _config(tmp_path: Path | None = None) -> pocket.PocketEngineConfig:
    """Build a config whose paths exist, since semantic_material resolves them."""
    root = tmp_path if tmp_path is not None else Path("/models/english")
    if tmp_path is not None:
        (root / "model").mkdir(mode=0o700, parents=True, exist_ok=True)
        (root / "model" / "english.yaml").write_text("model: local\n", encoding="utf-8")
        (root / "eponine.safetensors").write_bytes(b"x")
    model_root = str((root / "model").resolve()) if tmp_path else "/models/english"
    config_path = (
        str((root / "model" / "english.yaml").resolve())
        if tmp_path
        else "/models/english/english.yaml"
    )
    asset = (
        str((root / "eponine.safetensors").resolve())
        if tmp_path
        else "/cache/voices/english/eponine.safetensors"
    )
    return pocket.PocketEngineConfig(
        model_root=model_root,
        config_path=config_path,
        model_revision="revision-1",
        package_version="2.1.0",
        files=(pocket.PocketManifestFile("english.yaml", 4, "a" * 64),),
        voices=(
            pocket.VoiceAsset(
                path=asset,
                sha256="b" * 64,
                variety="built-in",
                provenance="kyutai catalog",
                license_id="CC-BY-4.0",
                rights="review required",
                commercial_use_allowed=False,
            ),
        ),
        cloning_capable=False,
        sample_rate_hz=24000,
    )


def test_semantic_material_includes_variety_and_asset_hash(tmp_path: Path) -> None:
    material = _config(tmp_path).semantic_material()
    voices = cast("tuple[dict[str, object], ...]", material["voices"])
    assert voices[0]["variety"] == "built-in"
    assert voices[0]["sha256"] == "b" * 64
    assert "voice_prompt_sha256" not in material


def test_semantic_material_changes_with_variety(tmp_path: Path) -> None:
    base = _config(tmp_path)
    other = dataclasses.replace(
        base,
        voices=(dataclasses.replace(base.voices[0], variety="pre-compiled"),),
    )
    assert base.semantic_material() != other.semantic_material()


def test_safetensors_header_is_validated() -> None:
    pocket._validate_safetensors((8).to_bytes(8, "little") + b'{"a":{}}')


def test_truncated_safetensors_is_rejected() -> None:
    with pytest.raises(VoiceError) as excinfo:
        pocket._validate_safetensors(b"\x08")
    assert excinfo.value.code is ErrorCode.POCKET_VOICE_INVALID


def test_safetensors_with_oversized_header_is_rejected() -> None:
    with pytest.raises(VoiceError) as excinfo:
        pocket._validate_safetensors((1 << 40).to_bytes(8, "little") + b"{}")
    assert excinfo.value.code is ErrorCode.POCKET_VOICE_INVALID


def test_safetensors_with_non_object_header_is_rejected() -> None:
    with pytest.raises(VoiceError) as excinfo:
        pocket._validate_safetensors((3).to_bytes(8, "little") + b"[1]")
    assert excinfo.value.code is ErrorCode.POCKET_VOICE_INVALID


class _Model:
    """Counts conditioning derivations and returns a fixed mono tensor."""

    sample_rate = 24000
    device = "cpu"

    def __init__(self) -> None:
        self.state_calls = 0

    def get_state_for_audio_prompt(self, conditioning: Path) -> object:
        self.state_calls += 1
        return {"conditioning": conditioning}

    def generate_audio(self, state: object, text: str) -> object:
        raise NotImplementedError


def _detached_engine(model: _Model) -> pocket.PocketTTSEngine:
    engine = pocket.PocketTTSEngine.__new__(pocket.PocketTTSEngine)
    object.__setattr__(engine, "_model", model)
    object.__setattr__(engine, "_config", _config())
    object.__setattr__(engine, "_state", None)
    # __del__ calls close(), which reads these; omitting them raises during
    # garbage collection and surfaces as an unraisable-exception warning.
    object.__setattr__(engine, "_snapshot", None)
    object.__setattr__(engine, "_reusable", True)
    return engine


def test_voice_state_is_derived_once_per_engine() -> None:
    model = _Model()
    engine = _detached_engine(model)
    first = engine._voice_state()
    second = engine._voice_state()
    assert first is second
    assert model.state_calls == 1


def test_voice_state_is_passed_a_path_not_a_string() -> None:
    """A str would let pocket-tts call download_if_necessary; a Path cannot."""
    model = _Model()
    state = _detached_engine(model)._voice_state()
    assert isinstance(state["conditioning"], Path)


def test_task_type_is_unchanged() -> None:
    task = SynthesisTask(
        segment_id="s1",
        chapter_id="c1",
        text="hello",
        sample_rate_hz=24000,
        channels=1,
        max_output_bytes=1024,
    )
    assert task.segment_id == "s1"
