"""Engine identity and voice assets are separate concerns.

One engine is ~225 MB of language weights; a voice is a ~6.5 MB speaker
embedding. Inlining a single voice into the engine config was the only reason
one engine could not speak twice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from kenkui._tts.pocket import PocketEngineConfig, PocketManifestFile, VoiceAsset
from kenkui.errors import VoiceError

if TYPE_CHECKING:
    from pathlib import Path

_SHA_A = "a" * 64
_SHA_B = "b" * 64


@pytest.fixture
def engine_root(tmp_path: Path) -> Path:
    """Build a real on-disk engine layout.

    semantic_material resolves model_root and config_path with
    resolve(strict=True), so these paths must exist. tmp_path is already
    resolved on macOS, where /var is a symlink to /private/var.
    """
    root = (tmp_path / "engines" / "english").resolve()
    root.mkdir(parents=True)
    (root / "english.yaml").write_text("model: english\n")
    voices = root / "voices"
    voices.mkdir()
    for name in ("eponine", "charles"):
        (voices / f"{name}.safetensors").write_bytes(b"\x00" * 8)
    return root


def _asset(root: Path, sha256: str, name: str) -> VoiceAsset:
    return VoiceAsset(
        path=str(root / "voices" / f"{name}.safetensors"),
        sha256=sha256,
        variety="built-in",
        provenance="kyutai/pocket-tts-without-voice-cloning",
        license_id="CC-BY-4.0",
        rights="Review the VCTK terms before commercial use.",
        commercial_use_allowed=False,
    )


def _config(root: Path, *assets: VoiceAsset) -> PocketEngineConfig:
    return PocketEngineConfig(
        model_root=str(root),
        config_path=str(root / "english.yaml"),
        model_revision="c" * 40,
        package_version="2.1.0",
        files=(PocketManifestFile("english.yaml", 10, "d" * 64),),
        voices=assets,
        cloning_capable=False,
        sample_rate_hz=24000,
    )


def test_config_holds_many_voices_and_resolves_by_digest(engine_root: Path) -> None:
    """One engine config carries a whole cast, addressable by digest."""
    config = _config(
        engine_root,
        _asset(engine_root, _SHA_A, "eponine"),
        _asset(engine_root, _SHA_B, "charles"),
    )
    assert config.voice_by_sha(_SHA_B).path.endswith("charles.safetensors")


def test_unknown_digest_is_rejected(engine_root: Path) -> None:
    """An unregistered digest must fail rather than fall back to some voice."""
    config = _config(engine_root, _asset(engine_root, _SHA_A, "eponine"))
    with pytest.raises(KeyError):
        config.voice_by_sha(_SHA_B)


def test_semantic_material_is_order_independent(engine_root: Path) -> None:
    """Cast ordering must not change the cache key."""
    first = _asset(engine_root, _SHA_A, "eponine")
    second = _asset(engine_root, _SHA_B, "charles")
    forward = _config(engine_root, first, second)
    reverse = _config(engine_root, second, first)
    assert forward.semantic_material() == reverse.semantic_material()


def test_semantic_material_separates_engine_from_voices(engine_root: Path) -> None:
    """Engine identity and voice identity are distinct keys in the material."""
    material = _config(
        engine_root, _asset(engine_root, _SHA_A, "eponine")
    ).semantic_material()
    assert "voices" in material
    assert "voice_asset_sha256" not in material


def test_duplicate_digests_are_rejected(engine_root: Path) -> None:
    """voice_by_sha must stay unambiguous; the worker routes on that key."""
    config = _config(
        engine_root,
        _asset(engine_root, _SHA_A, "eponine"),
        _asset(engine_root, _SHA_A, "charles"),
    )
    with pytest.raises(VoiceError):
        config.semantic_material()


def test_empty_cast_is_rejected(engine_root: Path) -> None:
    """A config that names no voice can render nothing."""
    with pytest.raises(VoiceError):
        _config(engine_root).semantic_material()
