"""One engine holds a conditioning state per voice.

A language engine is ~225 MB of weights against ~6.5 MB per speaker
embedding, so a cast should cost embeddings and not extra models. The worker
scheduler also rejects any result reporting engine_initializations != 1, so
N voices must share exactly one constructed model.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from kenkui._tts import pocket as pocket_module
from kenkui._tts.pocket import (
    PocketEngineConfig,
    PocketManifestFile,
    PocketTTSEngine,
    VoiceAsset,
)
from kenkui._tts.protocols import SynthesisTask
from kenkui.errors import VoiceError

if TYPE_CHECKING:
    from pathlib import Path


def _worker_token() -> object:
    """Return the sentinel proving construction happened in a spawned worker."""
    return pocket_module._WORKER_TOKEN  # noqa: SLF001


_TWO = 2


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safetensors_bytes(header: bytes) -> bytes:
    return len(header).to_bytes(8, "little") + header + b"\x00" * 16


class _CountingModel:
    """Record every state derivation so reuse is directly observable."""

    sample_rate = 24_000
    device = "cpu"
    derived: list[Path] = []  # noqa: RUF012
    loads = 0

    @classmethod
    def load_model(cls, *, config: Path, quantize: bool) -> _CountingModel:
        assert quantize is False
        assert config is not None
        cls.loads += 1
        return cls()

    def get_state_for_audio_prompt(self, prompt: Path) -> dict[str, str]:
        type(self).derived.append(prompt)
        return {"state": prompt.name}

    def generate_audio(self, state: object, text: str) -> _FakeTensor:
        assert text
        assert state is not None
        return _FakeTensor([0.0] * 48)  # >= 1 ms at 24 kHz


class _FakeTensor:
    """The exact CPU contiguous float32 mono shape tensor_to_pcm accepts."""

    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.dtype = "torch.float32"
        self.ndim = 1
        self.shape = (len(values),)
        self.device = "cpu"

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def tolist(self) -> list[float]:
        return self.values

    def is_contiguous(self) -> bool:
        return True

    def __getitem__(self, value: slice) -> _FakeTensor:
        return _FakeTensor(self.values[value])


@pytest.fixture
def two_voice_config(tmp_path: Path) -> tuple[PocketEngineConfig, str, str]:
    """Build a real two-voice engine config on disk."""
    root = (tmp_path / "model").resolve()
    root.mkdir()
    config_bytes = b"name: local\n"
    (root / "model.yaml").write_bytes(config_bytes)
    weights = b"approved local weights"
    (root / "model.safetensors").write_bytes(weights)

    assets: list[VoiceAsset] = []
    digests: list[str] = []
    for name in ("eponine", "charles"):
        payload = _safetensors_bytes(f'{{"{name}":{{}}}}'.encode())
        path = tmp_path / f"{name}.safetensors"
        path.write_bytes(payload)
        digest = _digest(payload)
        digests.append(digest)
        assets.append(
            VoiceAsset(
                path=str(path.resolve()),
                sha256=digest,
                variety="built-in",
                provenance="kyutai catalog",
                license_id="CC-BY-4.0",
                rights="review required",
                commercial_use_allowed=False,
            )
        )
    config = PocketEngineConfig(
        model_root=str(root),
        config_path=str((root / "model.yaml").resolve()),
        model_revision="approved-revision",
        package_version="2.1.0",
        files=(
            PocketManifestFile("model.safetensors", len(weights), _digest(weights)),
            PocketManifestFile("model.yaml", len(config_bytes), _digest(config_bytes)),
        ),
        voices=tuple(assets),
        cloning_capable=False,
        sample_rate_hz=24_000,
    )
    return config, digests[0], digests[1]


@pytest.fixture
def engine_harness(monkeypatch: pytest.MonkeyPatch) -> type[_CountingModel]:
    """Install the counting model and enter worker mode."""
    _CountingModel.derived = []
    _CountingModel.loads = 0
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "2.1.0")
    monkeypatch.setattr(
        "kenkui._tts.pocket.importlib.import_module",
        {
            "pocket_tts": SimpleNamespace(TTSModel=_CountingModel),
            "pocket_tts.models.tts_model": SimpleNamespace(
                download_if_necessary=lambda _value: None
            ),
            "torch": SimpleNamespace(Tensor=_FakeTensor),
        }.__getitem__,
    )
    monkeypatch.setattr(pocket_module, "_worker_marker", _worker_token())
    return _CountingModel


def _task(digest: str, text: str = "exact task text") -> SynthesisTask:
    return SynthesisTask("segment", "chapter", text, 24_000, 1, 1024, digest)


def test_task_carries_the_voice_digest() -> None:
    """A worker cannot route to a conditioning state it was never told about."""
    assert _task("a" * 64).voice_asset_sha256 == "a" * 64


def test_each_voice_derives_its_own_state(
    two_voice_config: tuple[PocketEngineConfig, str, str],
    engine_harness: type[_CountingModel],
) -> None:
    """Two cast voices must not share one conditioning state."""
    config, first, second = two_voice_config
    engine = PocketTTSEngine(config, reusable=True)
    engine.synthesize(_task(first))
    engine.synthesize(_task(second))
    assert len(engine_harness.derived) == _TWO
    assert engine_harness.derived[0] != engine_harness.derived[1]
    engine.close()


def test_repeated_use_of_one_voice_derives_once(
    two_voice_config: tuple[PocketEngineConfig, str, str],
    engine_harness: type[_CountingModel],
) -> None:
    """Deriving per segment was pure waste; the cache must survive reuse."""
    config, first, _ = two_voice_config
    engine = PocketTTSEngine(config, reusable=True)
    engine.synthesize(_task(first))
    engine.synthesize(_task(first, "another line"))
    assert len(engine_harness.derived) == 1
    engine.close()


def test_a_cast_constructs_exactly_one_model(
    two_voice_config: tuple[PocketEngineConfig, str, str],
    engine_harness: type[_CountingModel],
) -> None:
    """process_pool rejects any worker reporting engine_initializations != 1."""
    config, first, second = two_voice_config
    engine = PocketTTSEngine(config, reusable=True)
    engine.synthesize(_task(first))
    engine.synthesize(_task(second))
    assert engine_harness.loads == 1
    engine.close()


def test_unknown_digest_fails_rather_than_rendering_a_wrong_voice(
    two_voice_config: tuple[PocketEngineConfig, str, str],
    engine_harness: type[_CountingModel],
) -> None:
    """Silently falling back would produce valid audio in the wrong voice."""
    assert engine_harness.loads == 0
    config, _, _ = two_voice_config
    engine = PocketTTSEngine(config, reusable=True)
    with pytest.raises(VoiceError):
        engine.synthesize(_task("f" * 64))


def test_close_clears_derived_states(
    two_voice_config: tuple[PocketEngineConfig, str, str],
    engine_harness: type[_CountingModel],
) -> None:
    """A closed engine must not hand out states bound to a removed snapshot."""
    assert engine_harness.loads == 0
    config, first, _ = two_voice_config
    engine = PocketTTSEngine(config, reusable=True)
    engine.synthesize(_task(first))
    engine.close()
    assert engine._states == {}  # noqa: SLF001
