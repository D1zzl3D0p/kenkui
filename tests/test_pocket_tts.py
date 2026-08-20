"""WP9 Pocket adapter tests; all inference/package objects are lightweight fakes."""
# ruff: noqa: ANN401, D101, D102, D103, D105, EM101, PLC0415, PLR0913, PLR2004, S101, SLF001

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import os
import pickle
import struct
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from kenkui._domain.planning import VoicePlan
from kenkui._execution.process_pool import (
    EngineSpecification,
    FakeEngineConfig,
    render_spawned,
)
from kenkui._tts.pocket import (
    MAX_OUTPUT_SAMPLES,
    PocketEngineConfig,
    PocketManifestFile,
    PocketTTSEngine,
    preflight_pocket,
    tensor_to_pcm,
)
from kenkui._tts.production import pocket_production_bindings
from kenkui._tts.protocols import SynthesisTask
from kenkui.errors import ErrorCode, ModelError, RenderError, VoiceError
from kenkui.voices import Voice


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _wav_bytes() -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(24_000)
        target.writeframes(b"\x00\x00" * 24)
    return stream.getvalue()


def _fixture(tmp_path: Path) -> tuple[PocketEngineConfig, VoicePlan]:
    root = tmp_path / "approved-model"
    root.mkdir(parents=True)
    config_bytes = b"weights_path: model.safetensors\n"
    weights = b"approved local weights"
    config_path = root / "model.yaml"
    config_path.write_bytes(config_bytes)
    (root / "model.safetensors").write_bytes(weights)
    voice_bytes = _wav_bytes()
    voice_path = tmp_path / "authorized.wav"
    voice_path.write_bytes(voice_bytes)
    files = (
        PocketManifestFile("model.safetensors", len(weights), _digest(weights)),
        PocketManifestFile("model.yaml", len(config_bytes), _digest(config_bytes)),
    )
    config = PocketEngineConfig(
        str(root.resolve()),
        str(config_path.resolve()),
        "approved-revision",
        "2.1.0",
        files,
        str(voice_path.resolve()),
        _digest(voice_bytes),
        "wav",
        True,
        "locally recorded with documented consent",
        "research-only",
        "authorized for this local use",
        False,
        24_000,
    )
    voice = VoicePlan(
        "voice",
        "Voice",
        config.voice_asset_sha256,
        "en",
        config.voice_provenance,
        config.voice_license_id,
        False,
        (config.model_revision,),
    )
    return config, voice


def _installed(monkeypatch: pytest.MonkeyPatch, version: str = "2.1.0") -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: version)


def test_config_is_frozen_pickle_safe_and_manifest_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, voice = _fixture(tmp_path)
    _installed(monkeypatch)
    assert pickle.loads(
        pickle.dumps(EngineSpecification.pocket(config))
    ) == EngineSpecification.pocket(config)
    preflight_pocket(config, voice, config.model_revision)


@pytest.mark.parametrize(
    "failure",
    ["missing", "corrupt", "extra", "traversal", "symlink", "hardlink", "size"],
)
def test_manifest_failures_are_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    config, _ = _fixture(tmp_path)
    _installed(monkeypatch)
    root = Path(config.model_root)
    if failure == "missing":
        (root / "model.safetensors").unlink()
    elif failure == "corrupt":
        (root / "model.safetensors").write_bytes(b"x" * config.files[0].size)
    elif failure == "extra":
        (root / "extra.bin").write_bytes(b"extra")
    elif failure == "traversal":
        config = replace(
            config,
            files=(replace(config.files[0], relative_path="../bad"), config.files[1]),
        )
    elif failure == "symlink":
        target = root / "model.safetensors"
        target.unlink()
        target.symlink_to(Path(config.voice_asset_path))
    elif failure == "hardlink":
        os.link(root / "model.safetensors", root / "second-link")
        config = replace(
            config,
            files=config.files
            + (
                PocketManifestFile(
                    "second-link", config.files[0].size, config.files[0].sha256
                ),
            ),
        )
    else:
        config = replace(
            config,
            files=(
                replace(config.files[0], size=config.files[0].size + 1),
                config.files[1],
            ),
        )
    with pytest.raises(ModelError) as caught:
        preflight_pocket(config)
    assert caught.value.code == ErrorCode.POCKET_MODEL_INVALID
    assert caught.value.__cause__ is None
    assert config.model_root not in str(caught.value)


def test_manifest_bounds_are_rejected_before_filesystem_or_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kenkui._tts import pocket as module

    config, _ = _fixture(tmp_path)
    _installed(monkeypatch)
    walked = False
    hashed = False

    def forbidden_walk(*_args: object, **_kwargs: object) -> object:
        nonlocal walked
        walked = True
        raise AssertionError("walked")

    def forbidden_hash(*_args: object, **_kwargs: object) -> object:
        nonlocal hashed
        hashed = True
        raise AssertionError("hashed")

    monkeypatch.setattr("kenkui._tts.pocket.os.scandir", forbidden_walk)
    monkeypatch.setattr(module, "_verify_source", forbidden_hash)
    item = config.files[0]
    cases = (
        replace(config, files=(item,) * (module.MAX_MANIFEST_ENTRIES + 1)),
        replace(
            config,
            files=(
                replace(
                    item,
                    relative_path="/".join(
                        ["a"] * (module.MAX_RELATIVE_PATH_DEPTH + 1)
                    ),
                ),
            ),
        ),
        replace(
            config,
            files=(
                replace(
                    item, relative_path="a" * (module.MAX_RELATIVE_NAME_LENGTH + 1)
                ),
            ),
        ),
        replace(config, files=(replace(item, size=module.MAX_MODEL_TOTAL_BYTES), item)),
    )
    for malformed in cases:
        with pytest.raises(ModelError):
            preflight_pocket(malformed)
    assert not walked
    assert not hashed


def test_directory_entry_cap_aborts_incrementally_before_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kenkui._tts import pocket as module

    config, _ = _fixture(tmp_path)
    _installed(monkeypatch)
    monkeypatch.setattr(module, "MAX_DIRECTORY_ENTRIES", 1)
    monkeypatch.setattr(
        module,
        "_verify_source",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("hashed")),
    )
    with pytest.raises(ModelError) as caught:
        preflight_pocket(config)
    assert caught.value.code == ErrorCode.POCKET_MODEL_INVALID


def test_weights_are_streamed_and_only_selected_yaml_and_voice_are_captured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kenkui._tts import pocket as module

    config, _ = _fixture(tmp_path)
    _installed(monkeypatch)
    original = module._verify_source
    captures: list[tuple[str, object]] = []

    def tracking(
        path: Path,
        expected_size: int,
        expected_hash: str,
        limit: int,
        *,
        voice: bool = False,
        capture_limit: int | None = None,
    ) -> bytes | None:
        captures.append((path.name, capture_limit))
        return original(
            path,
            expected_size,
            expected_hash,
            limit,
            voice=voice,
            capture_limit=capture_limit,
        )

    monkeypatch.setattr(module, "_verify_source", tracking)
    preflight_pocket(config)
    assert captures == [
        ("model.safetensors", None),
        ("model.yaml", module.MAX_CONFIG_BYTES),
        ("authorized.wav", module.MAX_VOICE_BYTES),
    ]


def test_package_absent_and_version_are_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _fixture(tmp_path)

    def absent(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", absent)
    with pytest.raises(ModelError) as caught:
        preflight_pocket(config)
    assert caught.value.code == ErrorCode.POCKET_PACKAGE_MISSING
    _installed(monkeypatch, "2.0.0")
    with pytest.raises(ModelError) as caught:
        preflight_pocket(config)
    assert caught.value.code == ErrorCode.POCKET_VERSION_UNSUPPORTED


def test_voice_hash_wav_and_plan_identity_are_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, voice = _fixture(tmp_path)
    _installed(monkeypatch)
    Path(config.voice_asset_path).write_bytes(b"not a wav!!")
    bad = replace(config, voice_asset_sha256=_digest(b"not a wav!!"))
    with pytest.raises(VoiceError) as caught:
        preflight_pocket(bad)
    assert caught.value.code == ErrorCode.POCKET_VOICE_INVALID
    config, voice = _fixture(tmp_path / "new")
    with pytest.raises(VoiceError):
        preflight_pocket(
            config, replace(voice, content_fingerprint="0" * 64), config.model_revision
        )


class FakeTensor:
    def __init__(
        self,
        values: list[float],
        *,
        dtype: str = "torch.float32",
        ndim: int = 1,
        shape: tuple[int, ...] | None = None,
    ) -> None:
        self.values = values
        self.dtype = dtype
        self.ndim = ndim
        self.shape = shape or (len(values),)
        self.device = "cpu"

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def tolist(self) -> list[float]:
        return self.values

    def is_contiguous(self) -> bool:
        return True

    def __getitem__(self, value: slice) -> FakeTensor:
        return FakeTensor(self.values[value])


def _task(max_bytes: int = 100) -> SynthesisTask:
    return SynthesisTask("segment", "chapter", "exact task text", 24_000, 1, max_bytes)


def test_pcm_conversion_golden_is_deterministic() -> None:
    audio = tensor_to_pcm(
        FakeTensor([-1.0, -0.5, 0.0, 0.5, 1.0] * 5), FakeTensor, _task(), 24_000
    )
    expected_five = struct.pack("<hhhhh", -32767, -16384, 0, 16384, 32767)
    assert audio.pcm_s16le == expected_five * 5
    assert (audio.channels, audio.frame_count, audio.duration_ms) == (1, 25, 1)


@pytest.mark.parametrize(
    "output",
    [
        FakeTensor([float("nan")] * 24),
        FakeTensor([float("inf")] * 24),
        FakeTensor([1.01] * 24),
        FakeTensor([0.0] * 24, dtype="torch.float64"),
        FakeTensor([0.0] * 24, ndim=2, shape=(1, 24)),
        object(),
    ],
)
def test_invalid_output_is_sanitized(output: object) -> None:
    with pytest.raises(RenderError) as caught:
        tensor_to_pcm(output, FakeTensor, _task(), 24_000)
    assert caught.value.code == ErrorCode.INVALID_AUDIO
    assert caught.value.__cause__ is None


def test_output_bounds_checked_before_materialization() -> None:
    huge = FakeTensor([], shape=(MAX_OUTPUT_SAMPLES + 1,))
    with pytest.raises(RenderError):
        tensor_to_pcm(huge, FakeTensor, _task(), 24_000)
    with pytest.raises(RenderError):
        tensor_to_pcm(FakeTensor([0.0] * 24), FakeTensor, _task(2), 24_000)


class FakeModel:
    sample_rate = 24_000
    device = "cpu"
    loaded_config: Path | None = None
    prompt: Path | None = None
    text: str | None = None
    fail: str | None = None

    @classmethod
    def load_model(cls, *, config: Path, quantize: bool) -> FakeModel:
        assert quantize is False
        cls.loaded_config = config
        if cls.fail == "load":
            raise RuntimeError("secret provider path")
        return cls()

    def get_state_for_audio_prompt(self, prompt: Path) -> dict[str, str]:
        type(self).prompt = prompt
        if type(self).fail == "voice":
            raise RuntimeError("secret voice path")
        return {"state": "local"}

    def generate_audio(self, _state: object, text: str) -> FakeTensor:
        type(self).text = text
        if type(self).fail == "inference":
            raise RuntimeError("secret input text")
        return FakeTensor([0.0] * 24)


def test_adapter_exact_local_api_and_sanitized_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _fixture(tmp_path)
    _installed(monkeypatch)
    pocket = SimpleNamespace(TTSModel=FakeModel)
    implementation = SimpleNamespace(download_if_necessary=lambda _value: None)
    torch = SimpleNamespace(Tensor=FakeTensor)
    modules = {
        "pocket_tts": pocket,
        "pocket_tts.models.tts_model": implementation,
        "torch": torch,
    }
    monkeypatch.setattr(
        "kenkui._tts.pocket.importlib.import_module", modules.__getitem__
    )
    from kenkui._tts import pocket as pocket_module

    monkeypatch.setattr(pocket_module, "_worker_marker", pocket_module._WORKER_TOKEN)
    FakeModel.fail = None
    engine = PocketTTSEngine(config)
    audio = engine.synthesize(_task())
    assert FakeModel.loaded_config is not None
    assert FakeModel.loaded_config.name == Path(config.config_path).name
    assert FakeModel.prompt is not None
    assert FakeModel.prompt.name == "prompt.wav"
    assert FakeModel.text == "exact task text"
    assert len(audio.pcm_s16le) == 48
    for stage, code in (
        ("load", ErrorCode.POCKET_MODEL_LOAD_FAILED),
        ("voice", ErrorCode.POCKET_VOICE_LOAD_FAILED),
        ("inference", ErrorCode.POCKET_INFERENCE_FAILED),
    ):
        FakeModel.fail = stage
        with pytest.raises((ModelError, VoiceError, RenderError)) as caught:
            if stage == "load":
                PocketTTSEngine(config)
            else:
                PocketTTSEngine(config).synthesize(_task())
        assert caught.value.code == code
        assert "secret" not in str(caught.value)
    FakeModel.fail = None


def test_private_production_factory_requires_complete_approved_voice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _fixture(tmp_path)
    _installed(monkeypatch)
    voice = Voice(
        "voice",
        "Voice",
        True,
        config.voice_provenance,
        config.voice_license_id,
        False,
        "en",
        config.voice_asset_sha256,
        (config.model_revision,),
    )
    bindings = pocket_production_bindings(config, voice)
    assert bindings.engine_specification == EngineSpecification.pocket(config)
    assert bindings.voice is voice
    with pytest.raises(VoiceError) as caught:
        pocket_production_bindings(replace(config, commercial_use_allowed=True), voice)
    assert caught.value.code == ErrorCode.POCKET_VOICE_INVALID
    with pytest.raises(VoiceError) as disabled:
        pocket_production_bindings(config, replace(voice, enabled=False))
    assert disabled.value.code is ErrorCode.VOICE_DISABLED


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_root", 1),
        ("config_path", None),
        ("model_revision", []),
        ("model_revision", "   "),
        ("package_version", 2.1),
        ("files", []),
        ("files", (object(),)),
        ("device", b"cpu"),
        ("device", "cuda"),
        ("sample_rate_hz", True),
        ("sample_rate_hz", 7999),
        ("sample_rate_hz", 192001),
        ("timeout_seconds", 1),
        ("timeout_seconds", float("nan")),
        ("timeout_seconds", float("inf")),
        ("timeout_seconds", 3600.1),
    ],
)
def test_every_malformed_model_field_is_stably_model_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    config, _ = _fixture(tmp_path)
    _installed(monkeypatch)
    with pytest.raises(ModelError) as caught:
        preflight_pocket(replace(config, **{field: value}))  # type: ignore[arg-type]
    assert caught.value.code == ErrorCode.POCKET_MODEL_INVALID


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("voice_asset_path", 1),
        ("voice_asset_sha256", None),
        ("voice_asset_sha256", "A" * 64),
        ("voice_provenance", []),
        ("voice_license_id", "  "),
        ("voice_rights", False),
        ("commercial_use_allowed", 0),
    ],
)
def test_every_malformed_voice_field_is_stably_voice_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    config, _ = _fixture(tmp_path)
    _installed(monkeypatch)
    with pytest.raises(VoiceError) as caught:
        preflight_pocket(replace(config, **{field: value}))  # type: ignore[arg-type]
    assert caught.value.code == ErrorCode.POCKET_VOICE_INVALID


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("relative_path", 1),
        ("relative_path", "a\\b"),
        ("relative_path", "/absolute"),
        ("size", True),
        ("size", 0),
        ("sha256", None),
        ("sha256", "f" * 63),
    ],
)
def test_every_malformed_manifest_file_field_is_model_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    config, _ = _fixture(tmp_path)
    _installed(monkeypatch)
    bad_item = replace(config.files[0], **{field: value})  # type: ignore[arg-type]
    with pytest.raises(ModelError) as caught:
        preflight_pocket(replace(config, files=(bad_item, config.files[1])))
    assert caught.value.code == ErrorCode.POCKET_MODEL_INVALID


def test_wav_must_be_structurally_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _installed(monkeypatch)
    for index, malformed in enumerate(
        (b"RIFF\x04\x00\x00\x00WAVE", _wav_bytes()[:-1], _wav_bytes()[:36])
    ):
        config, _ = _fixture(tmp_path / str(index))
        Path(config.voice_asset_path).write_bytes(malformed)
        bad = replace(config, voice_asset_sha256=_digest(malformed))
        with pytest.raises(VoiceError) as caught:
            preflight_pocket(bad)
        assert caught.value.code == ErrorCode.POCKET_VOICE_INVALID


def test_yaml_is_bounded_local_and_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _installed(monkeypatch)
    for index, data in enumerate(
        (
            b"weights: /etc/passwd\n",
            b"weights: [../escape.bin]\n",
            b"source: file:///tmp/x\n",
        )
    ):
        config, _ = _fixture(tmp_path / str(index))
        path = Path(config.config_path)
        path.write_bytes(data)
        item = replace(config.files[1], size=len(data), sha256=_digest(data))
        with pytest.raises(ModelError):
            preflight_pocket(replace(config, files=(config.files[0], item)))


def test_semantic_identity_distinguishes_selected_yaml_without_absolute_paths(
    tmp_path: Path,
) -> None:
    config, _ = _fixture(tmp_path)
    root = Path(config.model_root)
    second = root / "other.yaml"
    data = Path(config.config_path).read_bytes()
    second.write_bytes(data)
    second_item = PocketManifestFile("other.yaml", len(data), _digest(data))
    first = replace(config, files=config.files + (second_item,))
    alternate = replace(first, config_path=str(second.resolve()))
    assert first.semantic_material() != alternate.semantic_material()
    assert str(tmp_path) not in repr(first.semantic_material())


def test_direct_parent_engine_construction_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _fixture(tmp_path)
    _installed(monkeypatch)
    from kenkui._tts import pocket as pocket_module

    monkeypatch.setattr(pocket_module, "_worker_marker", None)
    with pytest.raises(ModelError) as caught:
        PocketTTSEngine(config)
    assert caught.value.code == ErrorCode.POCKET_MODEL_LOAD_FAILED


def test_unsafe_writable_root_and_file_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _fixture(tmp_path)
    _installed(monkeypatch)
    root = Path(config.model_root)
    root.chmod(0o777)
    with pytest.raises(ModelError):
        preflight_pocket(config)
    root.chmod(0o700)
    (root / "model.safetensors").chmod(0o666)
    with pytest.raises(ModelError):
        preflight_pocket(config)


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), -float("inf"), 3601.0])
def test_scheduler_rejects_nonfinite_or_unreasonable_timeout_before_spawn(
    monkeypatch: pytest.MonkeyPatch, timeout: float
) -> None:
    def forbidden(_method: str) -> object:
        raise AssertionError("process context created")

    monkeypatch.setattr(
        "kenkui._execution.process_pool.multiprocessing.get_context", forbidden
    )
    spec = EngineSpecification.fake(FakeEngineConfig(timeout_seconds=timeout))
    with pytest.raises(RenderError) as caught:
        list(render_spawned((_task(),), spec, 1, None))
    assert caught.value.code == ErrorCode.SYNTHESIS_FAILED


def test_tensor_materialization_is_chunk_bounded() -> None:
    class TrackingTensor(FakeTensor):
        maximum = 0

        def __getitem__(self, value: slice) -> TrackingTensor:
            chunk = type(self)(self.values[value])
            type(self).maximum = max(type(self).maximum, len(chunk.values))
            return chunk

    samples = 65_537
    task = _task(samples * 2)
    audio = tensor_to_pcm(TrackingTensor([0.0] * samples), TrackingTensor, task, 24_000)
    assert len(audio.pcm_s16le) == samples * 2
    assert TrackingTensor.maximum <= 65_536


def test_network_audit_denies_only_network_events() -> None:
    from kenkui._tts import pocket as pocket_module

    pocket_module._network_audit("open", ())
    for event in (
        "socket.__new__",
        "socket.connect",
        "socket.bind",
        "socket.getaddrinfo",
        "socket.sendto",
        "socket.sendmsg",
    ):
        with pytest.raises(PermissionError):
            pocket_module._network_audit(event, ())


def test_private_fail_closed_validation_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kenkui._tts import pocket as module

    with pytest.raises(ModelError):
        module._validate_fields(object())
    with pytest.raises(ModelError):
        module._canonical_absolute("relative")
    with pytest.raises(VoiceError):
        module._canonical_absolute("relative", voice=True)
    with pytest.raises(ModelError):
        module._inspect_yaml(b"")
    with pytest.raises(ModelError):
        module._inspect_yaml(b"url: https://example.invalid/x")
    with pytest.raises(ModelError):
        module._inspect_yaml(b"x: \xff")
    with pytest.raises(ModelError):
        module._inspect_yaml(b"x: safe/../escape")
    module._inspect_yaml(b"# comment\nplain\nempty:\nlist: [safe.bin]\n")
    with pytest.raises(RuntimeError):
        module._enter_spawned_worker(object())
    monkeypatch.setattr("kenkui._tts.pocket.sys.addaudithook", lambda _hook: None)
    monkeypatch.setattr(module, "_audit_installed", False)
    module._enter_spawned_worker(module._WORKER_TOKEN)
    assert module._audit_installed

    config, _ = _fixture(tmp_path)
    outside = tmp_path / "outside.yaml"
    outside.write_bytes(b"x: safe\n")
    with pytest.raises(ModelError):
        replace(config, config_path=str(outside.resolve())).semantic_material()
    huge = PocketManifestFile("huge", module.MAX_MODEL_FILE_BYTES, "0" * 64)
    with pytest.raises(ModelError):
        module._validate_fields(replace(config, files=(huge, huge, huge)))


def test_manifest_selected_prompt_and_directory_rejections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _installed(monkeypatch)
    config, _ = _fixture(tmp_path / "duplicate")
    with pytest.raises(ModelError):
        preflight_pocket(replace(config, files=config.files + (config.files[0],)))

    config, _ = _fixture(tmp_path / "undeclared")
    with pytest.raises(ModelError):
        preflight_pocket(replace(config, files=(config.files[0],)))

    config, _ = _fixture(tmp_path / "inside")
    prompt = Path(config.model_root) / "prompt.wav"
    prompt.write_bytes(_wav_bytes())
    item = PocketManifestFile(
        "prompt.wav", prompt.stat().st_size, _digest(prompt.read_bytes())
    )
    bad = replace(
        config,
        files=config.files + (item,),
        voice_asset_path=str(prompt.resolve()),
        voice_asset_sha256=item.sha256,
    )
    with pytest.raises(VoiceError):
        preflight_pocket(bad)

    config, _ = _fixture(tmp_path / "extension")
    prompt = Path(config.voice_asset_path)
    renamed = prompt.with_suffix(".bin")
    prompt.rename(renamed)
    with pytest.raises(VoiceError):
        preflight_pocket(replace(config, voice_asset_path=str(renamed.resolve())))

    config, _ = _fixture(tmp_path / "directory")
    nested = Path(config.model_root) / "nested"
    nested.mkdir()
    nested.chmod(0o777)
    with pytest.raises(ModelError):
        preflight_pocket(config)


def test_downloader_allowlist_revalidates_snapshot_files(tmp_path: Path) -> None:
    from kenkui._tts import pocket as module

    root = tmp_path / "snapshot"
    root.mkdir(mode=0o700)
    approved = root / "approved.bin"
    approved.write_bytes(b"approved")
    approved.chmod(0o400)
    allowed = frozenset({approved.resolve()})
    assert (
        module._deny_remote(approved.resolve(), allowed, root.resolve())
        == approved.resolve()
    )
    for value in (object(), "relative", str(tmp_path / "missing")):
        with pytest.raises(RuntimeError):
            module._deny_remote(value, allowed, root.resolve())
    approved.chmod(0o666)
    with pytest.raises(RuntimeError):
        module._deny_remote(approved.resolve(), allowed, root.resolve())


def test_additional_tensor_contract_rejections() -> None:
    class BadChunk(FakeTensor):
        def __getitem__(self, _value: slice) -> object:  # type: ignore[override]
            return object()

    class BadList(FakeTensor):
        def __getitem__(self, value: slice) -> BadList:
            return type(self)(self.values[value])

        def tolist(self) -> tuple[float, ...]:  # type: ignore[override]
            return tuple(self.values)

    for output in (BadChunk([0.0] * 24), BadList([0.0] * 24)):
        with pytest.raises(RenderError):
            tensor_to_pcm(output, type(output), _task(), 24_000)
    too_short = FakeTensor([0.0])
    with pytest.raises(RenderError):
        tensor_to_pcm(too_short, FakeTensor, _task(), 24_000)


def test_snapshot_binds_loaded_bytes_despite_original_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _fixture(tmp_path)
    _installed(monkeypatch)
    original_yaml = Path(config.config_path).read_bytes()
    original_wav = Path(config.voice_asset_path).read_bytes()
    observed: dict[str, bytes | Path] = {}

    class SnapshotModel:
        sample_rate = 24_000
        device = "cpu"

        @classmethod
        def load_model(cls, *, config: Path, quantize: bool) -> SnapshotModel:
            assert quantize is False
            observed["yaml"] = config.read_bytes()
            observed["snapshot"] = config.parents[1]
            Path(config_path_name).write_bytes(b"replaced after snapshot")
            return cls()

        def get_state_for_audio_prompt(self, prompt: Path) -> object:
            observed["wav"] = prompt.read_bytes()
            return object()

        def generate_audio(self, _state: object, _text: str) -> FakeTensor:
            return FakeTensor([0.0] * 24)

    config_path_name = config.config_path
    modules = {
        "pocket_tts": SimpleNamespace(TTSModel=SnapshotModel),
        "pocket_tts.models.tts_model": SimpleNamespace(download_if_necessary=None),
        "torch": SimpleNamespace(Tensor=FakeTensor),
    }
    monkeypatch.setattr(
        "kenkui._tts.pocket.importlib.import_module", modules.__getitem__
    )
    from kenkui._tts import pocket as module

    monkeypatch.setattr(module, "_worker_marker", module._WORKER_TOKEN)
    engine = PocketTTSEngine(config)
    Path(config.voice_asset_path).unlink()
    Path(config.voice_asset_path).symlink_to(config_path_name)
    engine.synthesize(_task())
    assert observed["yaml"] == original_yaml
    assert observed["wav"] == original_wav
    assert not Path(observed["snapshot"]).exists()  # type: ignore[arg-type]
