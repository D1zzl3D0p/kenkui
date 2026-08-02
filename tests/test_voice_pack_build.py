from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from kenkui.voice_registry import load_manifest


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "build_voice_pack.py"
    spec = importlib.util.spec_from_file_location("build_voice_pack", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_voice_pack_copies_compiled_asset_and_writes_hashes(tmp_path, monkeypatch):
    tool = _load_tool_module()
    monkeypatch.setattr(tool.importlib.metadata, "version", lambda _package: "2.0.0")
    source_asset = tmp_path / "source.safetensors"
    source_asset.write_bytes(b"compiled")
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "voices": [
                    {
                        "voice_id": "demo_voice",
                        "display_name": "Demo Voice",
                        "gender": "Female",
                        "path": str(source_asset),
                        "dataset": "VCTK",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def _migrate(source_path, output_path, *, language):
        assert source_path == source_asset
        assert language == tool.DEFAULT_VOICE_PACK_LANGUAGE
        output_path.write_bytes(b"modern")
        return output_path

    with monkeypatch.context() as m:
        m.setattr(tool, "migrate_voice_asset", _migrate)
        manifest = tool.build_voice_pack(
            source_manifest,
            tmp_path / "pack",
            generate_previews=False,
            smoke_test=False,
            pocket_tts_version="2.0.0",
        )

    entries = load_manifest(manifest)
    assert entries[0].voice_id == "demo_voice"
    assert entries[0].origin == "kenkui_compiled"
    assert entries[0].sha256 is not None
    assert entries[0].size_bytes == len(b"modern")
    assert entries[0].path is not None
    assert entries[0].path.exists()
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_data["pocket_tts_version"] == "2.0.0"
    assert manifest_data["voice_pack_format_version"] == 2


def test_build_voice_pack_compiles_prompt_sources(tmp_path, monkeypatch):
    tool = _load_tool_module()
    monkeypatch.setattr(tool.importlib.metadata, "version", lambda _package: "2.0.0")
    prompt_source = tmp_path / "prompt.wav"
    prompt_source.write_bytes(b"prompt")
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "voices": [
                    {
                        "voice_id": "prompt_voice",
                        "display_name": "Prompt Voice",
                        "gender": "Female",
                        "prompt_source": str(prompt_source),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def _compile(source, output_path, *, language, truncate):
        assert source == str(prompt_source)
        assert language == tool.DEFAULT_VOICE_PACK_LANGUAGE
        assert truncate is True
        output_path.write_bytes(b"modern_prompt")
        return output_path

    with monkeypatch.context() as m:
        m.setattr(tool, "compile_audio_prompt_source", _compile)
        manifest = tool.build_voice_pack(
            source_manifest,
            tmp_path / "pack",
            generate_previews=False,
            smoke_test=False,
            pocket_tts_version="2.0.0",
        )

    entries = load_manifest(manifest)
    assert entries[0].voice_id == "prompt_voice"
    assert entries[0].path is not None and entries[0].path.exists()


def test_build_voice_pack_writes_complete_deterministic_preview_matrix(tmp_path, monkeypatch):
    tool = _load_tool_module()
    monkeypatch.setattr(tool.importlib.metadata, "version", lambda package: {
        "kenkui": "2.3.1", "pocket-tts": "2.0.0"
    }[package])
    source_manifest = tmp_path / "source_manifest.json"
    voices = []
    for voice_id in ("zeta", "alpha"):
        source_asset = tmp_path / f"{voice_id}.safetensors"
        source_asset.write_bytes(f"voice:{voice_id}".encode())
        voices.append({"voice_id": voice_id, "display_name": voice_id.title(),
                       "gender": "Female", "path": str(source_asset)})
    source_manifest.write_text(json.dumps({"voices": voices}), encoding="utf-8")
    generated = []
    model_loads = 0

    def generate(asset_path, phrase):
        generated.append((asset_path.stem, phrase.phrase_id))
        return 8_000, np.full(96_000, 0.1, dtype=np.float32)

    def load_generator():
        nonlocal model_loads
        model_loads += 1
        return generate

    def encode(wav_path, mp3_path):
        assert wav_path.exists()
        mp3_path.write_bytes(f"mp3:{wav_path.parent.name}:{wav_path.stem}".encode())

    def migrate(source, output, **_):
        output.write_bytes(source.read_bytes())
        return output

    monkeypatch.setattr(tool, "migrate_voice_asset", migrate)
    monkeypatch.setattr(tool, "_make_preview_generator", load_generator)
    tool.build_voice_pack(source_manifest, tmp_path / "pack", mp3_encoder=encode,
                          pocket_tts_version="2.0.0")

    preview_manifest = json.loads(
        (tmp_path / "pack" / "preview-manifest.json").read_text(encoding="utf-8")
    )
    pairs = [(asset["voice_id"], asset["phrase_id"]) for asset in preview_manifest["assets"]]
    assert pairs == sorted(pairs)
    assert len(pairs) == 6
    assert model_loads == 1
    assert set(generated) == set(pairs)
    assert preview_manifest["kenkui_version"] == "2.3.1"
    assert preview_manifest["pocket_tts_version"] == "2.0.0"
    assert preview_manifest["phrase_catalog_version"] == 1
    assert preview_manifest["voice_count"] == 2
    assert preview_manifest["phrase_count"] == 3
    assert preview_manifest["asset_count"] == 6
    for asset in preview_manifest["assets"]:
        assert asset["content_type"] == "audio/mpeg"
        assert asset["duration_ms"] == 12_000
        assert len(asset["voice_asset_sha256"]) == 64
        assert len(asset["sha256"]) == 64
        assert asset["size_bytes"] > 0
        assert (tmp_path / "pack" / asset["wav_path"]).exists()
        assert (tmp_path / "pack" / asset["mp3_path"]).exists()


def test_preview_matrix_rejects_partial_generation(tmp_path, monkeypatch):
    tool = _load_tool_module()
    monkeypatch.setattr(tool.importlib.metadata, "version", lambda package: {
        "kenkui": "2.3.1", "pocket-tts": "2.0.0"
    }[package])
    source_asset = tmp_path / "voice.safetensors"
    source_asset.write_bytes(b"voice")
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text(json.dumps({"voices": [{"voice_id": "alpha",
        "display_name": "Alpha", "gender": "Female", "path": str(source_asset)}]}),
        encoding="utf-8")
    output_dir = tmp_path / "pack"
    output_dir.mkdir()
    (output_dir / "preview-manifest.json").write_text("stale", encoding="utf-8")

    def migrate(source, output, **_):
        output.write_bytes(source.read_bytes())
        return output

    monkeypatch.setattr(tool, "migrate_voice_asset", migrate)
    calls = 0

    def generate(_asset_path, _phrase):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("generation failed")
        return 8_000, np.full(8_000, 0.1, dtype=np.float32)

    with pytest.raises(RuntimeError, match="complete preview matrix"):
        tool.build_voice_pack(source_manifest, output_dir, preview_generator=generate,
                              mp3_encoder=lambda _wav, mp3: mp3.write_bytes(b"mp3"),
                              pocket_tts_version="2.0.0")
    assert not (output_dir / "preview-manifest.json").exists()


@pytest.mark.parametrize("samples,error", [
    (np.array([], dtype=np.float32), "empty"),
    (np.array([np.nan] * 8_000, dtype=np.float32), "non-finite"),
    (np.ones(8_000, dtype=np.float32), "clipping"),
    (np.full(10, 0.1, dtype=np.float32), "short"),
    (np.full(8_000 * 31, 0.1, dtype=np.float32), "long"),
])
def test_preview_audio_validation_rejects_invalid_samples(samples, error):
    tool = _load_tool_module()
    with pytest.raises(ValueError, match=error):
        tool.validate_preview_audio(8_000, samples)


def test_preview_audio_validation_normalizes_sparse_hot_samples():
    tool = _load_tool_module()
    samples = np.full(8_000, 0.1, dtype=np.float32)
    samples[0] = 1.2

    normalized = tool.validate_preview_audio(8_000, samples)

    assert np.max(np.abs(normalized)) == pytest.approx(0.99)


def test_preview_generator_allows_complete_long_phrase_generation(monkeypatch):
    tool = _load_tool_module()
    calls = []

    class Audio:
        def squeeze(self):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.full(8_000, 0.1, dtype=np.float32)

    class Model:
        sample_rate = 8_000

        def get_state_for_audio_prompt(self, asset_path):
            return asset_path

        def generate_audio(self, state, text, **kwargs):
            calls.append((state, text, kwargs))
            return Audio()

    class TTSModel:
        @staticmethod
        def load_model(*, language, eos_threshold):
            assert language == tool.DEFAULT_VOICE_PACK_LANGUAGE
            assert eos_threshold == -2.0
            return Model()

    monkeypatch.setitem(__import__("sys").modules, "pocket_tts", type("PocketTTS", (), {
        "TTSModel": TTSModel,
    }))
    generate = tool._make_preview_generator()
    phrase = tool.PREVIEW_PHRASE_CATALOG.phrases[0]
    sample_rate, audio = generate(Path("voice.safetensors"), phrase)

    assert sample_rate == 8_000
    assert audio.shape == (8_000,)
    assert calls == [("voice.safetensors", phrase.text, {
        "max_tokens": 200,
        "frames_after_eos": 2,
    })]


def test_preview_generation_retries_implausibly_early_eos():
    tool = _load_tool_module()
    phrase = tool.PREVIEW_PHRASE_CATALOG.phrases[0]
    calls = 0

    def generate(_asset_path, _phrase):
        nonlocal calls
        calls += 1
        sample_count = 8_000 if calls == 1 else 96_000
        return 8_000, np.full(sample_count, 0.1, dtype=np.float32)

    sample_rate, audio = tool._generate_valid_preview(
        generate,
        Path("voice.safetensors"),
        phrase,
    )

    assert calls == 2
    assert sample_rate == 8_000
    assert audio.shape == (96_000,)


def test_preview_generation_exhaustion_rejects_early_eos():
    tool = _load_tool_module()
    phrase = tool.PREVIEW_PHRASE_CATALOG.phrases[0]
    calls = 0

    def generate(_asset_path, _phrase):
        nonlocal calls
        calls += 1
        return 8_000, np.full(8_000, 0.1, dtype=np.float32)

    with pytest.raises(ValueError, match="voice/pride-and-prejudice"):
        tool._generate_valid_preview(generate, Path("voice.safetensors"), phrase)

    assert calls == tool.PREVIEW_GENERATION_ATTEMPTS


def test_preview_generation_does_not_retry_structural_validation_failure():
    tool = _load_tool_module()
    phrase = tool.PREVIEW_PHRASE_CATALOG.phrases[0]
    calls = 0

    def generate(_asset_path, _phrase):
        nonlocal calls
        calls += 1
        return 0, np.full(8_000, 0.1, dtype=np.float32)

    with pytest.raises(ValueError, match="invalid sample rate"):
        tool._generate_valid_preview(generate, Path("voice.safetensors"), phrase)

    assert calls == 1
