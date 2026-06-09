from __future__ import annotations

import importlib.util
import json
from pathlib import Path

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
