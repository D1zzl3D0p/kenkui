"""Final production activation, chunked execution, callback, and logging contracts."""
# ruff: noqa: D103, EM101, PLR2004, TC003, TRY003, TRY004

from __future__ import annotations

import json
from pathlib import Path

import pytest

import kenkui as kk
from kenkui._audio.m4b import FakeArtifactAssembler
from kenkui._domain.planning import MAX_TTS_SEGMENT_CHARACTERS
from kenkui._execution.cache import CacheStore
from kenkui._execution.process_pool import EngineSpecification
from kenkui._tts.production import (
    MANIFEST_SCHEMA_VERSION,
    default_cache_root,
    production_bindings_from_environment,
)
from test_epub import make_epub, xhtml
from test_execution import _bind


def test_multisegment_chapter_is_one_chapter_for_events_and_assembly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    speech = "Sentence boundary. " * (MAX_TTS_SEGMENT_CHARACTERS // 4)
    source = make_epub(
        tmp_path / "long.epub",
        chapters={"one": xhtml(f"<h1>One</h1><p>{speech}</p>")},
        spine=("one",),
    )
    pipeline = kk.epub(source).assign_voice("narrator").tts()
    _bind(monkeypatch, EngineSpecification.fake(), FakeArtifactAssembler())
    serial_events: list[kk.ExecutionEvent] = []
    parallel_events: list[kk.ExecutionEvent] = []
    output = tmp_path / "long.m4b"

    serial = pipeline.write_m4b(output, workers=1, on_event=serial_events.append)
    serial_bytes = output.read_bytes()
    parallel = pipeline.write_m4b(
        output, workers=4, overwrite=True, on_event=parallel_events.append
    )

    assert serial == parallel
    assert output.read_bytes() == serial_bytes
    assert serial.stats.synthesized_segments > 1
    assert serial.stats.rendered_chapters == 1
    render = [
        event
        for event in serial_events
        if isinstance(event, kk.StageProgress) and event.stage == "render"
    ]
    assert [(event.completed, event.total) for event in render] == [(1, 1)]
    assert serial_events == parallel_events


def test_callback_precommit_fails_but_completed_is_best_effort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = make_epub(
        tmp_path / "book.epub",
        chapters={"one": xhtml("<h1>One</h1>Speech")},
        spine=("one",),
    )
    pipeline = kk.epub(source).assign_voice("narrator").tts()
    _bind(monkeypatch, EngineSpecification.fake(), FakeArtifactAssembler())
    output = tmp_path / "completed.m4b"

    def callback(event: kk.ExecutionEvent) -> None:
        if isinstance(event, kk.Completed):
            raise RuntimeError("private completed detail")

    result = pipeline.write_m4b(output, on_event=callback)
    assert result.output == output
    assert output.is_file()
    assert not list(tmp_path.glob(".kenkui-*"))


def test_structured_logs_are_generic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = make_epub(
        tmp_path / "secret-source.epub",
        chapters={"one": xhtml("<h1>Secret title</h1>secret speech")},
        spine=("one",),
    )
    pipeline = kk.epub(source).assign_voice("narrator").tts()
    _bind(monkeypatch, EngineSpecification.fake(), FakeArtifactAssembler())
    caplog.set_level("INFO", logger="kenkui")
    pipeline.write_m4b(tmp_path / "secret-output.m4b")

    text = caplog.text
    assert "execution_stage_started" in text
    assert "execution_stage_completed" in text
    assert "secret-source" not in text
    assert "secret-output" not in text
    assert "secret speech" not in text
    assert "Secret title" not in text


def _manifest(tmp_path: Path) -> Path:
    root = tmp_path / "model"
    root.mkdir(mode=0o700)
    config = root / "config.yaml"
    config.write_text("model: local\n", encoding="utf-8")
    prompt = tmp_path / "voice.wav"
    prompt.write_bytes(b"local fixture")
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "engines": {
            "english": {
                "language": "english",
                "model_root": str(root),
                "config_path": str(config),
                "model_revision": "fake-v1",
                "package_version": "2.1.0",
                "files": [
                    {
                        "relative_path": "config.yaml",
                        "size": config.stat().st_size,
                        "sha256": "a" * 64,
                    }
                ],
                "sample_rate_hz": 24000,
                "device": "cpu",
                "timeout_seconds": 300.0,
                "cloning_capable": False,
            }
        },
        "voices": {
            "narrator": {
                "variety": "built-in",
                "state": "loaded",
                "name": "Narrator",
                "enabled": True,
                "provenance": "project fixture",
                "license_id": "CC0-1.0",
                "commercial_use_allowed": True,
                "language": "english",
                "engine_id": "english",
                "compatible_model_revisions": ["fake-v1"],
                "asset_path": str(prompt),
                "asset_sha256": "c" * 64,
                "voice_rights": "project-owned",
            }
        },
    }
    path = tmp_path / "production.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)
    return path


def test_manifest_activation_resolves_voice_and_attaches_private_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    cache = tmp_path / "cache"
    monkeypatch.setenv("KENKUI_POCKET_MANIFEST", str(manifest))
    monkeypatch.setattr("kenkui._tts.production.default_cache_root", lambda: cache)
    monkeypatch.setattr("kenkui._tts.production.preflight_pocket", lambda *_: None)

    bindings = production_bindings_from_environment("narrator")

    assert bindings.voice.id == "narrator"
    # Schema v2 collapses content_fingerprint into asset_sha256 (spec section 5).
    assert bindings.voice.content_fingerprint == "c" * 64
    assert bindings.voice.compatible_model_revisions == ("fake-v1",)
    assert bindings.model_revision == "fake-v1"
    assert bindings.engine_specification.kind == "pocket"
    assert isinstance(bindings.cache_store, CacheStore)
    assert cache.stat().st_mode & 0o777 == 0o700


def test_manifest_unknown_voice_and_malformed_schema_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    monkeypatch.setenv("KENKUI_POCKET_MANIFEST", str(manifest))
    with pytest.raises(kk.VoiceError) as unknown:
        production_bindings_from_environment("unknown")
    assert unknown.value.code is kk.ErrorCode.VOICE_UNRESOLVED

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(kk.ModelError) as malformed:
        production_bindings_from_environment("narrator")
    assert malformed.value.code is kk.ErrorCode.POCKET_MODEL_INVALID


def test_default_cache_root_is_versioned_and_private_api_only() -> None:
    assert default_cache_root().parts[-2:] == ("kenkui", "v1")
    assert not hasattr(kk, "CacheStore")
