"""Write fails fast on an unprovisioned voice, before spawning workers."""
# ruff: noqa: D103, TC003

from __future__ import annotations

import json
from pathlib import Path

import pytest

import kenkui as kk
from kenkui import ErrorCode, VoiceError
from test_epub import make_epub, xhtml


@pytest.fixture(autouse=True)
def no_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-flight Ruling B: pipeline.py from-imports execute_sequential.

    Patching kenkui._execution.coordinator would leave pipeline's own binding
    untouched and silently run a real render.
    """

    def explode(*_args: object, **_kwargs: object) -> object:
        message = "write must fail before execution starts"
        raise AssertionError(message)

    monkeypatch.setattr(kk.pipeline, "execute_sequential", explode)


def _book(tmp_path: Path) -> Path:
    return make_epub(
        tmp_path / "book.epub",
        chapters={"one": xhtml("<h1>One</h1><p>Exact first.</p>")},
        spine=("one",),
    )


def _manifest(tmp_path: Path, state: str) -> Path:
    path = tmp_path / "manifest.json"
    voice = {
        "variety": "wav",
        "state": state,
        "name": "Mine",
        "enabled": True,
        "language": "english",
        "engine_id": "english",
        "provenance": "mine",
        "license_id": "proprietary",
        "commercial_use_allowed": True,
        "voice_rights": "owned",
        "source_path": str(tmp_path / "mine.wav"),
        "source_sha256": "c" * 64,
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": "kenkui-pocket-production-v2",
                "engines": {},
                "voices": {"mine": voice},
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def test_registered_voice_fails_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KENKUI_POCKET_MANIFEST", str(_manifest(tmp_path, "registered")))
    pipeline = kk.book(_book(tmp_path)).assign_voice("mine").tts()
    with pytest.raises(VoiceError) as excinfo:
        pipeline.write(tmp_path / "out.m4b")
    assert excinfo.value.code is ErrorCode.VOICE_NOT_PROVISIONED


def test_unknown_voice_fails_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KENKUI_POCKET_MANIFEST", str(_manifest(tmp_path, "registered")))
    pipeline = kk.book(_book(tmp_path)).assign_voice("nobody").tts()
    with pytest.raises(VoiceError) as excinfo:
        pipeline.write(tmp_path / "out.m4b")
    assert excinfo.value.code is ErrorCode.VOICE_UNRESOLVED
