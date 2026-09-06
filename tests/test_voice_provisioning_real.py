"""Opt-in end-to-end provisioning and rendering against real assets."""
# ruff: noqa: D103

from __future__ import annotations

import os
from pathlib import Path

import pytest

import kenkui as kk
from kenkui.voices import provision
from kenkui.voices.manifest import ManifestStore
from test_epub import make_epub, xhtml

pytestmark = pytest.mark.pocket_real

_ENABLED = os.environ.get("KENKUI_RUN_PROVISIONING_REAL") == "1"
_REASON = "set KENKUI_RUN_PROVISIONING_REAL=1 to download real assets"
_MIN_EMBEDDING_BYTES = 1_000_000


def _book(tmp_path: Path) -> Path:
    return make_epub(
        tmp_path / "book.epub",
        chapters={"one": xhtml("<h1>One</h1><p>Exact first sentence.</p>")},
        spine=("one",),
    )


@pytest.mark.skipif(not _ENABLED, reason=_REASON)
def test_load_builtin_voice_downloads_and_verifies(tmp_path: Path) -> None:
    voice = kk.load_voice("eponine", manifest=tmp_path / "manifest.json")
    assert voice.state == "loaded"
    assert voice.engine is not None
    assert voice.engine.cloning_capable is False
    assert voice.asset_bytes is not None
    assert voice.asset_bytes > _MIN_EMBEDDING_BYTES


@pytest.mark.skipif(not _ENABLED, reason=_REASON)
def test_second_load_is_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.json"
    kk.load_voice("eponine", manifest=manifest)

    def explode(url: str) -> Path:
        _ = url
        message = "an idempotent load must not fetch"
        raise AssertionError(message)

    monkeypatch.setattr(provision, "_fetch", explode)
    assert kk.load_voice("eponine", manifest=manifest).state == "loaded"


@pytest.mark.skipif(not _ENABLED, reason=_REASON)
def test_render_a_real_m4b(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "manifest.json"
    kk.load_voice("eponine", manifest=manifest)
    monkeypatch.setenv("KENKUI_POCKET_MANIFEST", str(manifest))
    output = tmp_path / "out.m4b"
    result = kk.book(_book(tmp_path)).assign_voice("eponine").tts().write(output)
    assert Path(result.output).is_file()
    assert Path(result.output).stat().st_size > 0


@pytest.mark.skipif(not _ENABLED, reason=_REASON)
def test_gated_compiled_embedding_renders_on_ungated_weights(tmp_path: Path) -> None:
    """Answer spec section 13 on gated-versus-ungated embedding compatibility.

    Requires accepted kyutai/pocket-tts terms and `hf auth login`, plus a local
    WAV prompt named by KENKUI_TEST_VOICE_WAV.
    """
    named = os.environ.get("KENKUI_TEST_VOICE_WAV")
    if named is None or not Path(named).is_file():
        pytest.skip("set KENKUI_TEST_VOICE_WAV to a readable local WAV prompt")
    manifest = tmp_path / "manifest.json"
    kk.add_voice(
        Path(named),
        voice_id="local-test",
        name="Local Test",
        language="english",
        provenance="local test fixture",
        license_id="proprietary",
        commercial_use_allowed=False,
        voice_rights="test only",
        manifest=manifest,
    )
    compiled = kk.load_voice("local-test", manifest=manifest)
    assert compiled.engine is not None
    assert compiled.engine.cloning_capable is True

    _, voices = ManifestStore(manifest).read()
    record = voices["local-test"]
    assert record.asset_path is not None
    assert Path(record.asset_path).is_file()
    # The compiling engine's revision is pinned here; the strict reader rejects
    # a mismatch, which is what makes the conservative design fail closed.
    assert record.compatible_model_revisions == (compiled.engine.model_revision,)
