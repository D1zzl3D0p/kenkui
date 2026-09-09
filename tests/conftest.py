"""Shared test isolation.

Provisioning writes to a real per-user cache. Without redirecting it, results
depend on whether the developer has ever run `load_voice`, and a suite run can
read — or worse, mutate — real assets. Every test gets its own cache root.
"""
# ruff: noqa: TC003

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import kenkui as kk
from helpers import make_epub, xhtml
from kenkui._audio.m4b import FakeArtifactAssembler
from kenkui._domain.planning import SpeakerSpan
from kenkui._epub.identifiers import chapter_id
from kenkui._execution.coordinator import ExecutionBindings
from kenkui._execution.process_pool import EngineSpecification
from kenkui._tts import production
from kenkui.inspection import ChapterInspection
from kenkui.voices import manifest as manifest_module


@pytest.fixture(autouse=True)
def isolated_cache_root(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Redirect the managed cache and manifest away from the real user cache."""
    root = tmp_path_factory.mktemp("kenkui-cache")
    monkeypatch.setattr(production, "default_cache_root", lambda: root)
    # Provisioning imported default_manifest_path directly. Its function still
    # resolves this module's cache-root binding, even when the function name
    # below has been replaced, so redirect both lookup paths.
    monkeypatch.setattr(manifest_module, "default_cache_root", lambda: root)
    monkeypatch.setattr(
        manifest_module, "default_manifest_path", lambda: root / "manifest.json"
    )
    monkeypatch.delenv("KENKUI_POCKET_MANIFEST", raising=False)
    return root


@pytest.fixture
def _real_cache_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the autouse redirect for tests of default_cache_root itself."""
    monkeypatch.undo()


def log_field(record: logging.LogRecord, name: str) -> object:
    """Read one structured field that log_event attached through ``extra``.

    LogRecord declares no such attributes, so reading them directly is
    invisible to the type checker even though they exist at runtime.
    """
    return getattr(record, name)


CH08_TEXT = 'Alpha one. Alpha two.\n\n"Beta," she said. "Gamma," he answered.'
CH08_BODY = '<p>Alpha one. Alpha two.</p><p>"Beta," she said. "Gamma," he answered.</p>'

# Chapter IDs are a hash of (member path, occurrence, fragment), not the
# manifest item names passed to make_epub -- see kenkui._epub.identifiers.
# make_epub's default hrefs are "text/<name>.xhtml", written under "OPS/", so
# these are the real, stable IDs epub_path's chapters parse to.
CH08_ID = chapter_id("OPS/text/ch08.xhtml", 0, "")
CH09_ID = chapter_id("OPS/text/ch09.xhtml", 0, "")


@pytest.fixture
def epub_path(tmp_path: Path) -> Path:
    """Build a two-chapter EPUB on disk with IDs ``CH08_ID`` and ``CH09_ID``."""
    return make_epub(
        tmp_path / "book.epub",
        chapters={"ch08": xhtml(CH08_BODY), "ch09": xhtml("<p>Delta.</p>")},
        spine=["ch08", "ch09"],
    )


@pytest.fixture
def chapter_ch08() -> ChapterInspection:
    """Return a chapter inspection with known text, matching ``epub_path``'s ch08."""
    return ChapterInspection(
        id=CH08_ID,
        index=8,
        title="Chapter Eight",
        speech_characters=None,
        text=CH08_TEXT,
    )


@pytest.fixture
def machine_spans(chapter_ch08: ChapterInspection) -> tuple[SpeakerSpan, ...]:
    """One narration span covering the chapter exactly, as attribution emits."""
    return (SpeakerSpan(CH08_ID, 0, len(chapter_ch08.text), None),)


@pytest.fixture
def siblings() -> dict[tuple[str, ...], int]:
    """Empty sibling counts, standing in for ``SiblingCounts``."""
    return {}


@pytest.fixture
def unresolved_book(epub_path: Path) -> kk.Pipeline:
    """Return a pipeline over ``epub_path`` that has not yet resolved a cast."""
    return kk.book(epub_path).assign_voice("ivy")


class _NoCharactersClient:
    """Answers roster discovery with no characters, so every quote narrates.

    An empty roster short-circuits attribution before any quote-attribution
    prompt is built: ``attribute_chapter`` returns every span as narration
    when there are no characters to assign, without a second model call.
    """

    def complete(self, model: str, prompt: str) -> str:
        """Return an empty character roster for any prompt."""
        assert model
        assert prompt
        return '{"characters": []}'


@pytest.fixture
def resolved_book(
    unresolved_book: kk.Pipeline, monkeypatch: pytest.MonkeyPatch
) -> kk.Pipeline:
    """``unresolved_book`` with quotes attributed and cast resolved, no network.

    Resolution reaches two model/asset boundaries: attribution (a model call)
    and execution bindings (a local voice manifest lookup that a redirected,
    empty test cache can never satisfy). Both are private seams other
    pipeline-resolution tests already monkeypatch -- see
    ``kenkui._resolution._attribution_client`` and
    ``kenkui._resolution._execution_bindings`` in test_resolution_inspection.py.
    """
    narrator = kk.Voice(
        id="ivy",
        name="Ivy",
        enabled=True,
        provenance="fixture",
        license_id="CC0-1.0",
        commercial_use_allowed=True,
        language="en",
        state="loaded",
        content_fingerprint="a" * 64,
        compatible_model_revisions=("fake-v1",),
    )
    bindings = ExecutionBindings(
        EngineSpecification.fake(), FakeArtifactAssembler(), narrator, "fake-v1"
    )
    monkeypatch.setattr(
        "kenkui._resolution._execution_bindings", lambda _voice_id, **_cast: bindings
    )
    monkeypatch.setattr("kenkui._resolution._attribution_client", _NoCharactersClient)
    return unresolved_book.attribute_quotes("fake/model").resolve()
