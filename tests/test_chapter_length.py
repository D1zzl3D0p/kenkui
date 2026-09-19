"""A chapter may be any length; an unusual one is reported, never refused."""
# ruff: noqa: TC003

from __future__ import annotations

from pathlib import Path

import pytest

import kenkui as kk
from helpers import make_epub, xhtml
from kenkui._audio.m4b import FakeArtifactAssembler
from kenkui._execution import coordinator
from kenkui._execution.cache import CacheStore
from kenkui._execution.coordinator import ExecutionBindings
from kenkui._execution.process_pool import EngineSpecification
from kenkui._tts.fake import DeterministicFakeEngine
from kenkui.limits import (
    LONG_CHAPTER_HOURS,
    TYPICAL_SPEECH_CHARACTERS_PER_SECOND,
    estimated_audio_hours,
    is_long_chapter,
)
from test_execution import _bind, _voice


def _book(tmp_path: Path, sentences: int = 1) -> kk.Pipeline:
    """Build a two-chapter book whose first chapter is as long as asked."""
    long_text = "Exact first sentence. " * sentences
    source = make_epub(
        tmp_path / "book.epub",
        chapters={
            "one": xhtml(f"<h1>Endnotes</h1><p>{long_text}</p>"),
            "two": xhtml("<h1>Two</h1><p>Exact second.</p>"),
        },
        spine=("one", "two"),
    )
    return kk.epub(source).assign_voice("narrator").tts()


def _warnings(events: list[kk.ExecutionEvent]) -> list[kk.Warning]:
    return [event for event in events if isinstance(event, kk.Warning)]


def test_a_long_chapter_is_reported_and_then_rendered_anyway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of the warning: it informs a decision, it does not make one."""
    pipeline = _book(tmp_path, sentences=40)
    _bind(monkeypatch, DeterministicFakeEngine(), FakeArtifactAssembler())
    remarkable = 100
    monkeypatch.setattr(
        coordinator, "is_long_chapter", lambda characters: characters > remarkable
    )
    events: list[kk.ExecutionEvent] = []
    output = tmp_path / "long.m4b"

    result = pipeline.write_m4b(output, on_event=events.append)

    assert result.output == output
    assert output.is_file()
    warned = _warnings(events)
    assert [warning.code for warning in warned] == ["long_chapter"]
    assert "'Endnotes'" in warned[0].message
    assert warned[0].stage == "planning"
    # Named, so a caller can act on the chapter rather than on the whole book.
    assert warned[0].chapter_id
    # Early enough to act on: it precedes the render stage, which is the last
    # moment stopping is still free.
    render_started = next(
        event.sequence
        for event in events
        if isinstance(event, kk.StageStarted) and event.stage == "render"
    )
    assert warned[0].sequence < render_started


def test_an_ordinary_book_says_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Silence is the common case, so the warning keeps its meaning."""
    pipeline = _book(tmp_path, sentences=3)
    _bind(monkeypatch, DeterministicFakeEngine(), FakeArtifactAssembler())
    events: list[kk.ExecutionEvent] = []

    pipeline.write_m4b(tmp_path / "ordinary.m4b", on_event=events.append)

    assert _warnings(events) == []


def test_a_chapter_reaches_disk_as_it_renders_rather_than_at_its_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Streaming is why length is free: the part exists while it is being filled."""
    pipeline = _book(tmp_path, sentences=200)
    _bind(monkeypatch, DeterministicFakeEngine(), FakeArtifactAssembler())
    writes: list[tuple[str, bool]] = []

    class Recording(coordinator._ChapterSpill):  # noqa: SLF001
        def write(self, payload: bytes) -> None:
            super().write(payload)
            writes.append((self.chapter_id, self.path.exists()))

    monkeypatch.setattr(coordinator, "_ChapterSpill", Recording)
    pipeline.write_m4b(tmp_path / "streamed.m4b")

    per_chapter = {chapter: 0 for chapter, _existed in writes}
    for chapter, _existed in writes:
        per_chapter[chapter] += 1
    assert max(per_chapter.values()) > 1, "a long chapter arrives segment by segment"
    assert all(existed for _chapter, existed in writes)


def test_a_run_past_the_whole_book_budget_says_the_book_is_too_long(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one remaining stop bounds worker output, and says so in its own words."""
    pipeline = _book(tmp_path, sentences=2)
    bindings = ExecutionBindings(
        EngineSpecification.fake(),
        FakeArtifactAssembler(),
        _voice(),
        "fake-v1",
        CacheStore(tmp_path / "private-cache"),
    )
    monkeypatch.setattr(
        "kenkui._resolution._execution_bindings", lambda _voice_id, **_cast: bindings
    )
    pipeline.write_m4b(tmp_path / "cold.m4b", keep_audio_cache=True)

    # Every segment is cached, so the run budget answers on its own.
    monkeypatch.setattr(coordinator, "MAX_TOTAL_PCM_BYTES", 1024)
    with pytest.raises(kk.RenderError) as caught:
        pipeline.write_m4b(tmp_path / "warm.m4b", keep_audio_cache=True)

    assert caught.value.code == kk.ErrorCode.BOOK_TOO_LONG
    assert not (tmp_path / "warm.m4b").exists()


def test_the_published_estimate_matches_the_threshold_it_reports_against() -> None:
    """The estimate and the threshold are one pair, so callers agree with runs."""
    per_hour = 3600 * TYPICAL_SPEECH_CHARACTERS_PER_SECOND
    threshold = int(LONG_CHAPTER_HOURS * per_hour)
    assert is_long_chapter(threshold)
    assert not is_long_chapter(threshold - per_hour)
    assert estimated_audio_hours(threshold) == pytest.approx(
        LONG_CHAPTER_HOURS, abs=0.01
    )
