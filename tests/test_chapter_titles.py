"""Announcements preserve source coordinates, pacing, and exact audio markers."""

# ruff: noqa: PLR2004
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

import kenkui as kk
from helpers import make_epub, xhtml
from kenkui._audio.m4b import FakeArtifactAssembler, chapter_frame_boundaries_ms
from kenkui._domain.operations import ChapterTitles
from kenkui._domain.planning import compile_execution_plan
from kenkui._execution.cache import CacheStore
from kenkui._execution.coordinator import ExecutionBindings
from kenkui._execution.process_pool import EngineSpecification
from kenkui._tts.protocols import SegmentAudio
from test_execution import _voice as execution_voice
from test_planning import MODEL_REVISION, SOURCE_HASH, _voice

if TYPE_CHECKING:
    from pathlib import Path

    from kenkui._domain.planning import ExecutionPlan


def _chapter(
    identity: str, title: str, text: str, *, heading: bool = False
) -> kk.ChapterInspection:
    return kk.ChapterInspection(
        identity,
        0,
        title,
        len(text),
        text,
        heading_ranges=((0, len(title)),) if heading else (),
        title_source="navigation",
    )


def _plan(
    chapters: tuple[kk.ChapterInspection, ...], pipeline: kk.Pipeline | None = None
) -> ExecutionPlan:
    pipeline = pipeline or kk.book("unused.epub").assign_voice("fixture").tts()
    return compile_execution_plan(
        pipeline,
        kk.BookInspection(kk.BookMetadata(), chapters),
        source_bytes_hash=SOURCE_HASH,
        resolved_voice=_voice(),
        model_revision=MODEL_REVISION,
    )


def test_new_synthesis_defaults_on_and_opt_out_preserves_body() -> None:
    """The default announces TOC-only labels; opt-out emits canonical text alone."""
    chapter = _chapter("c1", "Chapter 12", "An epigraph.\n\nThe story.")
    pipeline = kk.book("unused.epub").assign_voice("fixture").tts()
    assert next(
        op for op in pipeline.operations if isinstance(op, ChapterTitles)
    ).enabled
    plan = _plan((chapter,), pipeline)
    assert [s.text for s in plan.segments] == ["Chapter 12", chapter.text]
    assert plan.trailing_silence_ms == (750, 0)
    assert plan.total_speech_characters == len(chapter.text) + len(chapter.title)
    assert plan.output.chapters[0].speech_characters == plan.total_speech_characters
    assert all(
        s.chapter_id == chapter.id and s.voice_id == "fixture" for s in plan.segments
    )
    disabled = _plan((chapter,), pipeline.chapter_titles(enabled=False))
    assert [s.text for s in disabled.segments] == [chapter.text]
    assert disabled.total_speech_characters == len(chapter.text)


def test_title_and_chapter_pauses_have_exact_frame_boundaries() -> None:
    """Markers include added words and gaps once, and no final dead air."""
    chapters = (_chapter("a", "One", "Body one."), _chapter("b", "Two", "Body two."))
    pipeline = (
        kk.book("unused.epub").assign_voice("fixture").pauses(chapter_ms=1500).tts()
    )
    plan = _plan(chapters, pipeline)
    assert plan.trailing_silence_ms == (750, 1500, 750, 0)
    audio = tuple(
        SegmentAudio(s.id, s.chapter_id, 1000, 1, 1000 + gap, 1000 + gap)
        for s, gap in zip(plan.segments, plan.trailing_silence_ms, strict=True)
    )
    assert chapter_frame_boundaries_ms(plan, audio) == (0, 4250, 7000)
    retuned = _plan(chapters, pipeline.chapter_titles(pause_ms=250))
    assert [s.id for s in retuned.segments] == [s.id for s in plan.segments]
    assert retuned.semantic_fingerprint != plan.semantic_fingerprint
    assert retuned.trailing_silence_ms == (250, 1500, 250, 0)


def test_existing_heading_is_reused_and_manual_silence_wins() -> None:
    """Duplicate detection preserves original spelling and applies one title gap."""
    chapter = _chapter("c1", "Chapter 1", "Chapter 1\n\nBody.", heading=True)
    pipeline = (
        kk.book("unused.epub")
        .assign_voice("fixture")
        .pauses(heading_after_ms=1000)
        .tts()
    )
    plan = _plan((chapter,), pipeline)
    assert "".join(s.text for s in plan.segments) == chapter.text
    assert len(plan.segments) == 2
    assert plan.trailing_silence_ms == (1000, 0)
    manual = _plan(
        (chapter,),
        kk.book("unused.epub")
        .assign_voice("fixture")
        .silence(0, where={"paragraph": 1})
        .tts(),
    )
    assert manual.trailing_silence_ms == (0, 0)
    assert plan.total_speech_characters == len(chapter.text)


def test_resolver_skips_fallbacks_and_never_matches_later_headings() -> None:
    """Editorial overrides are explicit and epigraph credits are not duplicates."""
    chapters = (
        replace(_chapter("a", "Untitled section 1", "Body."), title_source="generated"),
        replace(
            _chapter("b", "Chapter 2", "An epigraph.\n\nChapter 2"),
            heading_ranges=((14, 23),),
        ),
        replace(_chapter("c", "Chapter 3", "Body."), title_source="unknown"),
    )
    inspection = kk.BookInspection(kk.BookMetadata(), chapters)
    assert [a.kind for a in kk.resolve_chapter_titles(inspection)] == [
        "omitted",
        "inserted",
        "omitted",
    ]
    resolved = kk.resolve_chapter_titles(
        inspection, overrides={"a": "The Welbeck Fragment", "b": None}
    )
    assert resolved[0].text == "The Welbeck Fragment"
    assert resolved[1].kind == "omitted"
    assert all(
        a.kind == "omitted"
        for a in kk.resolve_chapter_titles(inspection, enabled=False)
    )


def test_title_only_heading_has_no_trailing_pause() -> None:
    """A one-heading book must not end on silence."""
    plan = _plan((_chapter("a", "One", "One", heading=True),))
    assert [s.text for s in plan.segments] == ["One"]
    assert plan.trailing_silence_ms == (0,)


def test_parser_retains_title_provenance(tmp_path: Path) -> None:
    """Opening headings and generated labels remain distinguishable."""
    path = make_epub(
        tmp_path / "book.epub",
        chapters={
            "a": xhtml("<h1>One</h1><p>Body.</p>"),
            "b": xhtml("<p>Body.</p>", title=""),
        },
        spine=["a", "b"],
    )
    inspection = kk.book(path).inspect()
    assert [c.title_source for c in inspection.chapters] == ["heading", "generated"]


@pytest.mark.parametrize("pause", [-1, 60001, True, 1.5])
def test_title_pause_rejects_invalid_values(pause: int) -> None:
    """Do not coerce invalid timing into a render setting."""
    with pytest.raises(kk.ValidationError):
        kk.book("unused.epub").chapter_titles(pause_ms=pause)


def test_long_title_is_bounded_and_only_last_title_chunk_has_pause() -> None:
    """Authored labels still obey engine budgets without internal title gaps."""
    chapter = _chapter("a", "A very long title. " * 100, "Body.")
    plan = _plan((chapter,))
    titles = plan.segments[:-1]
    assert len(titles) > 1
    assert all(len(s.text) <= 1000 for s in titles)
    assert plan.trailing_silence_ms[:-2] == (0,) * (len(titles) - 1)
    assert plan.trailing_silence_ms[-2:] == (750, 0)


def test_title_only_chapter_keeps_interchapter_pause() -> None:
    """An automatic title gap cannot masquerade as a manual chapter override."""
    chapters = (
        _chapter("a", "One", "One", heading=True),
        _chapter("b", "Two", "Body."),
    )
    pipeline = (
        kk.book("unused.epub").assign_voice("fixture").pauses(chapter_ms=1500).tts()
    )
    assert _plan(chapters, pipeline).trailing_silence_ms == (1500, 750, 0)


def test_coincident_opening_heading_pause_uses_maximum() -> None:
    """An epigraph heading after an announcement adds no second silence."""
    chapter = replace(
        _chapter("a", "Chapter 1", "Epigraph\n\nBody."), heading_ranges=((0, 8),)
    )
    pipeline = (
        kk.book("unused.epub")
        .assign_voice("fixture")
        .pauses(heading_before_ms=1200)
        .tts()
    )
    plan = _plan((chapter,), pipeline)
    assert plan.trailing_silence_ms[0] == 1200


def test_announcements_always_use_narrator_in_a_character_cast() -> None:
    """Neither generated titles nor a reused heading inherits a character voice."""
    chapter = _chapter("a", "Chapter 1", "Chapter 1\n\nBody.", heading=True)
    pipeline = (
        kk.book("unused.epub")
        .assign_voices(narrator="fixture", cast={"character": "actor"})
        .tts()
    )
    actor = _voice(id="actor")
    plan = compile_execution_plan(
        pipeline,
        kk.BookInspection(kk.BookMetadata(), (chapter,)),
        source_bytes_hash=SOURCE_HASH,
        resolved_voice=_voice(),
        model_revision=MODEL_REVISION,
        cast_voices=(actor,),
        assignments={"character": "actor"},
        spans=(kk.SpeakerSpan("a", 0, len(chapter.text), "character"),),
    )
    assert plan.segments[0].voice_id == "fixture"
    assert plan.segments[-1].voice_id == "actor"


def test_rendered_duration_and_pause_only_cache_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real coordinator inserts each pause once and reuses title speech."""
    source = make_epub(
        tmp_path / "book.epub",
        chapters={"one": xhtml("<p>Body.</p>", title="One")},
        spine=["one"],
    )
    bindings = ExecutionBindings(
        EngineSpecification.fake(),
        FakeArtifactAssembler(),
        execution_voice(),
        "fake-v1",
        cache_store=CacheStore(tmp_path / "cache"),
    )
    monkeypatch.setattr(
        "kenkui._resolution._execution_bindings", lambda *_args, **_kwargs: bindings
    )
    pipeline = kk.book(source).assign_voice("narrator").tts()
    first = pipeline.write(tmp_path / "one.m4b", workers=1, keep_audio_cache=True)
    assert first.stats.duration_ms == (len("One") + len("Body.")) * 10 + 750

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("retuning title pause must not synthesize again")

    monkeypatch.setattr("kenkui._execution.coordinator.render_spawned", forbidden)
    second = pipeline.chapter_titles(pause_ms=250).write(
        tmp_path / "two.m4b", workers=1, keep_audio_cache=True
    )
    assert second.stats.duration_ms == first.stats.duration_ms - 500
