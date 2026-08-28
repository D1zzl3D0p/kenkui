"""Silence padding must agree with part size, chapter markers, and duration."""

from __future__ import annotations

import kenkui as kk
from kenkui._domain.planning import ExecutionPlan, compile_execution_plan
from kenkui._execution.coordinator import _padded
from kenkui._tts.protocols import SegmentAudio

SAMPLE_RATE = 24_000
MODEL_REVISION = "pocket-tts/model@0123456789abcdef"
CHAPTER_MS = 1000
HALF_SECOND_FRAMES = SAMPLE_RATE // 2


def audio(frames: int, chapter: str = "ch-1", segment: str = "seg-1") -> SegmentAudio:
    """Build one segment's post-render metadata."""
    return SegmentAudio(
        segment, chapter, SAMPLE_RATE, 1, frames, frames * 1000 // SAMPLE_RATE
    )


def test_padding_extends_byte_count_consistently() -> None:
    """byte_count is derived, so one padded field corrects the part-size check."""
    item, padding = _padded(audio(SAMPLE_RATE), 500)
    assert item.frame_count == SAMPLE_RATE + HALF_SECOND_FRAMES
    assert item.byte_count == item.frame_count * item.channels * 2
    assert item.duration_ms == 1500  # noqa: PLR2004 - one second plus 500ms
    assert len(padding) == HALF_SECOND_FRAMES * 2
    assert padding == bytes(len(padding))


def test_zero_silence_leaves_metadata_untouched() -> None:
    """A plain pipeline's audio metadata is bit-for-bit what the worker produced."""
    item = audio(SAMPLE_RATE)
    padded, padding = _padded(item, 0)
    assert padded == item
    assert padding == b""


def test_padding_bytes_match_the_declared_byte_count() -> None:
    """The spilled part must be exactly what the assembler will demand."""
    original = audio(SAMPLE_RATE)
    padded, padding = _padded(original, 250)
    assert original.byte_count + len(padding) == padded.byte_count


def two_chapter_plan(chapter_ms: int) -> ExecutionPlan:
    """Compile a real two-chapter plan so the arithmetic is not stubbed."""
    voice = kk.Voice(
        id="eponine",
        name="Eponine",
        enabled=True,
        provenance="Project-owned recording by Test Speaker",
        license_id="CC0-1.0",
        commercial_use_allowed=True,
        language="en-US",
        content_fingerprint="2" * 64,
        compatible_model_revisions=(MODEL_REVISION,),
        state="loaded",
    )
    texts = (("one", "He woke early.\n\nShe slept on."), ("two", "They left."))
    chapters = tuple(
        kk.ChapterInspection(f"ch-v1-{name}", index, f"Chapter {name}", len(text), text)
        for index, (name, text) in enumerate(texts)
    )
    pipeline = (
        kk.epub("book.epub").pauses(chapter_ms=chapter_ms).assign_voice("eponine").tts()
    )
    return compile_execution_plan(
        pipeline,
        kk.BookInspection(kk.BookMetadata("T", "A", cover_available=True), chapters),
        source_bytes_hash="1" * 64,
        resolved_voice=voice,
        model_revision=MODEL_REVISION,
    )


def test_inter_chapter_gap_lands_in_the_preceding_chapter() -> None:
    """Skipping to a chapter must land on speech, not on silence."""
    plan = two_chapter_plan(CHAPTER_MS)
    silence = plan.trailing_silence_ms
    last_of_first = max(
        index
        for index, segment in enumerate(plan.segments)
        if segment.chapter_id == "ch-v1-one"
    )
    assert silence[last_of_first] == CHAPTER_MS
    assert silence[-1] == 0
    assert sum(silence) == CHAPTER_MS
