"""Speaker spans partition chapter text; the frozen chunker runs inside each.

Attribution supplies spans rather than replacing segmentation, so the
"".join(chunks) == text invariant survives by construction: a partition of a
partition is still a partition.
"""

from __future__ import annotations

import kenkui as kk
from kenkui._domain.planning import (
    ExecutionPlan,
    SpeakerSpan,
    compile_execution_plan,
)
from kenkui._execution.cache import CacheStore
from kenkui._execution.process_pool import EngineSpecification
from kenkui._tts.protocols import SynthesisTask

SOURCE_HASH = "1" * 64
MODEL_REVISION = "pocket-tts/model@0123456789abcdef"

NARRATION_A = "The inspector waited by the door. "
DIALOGUE = '"You are late again," he said.'
NARRATION_B = " Nobody answered him."
TEXT = NARRATION_A + DIALOGUE + NARRATION_B


def _voice(voice_id: str, fingerprint: str) -> kk.Voice:
    """Build a resolved public Voice, which is what the planner converts."""
    return kk.Voice(
        id=voice_id,
        name=voice_id.title(),
        enabled=True,
        provenance="Project-owned recording by Test Speaker",
        license_id="CC0-1.0",
        commercial_use_allowed=True,
        language="en-US",
        content_fingerprint=fingerprint,
        compatible_model_revisions=(MODEL_REVISION,),
        state="loaded",
    )


NARRATOR = _voice("eponine", "2" * 64)
JAVERT = _voice("charles", "3" * 64)


def _inspection(text: str = TEXT) -> kk.BookInspection:
    chapter = kk.ChapterInspection("ch-v1-one", 3, "Chapter One", len(text), text)
    return kk.BookInspection(
        kk.BookMetadata("Source Title", "Source Author", cover_available=True),
        (chapter,),
    )


def _spans() -> tuple[SpeakerSpan, ...]:
    start = len(NARRATION_A)
    stop = start + len(DIALOGUE)
    return (
        SpeakerSpan("ch-v1-one", 0, start, None),
        SpeakerSpan("ch-v1-one", start, stop, "javert"),
        SpeakerSpan("ch-v1-one", stop, len(TEXT), None),
    )


def _compile(
    *,
    spans: tuple[SpeakerSpan, ...] = (),
    text: str = TEXT,
) -> ExecutionPlan:
    return compile_execution_plan(
        kk.epub("ignored-location.epub").assign_voice("eponine").tts(),
        _inspection(text),
        source_bytes_hash=SOURCE_HASH,
        resolved_voice=NARRATOR,
        model_revision=MODEL_REVISION,
        cast_voices=(JAVERT,),
        assignments={"javert": "charles"},
        spans=spans,
    )


def test_spans_partition_chapter_text_exactly() -> None:
    """A gap or overlap here silently drops or duplicates rendered audio."""
    plan = _compile(spans=_spans())
    assert "".join(segment.text for segment in plan.segments) == TEXT


def test_each_segment_carries_its_speaker_and_voice() -> None:
    """Every segment records who speaks and which voice renders it."""
    plan = _compile(spans=_spans())
    assert [s.speaker_id for s in plan.segments] == [None, "javert", None]
    assert [s.voice_id for s in plan.segments] == ["eponine", "charles", "eponine"]


def test_narration_uses_the_narrator_voice() -> None:
    """Unattributed prose stays with the narrator."""
    plan = _compile(spans=_spans())
    narration = [s for s in plan.segments if s.speaker_id is None]
    assert {s.voice_id for s in narration} == {"eponine"}


def test_unassigned_speaker_falls_to_the_unknown_voice() -> None:
    """Unknown is an explicit role, not a guess."""
    spans = (SpeakerSpan("ch-v1-one", 0, len(TEXT), "stranger"),)
    plan = _compile(spans=spans)
    assert plan.segments[0].voice_id == "eponine"


def test_identical_text_from_two_speakers_yields_distinct_segments() -> None:
    """Two characters saying the same words must not collide in the cache."""
    line = "I know."
    text = line + line
    spans = (
        SpeakerSpan("ch-v1-one", 0, len(line), "javert"),
        SpeakerSpan("ch-v1-one", len(line), len(text), None),
    )
    plan = _compile(spans=spans, text=text)
    first, second = plan.segments
    assert first.text == second.text
    assert first.id != second.id


def test_no_spans_means_one_narration_span_per_chapter() -> None:
    """The single-voice path must reach the chunker exactly as it did before."""
    plan = _compile()
    assert all(segment.speaker_id is None for segment in plan.segments)
    assert "".join(segment.text for segment in plan.segments) == TEXT


def test_cache_keys_differ_for_two_speakers_of_identical_text() -> None:
    """Voice identity must enter the key per segment, not per plan.

    Without this, a cast where two characters say the same words would serve
    one character's audio for the other, from a cache hit that looks correct.
    """
    line = "I know."
    text = line + line
    spans = (
        SpeakerSpan("ch-v1-one", 0, len(line), "javert"),
        SpeakerSpan("ch-v1-one", len(line), len(text), None),
    )
    plan = _compile(spans=spans, text=text)
    spec = EngineSpecification.fake()
    store = CacheStore.__new__(CacheStore)

    def key(index: int) -> str:
        segment = plan.segments[index]
        task = SynthesisTask(
            segment.id, segment.chapter_id, segment.text, 24_000, 1, 1024, ""
        )
        return store.key_for(plan, segment, task, spec)

    assert plan.segments[0].text == plan.segments[1].text
    assert key(0) != key(1)


def test_worker_task_carries_each_segment_s_own_voice_digest() -> None:
    """The bridge from a segment's voice_id to the digest a worker routes by.

    Getting this wrong renders every segment in whichever voice the engine
    happened to derive first, which is well-formed audio and therefore
    invisible to every check except listening.
    """
    plan = _compile(spans=_spans())
    digests = [
        plan.cast.voice_for(segment.speaker_id).content_fingerprint
        for segment in plan.segments
    ]
    assert digests == [
        NARRATOR.content_fingerprint,
        JAVERT.content_fingerprint,
        NARRATOR.content_fingerprint,
    ]
