"""Speaker spans partition chapter text; the frozen chunker runs inside each.

Attribution supplies spans rather than replacing segmentation, so the
"".join(chunks) == text invariant survives by construction: a partition of a
partition is still a partition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import kenkui as kk
from kenkui._domain.planning import (
    ExecutionPlan,
    SpeakerSpan,
    compile_execution_plan,
)
from kenkui._execution.cache import CacheStore
from kenkui._execution.process_pool import EngineSpecification
from kenkui._tts.pocket import (
    PocketEngineConfig,
    PocketManifestFile,
    VoiceAsset,
)
from kenkui._tts.protocols import SynthesisTask

if TYPE_CHECKING:
    from pathlib import Path

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
    paragraph_ms: int = 0,
) -> ExecutionPlan:
    source = kk.epub("ignored-location.epub")
    if paragraph_ms:
        source = source.pauses(paragraph_ms=paragraph_ms)
    return compile_execution_plan(
        source.assign_voice("eponine").tts(),
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


def _pocket_spec(root: Path, *digests: str) -> EngineSpecification:
    """Build a pocket specification whose cast holds exactly these digests."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "english.yaml").write_text("model: english\n")
    voices = root / "voices"
    voices.mkdir(exist_ok=True)
    assets = []
    for digest in digests:
        (voices / f"{digest[:4]}.safetensors").write_bytes(b"\x00" * 8)
        assets.append(
            VoiceAsset(
                path=str(voices / f"{digest[:4]}.safetensors"),
                sha256=digest,
                variety="built-in",
                provenance="Project-owned recording by Test Speaker",
                license_id="CC0-1.0",
                rights="Test rights.",
                commercial_use_allowed=True,
            )
        )
    return EngineSpecification.pocket(
        PocketEngineConfig(
            model_root=str(root),
            config_path=str(root / "english.yaml"),
            model_revision=MODEL_REVISION,
            package_version="2.1.0",
            files=(PocketManifestFile("english.yaml", 10, "d" * 64),),
            voices=tuple(assets),
            cloning_capable=False,
            sample_rate_hz=24_000,
        )
    )


def test_adding_a_cast_voice_does_not_rekey_narration(tmp_path: Path) -> None:
    """A re-cast must not re-synthesize prose the new voice never speaks.

    The engine config lists the whole cast. If all of it entered every
    segment's key, casting one more character would invalidate every
    narration entry already rendered and paid for.
    """
    plan = _compile(spans=_spans())
    store = CacheStore.__new__(CacheStore)
    narration = plan.segments[0]
    assert NARRATOR.content_fingerprint is not None
    assert JAVERT.content_fingerprint is not None
    task = SynthesisTask(
        narration.id,
        narration.chapter_id,
        narration.text,
        24_000,
        1,
        1024,
        NARRATOR.content_fingerprint,
    )
    alone = _pocket_spec((tmp_path / "alone").resolve(), NARRATOR.content_fingerprint)
    with_cast = _pocket_spec(
        (tmp_path / "cast").resolve(),
        NARRATOR.content_fingerprint,
        JAVERT.content_fingerprint,
    )
    assert store.key_for(plan, narration, task, alone) == store.key_for(
        plan, narration, task, with_cast
    )


def test_the_rendering_voice_still_enters_the_key(tmp_path: Path) -> None:
    """Narrowing the engine material must not drop voice identity entirely."""
    plan = _compile(spans=_spans())
    store = CacheStore.__new__(CacheStore)
    assert NARRATOR.content_fingerprint is not None
    assert JAVERT.content_fingerprint is not None
    spec = _pocket_spec(
        (tmp_path / "cast").resolve(),
        NARRATOR.content_fingerprint,
        JAVERT.content_fingerprint,
    )

    def key(index: int) -> str:
        segment = plan.segments[index]
        return store.key_for(
            plan,
            segment,
            SynthesisTask(
                segment.id, segment.chapter_id, segment.text, 24_000, 1, 1024, ""
            ),
            spec,
        )

    assert key(0) != key(1)


PARAGRAPH_MS = 500
PARA_NARRATION = "He waited.\n\n"
PARA_DIALOGUE = '"You are late."'
PARA_TAIL = " She left."
PARA_TEXT = PARA_NARRATION + PARA_DIALOGUE + PARA_TAIL


def _para_spans() -> tuple[SpeakerSpan, ...]:
    start = len(PARA_NARRATION)
    stop = start + len(PARA_DIALOGUE)
    return (
        SpeakerSpan("ch-v1-one", 0, start, None),
        SpeakerSpan("ch-v1-one", start, stop, "javert"),
        SpeakerSpan("ch-v1-one", stop, len(PARA_TEXT), None),
    )


def test_paragraph_pause_survives_a_span_boundary() -> None:
    """A structural gap belongs to the chapter, not to the span it falls in.

    Deciding pauses inside each span makes that span's last block look like
    the end of the text, so the gap is suppressed. A paragraph break before a
    line of dialogue is exactly that shape, which is most dialogue in a book.
    """
    plan = _compile(spans=_para_spans(), text=PARA_TEXT, paragraph_ms=PARAGRAPH_MS)
    assert plan.segments[0].text == PARA_NARRATION
    assert plan.trailing_silence_ms[0] == PARAGRAPH_MS


def test_spans_with_pauses_still_partition_chapter_text() -> None:
    """Deciding gaps chapter-wide must not disturb the partition."""
    plan = _compile(spans=_para_spans(), text=PARA_TEXT, paragraph_ms=PARAGRAPH_MS)
    assert "".join(segment.text for segment in plan.segments) == PARA_TEXT
