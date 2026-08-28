"""D6: a pipeline requesting nothing new must render byte-identically.

Segment identities are cache keys. If they shift for a pipeline that asked for
neither pronunciation nor pauses, every user silently re-synthesizes an entire
book. The golden file is captured from the pre-change planner; this test is the
only thing standing between a refactor and that outcome.
"""

from __future__ import annotations

import json
from pathlib import Path

import kenkui as kk
from kenkui._domain.planning import ExecutionPlan, compile_execution_plan

GOLDEN = Path(__file__).parent / "data" / "identity-golden.json"
SOURCE_HASH = "1" * 64
MODEL_REVISION = "pocket-tts/model@0123456789abcdef"
PARAGRAPH_MS = 250
_PARAGRAPH_ONE = (
    "It was 100,000 to one. The cello sounded in 1984, and the 3rd "
    "movement began. He waited by the door for a long while, thinking of "
    "nothing at all, and then he left without speaking to anyone. The "
    "colonel had promised 40% of the takings, which came to $1.50 a head,"
    " and nobody believed a word of it. "
)

_PARAGRAPH_TWO = (
    "Outside, the rain fell in the steady way it always did, and the road"
    " ran 5 km to the crossing where the lamps were lit. She counted the "
    "paces, one hundred and then another hundred, until the counting "
    "itself became the only thing holding her together. "
)

_PARAGRAPH_THREE = (
    "By the 21st of the month the money was gone. Chapter IV of the "
    "ledger recorded it plainly, in the hand of a clerk who had long "
    "since stopped caring what the numbers meant. "
)

# Three paragraphs, each repeated, so the fixture chunks into several
# segments. A single-segment fixture would pin identity without ever
# exercising the chunker it exists to protect.
TEXT = "\n\n".join((_PARAGRAPH_ONE * 3, _PARAGRAPH_TWO * 3, _PARAGRAPH_THREE * 3))


def voice(language: str = "en-US") -> kk.Voice:
    """Build the resolved voice the planner converts into a VoicePlan."""
    return kk.Voice(
        id="eponine",
        name="Eponine",
        enabled=True,
        provenance="Project-owned recording by Test Speaker",
        license_id="CC0-1.0",
        commercial_use_allowed=True,
        language=language,
        content_fingerprint="2" * 64,
        compatible_model_revisions=(MODEL_REVISION,),
        state="loaded",
    )


def inspection() -> kk.BookInspection:
    """Build a one-chapter inspection with prose the new stages would change."""
    chapter = kk.ChapterInspection("ch-v1-one", 0, "Chapter One", len(TEXT), TEXT)
    return kk.BookInspection(
        kk.BookMetadata("Source Title", "Source Author", cover_available=True),
        (chapter,),
    )


def compile_for(pipeline: kk.Pipeline, language: str = "en-US") -> ExecutionPlan:
    """Compile one pipeline against the shared fixture."""
    return compile_execution_plan(
        pipeline,
        inspection(),
        source_bytes_hash=SOURCE_HASH,
        resolved_voice=voice(language),
        model_revision=MODEL_REVISION,
    )


def plain_plan() -> ExecutionPlan:
    """Compile the plan for a pipeline that requests nothing new."""
    return compile_for(kk.epub("book.epub").assign_voice("eponine").tts())


def snapshot() -> dict[str, object]:
    """Reduce a plan to the values that must never drift."""
    plan = plain_plan()
    return {
        "fingerprint": plan.semantic_fingerprint,
        "segment_ids": [segment.id for segment in plan.segments],
        "segment_texts": [segment.text for segment in plan.segments],
        "total": plan.total_speech_characters,
    }


def test_plain_pipeline_identity_is_unchanged() -> None:
    """Segment IDs, texts, fingerprint, and billable total all hold steady."""
    expected = json.loads(GOLDEN.read_text("utf-8"))
    assert snapshot() == expected


def test_billable_total_equals_canonical_text_length() -> None:
    """normalized_speech_characters describes the source, not the spoken form."""
    assert plain_plan().total_speech_characters == len(TEXT)


def spoken_plan(language: str = "en-US") -> ExecutionPlan:
    """Compile the same book with a pronounce() request attached."""
    return compile_for(
        kk.epub("book.epub").pronounce().assign_voice("eponine").tts(),
        language,
    )


def test_spoken_form_changes_segment_text_but_not_the_bill() -> None:
    """The engine hears words; the meter still counts the source characters."""
    plan = spoken_plan()
    spoken_text = "".join(segment.text for segment in plan.segments)
    assert "one hundred thousand" in spoken_text
    assert "chello" in spoken_text
    assert "100,000" not in spoken_text
    assert plan.total_speech_characters == len(TEXT)


def test_spoken_form_changes_segment_identity() -> None:
    """Different spoken output must never reuse a plain pipeline's cache entry."""
    plain = {segment.id for segment in plain_plan().segments}
    spoken = {segment.id for segment in spoken_plan().segments}
    assert plain.isdisjoint(spoken)


def test_non_english_narrator_disables_the_stage() -> None:
    """A voice that cannot speak English number-words leaves text alone."""
    plan = spoken_plan("fr-FR")
    assert "100,000" in "".join(segment.text for segment in plan.segments)
    assert plan.schema_versions.spoken_form is None


def test_synthesized_characters_exceed_the_billable_total() -> None:
    """Spoken form is exactly what makes the two statistics diverge."""
    plan = spoken_plan()
    synthesized = sum(segment.character_count for segment in plan.segments)
    assert plan.total_speech_characters == len(TEXT)
    assert synthesized > plan.total_speech_characters


def test_spoken_form_absent_from_fingerprint_payload_when_off() -> None:
    """An explicit null would change the fingerprint of every plain pipeline.

    Regression guard: the schema_versions block must omit the key entirely
    when the stage is off, not emit it as null.
    """
    assert plain_plan().schema_versions.spoken_form is None
    assert spoken_plan().schema_versions.spoken_form is not None
    assert plain_plan().semantic_fingerprint != spoken_plan().semantic_fingerprint


def paused_plan(**kwargs: int) -> ExecutionPlan:
    """Compile the same book with a pauses() request attached."""
    return compile_for(
        kk.epub("book.epub").pauses(**kwargs).assign_voice("eponine").tts()
    )


def test_chapter_pause_alone_keeps_v2_identity() -> None:
    """chapter_ms forces no chunk break, so every cache entry stays valid."""
    plain = [segment.id for segment in plain_plan().segments]
    paused = paused_plan(chapter_ms=1500)
    assert [segment.id for segment in paused.segments] == plain


def test_paragraph_pause_changes_identity_and_adds_silence() -> None:
    """Enabling a break tier re-chunks and records a gap on the right segment."""
    plain = {segment.id for segment in plain_plan().segments}
    paused = paused_plan(paragraph_ms=PARAGRAPH_MS)
    assert {segment.id for segment in paused.segments}.isdisjoint(plain)
    silence = paused.trailing_silence_ms
    assert len(silence) == len(paused.segments)
    assert PARAGRAPH_MS in silence


def test_silence_table_is_all_zero_without_pauses() -> None:
    """A plain pipeline records no silence anywhere."""
    assert set(plain_plan().trailing_silence_ms) == {0}


def test_final_segment_has_no_trailing_silence() -> None:
    """A book must not end on dead air."""
    plan = paused_plan(chapter_ms=1500, paragraph_ms=250)
    assert plan.trailing_silence_ms[-1] == 0


def test_retuning_a_duration_does_not_change_identity() -> None:
    """Changing 250ms to 600ms must cost no re-synthesis."""
    first = [segment.id for segment in paused_plan(paragraph_ms=250).segments]
    second = [segment.id for segment in paused_plan(paragraph_ms=600).segments]
    assert first == second


def test_retuning_a_duration_does_change_the_fingerprint() -> None:
    """The audio differs, so plan identity must differ even though chunks do not."""
    first = paused_plan(paragraph_ms=250).semantic_fingerprint
    second = paused_plan(paragraph_ms=600).semantic_fingerprint
    assert first != second


def test_pieces_join_to_the_canonical_chapter_text() -> None:
    """Exactness survives the split: chunks still reconstruct the source."""
    plan = paused_plan(paragraph_ms=250, line_ms=100)
    assert "".join(segment.text for segment in plan.segments) == TEXT


def _cache_key_inputs(plan: ExecutionPlan) -> list[tuple[object, ...]]:
    """Project the plan onto exactly what CacheStore.key_for consumes.

    key_for reads the segment's id, chapter, ordinal, character count and
    content hash, plus the model revision and the normalization and render
    schemas. It never reads trailing_silence_ms or the structure schema, so
    this projection is what determines whether a render re-uses cached PCM.
    """
    return [
        (
            segment.id,
            segment.chapter_id,
            segment.ordinal,
            segment.character_count,
            segment.content_hash,
            plan.model_revision,
            plan.schema_versions.normalization,
            plan.schema_versions.render,
        )
        for segment in plan.segments
    ]


def test_retuning_a_duration_leaves_every_cache_key_input_identical() -> None:
    """Retuning pause length must reuse cached PCM, not re-synthesize it."""
    first = paused_plan(paragraph_ms=250)
    second = paused_plan(paragraph_ms=600)
    assert _cache_key_inputs(first) == _cache_key_inputs(second)
    assert first.trailing_silence_ms != second.trailing_silence_ms


def test_changing_the_number_tier_invalidates_every_cache_key_input() -> None:
    """Different spoken output must never reuse another tier's cached PCM."""
    conservative = compile_for(
        kk.epub("book.epub")
        .pronounce(numbers="conservative")
        .assign_voice("eponine")
        .tts()
    )
    standard = compile_for(
        kk.epub("book.epub").pronounce(numbers="standard").assign_voice("eponine").tts()
    )
    shared = {segment.id for segment in conservative.segments} & {
        segment.id for segment in standard.segments
    }
    assert not shared
