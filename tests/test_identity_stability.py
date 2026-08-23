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
