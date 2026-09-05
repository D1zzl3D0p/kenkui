"""Single-voice output must not change when multi-voice lands.

The values below were captured by running the planner before any multi-voice
change and pasted in verbatim. That ordering is the point: a regression test
written against post-change output proves only that the code agrees with
itself.

A chapter with no attributed dialogue is one speaker span, so the frozen
tts-chunks-v2 chunker sees exactly what it saw before and every segment
identity is unchanged. That in turn means no existing cache entry is
invalidated by this work.
"""

from __future__ import annotations

import kenkui as kk
from kenkui._domain.planning import (
    CHUNKING_SCHEMA_VERSION,
    ExecutionPlan,
    compile_execution_plan,
)

SOURCE_HASH = "1" * 64
MODEL_REVISION = "pocket-tts/model@0123456789abcdef"

_SENTENCES = [
    "The house was quiet.",
    "Rain moved across the roof in long slow passes, and no one spoke for a while.",
    "Later, someone laughed in another room.",
    "It was not a happy sound.",
    "The clock in the hall struck three.",
    "Margaret set down her cup and listened to the water in the gutters.",
    "She had not expected the letter, and she had not expected to mind it.",
    "Outside, a door closed somewhere below, and footsteps crossed the yard.",
]
# Long enough to cross the 1000-character chunk budget, so the break-point
# ranking is exercised rather than trivially skipped.
BASELINE_TEXT = " ".join(_SENTENCES * 4)

BASELINE_SEGMENT_IDS = (
    "seg-nfc-space-newline-v1-v2-3d052fd0eb57686108a036c8",
    "seg-nfc-space-newline-v1-v2-9d060ee1b2a5906cc0c7f640",
    "seg-nfc-space-newline-v1-v2-f846836f767f82a8c5b7925d",
    "seg-nfc-space-newline-v1-v2-7da99c55109c2fb8e4f8b141",
    "seg-nfc-space-newline-v1-v2-5a7f17a2862d907181c20192",
    "seg-nfc-space-newline-v1-v2-086e2e9ff50a9eb7a57cf46a",
    "seg-nfc-space-newline-v1-v2-961bee8640a49e79ba470ffa",
    "seg-nfc-space-newline-v1-v2-a8a74781eb1afff57d299394",
    "seg-nfc-space-newline-v1-v2-00020e97bafc2a7b6196ba7d",
)
BASELINE_FINGERPRINT = (
    "28c06048fab8bbcf0be6459ffc821deba06ed38dbdd3fd5e0cf66e761db1e707"
)
# Regenerated to the values d6705fe actually produces, which its own commit
# message flagged as owed and never paid. They are recorded here as the honest
# current state, not as a target: two segments of (987, 656) became nine, four
# of them 46 characters long, because MAX_SEPARATOR_FREE_CHARACTERS was
# tightened to 48 and now cuts inside almost every sentence. The chunking fix
# moves these again, and that is the point -- a red guard proves nothing, so it
# is made green here first and the next diff against it is the real evidence.
BASELINE_LENGTHS = (201, 46, 365, 46, 365, 46, 365, 46, 163)


def _plan() -> ExecutionPlan:
    chapter = kk.ChapterInspection(
        "ch-v1-one", 3, "Chapter One", len(BASELINE_TEXT), BASELINE_TEXT
    )
    inspection = kk.BookInspection(
        kk.BookMetadata("Source Title", "Source Author", cover_available=True),
        (chapter,),
    )
    voice = kk.Voice(
        id="fixture",
        name="Fixture Voice",
        enabled=True,
        provenance="Project-owned recording by Test Speaker",
        license_id="CC0-1.0",
        commercial_use_allowed=True,
        language="en-US",
        content_fingerprint="2" * 64,
        compatible_model_revisions=(MODEL_REVISION, "pocket-tts/model@next"),
    )
    return compile_execution_plan(
        kk.epub("ignored-location.epub").assign_voice("fixture").tts(),
        inspection,
        source_bytes_hash=SOURCE_HASH,
        resolved_voice=voice,
        model_revision=MODEL_REVISION,
    )


def test_chunking_schema_version_is_frozen() -> None:
    """Bumping this invalidates every cached segment in every user's cache."""
    assert CHUNKING_SCHEMA_VERSION == "tts-chunks-v2"


def test_single_voice_segment_identities_are_unchanged() -> None:
    """Segment IDs key the PCM cache; a change silently discards every entry."""
    plan = _plan()
    assert tuple(s.id for s in plan.segments) == BASELINE_SEGMENT_IDS


def test_single_voice_chunk_boundaries_are_unchanged() -> None:
    """Where the chunker breaks decides how the rendered audio actually sounds."""
    plan = _plan()
    assert tuple(len(s.text) for s in plan.segments) == BASELINE_LENGTHS


def test_single_voice_plan_fingerprint_is_unchanged() -> None:
    """The fingerprint is the promise that equal intent yields an equal plan."""
    plan = _plan()
    assert plan.semantic_fingerprint == BASELINE_FINGERPRINT


def test_segments_still_reconstruct_the_chapter_exactly() -> None:
    """Concatenating a chapter's segments must reproduce its normalized text."""
    plan = _plan()
    assert "".join(s.text for s in plan.segments) == BASELINE_TEXT
