"""Reusable inexpensive validation for public pipeline boundaries."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ._domain.operations import (
    AssignVoices,
    AttributeQuotes,
    InferCharacters,
    Operation,
    Series,
    SynthesizeSpeech,
    has_operation,
)
from .errors import ErrorCode

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ._characters.store import SeriesRecord


def source_validation_error(path: Path) -> ErrorCode | None:
    """Return a stable error code when a source cannot be read cheaply."""
    if not path.exists():
        return ErrorCode.SOURCE_NOT_FOUND
    if not path.is_file() or not os.access(path, os.R_OK):
        return ErrorCode.SOURCE_NOT_READABLE
    return None


def _names_an_unreachable_voice(operations: tuple[Operation, ...]) -> bool:
    """Return whether a voice only attributed spans could reach was named.

    A cast entry speaks when a span carries its character, and the unknown
    voice speaks when a span carries a character nobody placed. Neither span
    exists without attribution, so both voices are dead intent: the book
    renders entirely in the narrator's voice while the caller believes
    otherwise. A narrator-only assignment names no such voice, which keeps
    single-voice rendering free of this rule.
    """
    if has_operation(operations, AttributeQuotes):
        return False
    return any(
        bool(item.cast) or item.unknown_voice_id != item.narrator_voice_id
        for item in operations
        if isinstance(item, AssignVoices)
    )


def render_intent_errors(operations: tuple[Operation, ...]) -> tuple[ErrorCode, ...]:
    """Return missing explicit rendering requirements in stable order."""
    errors: list[ErrorCode] = []
    if not has_operation(operations, AssignVoices):
        errors.append(ErrorCode.VOICE_REQUIRED)
    # A presence rule rather than an ordering one: attribution needs a roster
    # to answer with, but the planner reads operations by type, so requiring a
    # particular chaining order would constrain callers for nothing.
    if has_operation(operations, AttributeQuotes) and not has_operation(
        operations, InferCharacters
    ):
        errors.append(ErrorCode.ATTRIBUTION_UNAVAILABLE)
    if _names_an_unreachable_voice(operations):
        errors.append(ErrorCode.CAST_UNATTRIBUTED)
    if not has_operation(operations, SynthesizeSpeech):
        errors.append(ErrorCode.TTS_REQUIRED)
    return tuple(errors)


def series_intent_errors(
    operations: Sequence[Operation],
    record: SeriesRecord | None,
    pool_ids: frozenset[str],
) -> tuple[ErrorCode, ...]:
    """Return the ways this render would contradict its series.

    Checked here rather than at render time because both are knowable from
    the store and the operations alone: failing after a book has been
    attributed spends a model pass to learn something free.
    """
    series = next((item for item in operations if isinstance(item, Series)), None)
    if series is None or record is None:
        return ()
    errors: list[ErrorCode] = []
    if not series.allow_recast and any(
        character.voice_id not in pool_ids for character in record.characters
    ):
        errors.append(ErrorCode.SERIES_VOICE_MISSING)
    narrator = next(
        (
            item.narrator_voice_id
            for item in operations
            if isinstance(item, AssignVoices)
        ),
        None,
    )
    if (
        not series.allow_narrator_change
        and narrator is not None
        and narrator != record.narrator_voice_id
    ):
        errors.append(ErrorCode.SERIES_NARRATOR_CHANGED)
    return tuple(errors)
