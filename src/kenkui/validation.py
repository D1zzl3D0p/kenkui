"""Reusable inexpensive validation for public pipeline boundaries."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ._domain.operations import (
    AssignVoices,
    AttributeQuotes,
    InferCharacters,
    Operation,
    SynthesizeSpeech,
    has_operation,
)
from .errors import ErrorCode

if TYPE_CHECKING:
    from pathlib import Path


def source_validation_error(path: Path) -> ErrorCode | None:
    """Return a stable error code when a source cannot be read cheaply."""
    if not path.exists():
        return ErrorCode.SOURCE_NOT_FOUND
    if not path.is_file() or not os.access(path, os.R_OK):
        return ErrorCode.SOURCE_NOT_READABLE
    return None


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
    if not has_operation(operations, SynthesizeSpeech):
        errors.append(ErrorCode.TTS_REQUIRED)
    return tuple(errors)
