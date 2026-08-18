"""Reusable inexpensive validation for public pipeline boundaries."""

from __future__ import annotations

import os
from pathlib import Path

from ._domain.operations import AssignVoice, Operation, SynthesizeSpeech, has_operation
from .errors import ErrorCode


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
    if not has_operation(operations, AssignVoice):
        errors.append(ErrorCode.VOICE_REQUIRED)
    if not has_operation(operations, SynthesizeSpeech):
        errors.append(ErrorCode.TTS_REQUIRED)
    return tuple(errors)
