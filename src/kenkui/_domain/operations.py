"""Immutable semantic operation records and pure chain validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias, TypeVar

from kenkui.errors import ErrorCode, ValidationError


@dataclass(frozen=True, slots=True)
class SelectChapters:
    """Select explicit stable chapter IDs in caller order."""

    chapter_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SelectChapterRange:
    """Select an inclusive range in materialized spine order."""

    start_id: str
    end_id: str


@dataclass(frozen=True, slots=True)
class NormalizeText:
    """Request the default versioned text normalization."""


@dataclass(frozen=True, slots=True)
class AssignVoice:
    """Assign one voice to all selected speech."""

    voice_id: str


@dataclass(frozen=True, slots=True)
class SynthesizeSpeech:
    """Request explicit TTS synthesis."""


@dataclass(frozen=True, slots=True)
class MetadataIntent:
    """Semantic output metadata overrides and cover choice."""

    title: str | None = None
    author: str | None = None
    cover: Literal["source"] | None = "source"


Operation: TypeAlias = (
    SelectChapters
    | SelectChapterRange
    | NormalizeText
    | AssignVoice
    | SynthesizeSpeech
    | MetadataIntent
)
_OperationT = TypeVar("_OperationT", bound=Operation)


def append_unique(
    operations: tuple[Operation, ...],
    operation: _OperationT,
    *,
    before_tts: bool = False,
) -> tuple[Operation, ...]:
    """Append one operation after enforcing type uniqueness and ordering."""
    if any(type(item) is type(operation) for item in operations):
        raise ValidationError(ErrorCode.DUPLICATE_OPERATION)
    if before_tts and any(isinstance(item, SynthesizeSpeech) for item in operations):
        raise ValidationError(ErrorCode.INVALID_OPERATION_ORDER)
    return (*operations, operation)


def has_operation(operations: tuple[Operation, ...], kind: type[Operation]) -> bool:
    """Return whether an operation chain contains an exact operation family."""
    return any(isinstance(item, kind) for item in operations)
