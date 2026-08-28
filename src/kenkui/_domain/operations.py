"""Immutable semantic operation records and pure chain validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias, TypeVar

from kenkui.errors import ErrorCode, ValidationError

if TYPE_CHECKING:
    from pathlib import Path


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
class SpokenForm:
    """How canonical text becomes the string the engine actually speaks.

    Never affects the canonical text, and therefore never affects billing,
    inspection, chapter identity, or attribution offsets.
    """

    numbers: str = "conservative"
    builtin_lexicon: bool = True
    # Sorted pairs rather than a mapping: an operation record must be hashable
    # and compare equal regardless of how the caller ordered it.
    lexicon: tuple[tuple[str, str], ...] = ()
    # Per-feature overrides on the tier, same reasoning. Empty means the tier
    # decides everything, which is what every caller before these existed did.
    features: tuple[tuple[str, bool], ...] = ()


@dataclass(frozen=True, slots=True)
class Pauses:
    """Silence durations for each structural boundary, in milliseconds.

    Zero disables a tier completely, including the chunk-break cost it would
    otherwise impose. Structurally satisfies the domain ``PauseSpec`` protocol.
    """

    chapter_ms: int = 0
    heading_before_ms: int = 0
    heading_after_ms: int = 0
    paragraph_ms: int = 0
    line_ms: int = 0


@dataclass(frozen=True, slots=True)
class InferCharacters:
    """Derive a character roster with the named model."""

    model_id: str


@dataclass(frozen=True, slots=True)
class AttributeQuotes:
    """Assign a speaker to each quoted run with the named model."""

    model_id: str


@dataclass(frozen=True, slots=True)
class AssignVoices:
    """Cast narrator, unknown, and characters to voices.

    Single voice is the degenerate case, not a separate operation: one
    VoicePlan and one renderer serve both, which is what keeps multi-voice a
    composition rather than a second architecture.
    """

    narrator_voice_id: str
    unknown_voice_id: str
    # Sorted pairs rather than a mapping: an operation record has to be
    # hashable and compare equal regardless of how the caller ordered it.
    cast: tuple[tuple[str, str], ...] = ()
    method: str = "gendered"


@dataclass(frozen=True, slots=True)
class SynthesizeSpeech:
    """Request explicit TTS synthesis."""


@dataclass(frozen=True, slots=True)
class MetadataIntent:
    """Semantic output metadata overrides and cover choice."""

    title: str | None = None
    author: str | None = None
    cover: Literal["source"] | Path | None = "source"


@dataclass(frozen=True, slots=True)
class Series:
    """Which series this book belongs to, and how strictly to honour it.

    Declared, never derived: no EPUB in practice carries series metadata,
    so there is nothing to read it from.
    """

    series_id: str
    book: int | None = None
    allow_recast: bool = False
    allow_narrator_change: bool = False


Operation: TypeAlias = (
    SelectChapters
    | SelectChapterRange
    | SpokenForm
    | Pauses
    | InferCharacters
    | AttributeQuotes
    | AssignVoices
    | SynthesizeSpeech
    | MetadataIntent
    | Series
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
