"""Immutable public execution events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class Started:
    """An execution run started."""

    sequence: int


@dataclass(frozen=True, slots=True)
class StageStarted:
    """A named execution stage started."""

    sequence: int
    stage: str


@dataclass(frozen=True, slots=True)
class StageProgress:
    """A named stage made measurable progress."""

    sequence: int
    stage: str
    completed: int
    total: int
    chapter_id: str | None = None


@dataclass(frozen=True, slots=True)
class StageCompleted:
    """A named stage completed."""

    sequence: int
    stage: str
    elapsed_ms: int | None = None


@dataclass(frozen=True, slots=True)
class CastResolved:
    """The character-to-voice assignment resolved for this run.

    Emitted before any worker exists, so a caller may inspect the cast and
    cancel without paying for a render. Single-voice runs emit it with no
    assignments, keeping one event shape for both castings.
    """

    sequence: int
    stage: str
    narrator_voice_id: str
    unknown_voice_id: str
    assignments: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class Warning:
    """A recoverable condition was observed."""

    sequence: int
    stage: str
    code: str
    message: str
    chapter_id: str | None = None


@dataclass(frozen=True, slots=True)
class Completed:
    """An execution run completed successfully."""

    sequence: int
    elapsed_ms: int | None = None


ExecutionEvent: TypeAlias = (
    CastResolved
    | Started
    | StageStarted
    | StageProgress
    | StageCompleted
    | Warning
    | Completed
)
