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
    Started | StageStarted | StageProgress | StageCompleted | Warning | Completed
)
