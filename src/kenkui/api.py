"""Thin public constructors and frozen result values."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import ErrorCode, SourceError

if TYPE_CHECKING:
    import os

    from .pipeline import Pipeline


def epub(path: str | os.PathLike[str]) -> Pipeline:
    """Record EPUB source intent without opening or parsing the path."""
    from .pipeline import Pipeline, Source

    return Pipeline(Source(Path(path), "epub"))


def book(path: str | os.PathLike[str]) -> Pipeline:
    """Dispatch a supported source by its format marker without reading it."""
    source_path = Path(path)
    if source_path.suffix.lower() != ".epub":
        raise SourceError(ErrorCode.UNSUPPORTED_FORMAT)
    return epub(source_path)


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One inexpensive pipeline validation finding."""

    code: ErrorCode
    message: str


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """All inexpensive validation findings for a pipeline."""

    issues: tuple[ValidationIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        """Whether validation found no issues."""
        return not self.issues


@dataclass(frozen=True, slots=True)
class ExecutionStats:
    """Stable technical statistics for one successful execution."""

    normalized_speech_characters: int
    synthesized_characters: int
    synthesized_segments: int
    rendered_chapters: int
    duration_ms: int


@dataclass(frozen=True, slots=True)
class Result:
    """A published artifact and its execution statistics."""

    output: Path
    stats: ExecutionStats
