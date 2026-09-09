"""Thin public constructors and frozen result values."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

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


def magic_run(
    book_path: str | os.PathLike[str],
    *,
    narrator: str,
    multi: bool = False,
    model: str = "openrouter/deepseek/deepseek-v4-flash",
) -> Result:
    """Render an EPUB with one narrator or an automatically assigned cast."""
    output = Path(book_path).with_suffix(".m4b")
    pipeline = book(book_path)
    if multi:
        pipeline = (
            pipeline.infer_characters(model)
            .attribute_quotes(model)
            .assign_voices(narrator=narrator)
        )
    else:
        pipeline = pipeline.assign_voice(narrator)
    return pipeline.tts().write(output)


def read_lexicon(path: str | os.PathLike[str]) -> dict[str, str]:
    """Read a pronunciation table from a JSON file, validated as a literal is.

    Accepts a plain object of word to replacement, or the shape the shipped
    table uses. The result is an ordinary dict: merge it, edit it, or pass it
    straight to ``Pipeline.pronounce``.
    """
    from ._domain.spoken.lexicon import read_entries

    return read_entries(Path(path))


def builtin_lexicon() -> dict[str, str]:
    """Return a copy of the pronunciation table Kenkui ships.

    A copy rather than the cached table itself, so a caller extending it
    cannot corrupt what every later pipeline in the process reads.
    """
    from ._domain.spoken.lexicon import builtin_entries

    return dict(builtin_entries())


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One inexpensive pipeline validation finding."""

    code: ErrorCode
    message: str
    severity: Literal["error", "warning"] = "error"


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """All inexpensive validation findings for a pipeline."""

    issues: tuple[ValidationIssue, ...] = ()

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        """Issues severe enough to block rendering."""
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        """Issues worth surfacing but not severe enough to block rendering."""
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def is_valid(self) -> bool:
        """Whether validation found no errors."""
        return not self.errors


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
