"""Lazy immutable public audiobook pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ._domain.operations import (
    AssignVoice,
    MetadataIntent,
    NormalizeText,
    Operation,
    SelectChapterRange,
    SelectChapters,
    SynthesizeSpeech,
    append_unique,
    has_operation,
)
from ._domain.selection import select_chapters, select_range
from ._epub.parser import inspect_epub
from ._execution.coordinator import ExecutionBindings, execute_sequential
from .api import Result, ValidationIssue, ValidationResult
from .errors import (
    EncodingError,
    ErrorCode,
    SourceError,
    ValidationError,
)
from .voices import Voice

if TYPE_CHECKING:
    from collections.abc import Callable

    from .cancellation import CancellationToken
    from .events import ExecutionEvent
    from .inspection import BookInspection


@dataclass(frozen=True, slots=True)
class Source:
    """A lazily recorded readable source."""

    path: Path
    format: Literal["epub"]


@dataclass(frozen=True, slots=True)
class Pipeline:
    """A source plus ordered immutable audiobook intent."""

    source: Source
    operations: tuple[Operation, ...] = ()

    @property
    def metadata_intent(self) -> MetadataIntent | None:
        """Return recorded metadata intent, if present."""
        return next(
            (item for item in self.operations if isinstance(item, MetadataIntent)),
            None,
        )

    def select_chapters(self, *chapter_ids: str) -> Pipeline:
        """Return a branch selecting explicit stable chapter IDs."""
        normalized = tuple(chapter_id.strip() for chapter_id in chapter_ids)
        if not normalized or any(not chapter_id for chapter_id in normalized):
            raise ValidationError(ErrorCode.EMPTY_SELECTION)
        if len(set(normalized)) != len(normalized):
            raise ValidationError(ErrorCode.DUPLICATE_CHAPTER_ID)
        if has_operation(self.operations, SelectChapterRange):
            raise ValidationError(ErrorCode.DUPLICATE_OPERATION)
        return self._append(SelectChapters(normalized), before_tts=True)

    def select_chapter_range(self, start_id: str, end_id: str) -> Pipeline:
        """Return a branch selecting an inclusive stable-ID spine range."""
        start = start_id.strip()
        end = end_id.strip()
        if not start or not end:
            raise ValidationError(ErrorCode.EMPTY_SELECTION)
        if has_operation(self.operations, SelectChapters):
            raise ValidationError(ErrorCode.DUPLICATE_OPERATION)
        return self._append(SelectChapterRange(start, end), before_tts=True)

    def normalize_text(self) -> Pipeline:
        """Return a branch requesting default deterministic normalization."""
        return self._append(NormalizeText(), before_tts=True)

    def assign_voice(self, voice: str | Voice) -> Pipeline:
        """Return a branch assigning one stable voice ID."""
        voice_id = voice.id if isinstance(voice, Voice) else voice
        voice_id = voice_id.strip()
        if not voice_id:
            raise ValidationError(ErrorCode.INVALID_VOICE)
        return self._append(AssignVoice(voice_id), before_tts=True)

    def tts(self) -> Pipeline:
        """Return a branch with explicit synthesis intent."""
        if not has_operation(self.operations, AssignVoice):
            raise ValidationError(ErrorCode.VOICE_REQUIRED)
        return self._append(SynthesizeSpeech())

    def metadata(
        self,
        *,
        title: str | None = None,
        author: str | None = None,
        cover: Literal["source"] | None = "source",
    ) -> Pipeline:
        """Return a branch with output metadata inheritance and overrides."""
        if cover not in ("source", None):
            raise ValidationError(ErrorCode.INVALID_METADATA)
        if title is not None and not title.strip():
            raise ValidationError(ErrorCode.INVALID_METADATA)
        if author is not None and not author.strip():
            raise ValidationError(ErrorCode.INVALID_METADATA)
        return self._append(MetadataIntent(title, author, cover))

    def validate(self) -> ValidationResult:
        """Perform inexpensive source and operation validation without parsing."""
        issues: list[ValidationIssue] = []
        path = self.source.path
        if not path.exists():
            issues.append(_issue(ErrorCode.SOURCE_NOT_FOUND))
        elif not path.is_file() or not os.access(path, os.R_OK):
            issues.append(_issue(ErrorCode.SOURCE_NOT_READABLE))
        if not has_operation(self.operations, AssignVoice):
            issues.append(_issue(ErrorCode.VOICE_REQUIRED))
        if not has_operation(self.operations, SynthesizeSpeech):
            issues.append(_issue(ErrorCode.TTS_REQUIRED))
        return ValidationResult(tuple(issues))

    def inspect(self) -> BookInspection:
        """Parse and select immutable normalized source data without synthesis."""
        if not self.source.path.exists():
            raise SourceError(ErrorCode.SOURCE_NOT_FOUND)
        if not self.source.path.is_file() or not os.access(self.source.path, os.R_OK):
            raise SourceError(ErrorCode.SOURCE_NOT_READABLE)
        inspection = inspect_epub(self.source.path)
        explicit = next(
            (item for item in self.operations if isinstance(item, SelectChapters)), None
        )
        range_selection = next(
            (item for item in self.operations if isinstance(item, SelectChapterRange)),
            None,
        )
        if explicit is not None:
            chapters = select_chapters(inspection.chapters, explicit.chapter_ids)
        elif range_selection is not None:
            chapters = select_range(
                inspection.chapters,
                range_selection.start_id,
                range_selection.end_id,
            )
        else:
            chapters = inspection.chapters
        return type(inspection)(inspection.metadata, chapters)

    def write(
        self,
        output: str | os.PathLike[str],
        *,
        on_event: Callable[[ExecutionEvent], None] | None = None,
        cancel: CancellationToken | None = None,
        workers: int | Literal["auto"] = "auto",
        overwrite: bool = False,
    ) -> Result:
        """Write an M4B through the private execution orchestration boundary."""
        return self.write_m4b(
            output,
            on_event=on_event,
            cancel=cancel,
            workers=workers,
            overwrite=overwrite,
        )

    def write_m4b(
        self,
        output: str | os.PathLike[str],
        *,
        on_event: Callable[[ExecutionEvent], None] | None = None,
        cancel: CancellationToken | None = None,
        workers: int | Literal["auto"] = "auto",
        overwrite: bool = False,
    ) -> Result:
        """Validate controls and execute using privately bound rendering resources."""
        validation = self.validate()
        if not validation.is_valid:
            issue = validation.issues[0]
            if issue.code in (
                ErrorCode.SOURCE_NOT_FOUND,
                ErrorCode.SOURCE_NOT_READABLE,
            ):
                raise SourceError(issue.code)
            raise ValidationError(issue.code)
        if workers != "auto" and (
            isinstance(workers, bool) or not isinstance(workers, int) or workers < 1
        ):
            raise ValidationError(ErrorCode.INVALID_WORKERS)
        output_path = Path(output)
        if output_path.suffix.lower() != ".m4b" or not output_path.parent.is_dir():
            raise ValidationError(ErrorCode.INVALID_OUTPUT)
        if output_path.exists() and not output_path.is_file():
            raise ValidationError(ErrorCode.INVALID_OUTPUT)
        if output_path.exists() and not overwrite:
            raise EncodingError(ErrorCode.OUTPUT_EXISTS)
        if cancel is not None:
            cancel.raise_if_cancelled()
        return execute_sequential(
            self,
            output_path,
            bindings=_resolved_execution_bindings(self._assigned_voice_id()),
            on_event=on_event,
            cancel=cancel,
            workers=workers,
            overwrite=overwrite,
        )

    def _append(self, operation: Operation, *, before_tts: bool = False) -> Pipeline:
        """Create a branch with one pure validated operation append."""
        return Pipeline(
            self.source,
            append_unique(self.operations, operation, before_tts=before_tts),
        )

    def _assigned_voice_id(self) -> str:
        assigned = next(
            item for item in self.operations if isinstance(item, AssignVoice)
        )
        return assigned.voice_id


def _issue(code: ErrorCode) -> ValidationIssue:
    """Build a validation issue from the matching sanitized public error text."""
    return ValidationIssue(code, str(ValidationError(code)))


def _resolved_execution_bindings(voice_id: str) -> ExecutionBindings:
    """Compatibility seam for private tests that replace the zero-argument binding."""
    try:
        return _execution_bindings(voice_id)
    except TypeError:
        return _execution_bindings()  # type: ignore[call-arg]


def _execution_bindings(voice_id: str) -> ExecutionBindings:
    """Resolve explicit local production resources without network discovery."""
    from ._tts.production import production_bindings_from_environment  # noqa: PLC0415

    return production_bindings_from_environment(voice_id)
