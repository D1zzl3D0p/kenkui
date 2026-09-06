"""Lazy immutable public audiobook pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ._characters.continuity import eligible_series_voice_ids
from ._domain.casting import validate_method
from ._domain.operations import (
    AssignVoices,
    AttributeQuotes,
    InferCharacters,
    MetadataIntent,
    Operation,
    Pauses,
    SelectChapterRange,
    SelectChapters,
    Series,
    SpokenForm,
    SynthesizeSpeech,
    append_unique,
    has_operation,
)
from ._domain.selection import select_chapters, select_range
from ._epub.parser import inspect_epub
from ._execution.coordinator import execute_sequential
from ._resolution import log_collisions, resolve_inputs
from .api import Result, ValidationIssue, ValidationResult
from .errors import (
    EncodingError,
    ErrorCode,
    SourceError,
    ValidationError,
)
from .observability import get_logger, log_event
from .validation import (
    render_intent_errors,
    series_intent_errors,
    source_validation_error,
)
from .voices import Voice

if TYPE_CHECKING:
    import os
    from collections.abc import Callable, Mapping

    from ._characters.models import SeriesRecord
    from ._resolution import Resolved
    from .cancellation import CancellationToken
    from .events import ExecutionEvent
    from .inspection import BookInspection

_LOGGER = get_logger(__name__)
_NUMBER_TIERS = frozenset({"off", "conservative", "standard", "aggressive"})
_MAX_PAUSE_MS = 60_000


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
    # Not intent, and never part of the plan or its fingerprint: resolve()
    # parks finished values here so write() can skip re-deriving them.
    _resolved: Resolved | None = None

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

    def pronounce(
        self,
        lexicon: Mapping[str, str] | None = None,
        *,
        numbers: str = "conservative",
        builtin: bool = True,
        **features: bool,
    ) -> Pipeline:
        """Return a branch shaping what the engine says, not what it counts.

        This is the only stage that rewrites text for speech. It runs last,
        per segment, and its output is thrown away after synthesis: canonical
        text, the billable character count, attribution offsets, and chapter
        identity are all unaffected. Normalization -- NFC, line endings,
        Unicode spaces, whitespace runs -- is not part of this and is not
        optional; it happens when the source is parsed.

        Off unless called. What it does when called:

        * replaces entries from ``lexicon``, then a small built-in
          pronunciation table unless ``builtin=False``;
        * reads numbers aloud, at the depth ``numbers`` selects --
          ``"off"``, ``"conservative"``, ``"standard"`` or ``"aggressive"``.

        ``numbers`` is a preset over individually switchable features, each
        of which may be overridden by keyword: ``currency``, ``percent``,
        ``ordinals``, ``units``, ``decimals``, ``integers``, ``years``,
        ``clock``, ``roman``, ``fractions``, ``numbered``. Passing ``False``
        declines a feature the tier supplies; passing ``True`` asks for one
        it does not, without accepting the rest of the tier that carries it.

            pipeline.pronounce(numbers="standard", roman=False)

        Features compose rather than nest, so declining a specific form
        leaves a general one free to match inside it: ``currency=False``
        alone reads "£5" as "£five", because the integer rule still applies.
        Decline ``integers`` too to leave the digits alone.
        """
        from ._domain.spoken.lexicon import validate_entries  # noqa: PLC0415
        from ._domain.spoken.numbers import FEATURES  # noqa: PLC0415

        if numbers not in _NUMBER_TIERS:
            raise ValidationError(ErrorCode.INVALID_PRONUNCIATION)
        # Rejected here rather than at render time: a misspelled feature is a
        # caller's typo, and silently ignoring it renders a book that does
        # not sound like what was asked for.
        for name, value in features.items():
            if name not in FEATURES or not isinstance(value, bool):
                raise ValidationError(ErrorCode.INVALID_PRONUNCIATION)
        return self._append(
            SpokenForm(
                numbers=numbers,
                builtin_lexicon=builtin,
                lexicon=validate_entries(lexicon or {}),
                features=tuple(sorted(features.items())),
            ),
            before_tts=True,
        )

    def pauses(
        self,
        *,
        chapter_ms: int = 0,
        heading_before_ms: int = 0,
        heading_after_ms: int = 0,
        paragraph_ms: int = 0,
        line_ms: int = 0,
    ) -> Pipeline:
        """Return a branch requesting silence at structural boundaries.

        Off unless called. Durations are retunable without re-synthesis: only
        turning a tier on or off changes segment identity, because only that
        changes where a segment ends.
        """
        requested = (
            chapter_ms,
            heading_before_ms,
            heading_after_ms,
            paragraph_ms,
            line_ms,
        )
        for duration in requested:
            if (
                isinstance(duration, bool)
                or not isinstance(duration, int)
                or duration < 0
                or duration > _MAX_PAUSE_MS
            ):
                raise ValidationError(ErrorCode.INVALID_PAUSE)
        return self._append(Pauses(*requested), before_tts=True)

    def infer_characters(self, model: str) -> Pipeline:
        """Return a branch that will derive a character roster."""
        return self._append(InferCharacters(_model_id(model)), before_tts=True)

    def attribute_quotes(self, model: str) -> Pipeline:
        """Return a branch that will assign a speaker to each quoted run."""
        return self._append(AttributeQuotes(_model_id(model)), before_tts=True)

    def assign_voice(self, voice: str | Voice) -> Pipeline:
        """Return a branch assigning one stable voice ID to all speech.

        The degenerate cast: narrator only. One VoicePlan and one renderer
        serve this and a full character cast alike.
        """
        return self.assign_voices(narrator=voice)

    def assign_voices(
        self,
        *,
        narrator: str | Voice,
        unknown: str | Voice | None = None,
        cast: Mapping[str, str | Voice] | None = None,
        method: str = "gendered",
    ) -> Pipeline:
        """Return a branch casting narrator, unknown, and named characters.

        ``unknown`` defaults to the narrator's voice, so a line nobody could
        place sounds like narration rather than vanishing.
        """
        narrator_id = _voice_id(narrator)
        # Checked here rather than only in the solver: a typo must fail when
        # the caller writes it, not silently render single-voice because no
        # character ever reached a method.
        validate_method(method)
        return self._append(
            AssignVoices(
                narrator_voice_id=narrator_id,
                unknown_voice_id=_voice_id(unknown) if unknown else narrator_id,
                cast=tuple(
                    sorted(
                        (character, _voice_id(voice))
                        for character, voice in (cast or {}).items()
                    )
                ),
                method=method,
            ),
            before_tts=True,
        )

    def tts(self) -> Pipeline:
        """Return a branch with explicit synthesis intent."""
        if not has_operation(self.operations, AssignVoices):
            raise ValidationError(ErrorCode.VOICE_REQUIRED)
        return self._append(SynthesizeSpeech())

    def metadata(
        self,
        *,
        title: str | None = None,
        author: str | None = None,
        cover: Literal["source"] | os.PathLike[str] | None = "source",
    ) -> Pipeline:
        """Return a branch with output metadata inheritance and overrides."""
        if isinstance(cover, str) and cover != "source":
            raise ValidationError(ErrorCode.INVALID_METADATA)
        if cover is not None and cover != "source":
            try:
                cover = Path(cover)
            except TypeError:
                # Anything that is not a path is invalid intent, not a crash.
                raise ValidationError(ErrorCode.INVALID_METADATA) from None
        if title is not None and not title.strip():
            raise ValidationError(ErrorCode.INVALID_METADATA)
        if author is not None and not author.strip():
            raise ValidationError(ErrorCode.INVALID_METADATA)
        return self._append(MetadataIntent(title, author, cover))

    def series(
        self,
        series_id: str,
        *,
        book: int | None = None,
        allow_recast: bool = False,
        allow_narrator_change: bool = False,
    ) -> Pipeline:
        """Return a branch tying this book to a series.

        A character the series already cast keeps their voice, and voices
        continue spreading across volumes rather than restarting. ``book`` is
        recorded for ordering only: continuity is decided by identity, not by
        volume number.

        Two ways a series can be contradicted fail validation before any
        model call: a pinned voice missing from the pool, and a narrator
        differing from the one the series recorded. ``allow_recast`` and
        ``allow_narrator_change`` each waive their matching failure so the
        render proceeds instead of refusing.
        """
        name = series_id.strip()
        if not name or (book is not None and book < 1):
            raise ValidationError(ErrorCode.INVALID_SERIES)
        return self._append(
            Series(
                series_id=name,
                book=book,
                allow_recast=allow_recast,
                allow_narrator_change=allow_narrator_change,
            ),
            before_tts=True,
        )

    def resolve(self, *, cancel: CancellationToken | None = None) -> Pipeline:
        """Resolve voices, attribution, and casting, returning a new Pipeline.

        Optional. ``write()`` resolves internally, so this exists only to pay
        the model cost early and inspect the outcome. It is an effect -- it
        reaches the network and writes the store -- but it is immutable,
        idempotent, and leaves intent untouched.

        Pass a cancellation token to stop between model calls and before
        committing series changes. A running provider call must return before
        cooperative cancellation can take effect.
        """
        if cancel is not None:
            cancel.raise_if_cancelled()
        if self._resolved is not None:
            return self
        return Pipeline(self.source, self.operations, resolve_inputs(self, cancel))

    def validate(self) -> ValidationResult:
        """Perform inexpensive source and operation validation without parsing."""
        issues: list[ValidationIssue] = []
        source_error = source_validation_error(self.source.path)
        if source_error is not None:
            issues.append(_issue(source_error))
        issues.extend(_issue(code) for code in render_intent_errors(self.operations))
        series = next(
            (item for item in self.operations if isinstance(item, Series)), None
        )
        # The series checks are all about the voices this render would cast
        # with, so a pipeline that declares a series but never calls
        # `.assign_voices()` has nothing for them to read -- and
        # `render_intent_errors` has already queued the missing narrator as
        # an issue. Reporting that is validate()'s whole contract; asking
        # `_casting()` for an operation that is not there turned the report
        # into a StopIteration and told the caller nothing at all.
        casting = next(
            (item for item in self.operations if isinstance(item, AssignVoices)), None
        )
        if series is not None and casting is not None:
            from ._characters import store  # noqa: PLC0415 - see below
            from .voices.provision import list_voices  # noqa: PLC0415
            # Both are import-time cost callers who never declare a series
            # should not pay: the store and the voice manifest are only
            # touched once a pipeline actually names one.

            stored = store.read_series(series.series_id)
            codes = series_intent_errors(
                self.operations,
                stored,
                eligible_series_voice_ids(
                    tuple(list_voices()),
                    casting.narrator_voice_id,
                    casting.unknown_voice_id,
                ),
            )
            if ErrorCode.SERIES_VOICE_MISSING in codes and stored is not None:
                _log_missing_series_voices(
                    series.series_id,
                    stored,
                    eligible_series_voice_ids(
                        tuple(list_voices()),
                        casting.narrator_voice_id,
                        casting.unknown_voice_id,
                    ),
                )
            issues.extend(_issue(code) for code in codes)
        return ValidationResult(tuple(issues))

    def inspect(self) -> BookInspection:
        """Parse and select immutable normalized source data without synthesis."""
        source_error = source_validation_error(self.source.path)
        if source_error is not None:
            raise SourceError(source_error)
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
        selected = type(inspection)(inspection.metadata, chapters)
        log_event(
            _LOGGER,
            "inspection_completed",
            context={"boundary": "parse", "chapter_count": len(chapters)},
        )
        return selected

    def write(  # noqa: PLR0913 - explicit orchestration boundary.
        self,
        output: str | os.PathLike[str],
        *,
        on_event: Callable[[ExecutionEvent], None] | None = None,
        cancel: CancellationToken | None = None,
        workers: int | Literal["auto"] = "auto",
        overwrite: bool = False,
        keep_audio_cache: bool = False,
    ) -> Result:
        """Write an M4B through the private execution orchestration boundary."""
        return self.write_m4b(
            output,
            on_event=on_event,
            cancel=cancel,
            workers=workers,
            overwrite=overwrite,
            keep_audio_cache=keep_audio_cache,
        )

    def write_m4b(  # noqa: PLR0913 - explicit orchestration boundary.
        self,
        output: str | os.PathLike[str],
        *,
        on_event: Callable[[ExecutionEvent], None] | None = None,
        cancel: CancellationToken | None = None,
        workers: int | Literal["auto"] = "auto",
        overwrite: bool = False,
        keep_audio_cache: bool = False,
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
        # Resolved before the call, not inside the argument list: an
        # unprovisioned or unknown voice must fail before any worker spawns,
        # and that ordering should be legible rather than an artifact of
        # argument-evaluation order.
        resolved = (
            self._resolved
            if self._resolved is not None
            else resolve_inputs(self, cancel)
        )
        log_collisions(resolved.collisions)
        return execute_sequential(
            self,
            output_path,
            bindings=resolved.bindings,
            on_event=on_event,
            cancel=cancel,
            workers=workers,
            overwrite=overwrite,
            keep_audio_cache=keep_audio_cache,
            assignments=resolved.cast_assignments,
            unknown_voice_id=resolved.unknown_voice_id,
            spans=resolved.spans,
        )

    def _append(self, operation: Operation, *, before_tts: bool = False) -> Pipeline:
        """Create a branch with one pure validated operation append.

        Resolved values are dropped: changing intent invalidates them, and
        re-resolving against a populated store is a lookup, so the
        conservative rule costs nothing.
        """
        return Pipeline(
            self.source,
            append_unique(self.operations, operation, before_tts=before_tts),
        )

    def _casting(self) -> AssignVoices:
        return next(item for item in self.operations if isinstance(item, AssignVoices))

    def _assigned_voice_id(self) -> str:
        return self._casting().narrator_voice_id


def _voice_id(voice: str | Voice) -> str:
    """Return a non-empty stable voice ID from either accepted form."""
    voice_id = (voice.id if isinstance(voice, Voice) else voice).strip()
    if not voice_id:
        raise ValidationError(ErrorCode.INVALID_VOICE)
    return voice_id


def _model_id(model: str) -> str:
    """Return a non-empty model identifier, checking a spaCy id's shape.

    Only the shape: whether the named pipeline is actually installed is a
    question for resolution, which is where loading it happens. But "spacy:"
    names nothing at all, and that is a typo the caller should see where they
    wrote it rather than an hour later.
    """
    from ._characters.spacy_roster import is_spacy, pipeline_for  # noqa: PLC0415

    model_id = model.strip()
    if not model_id:
        raise ValidationError(ErrorCode.INVALID_MODEL)
    if is_spacy(model_id) and pipeline_for(model_id) is None:
        raise ValidationError(ErrorCode.INVALID_MODEL)
    return model_id


def _issue(code: ErrorCode) -> ValidationIssue:
    """Build a validation issue from the matching sanitized public error text."""
    return ValidationIssue(code, str(ValidationError(code)))


def _log_missing_series_voices(
    series_id: str, stored: SeriesRecord, pool_ids: frozenset[str]
) -> None:
    """Name the characters a series can no longer honour, for an operator.

    The refusal itself is right and stays: a voice this series already cast
    has gone, and that is the operator's to fix. But `ValidationIssue`
    carries a stable code and sanitized public text and nothing else, so an
    operator told only `series_voice_missing` has the whole cast to search
    by hand. The spec asks for the characters and the lost voice by name,
    which is an operator's log line -- the same channel, and for the same
    reason, as a cast collision.
    """
    affected = [
        character
        for character in stored.characters
        if character.voice_id not in pool_ids
    ]
    if not affected:
        return
    log_event(
        _LOGGER,
        "series_voice_missing",
        level=logging.WARNING,
        context={
            "boundary": "series",
            "series_id": series_id,
            "characters": ", ".join(c.display_name for c in affected),
            "voice_ids": ", ".join(sorted({c.voice_id for c in affected})),
        },
    )
