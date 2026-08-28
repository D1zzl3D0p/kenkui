"""Lazy immutable public audiobook pipeline."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from ._domain.casting import (
    CastingMethod,
    CastingRequest,
    CharacterProfile,
    Collision,
    ungendered_pool_characters,
    validate_method,
)
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
from ._domain.planning import SpeakerSpan  # noqa: TC001 - dataclass field
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
from .observability import get_logger, log_event
from .validation import (
    render_intent_errors,
    series_intent_errors,
    source_validation_error,
)
from .voices import Voice

if TYPE_CHECKING:
    import os
    from collections.abc import Callable, Mapping, Sequence

    from ._characters.llm import Client
    from ._characters.store import SeriesRecord
    from .cancellation import CancellationToken
    from .events import ExecutionEvent
    from .inspection import BookInspection

_LOGGER = get_logger(__name__)
_HASH_CHUNK_BYTES = 1024 * 1024
_NUMBER_TIERS = frozenset({"off", "conservative", "standard", "aggressive"})
_MAX_PAUSE_MS = 60_000


@dataclass(frozen=True, slots=True)
class Source:
    """A lazily recorded readable source."""

    path: Path
    format: Literal["epub"]


@dataclass(frozen=True, slots=True)
class Resolved:
    """Everything the pure planner needs that only the shell can obtain.

    Voice resolution reads the manifest, attribution may call a model, and
    casting is pure over both. The planner receives finished values and
    reaches neither.
    """

    cast_assignments: Mapping[str, str]
    unknown_voice_id: str
    spans: tuple[SpeakerSpan, ...]
    collisions: tuple[Collision, ...]
    bindings: ExecutionBindings


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

    def resolve(self) -> Pipeline:
        """Resolve voices, attribution, and casting, returning a new Pipeline.

        Optional. ``write()`` resolves internally, so this exists only to pay
        the model cost early and inspect the outcome. It is an effect -- it
        reaches the network and writes the store -- but it is immutable,
        idempotent, and leaves intent untouched.
        """
        if self._resolved is not None:
            return self
        return Pipeline(self.source, self.operations, _resolve_all(self))

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
        if series is not None:
            from ._characters import store  # noqa: PLC0415 - see below
            from .voices.provision import list_voices  # noqa: PLC0415
            # Both are import-time cost callers who never declare a series
            # should not pay: the store and the voice manifest are only
            # touched once a pipeline actually names one.

            issues.extend(
                _issue(code)
                for code in series_intent_errors(
                    self.operations,
                    store.read_series(series.series_id),
                    frozenset(
                        voice.id for voice in list_voices() if voice.state == "loaded"
                    ),
                )
            )
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
        # Resolved before the call, not inside the argument list: an
        # unprovisioned or unknown voice must fail before any worker spawns,
        # and that ordering should be legible rather than an artifact of
        # argument-evaluation order.
        resolved = (
            self._resolved if self._resolved is not None else _resolve_all(self, cancel)
        )
        _log_collisions(resolved.collisions)
        return execute_sequential(
            self,
            output_path,
            bindings=resolved.bindings,
            on_event=on_event,
            cancel=cancel,
            workers=workers,
            overwrite=overwrite,
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
    """Return a non-empty model identifier."""
    model_id = model.strip()
    if not model_id:
        raise ValidationError(ErrorCode.INVALID_VOICE)
    return model_id


def _issue(code: ErrorCode) -> ValidationIssue:
    """Build a validation issue from the matching sanitized public error text."""
    return ValidationIssue(code, str(ValidationError(code)))


def _resolve_all(
    pipeline: Pipeline, cancel: CancellationToken | None = None
) -> Resolved:
    """Resolve voices, attribution, and casting for one pipeline.

    The single resolution implementation. ``resolve()`` and ``write()`` are
    two entry points into it, not two code paths.

    ``cancel`` reaches attribution, which is where a long book spends hours in
    model calls. Without it a caller who cancels waits for the whole pass.
    """
    from ._characters import (  # noqa: PLC0415 - see below
        resolve_attribution,
        resolve_cast,
        store,
    )
    from ._characters.series import match_roster, merged_series  # noqa: PLC0415
    from .voices.provision import list_voices  # noqa: PLC0415 - resolved per
    # call so the managed cache root can be redirected, as the store is.

    casting = pipeline._casting()  # noqa: SLF001 - same module, private by design
    bindings = _resolved_execution_bindings(casting.narrator_voice_id)

    attributing = next(
        (item for item in pipeline.operations if isinstance(item, AttributeQuotes)),
        None,
    )
    if attributing is None:
        return Resolved(
            cast_assignments={},
            unknown_voice_id=casting.unknown_voice_id,
            spans=(),
            collisions=(),
            bindings=bindings,
        )

    inferring = next(
        (item for item in pipeline.operations if isinstance(item, InferCharacters)),
        None,
    )
    inspection = pipeline.inspect()
    digest = _source_digest(pipeline.source.path)
    record = resolve_attribution(
        inspection,
        digest,
        attributing.model_id,
        roster_model_id=inferring.model_id if inferring is not None else None,
        client=_attribution_client(),
        cancel=cancel,
    )
    pool = tuple(
        voice
        for voice in list_voices()
        if voice.state == "loaded" and voice.language == bindings.voice.language
    )
    _log_ungendered_cast(casting.method, record.characters, pool)

    # A book carrying no .series() call touches none of this: `series` is
    # None, `stored` stays None, and `explicit`/`prior_load` end up exactly
    # what they always were. That is load-bearing, not incidental -- a
    # regression here would silently re-cast every book anyone has already
    # rendered.
    series = next(
        (item for item in pipeline.operations if isinstance(item, Series)), None
    )
    stored = store.read_series(series.series_id) if series is not None else None
    explicit = dict(casting.cast)
    prior_load: dict[str, int] = {}
    dropped_pins: list[str] = []
    if stored is not None:
        pool_ids = {voice.id for voice in pool}
        by_canonical = {c.canonical_id: c for c in stored.characters}
        for book_id, canonical in match_roster(stored, record.characters).items():
            known = by_canonical[canonical]
            # A pin the pool cannot honour only reaches here under
            # allow_recast; validate() refuses it otherwise. Dropping it
            # lets the solver choose afresh, which is what was asked for.
            if known.voice_id in pool_ids:
                explicit[book_id] = known.voice_id
            else:
                dropped_pins.append(book_id)
        for known in stored.characters:
            prior_load[known.voice_id] = (
                prior_load.get(known.voice_id, 0) + known.spoken_characters
            )
    if series is not None:
        _log_series_overrides(series, stored, dropped_pins, casting.narrator_voice_id)

    # Stored, not just solved: the cast is what list_castings names and what
    # remove_casting discards, and neither can see a cast that only ever
    # existed for the duration of one render.
    outcome = resolve_cast(
        record,
        CastingRequest(
            characters=record.characters,
            pool=pool,
            explicit=explicit,
            narrator_voice_id=casting.narrator_voice_id,
            unknown_voice_id=casting.unknown_voice_id,
            method=cast("CastingMethod", casting.method),
            prior_load=prior_load,
        ),
    )
    if series is not None:
        # book_digest makes a re-render of this exact volume replace its own
        # contribution rather than add to it -- see merged_series.
        store.write_series(
            merged_series(
                stored,
                record.characters,
                outcome.assignments,
                casting.narrator_voice_id,
                series.series_id,
                book_digest=digest,
            )
        )
    # Re-resolve with the whole cast so every assigned voice's asset reaches
    # the engine config; one worker then holds one model and N states.
    return Resolved(
        cast_assignments=outcome.assignments,
        unknown_voice_id=casting.unknown_voice_id,
        spans=record.spans,
        collisions=outcome.collisions,
        bindings=_resolved_execution_bindings(
            casting.narrator_voice_id,
            also=(casting.unknown_voice_id, *sorted(set(outcome.assignments.values()))),
        ),
    )


def _source_digest(path: Path) -> str:
    """Hash the source bytes in bounded chunks.

    The same identity the plan uses, so attribution stored for a book is found
    again on a later render of that same book and not of an edited copy.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _log_ungendered_cast(
    method: str,
    characters: tuple[CharacterProfile, ...],
    pool: tuple[Voice, ...],
) -> None:
    """Record a gendered cast the pool could not honour, for an operator.

    `candidates` falls back to the whole pool rather than dropping the
    speech, which is right at render time and silent by nature: a gendered
    cast becomes a random one with nothing to show for it. That silence is
    how a voice pool carrying no gender traits at all rendered whole books
    in arbitrary voices without a single failing check.

    Logged rather than raised as a Warning event, for the same reason as
    `_log_collisions`: the fix is to the operator's voice pool, not to
    anything the caller passed.
    """
    affected = ungendered_pool_characters(method, characters, pool)
    if not affected:
        return
    log_event(
        _LOGGER,
        "ungendered_cast",
        level=logging.WARNING,
        context={
            "boundary": "casting",
            "method": method,
            "characters": len(affected),
            "sample": ", ".join(affected[:5]),
            "traited_voices": sum(v.perceived_gender is not None for v in pool),
            "pool": len(pool),
        },
    )


def _attribution_client() -> Client | None:
    """Compatibility seam for private tests that replace the model boundary.

    `resolve_attribution` accepts a `client` explicitly so a test never
    reaches a provider, but `Pipeline` exposes no public argument for one --
    adding a caller-facing parameter just to satisfy a test would put a model
    concern in front of every user of the public API. Tests instead
    monkeypatch this function, the same seam `_execution_bindings` already
    is for the rendering side.
    """
    return None


def _log_series_overrides(
    series: Series,
    stored: SeriesRecord | None,
    dropped_pins: Sequence[str],
    narrator_voice_id: str,
) -> None:
    """Record a forced series override for an operator, not for the caller.

    `allow_recast` and `allow_narrator_change` exist to let a render proceed
    over a contradiction `validate()` would otherwise refuse. Proceeding
    silently would leave nobody able to tell that it happened -- a forced
    render must still say what it did.
    """
    narrator_changed = (
        stored is not None and stored.narrator_voice_id != narrator_voice_id
    )
    if not dropped_pins and not narrator_changed:
        return
    log_event(
        _LOGGER,
        "series_override",
        level=logging.WARNING,
        context={
            "boundary": "series",
            "series_id": series.series_id,
            "dropped_pins": ", ".join(dropped_pins),
            "narrator_changed": narrator_changed,
        },
    )


def _log_collisions(collisions: tuple[Collision, ...]) -> None:
    """Record same-chapter voice clashes for an operator, not for the caller.

    Deliberately not a Warning execution event: that is sequenced into
    on_event and would surface in a browser. A collision is a signal that the
    voice pool ran short, which is an operator's problem to fix.
    """
    for collision in collisions:
        log_event(
            _LOGGER,
            "cast_collision",
            level=logging.WARNING,
            context={
                "boundary": "casting",
                "chapter_id": collision.chapter_id,
                "first": collision.first,
                "second": collision.second,
                "voice_id": collision.voice_id,
            },
        )


def _resolved_execution_bindings(
    voice_id: str, *, also: Sequence[str] = ()
) -> ExecutionBindings:
    """Compatibility seam for private tests that replace the binding factory."""
    try:
        return _execution_bindings(voice_id, also=also)
    except TypeError:
        # Older test doubles accept fewer arguments. Falling back keeps them
        # working, and a single-voice run needs nothing more. Logged because a
        # TypeError raised *inside* resolution looks identical here, and
        # silently dropping the cast would render the book in one voice.
        log_event(
            _LOGGER,
            "cast_binding_fallback",
            level=logging.WARNING,
            context={"boundary": "casting", "cast_size": len(tuple(also))},
        )
        try:
            return _execution_bindings(voice_id)
        except TypeError:
            return _execution_bindings()  # type: ignore[call-arg]


def _execution_bindings(
    voice_id: str, *, also: Sequence[str] = ()
) -> ExecutionBindings:
    """Resolve explicit local production resources without network discovery."""
    from ._tts.production import production_bindings_from_environment  # noqa: PLC0415

    return production_bindings_from_environment(voice_id, also=also)
