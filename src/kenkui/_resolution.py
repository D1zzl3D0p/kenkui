"""Imperative resolution of pipeline intent into reusable rendering inputs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from ._characters.continuity import eligible_series_voice_ids, prepare_series_cast
from ._domain.casting import (
    CastingMethod,
    CastingRequest,
    CharacterProfile,
    Collision,
    ungendered_pool_characters,
)
from ._domain.operations import AssignVoices, AttributeQuotes, InferCharacters, Series
from ._domain.planning import SpeakerSpan  # noqa: TC001 - dataclass field
from ._source import snapshot_source, source_digest
from .errors import ErrorCode, SourceError, ValidationError
from .inspection import BookInspection, CastingInspection
from .observability import get_logger, log_event
from .validation import series_intent_errors

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ._characters.llm import Client
    from ._characters.models import AttributionRecord, SeriesRecord
    from ._domain.casting import CastingOutcome
    from ._execution.coordinator import ExecutionBindings
    from .cancellation import CancellationToken
    from .pipeline import Pipeline
    from .voices import Voice

_LOGGER = get_logger(__name__)
_MAX_SERIES_ATTEMPTS = 16


@dataclass(frozen=True, slots=True)
class RosterCheckpoint:
    """Character discovery bound to the exact source and selected chapters."""

    inspection: BookInspection
    source_hash: str
    reviewed: bool = False


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
    inspection: BookInspection
    source_hash: str

    def __post_init__(self) -> None:
        """Prevent a retained assignment mapping from changing later renders."""
        object.__setattr__(
            self, "cast_assignments", MappingProxyType(dict(self.cast_assignments))
        )


def resolve_inputs(
    pipeline: Pipeline,
    cancel: CancellationToken | None = None,
    *,
    validate_series: bool = False,
) -> Resolved:
    """Resolve voices, attribution, and casting for one pipeline.

    The single resolution implementation. ``resolve()`` and ``write()`` are
    two entry points into it, not two code paths.

    ``cancel`` reaches attribution, which is where a long book spends hours in
    model calls. Without it a caller who cancels waits for the whole pass.
    """
    from ._characters import (  # noqa: PLC0415 - see below
        resolve_attribution,
    )
    from .voices.provision import list_voices  # noqa: PLC0415 - resolved per
    # call so the managed cache root can be redirected, as the store is.

    casting = next(
        (item for item in pipeline.operations if isinstance(item, AssignVoices)), None
    )
    if casting is None:
        raise ValidationError(ErrorCode.VOICE_REQUIRED)
    bindings = _execution_bindings(casting.narrator_voice_id)
    if cancel is not None:
        cancel.raise_if_cancelled()

    inspection, digest = inspect_source(pipeline, cancel)
    checkpoint = pipeline._roster  # noqa: SLF001 - checkpoint orchestration
    inspection = _with_roster_checkpoint(inspection, digest, checkpoint)

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
            inspection=replace(
                inspection,
                casting=CastingInspection(
                    casting.narrator_voice_id, casting.unknown_voice_id
                ),
            ),
            source_hash=digest,
        )

    inferring = next(
        (item for item in pipeline.operations if isinstance(item, InferCharacters)),
        None,
    )
    record = resolve_attribution(
        inspection,
        digest,
        attributing.model_id,
        roster_model_id=inferring.model_id if inferring is not None else None,
        client=_attribution_client(),
        cancel=cancel,
        roster=inspection.roster,
        reviewed=checkpoint is not None and checkpoint.reviewed,
    )
    if cancel is not None:
        cancel.raise_if_cancelled()
    pool = tuple(
        voice
        for voice in list_voices()
        if voice.state == "loaded" and voice.language == bindings.voice.language
    )
    _log_ungendered_cast(casting.method, record.characters, pool)

    series = next(
        (item for item in pipeline.operations if isinstance(item, Series)), None
    )
    outcome, final_bindings = _resolve_cast_inputs(
        record, casting, pool, digest, series, cancel, validate_series=validate_series
    )
    # Re-resolve with the whole cast so every assigned voice's asset reaches
    # the engine config; one worker then holds one model and N states.
    return Resolved(
        cast_assignments=outcome.assignments,
        unknown_voice_id=casting.unknown_voice_id,
        spans=record.spans,
        collisions=outcome.collisions,
        bindings=final_bindings,
        inspection=replace(
            inspection,
            casting=CastingInspection(
                narrator_voice_id=casting.narrator_voice_id,
                unknown_voice_id=casting.unknown_voice_id,
                characters=record.characters,
                assignments=tuple(sorted(outcome.assignments.items())),
                spans=record.spans,
                collisions=outcome.collisions,
            ),
        ),
        source_hash=digest,
    )


def _resolve_cast_inputs(  # noqa: PLR0913, PLR0917 - resolved values, no provider work.
    record: AttributionRecord,
    casting: AssignVoices,
    pool: tuple[Voice, ...],
    digest: str,
    series: Series | None,
    cancel: CancellationToken | None,
    *,
    validate_series: bool = False,
) -> tuple[CastingOutcome, ExecutionBindings]:
    """Retry casting if another volume changed its continuity inputs.

    Attribution happens once before this boundary. Only cheap casting and
    asset binding repeat, outside a database transaction. The conditional
    commit ensures that accepted casts see all previously committed volumes.
    """
    from ._characters import resolve_cast, store  # noqa: PLC0415
    from ._characters.series import merged_series, series_members  # noqa: PLC0415

    for _attempt in range(_MAX_SERIES_ATTEMPTS):
        if cancel is not None:
            cancel.raise_if_cancelled()
        stored = store.read_series(series.series_id) if series is not None else None
        if validate_series and series is not None:
            errors = series_intent_errors(
                (series, casting),
                stored,
                eligible_series_voice_ids(
                    pool, casting.narrator_voice_id, casting.unknown_voice_id
                ),
            )
            if errors:
                raise ValidationError(errors[0])
        # Minted roles are chapter-scoped by construction and are not people a
        # series remembers; see `series_members`.
        members = series_members(record.characters)
        pins = prepare_series_cast(stored, members, casting, pool, digest)

        # Stored, not just solved: the cast is what list_castings names and what
        # remove_casting discards, and neither can see a cast that only ever
        # existed for the duration of one render.
        outcome = resolve_cast(
            record,
            CastingRequest(
                characters=record.characters,
                pool=pool,
                explicit=dict(pins.explicit),
                narrator_voice_id=casting.narrator_voice_id,
                unknown_voice_id=casting.unknown_voice_id,
                method=cast("CastingMethod", casting.method),
                prior_load=dict(pins.prior_load),
            ),
        )
        # Resolved before the series is written, not after: this is what can
        # fail on an unprovisioned voice, and a doomed render must not persist a
        # cast it never actually produced.
        final_bindings = _execution_bindings(
            casting.narrator_voice_id,
            also=(casting.unknown_voice_id, *sorted(set(outcome.assignments.values()))),
        )
        if cancel is not None:
            cancel.raise_if_cancelled()
        if series is not None:
            persisted_assignments = dict(outcome.assignments)
            if pins.dropped_pins and not series.allow_recast:
                # Reached only through resolve(), which never validates -- write()
                # would have refused this render before here. The render still
                # has to use whatever the solver chose (the speech cannot just
                # vanish), but persisting that choice would erase the operator's
                # only signal that a voice went missing: the next validate()
                # would no longer report it. It would also be wrong for a pin
                # dropped only because it is the wrong language for *this*
                # volume -- that voice is still correct for others. Either way
                # the series keeps what it already had.
                persisted_assignments.update(pins.dropped_voice_ids)
            # book_digest makes a re-render of this exact volume replace its own
            # contribution rather than add to it -- see merged_series.
            replacement = merged_series(
                stored,
                members,
                persisted_assignments,
                (
                    casting.narrator_voice_id
                    if stored is None or series.allow_narrator_change
                    else stored.narrator_voice_id
                ),
                series.series_id,
                book_digest=digest,
            )
            if not store.compare_and_write_series(stored, replacement):
                continue
        if series is not None:
            _log_series_overrides(
                series,
                stored,
                pins.dropped_pins,
                pins.overridden_pins,
                casting.narrator_voice_id,
            )

        return outcome, final_bindings
    message = "series changed repeatedly during resolution; retry the operation"
    raise OSError(message)


def _with_roster_checkpoint(
    inspection: BookInspection, digest: str, checkpoint: RosterCheckpoint | None
) -> BookInspection:
    """Attach reviewed input only to the exact source for which it was prepared."""
    if checkpoint is None:
        return inspection
    if checkpoint.source_hash != digest:
        raise SourceError(ErrorCode.SOURCE_CHANGED)
    return replace(inspection, roster=checkpoint.inspection.roster)


def inspect_source(
    pipeline: Pipeline, cancel: CancellationToken | None
) -> tuple[BookInspection, str]:
    """Read source information and its identity from the same private copy."""
    with TemporaryDirectory(prefix="kenkui-resolution-") as workspace:
        snapshot = Path(workspace) / "source.epub"
        digest = snapshot_source(pipeline.source.path, snapshot, cancel)
        snapshot_pipeline = replace(
            pipeline,
            source=replace(pipeline.source, path=snapshot),
            _resolved=None,
            _roster=None,
        )
        return snapshot_pipeline.inspect(), digest


def resolve_characters(
    pipeline: Pipeline, cancel: CancellationToken | None
) -> RosterCheckpoint:
    """Stop after character discovery, without binding voices or assigning quotes."""
    from ._characters import discover_characters  # noqa: PLC0415 - model boundary

    inferring = next(
        (item for item in pipeline.operations if isinstance(item, InferCharacters)),
        None,
    )
    if inferring is None:
        raise ValidationError(ErrorCode.ATTRIBUTION_UNAVAILABLE)
    checkpoint = pipeline._roster  # noqa: SLF001 - checkpoint orchestration
    if (
        checkpoint is not None
        and source_digest(pipeline.source.path) == checkpoint.source_hash
    ):
        return checkpoint
    inspection, digest = inspect_source(pipeline, cancel)
    roster = discover_characters(
        inspection, inferring.model_id, client=_attribution_client(), cancel=cancel
    )
    return RosterCheckpoint(replace(inspection, roster=roster), digest)


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
    `log_collisions`: the fix is to the operator's voice pool, not to
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
    overridden_pins: Sequence[str],
    narrator_voice_id: str,
) -> None:
    """Record a forced series override for an operator, not for the caller.

    Three things can make this volume's cast disagree with what the series
    remembers: a pin the pool can no longer honour, a pin this render's
    caller chose to override with an explicit `cast=`, and a narrator this
    render used that differs from the one the series recorded. Each is a
    legitimate outcome -- `allow_recast`, an explicit override, and
    `allow_narrator_change` all exist to let exactly this happen -- but
    proceeding silently would leave nobody able to tell that it did. A
    forced render must still say what it did.
    """
    narrator_changed = (
        stored is not None and stored.narrator_voice_id != narrator_voice_id
    )
    if not dropped_pins and not overridden_pins and not narrator_changed:
        return
    log_event(
        _LOGGER,
        "series_override",
        level=logging.WARNING,
        context={
            "boundary": "series",
            "series_id": series.series_id,
            "dropped_pins": ", ".join(dropped_pins),
            "overridden_pins": ", ".join(overridden_pins),
            "narrator_changed": narrator_changed,
        },
    )


def log_collisions(collisions: tuple[Collision, ...]) -> None:
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


def _execution_bindings(
    voice_id: str, *, also: Sequence[str] = ()
) -> ExecutionBindings:
    """Resolve explicit local production resources without network discovery."""
    from ._tts.production import production_bindings_from_environment  # noqa: PLC0415

    return production_bindings_from_environment(voice_id, also=also)
