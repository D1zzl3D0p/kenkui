"""Pure immutable compilation of validated audiobook intent."""

from __future__ import annotations

import hashlib
import json
import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import pairwise
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, TypeAlias, TypeVar, cast

from kenkui._domain.grid import (
    GapReason,
    StructuralIndex,
    build_grid,
    build_structure_index,
    sibling_counts,
)
from kenkui._domain.grid_packing import (
    FallbackCut,
    PackedRange,
    PackingInput,
    SpokenMapping,
    SpokenRegion,
    pack_grid,
)
from kenkui._domain.operations import (
    AssignVoices,
    Attributions,
    MetadataIntent,
    Operation,
    Pauses,
    Pronunciations,
    Silences,
    SpokenForm,
    SynthesizeSpeech,
)
from kenkui._domain.paths import LEVELS, matches
from kenkui._domain.selection import selected_patterns, selected_ranges, selected_unit
from kenkui._domain.spoken import (
    SPOKEN_FORM_VERSION,
    spoken_identity,
    to_spoken,
)
from kenkui._domain.text import NORMALIZATION_VERSION
from kenkui._domain.tuning import layered_rules, resolve_rules
from kenkui.errors import (
    ErrorCode,
    ModelError,
    ValidationError,
    VoiceError,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from kenkui._domain.grid import Unit
    from kenkui._domain.paths import Pattern, SiblingCounts
    from kenkui._domain.spoken.numbers import NumberTier
    from kenkui._domain.tuning import Rule
    from kenkui.inspection import BookInspection, ChapterInspection
    from kenkui.voices import Voice

PARSER_SCHEMA_VERSION = "epub-visible-text-v1"
NORMALIZATION_SCHEMA_VERSION = NORMALIZATION_VERSION
PLANNING_SCHEMA_VERSION = "execution-plan-v2"
RENDER_SCHEMA_VERSION = "m4b-render-v1"
GRID_CHUNKING_SCHEMA_VERSION = "grid-v1"
_NO_PAUSES = Pauses()
MAX_TTS_SEGMENT_CHARACTERS = 1000
_SEGMENT_ID_VERSION = "v2"
_UTF8_HASH_CHUNK_CHARACTERS = 64 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")
_OperationT = TypeVar("_OperationT", bound=Operation)
_RuleOperationT = TypeVar("_RuleOperationT", Attributions, Silences, Pronunciations)
_Entries: TypeAlias = tuple[tuple[str, str], ...]
# One canonical run of a chapter and the lexicon that speaks it: (start, end,
# entries). Empty means no scoped lexicon exists and the whole chapter speaks
# under ``SpokenForm.lexicon``, which is the pre-tuning path exactly.
_LexiconRegions: TypeAlias = tuple[tuple[int, int, _Entries], ...]


def _gap_ms(reasons: GapReason, pauses: Pauses) -> int:
    """Translate pure grid reasons to one derived duration after packing.

    A line pause applies only to a single-newline boundary; paragraph and
    chapter boundaries close nested line ranges too, but retain their distinct
    legacy pause semantics. Coincident pause-policy reasons take the maximum.
    """
    if GapReason.CHAPTER in reasons:
        return pauses.chapter_ms
    durations: list[int] = []
    if GapReason.PARAGRAPH in reasons:
        durations.append(pauses.paragraph_ms)
        if GapReason.HEADING_BEFORE in reasons:
            durations.append(pauses.heading_before_ms)
        if GapReason.HEADING_AFTER in reasons:
            durations.append(pauses.heading_after_ms)
        if GapReason.SCENE in reasons:
            durations.append(pauses.scene_ms)
    elif GapReason.LINE in reasons:
        durations.append(pauses.line_ms)
    return max(durations, default=0)


def _gap_enabled(reasons: GapReason, pauses: Pauses) -> bool:
    """Whether an effective pause makes this grid gap a mandatory cut."""
    if GapReason.PARAGRAPH in reasons:
        return bool(
            pauses.paragraph_ms
            or (GapReason.HEADING_BEFORE in reasons and pauses.heading_before_ms)
            or (GapReason.HEADING_AFTER in reasons and pauses.heading_after_ms)
            or (GapReason.SCENE in reasons and pauses.scene_ms)
        )
    return GapReason.LINE in reasons and bool(pauses.line_ms)


class _HashDigest(Protocol):
    """Minimal incremental digest interface used by bounded text hashing."""

    def update(self, value: bytes, /) -> None:
        """Add bytes to the digest."""
        ...

    def hexdigest(self) -> str:
        """Return the hexadecimal digest."""
        ...


class _PipelineIntent(Protocol):
    """Minimal structural boundary that excludes source and shell controls."""

    @property
    def operations(self) -> tuple[Operation, ...]:
        """Return ordered semantic operations."""
        ...


class CoverIntent(StrEnum):
    """Renderer-neutral choice for source cover inheritance."""

    SOURCE = "source"
    FILE = "file"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class SchemaVersions:
    """Versions of every stage whose semantics can affect an artifact."""

    parser: str
    normalization: str
    planning: str
    render: str
    spoken_form: str | None = None


@dataclass(frozen=True, slots=True)
class VoicePlan:
    """Resolved single-voice content, compatibility, and rights metadata."""

    id: str
    name: str
    content_fingerprint: str
    language: str
    provenance: str
    license_id: str
    commercial_use_allowed: bool
    compatible_model_revisions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SpeakerSpan:
    """One contiguous run of a chapter's normalized text with a single speaker."""

    chapter_id: str
    start: int
    end: int
    character_id: str | None  # None is narration


@dataclass(frozen=True, slots=True)
class CastPlan:
    """Resolved narrator, unknown, and per-character voices for one run."""

    narrator: VoicePlan
    unknown: VoicePlan
    voices: tuple[VoicePlan, ...]
    assignments: Mapping[str, str]  # character id -> voice id

    def __post_init__(self) -> None:
        """Snapshot assignments so later caller edits cannot change speech."""
        object.__setattr__(
            self, "assignments", MappingProxyType(dict(self.assignments))
        )

    def __reduce__(
        self,
    ) -> tuple[
        type[CastPlan],
        tuple[VoicePlan, VoicePlan, tuple[VoicePlan, ...], dict[str, str]],
    ]:
        """Reconstruct the immutable snapshot in spawned render processes."""
        return type(self), (
            self.narrator,
            self.unknown,
            self.voices,
            dict(self.assignments),
        )

    @classmethod
    def single(cls, voice: VoicePlan) -> CastPlan:
        """Return the degenerate cast: one voice narrates everything."""
        return cls(narrator=voice, unknown=voice, voices=(voice,), assignments={})

    def voice_for(self, speaker_id: str | None) -> VoicePlan:
        """Resolve the voice a speaker renders in.

        Narration takes the narrator. A character with no assignment takes the
        unknown voice, which defaults to the narrator's, so speech never
        silently vanishes because casting missed someone.
        """
        if speaker_id is None:
            return self.narrator
        voice_id = self.assignments.get(speaker_id)
        if voice_id is None:
            return self.unknown
        for voice in self.voices:
            if voice.id == voice_id:
                return voice
        return self.unknown


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    """One exact ordered synthesis input for an M1 spine chapter."""

    id: str
    chapter_id: str
    ordinal: int
    text: str
    character_count: int
    content_hash: str
    speaker_id: str | None = None
    voice_id: str = ""


@dataclass(frozen=True, slots=True)
class OutputChapter:
    """Semantic chapter marker metadata independent of output destination."""

    id: str
    source_index: int
    title: str
    speech_characters: int


@dataclass(frozen=True, slots=True)
class OutputMetadata:
    """Resolved bibliographic values and source-cover intent."""

    title: str | None
    author: str | None
    chapters: tuple[OutputChapter, ...]
    cover: CoverIntent
    source_cover_available: bool
    # The content digest, never the path: core spec 14 requires that where an
    # output lives cannot change what it means.
    cover_content_hash: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Complete renderer-neutral semantic plan safe to pickle for spawned work."""

    schema_versions: SchemaVersions
    source_bytes_hash: str
    model_revision: str
    # No separate singular voice: a lone VoicePlan beside the cast is a second
    # source of truth that can disagree with it, and disagreement here renders
    # well-formed audio in the wrong voice. cast.narrator is the authority.
    cast: CastPlan
    segments: tuple[SpeechSegment, ...]
    output: OutputMetadata
    total_speech_characters: int
    semantic_fingerprint: str
    # One entry per segment: the gap AFTER that segment, in milliseconds.
    # Excluded from segment identity -- silence never reaches a worker or the
    # cache, so retuning a duration costs no re-synthesis.
    trailing_silence_ms: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class _PlanMaterial:
    """Semantic fields canonicalized before attaching their fingerprint."""

    schemas: SchemaVersions
    source_bytes_hash: str
    model_revision: str
    voice: VoicePlan
    segments: tuple[SpeechSegment, ...]
    output: OutputMetadata
    total: int
    trailing_silence: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class _SegmentSource:
    """Temporary source coordinates; excluded from plans and cache identities."""

    start: int
    canonical: str
    rendered: str
    chunk_start: int
    chunk_end: int
    chunk_index: int
    offsets: tuple[tuple[int, int, int, int], ...]
    canonical_start: int = 0
    canonical_end: int = 0
    fallback_cut_after: FallbackCut | None = None


def compile_execution_plan(  # noqa: PLR0913 - explicit compilation boundary.
    pipeline: _PipelineIntent,
    inspection: BookInspection,
    *,
    source_bytes_hash: str,
    resolved_voice: Voice | None,
    model_revision: str,
    cast_voices: tuple[Voice, ...] = (),
    assignments: Mapping[str, str] | None = None,
    unknown_voice_id: str | None = None,
    spans: tuple[SpeakerSpan, ...] = (),
    cover_content_hash: str | None = None,
) -> ExecutionPlan:
    """Compile supplied material without filesystem, provider, or process effects."""
    source_hash = _validated_hash(source_bytes_hash)
    revision = model_revision.strip()
    if not revision:
        raise ModelError(ErrorCode.INVALID_MODEL_REVISION)

    assigned = _one_operation(pipeline.operations, AssignVoices)
    if assigned is None:
        raise VoiceError(ErrorCode.VOICE_UNRESOLVED)
    if _one_operation(pipeline.operations, SynthesizeSpeech) is None:
        raise ValidationError(ErrorCode.TTS_REQUIRED)
    narrator = _resolve_voice(assigned.narrator_voice_id, resolved_voice, revision)
    # Single voice is the degenerate cast, not a separate path: one renderer
    # and one set of segment identities serve both.
    others = tuple(_resolve_voice(item.id, item, revision) for item in cast_voices)
    by_id = {plan.id: plan for plan in (narrator, *others)}
    cast = CastPlan(
        narrator=narrator,
        unknown=by_id.get(unknown_voice_id or narrator.id, narrator),
        voices=(narrator, *others),
        assignments=dict(assignments or {}),
    )
    voice = narrator

    spoken = effective_spoken_form(pipeline.operations)
    if spoken is not None and not narrator.language.lower().startswith("en"):
        # The number words and lexicon are English. Mangling a French book is
        # worse than leaving it, so the stage disables itself rather than
        # asking the caller to know this.
        spoken = None

    pauses = _one_operation(pipeline.operations, Pauses) or _NO_PAUSES
    patterns = selected_patterns(pipeline.operations)
    origins: list[_SegmentSource] | None = [] if patterns else None
    grids: dict[str, tuple[Unit, ...]] = {}
    structures: dict[str, StructuralIndex] = {}
    source_chapters = inspection._planning_chapters or inspection.chapters  # noqa: SLF001
    segments, trailing_silence = _compile_segments(
        source_chapters,
        spans,
        cast,
        spoken,
        pauses,
        operations=pipeline.operations,
        origins=origins,
        grids=grids,
        structures=structures,
    )
    if origins is not None:
        segments, trailing_silence = _select_segments(
            segments,
            origins,
            source_chapters,
            pipeline.operations,
            spoken,
            pauses,
            grids,
            structures,
        )
    if not segments:
        raise ValidationError(ErrorCode.EMPTY_SPEECH)
    # Canonical characters, not spoken characters: this is the bill, and it
    # must describe the book the caller supplied.
    counts = {
        chapter.id: sum(
            end - start
            for start, end in selected_ranges(chapter, patterns, grid=grids[chapter.id])
        )
        for chapter in inspection.chapters
    }
    total = sum(counts.values())
    metadata = _output_metadata(
        inspection,
        _one_operation(pipeline.operations, MetadataIntent),
        cover_content_hash,
    )
    if patterns:
        rendered_ids = {segment.chapter_id for segment in segments}
        metadata = replace(
            metadata,
            chapters=tuple(
                replace(chapter, speech_characters=counts[chapter.id])
                for chapter in metadata.chapters
                if chapter.id in rendered_ids
            ),
        )
    schemas = SchemaVersions(
        parser=PARSER_SCHEMA_VERSION,
        normalization=NORMALIZATION_SCHEMA_VERSION,
        planning=PLANNING_SCHEMA_VERSION,
        render=RENDER_SCHEMA_VERSION,
        spoken_form=SPOKEN_FORM_VERSION if spoken is not None else None,
    )
    material = _PlanMaterial(
        schemas,
        source_hash,
        revision,
        voice,
        segments,
        metadata,
        total,
        trailing_silence,
    )
    return ExecutionPlan(
        schema_versions=schemas,
        source_bytes_hash=source_hash,
        model_revision=revision,
        cast=cast,
        segments=segments,
        output=metadata,
        total_speech_characters=total,
        semantic_fingerprint=_fingerprint(material),
        trailing_silence_ms=trailing_silence,
    )


def _one_operation(
    operations: tuple[Operation, ...], kind: type[_OperationT]
) -> _OperationT | None:
    found = tuple(item for item in operations if isinstance(item, kind))
    if len(found) > 1:
        raise ValidationError(ErrorCode.DUPLICATE_OPERATION)
    return found[0] if found else None


def _validated_hash(value: str) -> str:
    normalized = value.strip().lower()
    if _SHA256.fullmatch(normalized) is None:
        raise ValidationError(ErrorCode.INVALID_SOURCE_HASH)
    return normalized


def _resolve_voice(
    voice_id: str, voice: Voice | None, model_revision: str
) -> VoicePlan:
    if voice is None or voice.id != voice_id:
        raise VoiceError(ErrorCode.VOICE_UNRESOLVED)
    commercial_use_allowed = voice.commercial_use_allowed
    if not isinstance(voice.enabled, bool) or not isinstance(
        commercial_use_allowed, bool
    ):
        raise VoiceError(ErrorCode.VOICE_PROVENANCE_REQUIRED)
    if not voice.enabled:
        raise VoiceError(ErrorCode.VOICE_DISABLED)
    required_strings = (
        voice.id,
        voice.name,
        voice.provenance,
        voice.license_id,
        voice.language,
        voice.content_fingerprint,
    )
    if (
        any(value is None or not value.strip() for value in required_strings)
        or not voice.compatible_model_revisions
        or voice.content_fingerprint is None
        or _SHA256.fullmatch(voice.content_fingerprint.strip().lower()) is None
        or any(not revision.strip() for revision in voice.compatible_model_revisions)
    ):
        raise VoiceError(ErrorCode.VOICE_PROVENANCE_REQUIRED)
    compatibility = tuple(sorted(set(voice.compatible_model_revisions)))
    if model_revision not in compatibility:
        raise VoiceError(ErrorCode.VOICE_INCOMPATIBLE)
    return VoicePlan(
        id=voice.id,
        name=voice.name.strip(),
        content_fingerprint=voice.content_fingerprint.strip().lower(),
        language=cast("str", voice.language).strip(),
        provenance=cast("str", voice.provenance).strip(),
        license_id=cast("str", voice.license_id).strip(),
        commercial_use_allowed=commercial_use_allowed,
        compatible_model_revisions=compatibility,
    )


def effective_spans(
    chapter: ChapterInspection,
    machine_spans: tuple[SpeakerSpan, ...],
    operations: tuple[Operation, ...],
    *,
    grid: tuple[Unit, ...] | None = None,
) -> tuple[SpeakerSpan, ...]:
    """Merge machine attribution with tuning rules into the spans planning compiles.

    Deliberately here and not in resolution. Resolution produces the machine
    layer only, which is what lets a correction preserve a resolved pipeline
    and cost no model calls, no store read, and no re-resolution -- just a
    re-plan and the two or three segments whose identity actually changed.

    With no attribution rules the machine spans are returned untouched, so
    every book rendered before this stage existed still plans byte for byte
    the same way.

    Once any rule exists the chapter is re-tiled from grid units, and a rule
    matching nothing must still return the machine spans byte for byte. That
    holds on an attribution invariant: it emits a narration span between any
    two quote runs, so no two machine spans ever share a speaker across a
    boundary and ``_coalesce`` has nothing to merge that attribution left
    separate. ``test_tuning_merge`` pins it.
    """
    rules = _rules_of(operations, Attributions)
    if not rules:
        return machine_spans
    units = build_grid(chapter) if grid is None else grid
    siblings = sibling_counts(units)
    speaker = _machine_lookup(chapter.id, machine_spans)
    decided = [
        resolve_rules(unit, speaker(unit), rules, siblings).value for unit in units
    ]
    return _coalesce(chapter.id, units, decided)


def effective_spoken_form(operations: tuple[Operation, ...]) -> SpokenForm | None:
    """Return the spoken-form settings planning actually speaks under.

    Folds unscoped pronunciation rules into the declared style settings.

    A rule whose pattern matches every unit needs no grid to place, so folding
    it here keeps the whole-book lexicon on one code path -- and keeps its
    segment identity byte-identical whether the entries were written inline or
    reloaded from the sidecar.
    """
    spoken = _one_operation(operations, SpokenForm)
    rules = _rules_of(operations, Pronunciations)
    if not rules:
        return spoken
    if spoken is None:
        # A sidecar can carry pronunciations into a pipeline that never called
        # pronounce(). Honour exactly those: turning on number reading or the
        # built-in table would speak words nobody asked about.
        spoken = SpokenForm(numbers="off", builtin_lexicon=False)
    whole = tuple(rule for rule in rules if rule.where.is_whole_book())
    if not whole:
        return spoken
    return replace(spoken, lexicon=_merged_entries(spoken.lexicon, whole))


def _merged_entries(base: _Entries, rules: tuple[Rule, ...]) -> _Entries:
    """Layer ordered rule payloads over a base table, later entries winning.

    Keyed by the folded form because matching is case-insensitive: two rules
    writing ``Lead`` and ``lead`` name one position, and keeping both would
    leave which of them speaks to dictionary order rather than to precedence.
    """
    merged = {key.casefold(): (key, value) for key, value in base}
    for rule in rules:
        for key, value in cast("_Entries", rule.value):
            merged[key.casefold()] = (key, value)
    return tuple(sorted(merged.values()))


def _lexicon_regions(
    chapter: ChapterInspection,
    operations: tuple[Operation, ...],
    spoken: SpokenForm | None,
    grid: tuple[Unit, ...] | None = None,
) -> _LexiconRegions:
    """Return each canonical run of a chapter beside the lexicon speaking it.

    Empty whenever no pronunciation rule is scoped to part of the book, so a
    whole-book table -- the only shape that existed before tuning -- still
    reaches ``to_spoken`` as one call over one fragment and cannot lose a
    multi-word entry to a unit boundary.
    """
    if spoken is None:
        return ()
    scoped = tuple(
        rule
        for rule in _rules_of(operations, Pronunciations)
        if not rule.where.is_whole_book()
    )
    if not scoped:
        return ()
    units = build_grid(chapter) if grid is None else grid
    siblings = sibling_counts(units)
    regions: list[tuple[int, int, _Entries]] = []
    for unit in units:
        entries = _merged_entries(spoken.lexicon, layered_rules(unit, scoped, siblings))
        if regions and regions[-1][2] == entries and regions[-1][1] == unit.start:
            regions[-1] = (regions[-1][0], unit.end, entries)
        else:
            regions.append((unit.start, unit.end, entries))
    return tuple(regions)


def _speak(
    text: str,
    start: int,
    spoken: SpokenForm,
    regions: _LexiconRegions,
    offsets: list[tuple[int, int, int, int]] | None = None,
) -> str:
    """Apply spoken form to one canonical run, honouring scoped lexicons.

    ``start`` is the run's offset in the chapter, which is what locates it
    among the regions. Each region is transformed separately, so a scoped
    entry can never reach text outside its scope; ``offsets`` is rebased onto
    the whole run so selection clipping still sees one edit list.
    """
    numbers = cast("NumberTier", spoken.numbers)
    features = dict(spoken.features)
    if not regions:
        return to_spoken(
            text,
            numbers=numbers,
            lexicon=spoken.lexicon,
            builtin=spoken.builtin_lexicon,
            features=features,
            offsets=offsets,
        )
    pieces: list[str] = []
    produced = 0
    for lower, upper, entries in regions:
        first, last = max(lower, start), min(upper, start + len(text))
        if first >= last:
            continue
        edits: list[tuple[int, int, int, int]] | None = (
            [] if offsets is not None else None
        )
        piece = to_spoken(
            text[first - start : last - start],
            numbers=numbers,
            lexicon=entries,
            builtin=spoken.builtin_lexicon,
            features=features,
            offsets=edits,
        )
        if offsets is not None:
            offsets.extend(
                (
                    source_start + first - start,
                    source_end + first - start,
                    rendered_start + produced,
                    rendered_end + produced,
                )
                for source_start, source_end, rendered_start, rendered_end in edits
                or ()
            )
        pieces.append(piece)
        produced += len(piece)
    return "".join(pieces)


def manual_gaps(
    chapter: ChapterInspection, operations: tuple[Operation, ...]
) -> Mapping[int, int]:
    """Return each declared silence as a grid unit index and a forced duration.

    A silence anchored to a subtree normalizes to that subtree's last leaf, so
    ``{paragraph: 3}`` and ``{paragraph: 3, sentence: -1}`` name one physical
    gap and produce one key. The duration replaces whatever the derived tier
    model computed there, which is what makes zero an instruction.
    """
    rules = _rules_of(operations, Silences)
    if not rules:
        return {}
    return _gaps_over(build_grid(chapter), rules)


def _rules_of(
    operations: tuple[Operation, ...], kind: type[_RuleOperationT]
) -> tuple[Rule, ...]:
    """Return one tuning family's ordered rules, or none when it is absent."""
    operation = _one_operation(operations, kind)
    return () if operation is None else operation.rules


def _machine_lookup(
    chapter_id: str, spans: tuple[SpeakerSpan, ...]
) -> Callable[[Unit], object | None]:
    """Return the machine speaker covering a unit, by its starting offset.

    Grid units are cut at quote edges and machine spans begin at them, so a
    unit never straddles two machine speakers.
    """
    owned = sorted(
        (span for span in spans if span.chapter_id == chapter_id),
        key=lambda span: span.start,
    )
    starts = [span.start for span in owned]

    def speaker(unit: Unit) -> object | None:
        index = bisect_right(starts, unit.start) - 1
        if index < 0 or unit.start >= owned[index].end:
            return None
        return owned[index].character_id

    return speaker


def _coalesce(
    chapter_id: str, grid: tuple[Unit, ...], decided: list[object | None]
) -> tuple[SpeakerSpan, ...]:
    """Join adjacent units sharing a speaker, so one rule is not one span each.

    Segment count follows span count, so emitting a span per grid unit would
    multiply synthesis overhead across a book for no audible difference.
    """
    spans: list[SpeakerSpan] = []
    for unit, value in zip(grid, decided, strict=True):
        character = cast("str | None", value)
        if (
            spans
            and spans[-1].character_id == character
            and spans[-1].end == unit.start
        ):
            spans[-1] = replace(spans[-1], end=unit.end)
        else:
            spans.append(SpeakerSpan(chapter_id, unit.start, unit.end, character))
    return tuple(spans)


def _gaps_over(grid: tuple[Unit, ...], rules: tuple[Rule, ...]) -> dict[int, int]:
    """Resolve every rule's anchors, then settle collisions by precedence."""
    siblings = sibling_counts(grid)
    anchors: set[int] = set()
    for rule in rules:
        anchors.update(_last_leaves(rule.where, grid, siblings))
    gaps: dict[int, int] = {}
    for index in sorted(anchors):
        value = resolve_rules(grid[index], None, rules, siblings).value
        if isinstance(value, int) and not isinstance(value, bool):
            gaps[index] = value
    return gaps


def _last_leaves(
    pattern: Pattern, grid: tuple[Unit, ...], siblings: SiblingCounts
) -> Iterable[int]:
    """Return the index of the final leaf of each subtree a pattern matches."""
    depth = max((LEVELS.index(level) + 1 for level in pattern), default=0)
    last: dict[tuple[str | int, ...], int] = {}
    for index, unit in enumerate(grid):
        if matches(pattern, unit, siblings):
            last[_subtree(unit, depth)] = index
    return last.values()


def _subtree(unit: Unit, depth: int) -> tuple[str | int, ...]:
    """Return the path prefix identifying the subtree a unit sits in."""
    return (unit.chapter_id, unit.paragraph, unit.line, unit.sentence, unit.phrase)[
        :depth
    ]


def _manual_offsets(
    grid: tuple[Unit, ...], operations: tuple[Operation, ...]
) -> Mapping[int, int]:
    """Return declared silences keyed by the text offset they follow."""
    rules = _rules_of(operations, Silences)
    if not rules:
        return {}
    return {grid[index].end: value for index, value in _gaps_over(grid, rules).items()}


def _compile_segments(  # noqa: PLR0913 - one call site, all state explicit.
    chapters: tuple[ChapterInspection, ...],
    spans: tuple[SpeakerSpan, ...],
    cast_plan: CastPlan,
    spoken: SpokenForm | None = None,
    pauses: Pauses = _NO_PAUSES,
    *,
    operations: tuple[Operation, ...] = (),
    origins: list[_SegmentSource] | None = None,
    grids: dict[str, tuple[Unit, ...]] | None = None,
    structures: dict[str, StructuralIndex] | None = None,
) -> tuple[tuple[SpeechSegment, ...], tuple[int, ...]]:
    """Pack each chapter while assigning one global plan-order ordinal.

    Speaker spans partition a chapter and become mandatory cuts in its single
    hierarchical packing request. Concatenating every packed segment therefore
    reproduces the chapter exactly. A chapter with no spans is one narration
    region, preserving the behavior that existed before attribution.

    A piece holding no speakable character is never a segment of its own. Two
    adjacent quotations are separated by exactly such a span, and an engine
    handed pure whitespace returns no samples. Such spans are assigned to the
    next effective speaker before packing, preserving the legacy concatenated
    text without recombining already-bounded packer output.
    """
    result: list[SpeechSegment] = []
    silence: list[int] = []
    for chapter_index, chapter in enumerate(chapters):
        if (
            not chapter.text
            or chapter.speech_characters is None
            or chapter.speech_characters != len(chapter.text)
        ):
            raise ValidationError(ErrorCode.EMPTY_SPEECH)
        grid = build_grid(chapter)
        if grids is not None:
            grids[chapter.id] = grid
        structure = build_structure_index(grid)
        if structures is not None:
            structures[chapter.id] = structure
        manual = _manual_offsets(grid, operations)
        _append_chapter(
            chapter,
            _spans_for(chapter, effective_spans(chapter, spans, operations, grid=grid)),
            cast_plan,
            grid=grid,
            structure=structure,
            spoken=spoken,
            pauses=pauses,
            gaps=manual,
            regions=_lexicon_regions(chapter, operations, spoken, grid),
            result=result,
            silence=silence,
            origins=origins,
        )
        # The inter-chapter gap folds into this chapter's last segment, so
        # chapter N+1 begins exactly on its first spoken word. Max, not sum:
        # a chapter end meeting a heading-before pause is one gap.
        if silence and chapter_index + 1 < len(chapters):
            chapter_manual = manual.get(grid[-1].end) if grid else None
            if chapter_manual is None:
                silence[-1] = max(silence[-1], _gap_ms(GapReason.CHAPTER, pauses))
    if silence:
        silence[-1] = 0  # A book must not end on dead air.
    return tuple(result), tuple(silence)


def _semantic_cuts(  # noqa: PLR0913, PLR0917 - complete semantic boundary set.
    grid: tuple[Unit, ...],
    spans: tuple[SpeakerSpan, ...],
    pauses: Pauses,
    gaps: Mapping[int, int],
    reasons_by_offset: Mapping[int, GapReason],
    regions: _LexiconRegions,
) -> frozenset[int]:
    """Return every canonical edge the packer is forbidden to cross.

    Speaker spans also cover voice changes because the cast is a pure lookup
    from effective speaker to voice.  Pronunciation regions are already
    coalesced by effective lexicon, while manual gaps remain mandatory even
    when their replacement duration is zero.
    """
    chapter_start = grid[0].start
    chapter_end = grid[-1].end
    cuts = {
        chapter_start,
        chapter_end,
        *(span.start for span in spans),
        *(span.end for span in spans),
        *(edge for region in regions for edge in region[:2]),
        *gaps,
    }
    cuts.update(
        offset
        for offset, reasons in reasons_by_offset.items()
        if _gap_enabled(reasons, pauses)
    )
    return frozenset(cuts)


def _spoken_regions(  # noqa: PLR0913 - complete transformation boundary.
    chapter: ChapterInspection,
    spoken: SpokenForm | None,
    lexicon_regions: _LexiconRegions,
    cuts: Iterable[int],
    *,
    lower: int = 0,
    upper: int | None = None,
    ornaments: frozenset[tuple[int, int]] = frozenset(),
) -> tuple[SpokenRegion, ...]:
    """Transform each semantic interval and retain exact canonical mappings.

    An ornament speaks as nothing, whether or not a spoken form was asked for.
    Suppression cannot sit behind that setting: a pipeline that never called
    ``pronounce()`` is precisely the one handing ``* * *`` to an engine today.
    The canonical range is untouched, so offsets, billing and selection are
    unaffected -- only the string the engine receives loses it.
    """
    last = len(chapter.text) if upper is None else upper
    boundaries = (
        lower,
        *sorted(edge for edge in set(cuts) if lower < edge < last),
        last,
    )
    regions: list[SpokenRegion] = []
    for start, end in pairwise(boundaries):
        canonical = chapter.text[start:end]
        offsets: list[tuple[int, int, int, int]] = []
        if (start, end) in ornaments:
            # One mapping stating the whole range speaks as nothing: the
            # packer requires every divergence to be accounted for.
            rendered = ""
            offsets.append((0, end - start, 0, 0))
        elif spoken is None:
            rendered = canonical
        else:
            rendered = _speak(canonical, start, spoken, lexicon_regions, offsets)
        mappings = tuple(
            SpokenMapping(
                start + source_start, start + source_end, spoken_start, spoken_end
            )
            for source_start, source_end, spoken_start, spoken_end in offsets
        )
        regions.append(SpokenRegion(start, end, rendered, mappings))
    return tuple(regions)


def _global_offsets(
    regions: tuple[SpokenRegion, ...],
) -> tuple[tuple[int, int, int, int], ...]:
    """Flatten region-local replacement mappings into chapter coordinates."""
    offsets: list[tuple[int, int, int, int]] = []
    spoken_start = 0
    for region in regions:
        offsets.extend(
            (
                mapping.canonical_start,
                mapping.canonical_end,
                spoken_start + mapping.spoken_start,
                spoken_start + mapping.spoken_end,
            )
            for mapping in region.mappings
        )
        spoken_start += len(region.text)
    return tuple(offsets)


def _span_at(spans: tuple[SpeakerSpan, ...], offset: int) -> SpeakerSpan:
    """Return the effective speaker span owning one packed canonical start."""
    starts = [span.start for span in spans]
    index = bisect_right(starts, offset) - 1
    if index < 0 or not spans[index].start <= offset < spans[index].end:
        raise ValidationError(ErrorCode.EMPTY_SPEECH)
    return spans[index]


def _absorb_unspeakable_spans(
    canonical: str, spans: tuple[SpeakerSpan, ...]
) -> tuple[SpeakerSpan, ...]:
    """Assign whitespace-only source spans to adjacent effective speech.

    Whitespace has no synthesizable speaker of its own. Phase 1 carried such a
    run onto the next spoken fragment before chunking; moving its canonical
    start onto that fragment preserves the same content/voice ordering while
    letting the grid packer enforce the budget and record every fallback cut.
    A trailing whitespace-only run belongs to the preceding effective span.
    """
    result: list[SpeakerSpan] = []
    pending_start: int | None = None
    for span in spans:
        if not canonical[span.start : span.end].strip():
            if pending_start is None:
                pending_start = span.start
            continue
        effective = (
            span if pending_start is None else replace(span, start=pending_start)
        )
        pending_start = None
        result.append(effective)
    if pending_start is not None and result:
        result[-1] = replace(result[-1], end=spans[-1].end)
    return tuple(result)


def _ornament_edges(
    grid: tuple[Unit, ...], canonical: str
) -> frozenset[tuple[int, int]]:
    """Return the canonical span of every ornament leaf worth suppressing.

    A chapter holding nothing but ornaments would be left with no spoken text
    at all, which the packer rejects outright. Such a chapter is a separator
    page rather than a scene break, so leave it exactly as it was.
    """
    ornaments = frozenset((unit.start, unit.end) for unit in grid if unit.is_ornament)
    speakable = any(
        canonical[unit.start : unit.end].strip()
        for unit in grid
        if not unit.is_ornament
    )
    return ornaments if speakable else frozenset()


def _unspeakable_ranges(
    canonical: str,
    spans: tuple[SpeakerSpan, ...],
    grid: tuple[Unit, ...],
) -> tuple[tuple[int, int], ...]:
    """Return coalesced canonical runs that cannot produce speech.

    Whitespace is the obvious case. An ornament leaf is the other: it holds
    real characters, and therefore real offsets, but it stands in for a scene
    boundary rather than for anything a voice should say.
    """
    ornaments = {(unit.start, unit.end) for unit in grid if unit.is_ornament}
    candidates = sorted(
        {
            (start, end)
            for start, end in (
                *((span.start, span.end) for span in spans),
                *((unit.start, unit.end) for unit in grid),
            )
            if not canonical[start:end].strip() or (start, end) in ornaments
        }
    )
    result: list[tuple[int, int]] = []
    for start, end in candidates:
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return tuple(result)


def _settle_unspeakable_gaps(
    unspeakable: tuple[tuple[int, int], ...], gaps: Mapping[int, int]
) -> dict[int, int]:
    """Move gaps inside inaudible spans to the preceding effective boundary."""
    settled = dict(gaps)
    for start, end in unspeakable:
        offsets = sorted(edge for edge in settled if start < edge <= end)
        for edge in offsets:
            settled[start] = settled.pop(edge)
    return settled


def _settle_unspeakable_reasons(
    unspeakable: tuple[tuple[int, int], ...],
    grid: tuple[Unit, ...],
    structure: StructuralIndex,
) -> dict[int, GapReason]:
    """Fold pure gap reasons across spans that cannot produce speech."""
    return _settle_reasons_over(
        unspeakable,
        {unit.end: reasons for unit, reasons in zip(grid, structure.gaps, strict=True)},
    )


def _settle_reasons_over(
    unspeakable: tuple[tuple[int, int], ...],
    reasons: Mapping[int, GapReason],
) -> dict[int, GapReason]:
    """Fold selected pure gap reasons across inaudible canonical spans."""
    settled = dict(reasons)
    for start, end in unspeakable:
        offsets = sorted(edge for edge in settled if start < edge <= end)
        for edge in offsets:
            settled[start] = settled.get(start, GapReason.NONE) | settled.pop(edge)
    return settled


def _redistribute_unspeakable_ranges(
    packed: tuple[PackedRange, ...],
    spoken_text: str,
    offsets: tuple[tuple[int, int, int, int], ...],
) -> tuple[PackedRange, ...]:
    """Attach bounded whitespace runs to adjacent synthesizable ranges.

    Prefer the following range, matching phase 1's carry direction. If a long
    run does not fit there, its prefix fills the preceding range. Only an
    unrepresentable middle remainder is omitted; that requires more whitespace
    than both adjacent speech segments can hold and phase 1 also discarded the
    resulting whitespace-only chunks.
    """
    items = list(packed)
    result: list[PackedRange] = []
    index = 0
    while index < len(items):
        item = items[index]
        if spoken_text[item.spoken_start : item.spoken_end].strip():
            result.append(item)
            index += 1
            continue

        past = index + 1
        while (
            past < len(items)
            and not spoken_text[
                items[past].spoken_start : items[past].spoken_end
            ].strip()
        ):
            past += 1
        run_start = item.spoken_start
        run_end = items[past - 1].spoken_end

        suffix_start = run_end
        if past < len(items):
            following = items[past]
            capacity = MAX_TTS_SEGMENT_CHARACTERS - (
                following.spoken_end - following.spoken_start
            )
            suffix_start = max(run_start, run_end - capacity)
            if suffix_start < run_end:
                items[past] = replace(
                    following,
                    canonical_start=_translated_offset(
                        offsets, suffix_start, reverse=True
                    ),
                    spoken_start=suffix_start,
                )

        if result and run_start < suffix_start:
            preceding = result[-1]
            capacity = MAX_TTS_SEGMENT_CHARACTERS - (
                preceding.spoken_end - preceding.spoken_start
            )
            prefix_end = min(suffix_start, run_start + capacity)
            if run_start < prefix_end:
                result[-1] = replace(
                    preceding,
                    canonical_end=_translated_offset(
                        offsets, prefix_end, reverse=True, upper_edge=True
                    ),
                    spoken_end=prefix_end,
                    fallback_cut_after=(
                        FallbackCut.WHITESPACE
                        if prefix_end < run_end
                        else preceding.fallback_cut_after
                    ),
                )
        index = past
    return tuple(result)


def _append_chapter(  # noqa: PLR0913 - one call site, all state explicit.
    chapter: ChapterInspection,
    spans: tuple[SpeakerSpan, ...],
    cast_plan: CastPlan,
    *,
    grid: tuple[Unit, ...],
    structure: StructuralIndex,
    spoken: SpokenForm | None,
    pauses: Pauses,
    gaps: Mapping[int, int],
    regions: _LexiconRegions,
    result: list[SpeechSegment],
    silence: list[int],
    origins: list[_SegmentSource] | None = None,
) -> None:
    """Append one chapter packed over its grid and semantic boundaries."""
    unspeakable = _unspeakable_ranges(chapter.text, spans, grid)
    gaps = _settle_unspeakable_gaps(unspeakable, gaps)
    reasons_by_offset = _settle_unspeakable_reasons(unspeakable, grid, structure)
    spans = _absorb_unspeakable_spans(chapter.text, spans)
    cuts = _semantic_cuts(grid, spans, pauses, gaps, reasons_by_offset, regions)
    ornaments = _ornament_edges(grid, chapter.text)
    # Region boundaries, not packing boundaries. An ornament has to be a region
    # of its own to be suppressed as one, but it must never become a mandatory
    # interval: those are required to hold spoken text, and this one holds none.
    spoken_regions = _spoken_regions(
        chapter,
        spoken,
        regions,
        cuts.union(edge for span in ornaments for edge in span),
        ornaments=ornaments,
    )
    spoken_text = "".join(region.text for region in spoken_regions)
    packed = pack_grid(
        PackingInput(
            leaves=grid,
            ranges=structure,
            spoken_regions=spoken_regions,
            mandatory_cuts=cuts,
            character_budget=MAX_TTS_SEGMENT_CHARACTERS,
        )
    )
    source_offsets = _global_offsets(spoken_regions)
    packed = _redistribute_unspeakable_ranges(packed, spoken_text, source_offsets)
    chapter_segment_start = len(result)
    emitted: list[PackedRange] = []
    for chunk_index, item in enumerate(packed):
        text = spoken_text[item.spoken_start : item.spoken_end]
        chunk_start = item.spoken_start
        canonical_start = item.canonical_start
        span = _span_at(spans, item.canonical_start)
        voice = cast_plan.voice_for(span.character_id)
        result.append(
            _segment(
                chapter,
                len(result),
                chunk_index,
                text,
                speaker_id=span.character_id,
                voice_id=voice.id,
                spoken=spoken,
            )
        )
        silence.append(0)
        emitted.append(item)
        if origins is not None:
            origins.append(
                _SegmentSource(
                    0,
                    chapter.text,
                    spoken_text,
                    chunk_start,
                    item.spoken_end,
                    chunk_index,
                    source_offsets,
                    canonical_start,
                    item.canonical_end,
                    item.fallback_cut_after,
                )
            )
    starts = [item.canonical_start for item in emitted]
    for offset in sorted(reasons_by_offset.keys() | gaps.keys()):
        owner = bisect_left(starts, offset) - 1
        if owner >= 0:
            _apply_gap_at(
                silence,
                chapter_segment_start + owner,
                reasons_by_offset.get(offset, GapReason.NONE),
                pauses,
                gaps.get(offset),
            )


def grid_silences(
    chapter: ChapterInspection,
    units: tuple[Unit, ...],
    operations: tuple[Operation, ...],
    *,
    structure: StructuralIndex | None = None,
) -> dict[int, int]:
    """Settle selected grid gaps onto speech-bearing leaves, shared with Script."""
    pauses = _one_operation(operations, Pauses) or _NO_PAUSES
    derived = (build_structure_index(units) if structure is None else structure).gaps
    rules = _rules_of(operations, Silences)
    manual = _gaps_over(units, rules) if rules else {}
    patterns = selected_patterns(operations)
    siblings = sibling_counts(units)
    selected = tuple(
        index
        for index, unit in enumerate(units)
        if selected_unit(unit, patterns, siblings)
    )
    selected_units = tuple(units[index] for index in selected)
    unspeakable = _unspeakable_ranges(chapter.text, (), selected_units)
    reasons_by_offset = _settle_reasons_over(
        unspeakable, {units[index].end: derived[index] for index in selected}
    )
    gaps = _settle_unspeakable_gaps(
        unspeakable,
        {units[index].end: manual[index] for index in selected if index in manual},
    )
    indices: list[int] = []
    silence: list[int] = []
    for index in selected:
        unit = units[index]
        if chapter.text[unit.start : unit.end].strip():
            indices.append(index)
            silence.append(0)
            _apply_gap(
                silence,
                reasons_by_offset.get(unit.end, GapReason.NONE),
                pauses,
                gaps.get(unit.end),
            )
    return dict(zip(indices, silence, strict=True))


def _translated_offset(
    offsets: tuple[tuple[int, int, int, int], ...],
    offset: int,
    *,
    reverse: bool = False,
    upper_edge: bool = False,
) -> int:
    """Locate a canonical grid edge in globally spoken-form-transformed text.

    Equal runs preserve exact positions. Replacement runs belong to the
    source run they replace; a boundary at either end stays at that end.
    A boundary inside a replacement snaps outwards; ``_selected_text`` then
    replaces that complete edit with the selected canonical fragment. No
    guessed character correspondence within a replacement is used.
    """
    shift = 0
    for entry in offsets:
        start, end, lower, upper = (
            (entry[2], entry[3], entry[0], entry[1]) if reverse else entry
        )
        if offset < start:
            break
        if offset <= end:
            if offset == end:
                return upper
            return upper if upper_edge and offset > start else lower
        shift = upper - end
    return offset + shift


def _selected_text(
    origin: _SegmentSource,
    first: int,
    last: int,
    spoken: SpokenForm | None,
    regions: _LexiconRegions = (),
) -> tuple[int, int, str]:
    """Clip at exact edit boundaries, re-speaking only partial replacement edges."""
    start = max(origin.chunk_start, _translated_offset(origin.offsets, first))
    end = min(
        origin.chunk_end, _translated_offset(origin.offsets, last, upper_edge=True)
    )
    pieces: list[str] = []
    position = start
    for lower, upper, rendered_start, rendered_end in origin.offsets:
        if rendered_end <= start or rendered_start >= end:
            continue
        if first <= lower and upper <= last:
            continue
        pieces.append(origin.rendered[position : max(position, rendered_start)])
        if start <= rendered_start < end and spoken is not None:
            # An edit never straddles two regions -- each was transformed on
            # its own -- so re-speaking its surviving part under the region
            # holding its left edge reproduces exactly the scoped lexicon
            # that produced it.
            edge = max(first, lower)
            pieces.append(
                _speak(
                    origin.canonical[edge : min(last, upper)],
                    origin.start + edge,
                    spoken,
                    regions,
                )
            )
        position = min(end, rendered_end)
    pieces.append(origin.rendered[position:end])
    return start, end, "".join(pieces)


def _selected_edge_segments(  # noqa: PLR0913, PLR0917 - one edge context.
    segment: SpeechSegment,
    origin: _SegmentSource,
    chapter: ChapterInspection,
    text: str,
    start: int,
    end: int,
    first: int,
    last: int,
    spoken: SpokenForm | None,
    grid: tuple[Unit, ...],
    structure: StructuralIndex,
) -> tuple[tuple[SpeechSegment, int], ...]:
    """Return one bounded selected edge and its canonical source starts."""
    if start == origin.chunk_start and end == origin.chunk_end and text == segment.text:
        source_start = origin.start + _translated_offset(
            origin.offsets, start, reverse=True
        )
        return ((segment, source_start),)
    if len(text) <= MAX_TTS_SEGMENT_CHARACTERS:
        source_start = origin.start + _translated_offset(
            origin.offsets, start, reverse=True
        )
        return (
            (
                _segment(
                    chapter,
                    segment.ordinal,
                    origin.chunk_index,
                    text,
                    speaker_id=segment.speaker_id,
                    voice_id=segment.voice_id,
                    spoken=spoken,
                    selection=(first, last),
                ),
                source_start,
            ),
        )

    # A selection may expose canonical text that a full-plan pronunciation
    # contracted below the ceiling. Re-speak and hierarchically pack only this
    # changed edge; bounded and wholly-contained segments retain their IDs.
    source_first = max(origin.canonical_start, origin.start + first)
    source_last = min(origin.canonical_end, origin.start + last)
    # Preserve the exact edge text already produced by `_selected_text`.
    # Re-speaking the wider selected range can choose a different overlapping
    # longest-match lexicon entry than the partial edit did. Treat this changed
    # edge as one mapped replacement so the packer owns only its hierarchy,
    # hard ceiling, and emergency cuts—not a second transformation decision.
    selected_regions = (
        SpokenRegion(
            source_first,
            source_last,
            text,
            (SpokenMapping(source_first, source_last, 0, len(text)),),
        ),
    )
    repacked = pack_grid(
        PackingInput(
            leaves=grid,
            ranges=structure,
            spoken_regions=selected_regions,
            mandatory_cuts=frozenset({source_first, source_last}),
            character_budget=MAX_TTS_SEGMENT_CHARACTERS,
        )
    )
    selected_spoken = text
    repacked = _redistribute_unspeakable_ranges(
        repacked, selected_spoken, _global_offsets(selected_regions)
    )
    return tuple(
        (
            _segment(
                chapter,
                segment.ordinal,
                origin.chunk_index,
                selected_spoken[item.spoken_start : item.spoken_end],
                speaker_id=segment.speaker_id,
                voice_id=segment.voice_id,
                spoken=spoken,
                selection=(first, last, item.spoken_start, item.spoken_end),
            ),
            item.canonical_start,
        )
        for item in repacked
    )


def _select_segments(  # noqa: C901, PLR0913, PLR0917 - pure selection inputs.
    segments: tuple[SpeechSegment, ...],
    origins: list[_SegmentSource],
    chapters: tuple[ChapterInspection, ...],
    operations: tuple[Operation, ...],
    spoken: SpokenForm | None,
    pauses: Pauses,
    grids: Mapping[str, tuple[Unit, ...]],
    structures: Mapping[str, StructuralIndex],
) -> tuple[tuple[SpeechSegment, ...], tuple[int, ...]]:
    """Apply selection edges as mandatory clips over the stable full-grid packing.

    Repacking greedily from a selection start would move later boundaries and
    churn wholly-contained cache identities. Clipping only the intersected
    full-plan edge chunks makes the selection edges mandatory while retaining
    every interior segment byte-for-byte.
    """
    patterns = selected_patterns(operations)
    by_id = {chapter.id: chapter for chapter in chapters}
    ranges = {
        chapter.id: selected_ranges(chapter, patterns, grid=grids[chapter.id])
        for chapter in chapters
    }
    regions = {
        chapter.id: _lexicon_regions(chapter, operations, spoken, grids[chapter.id])
        for chapter in chapters
    }
    result: list[SpeechSegment] = []
    starts: dict[str, list[int]] = {}
    positions: dict[str, list[int]] = {}
    for segment, origin in zip(segments, origins, strict=True):
        for lower, upper in ranges[segment.chapter_id]:
            first = max(0, lower - origin.start)
            last = min(len(origin.canonical), upper - origin.start)
            if first >= last:
                continue
            start, end, text = _selected_text(
                origin, first, last, spoken, regions[segment.chapter_id]
            )
            if not text.strip():
                continue
            chapter = by_id[segment.chapter_id]
            selected_segments = _selected_edge_segments(
                segment,
                origin,
                chapter,
                text,
                start,
                end,
                first,
                last,
                spoken,
                grids[segment.chapter_id],
                structures[segment.chapter_id],
            )
            for selected_segment, source_start in selected_segments:
                positions.setdefault(segment.chapter_id, []).append(len(result))
                starts.setdefault(segment.chapter_id, []).append(source_start)
                result.append(selected_segment)
    silence = [0] * len(result)
    for chapter_id, indices in positions.items():
        chapter = by_id[chapter_id]
        grid = grids[chapter_id]
        for unit_index, value in grid_silences(
            chapter,
            grid,
            operations,
            structure=structures[chapter_id],
        ).items():
            local = bisect_right(starts[chapter_id], grid[unit_index].end - 1) - 1
            if local >= 0:
                silence[indices[local]] = value
        rules = _rules_of(operations, Silences)
        manual = _gaps_over(grid, rules) if rules else {}
        final_gap = len(grid) - 1
        if final_gap not in manual:
            silence[indices[-1]] = max(silence[indices[-1]], pauses.chapter_ms)
    if silence:
        silence[-1] = 0
    return tuple(result), tuple(silence)


def _apply_gap(
    silence: list[int],
    reasons: GapReason,
    pauses: Pauses,
    manual: int | None,
) -> None:
    """Settle one gap onto the segment before it.

    Derived reasons take the maximum among themselves, never the sum. A
    declared silence replaces that result outright rather than adding to it,
    which is the only way zero can remove a pause.
    """
    if not silence:
        return
    _apply_gap_at(silence, len(silence) - 1, reasons, pauses, manual)


def _apply_gap_at(
    silence: list[int],
    index: int,
    reasons: GapReason,
    pauses: Pauses,
    manual: int | None,
) -> None:
    """Settle one effective gap onto an explicit preceding segment."""
    silence[index] = (
        manual if manual is not None else max(silence[index], _gap_ms(reasons, pauses))
    )


def _spans_for(
    chapter: ChapterInspection, spans: tuple[SpeakerSpan, ...]
) -> tuple[SpeakerSpan, ...]:
    """Return a chapter's ordered spans, or one narration span covering it."""
    owned = tuple(span for span in spans if span.chapter_id == chapter.id)
    if not owned:
        return (SpeakerSpan(chapter.id, 0, len(chapter.text), None),)
    ordered = tuple(sorted(owned, key=lambda span: span.start))
    covered = "".join(chapter.text[span.start : span.end] for span in ordered)
    if covered != chapter.text:
        raise ValidationError(ErrorCode.EMPTY_SPEECH)
    return ordered


def _segment(  # noqa: PLR0913 - each field is part of a distinct identity.
    chapter: ChapterInspection,
    ordinal: int,
    chunk_index: int,
    text: str,
    *,
    speaker_id: str | None = None,
    voice_id: str = "",
    spoken: SpokenForm | None = None,
    selection: tuple[int, ...] | None = None,
) -> SpeechSegment:
    content_hash = _hash_utf8(text)
    fields: dict[str, object] = {
        "chapter_id": _string_identity(chapter.id),
        "chunk_index": chunk_index,
        "chunking_schema": GRID_CHUNKING_SCHEMA_VERSION,
        "content_hash": content_hash,
        "normalization": NORMALIZATION_SCHEMA_VERSION,
        "ordinal": ordinal,
        "segment_id_version": _SEGMENT_ID_VERSION,
    }
    if selection is not None:
        # Distinguish equal-text disjoint selections inside the same original
        # chunk. Unclipped segments never acquire this field.
        fields["selection"] = selection
    if speaker_id is not None:
        # Added only for attributed speech: narration takes its voice from the
        # plan, while attributed speech must distinguish speaker/cast changes.
        fields["speaker_id"] = _string_identity(speaker_id)
        fields["voice_id"] = _string_identity(voice_id)
    if spoken is not None:
        # Added only when the stage is active. Its configuration changes the
        # synthesis input even where a particular transformed string happens
        # to compare equal to the canonical text.
        fields.update(
            spoken_identity(
                numbers=cast("NumberTier", spoken.numbers),
                lexicon=spoken.lexicon,
                builtin=spoken.builtin_lexicon,
                features=dict(spoken.features),
            )
        )
    identity = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    digest = _hash_utf8(identity)[:24]
    return SpeechSegment(
        id=f"seg-{NORMALIZATION_SCHEMA_VERSION}-{_SEGMENT_ID_VERSION}-{digest}",
        chapter_id=chapter.id,
        ordinal=ordinal,
        text=text,
        character_count=len(text),
        content_hash=content_hash,
        speaker_id=speaker_id,
        voice_id=voice_id,
    )


def _output_metadata(
    inspection: BookInspection,
    intent: MetadataIntent | None,
    cover_content_hash: str | None = None,
) -> OutputMetadata:
    title = inspection.metadata.title
    author = inspection.metadata.author
    cover = CoverIntent.SOURCE
    if intent is not None:
        title = intent.title.strip() if intent.title is not None else title
        author = intent.author.strip() if intent.author is not None else author
        if intent.cover == "source":
            cover = CoverIntent.SOURCE
        elif intent.cover is None:
            cover = CoverIntent.NONE
        else:
            cover = CoverIntent.FILE
    chapters = tuple(
        OutputChapter(
            id=chapter.id,
            source_index=chapter.index,
            title=chapter.title,
            speech_characters=len(chapter.text),
        )
        for chapter in inspection.chapters
    )
    return OutputMetadata(
        title=title,
        author=author,
        chapters=chapters,
        cover=cover,
        source_cover_available=inspection.metadata.cover_available,
        cover_content_hash=cover_content_hash,
    )


def _new_sha256() -> _HashDigest:
    """Create a SHA-256 digest; split out as a narrow test seam."""
    return hashlib.sha256()


def _hash_utf8(value: str) -> str:
    """Hash UTF-8 text while bounding each temporary encoded allocation."""
    digest = _new_sha256()
    for start in range(0, len(value), _UTF8_HASH_CHUNK_CHARACTERS):
        chunk = value[start : start + _UTF8_HASH_CHUNK_CHARACTERS]
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def _string_identity(value: str) -> dict[str, int | str]:
    """Return a bounded collision-resistant identity for an arbitrary string."""
    return {"characters": len(value), "sha256": _hash_utf8(value)}


def _optional_string_identity(value: str | None) -> dict[str, int | str] | None:
    return None if value is None else _string_identity(value)


def _fingerprint(material: _PlanMaterial) -> str:
    schemas = material.schemas
    voice = material.voice
    segments = material.segments
    output = material.output
    schema_versions: dict[str, object] = {
        "parser": schemas.parser,
        "normalization": schemas.normalization,
        "planning": schemas.planning,
        "render": schemas.render,
    }
    if schemas.spoken_form is not None:
        # Present only when the stage is active. Emitting an explicit null
        # would change the canonical JSON -- and so the fingerprint -- for
        # every pipeline that never asked for spoken form.
        schema_versions["spoken_form"] = schemas.spoken_form
    payload = {
        "schema_versions": schema_versions,
        "source_bytes_hash": material.source_bytes_hash,
        "model_revision": _string_identity(material.model_revision),
        "voice": {
            "id": _string_identity(voice.id),
            "name": _string_identity(voice.name),
            "content_fingerprint": voice.content_fingerprint,
            "language": _string_identity(voice.language),
            "provenance": _string_identity(voice.provenance),
            "license_id": _string_identity(voice.license_id),
            "commercial_use_allowed": voice.commercial_use_allowed,
            "compatible_model_revisions": [
                _string_identity(revision)
                for revision in voice.compatible_model_revisions
            ],
        },
        "segments": [
            {
                "id": _string_identity(segment.id),
                "chapter_id": _string_identity(segment.chapter_id),
                "ordinal": segment.ordinal,
                "content_hash": segment.content_hash,
                "character_count": segment.character_count,
            }
            for segment in segments
        ],
        "output": {
            "title": _optional_string_identity(output.title),
            "author": _optional_string_identity(output.author),
            "chapters": [
                {
                    "id": _string_identity(chapter.id),
                    "source_index": chapter.source_index,
                    "title": _string_identity(chapter.title),
                    "speech_characters": chapter.speech_characters,
                }
                for chapter in output.chapters
            ],
            "cover": output.cover.value,
            "source_cover_available": output.source_cover_available,
        },
        "total_speech_characters": material.total,
    }
    if any(material.trailing_silence):
        payload["trailing_silence_ms"] = list(material.trailing_silence)
    if output.cover_content_hash is not None:
        # Absent rather than null: an explicit null would change the
        # fingerprint of every pipeline that never supplied a cover file.
        payload["cover_content_hash"] = output.cover_content_hash
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _hash_utf8(canonical)
