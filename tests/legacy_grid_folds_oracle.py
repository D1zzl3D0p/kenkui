"""Test-only observations of the partitioners that predate grid folds.

This module deliberately lives under ``tests`` and imports private production
helpers only to freeze their current output.  Production code must never import
it.  The migration deletes it after the differential checks are complete.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from kenkui._characters.quotes import extract_spans
from kenkui._domain import planning
from kenkui._domain.grid import Unit, build_grid
from kenkui._domain.operations import Operation, Pauses
from kenkui._domain.planning import SpeakerSpan, compile_execution_plan
from kenkui._domain.structure import PauseSpec, split_structural
from kenkui.inspection import BookInspection, ChapterInspection

if TYPE_CHECKING:
    from collections.abc import Mapping

    from kenkui.voices import Voice


class PipelineIntent(Protocol):
    """The narrow pipeline surface accepted by the legacy planner."""

    @property
    def operations(self) -> tuple[Operation, ...]:
        """Return the pipeline's immutable semantic operations."""
        ...


@dataclass(frozen=True, slots=True)
class QuoteObservation:
    """One legacy quote-partition span."""

    start: int
    end: int
    dialogue: bool


@dataclass(frozen=True, slots=True)
class StructureObservation:
    """One legacy structure piece and the gap reasons after it."""

    text: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PlanObservation:
    """Observable content and coordinates of one legacy execution plan."""

    spoken_text: str
    segment_texts: tuple[str, ...]
    speakers: tuple[str | None, ...]
    voices: tuple[str, ...]
    canonical_ranges: tuple[tuple[int, int], ...]
    spoken_ranges: tuple[tuple[int, int], ...]
    effective_silences_ms: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BreakQualityObservation:
    """Counts of normal grid-edge breaks and within-leaf emergency cuts."""

    characters: int
    chunks: int
    internal_boundaries: tuple[int, ...]
    structural_or_clause_grid_edges: tuple[tuple[str, int], ...]
    emergency_within_leaf: tuple[tuple[str, int], ...]


def legacy_quote_partition(chapter_id: str, text: str) -> tuple[QuoteObservation, ...]:
    """Return the current quote scanner's exact partition as plain test values."""
    return tuple(
        QuoteObservation(span.start, span.end, span.is_dialogue)
        for span in extract_spans(chapter_id, text)
    )


def legacy_structure_partition(
    text: str, headings: frozenset[str], pauses: PauseSpec
) -> tuple[StructureObservation, ...]:
    """Return the current structural split without exposing production records."""
    return tuple(
        StructureObservation(piece.text, tuple(sorted(piece.reasons)))
        for piece in split_structural(text, headings, pauses)
    )


def legacy_chunks(chapter_id: str, text: str) -> tuple[str, ...]:
    """Run the current ``_chunk_span`` policy behind a test-only boundary."""
    chapter = ChapterInspection(chapter_id, 0, "Fixture", len(text), text)
    return planning._chunk_span(chapter, text)  # noqa: SLF001


def observe_legacy_plan(  # noqa: PLR0913
    pipeline: PipelineIntent,
    inspection: BookInspection,
    *,
    source_bytes_hash: str,
    resolved_voice: Voice,
    model_revision: str,
    cast_voices: tuple[Voice, ...] = (),
    assignments: Mapping[str, str] | None = None,
    unknown_voice_id: str | None = None,
    spans: tuple[SpeakerSpan, ...] = (),
) -> PlanObservation:
    """Compile and expose the legacy plan's semantic sequence and coordinates.

    The public plan intentionally omits canonical offsets.  A second invocation
    of the same pure legacy segment compiler asks it for its temporary source
    records, then verifies those records reproduce the public plan exactly.
    """
    plan = compile_execution_plan(
        pipeline,
        inspection,
        source_bytes_hash=source_bytes_hash,
        resolved_voice=resolved_voice,
        model_revision=model_revision,
        cast_voices=cast_voices,
        assignments=assignments,
        unknown_voice_id=unknown_voice_id,
        spans=spans,
    )
    spoken = planning.effective_spoken_form(pipeline.operations)
    if spoken is not None and not plan.cast.narrator.language.lower().startswith("en"):
        spoken = None
    pauses = planning._one_operation(pipeline.operations, Pauses) or Pauses()  # noqa: SLF001
    origins: list[planning._SegmentSource] = []
    chapters = inspection._planning_chapters or inspection.chapters  # noqa: SLF001
    replayed, replayed_silences = planning._compile_segments(  # noqa: SLF001
        chapters,
        spans,
        plan.cast,
        spoken,
        pauses,
        operations=pipeline.operations,
        origins=origins,
    )
    assert replayed == plan.segments
    assert replayed_silences == plan.trailing_silence_ms

    canonical_ranges: list[tuple[int, int]] = []
    spoken_ranges: list[tuple[int, int]] = []
    spoken_position = 0
    for segment, origin in zip(plan.segments, origins, strict=True):
        canonical_start = origin.start + planning._translated_offset(  # noqa: SLF001
            origin.offsets, origin.chunk_start, reverse=True
        )
        canonical_end = origin.start + planning._translated_offset(  # noqa: SLF001
            origin.offsets,
            origin.chunk_end,
            reverse=True,
            upper_edge=True,
        )
        canonical_ranges.append((canonical_start, canonical_end))
        spoken_ranges.append((spoken_position, spoken_position + len(segment.text)))
        spoken_position += len(segment.text)

    return PlanObservation(
        spoken_text="".join(segment.text for segment in plan.segments),
        segment_texts=tuple(segment.text for segment in plan.segments),
        speakers=tuple(segment.speaker_id for segment in plan.segments),
        voices=tuple(segment.voice_id for segment in plan.segments),
        canonical_ranges=tuple(canonical_ranges),
        spoken_ranges=tuple(spoken_ranges),
        effective_silences_ms=plan.trailing_silence_ms,
    )


def _grid_edge_kind(left: Unit, right: Unit) -> str:
    """Name the strongest hierarchy or semantic reason for a grid edge."""
    if left.paragraph != right.paragraph:
        return "paragraph"
    if left.line != right.line:
        return "line"
    if left.sentence != right.sentence:
        return "sentence"
    if left.is_dialogue != right.is_dialogue:
        return "dialogue"
    return "phrase"


def _emergency_kind(text: str, boundary: int) -> str:
    """Characterize a legacy cut that falls strictly inside a grid leaf."""
    preceding = text[boundary - 1]
    if preceding in ".!?,;:-\u2010\u2011\u2012\u2013\u2014\u2015":
        return "punctuation_or_hyphen"
    if preceding.isspace():
        return "whitespace"
    return "hard_token"


def legacy_break_quality(chapter_id: str, text: str) -> BreakQualityObservation:
    """Measure legacy breaks without treating its current percentage as a target."""
    chunks = legacy_chunks(chapter_id, text)
    boundaries: list[int] = []
    position = 0
    for chunk in chunks[:-1]:
        position += len(chunk)
        boundaries.append(position)

    chapter = ChapterInspection(chapter_id, 0, "Fixture", len(text), text)
    units = build_grid(chapter)
    grid_edges = {unit.end: index for index, unit in enumerate(units[:-1])}
    ordinary: Counter[str] = Counter()
    emergency: Counter[str] = Counter()
    for boundary in boundaries:
        left_index = grid_edges.get(boundary)
        if left_index is None:
            emergency[_emergency_kind(text, boundary)] += 1
        else:
            ordinary[_grid_edge_kind(units[left_index], units[left_index + 1])] += 1
    return BreakQualityObservation(
        characters=len(text),
        chunks=len(chunks),
        internal_boundaries=tuple(boundaries),
        structural_or_clause_grid_edges=tuple(sorted(ordinary.items())),
        emergency_within_leaf=tuple(sorted(emergency.items())),
    )
