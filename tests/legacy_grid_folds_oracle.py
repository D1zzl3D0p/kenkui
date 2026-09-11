"""Test-only observations of the partitioners that predate grid folds.

This module deliberately lives under ``tests`` and imports private production
helpers only to freeze their current output.  Production code must never import
it.  The migration deletes it after the differential checks are complete.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from kenkui._domain import planning
from kenkui._domain.grid import Unit, build_grid
from kenkui._domain.quotes import extract_spans
from kenkui.inspection import ChapterInspection


class PauseSpec(Protocol):
    """Legacy pause-policy shape used only by the frozen structure oracle."""

    @property
    def chapter_ms(self) -> int:
        """Legacy chapter duration."""
        ...

    @property
    def heading_before_ms(self) -> int:
        """Legacy pre-heading duration."""
        ...

    @property
    def heading_after_ms(self) -> int:
        """Legacy post-heading duration."""
        ...

    @property
    def paragraph_ms(self) -> int:
        """Legacy paragraph duration."""
        ...

    @property
    def line_ms(self) -> int:
        """Legacy line duration."""
        ...


HEADING_BEFORE = "heading_before"
HEADING_AFTER = "heading_after"
PARAGRAPH = "paragraph"
LINE = "line"
_BLOCK = re.compile(r"\n{2,}")


@dataclass(frozen=True, slots=True)
class _LegacyPiece:
    text: str
    reasons: frozenset[str]


def _legacy_break_tiers(pauses: PauseSpec) -> tuple[str, ...]:
    tiers: list[str] = []
    if pauses.heading_before_ms or pauses.heading_after_ms:
        tiers.append("heading")
    if pauses.paragraph_ms:
        tiers.append(PARAGRAPH)
    if pauses.line_ms:
        tiers.append(LINE)
    return tuple(sorted(tiers))


def _legacy_blocks(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    position = 0
    for match in _BLOCK.finditer(text):
        out.append((text[position : match.start()], text[position : match.end()]))
        position = match.end()
    if position < len(text) or not out:
        out.append((text[position:], text[position:]))
    return out


def _legacy_lines(chunk: str, body: str) -> list[str]:
    separator = chunk[len(body) :]
    parts: list[str] = []
    position = 0
    for match in re.finditer(r"\n", body):
        parts.append(body[position : match.end()])
        position = match.end()
    parts.append(body[position:])
    parts = [part for part in parts if part] or [""]
    parts[-1] = f"{parts[-1]}{separator}"
    return parts


def _legacy_split_structural(
    text: str, headings: frozenset[str], pauses: PauseSpec
) -> tuple[_LegacyPiece, ...]:
    if not _legacy_break_tiers(pauses) or not text:
        return (_LegacyPiece(text, frozenset()),)
    blocks = _legacy_blocks(text)
    pieces: list[_LegacyPiece] = []
    for index, (body, chunk) in enumerate(blocks):
        last = index + 1 == len(blocks)
        reasons: set[str] = set()
        if pauses.paragraph_ms and not last:
            reasons.add(PARAGRAPH)
        if pauses.heading_after_ms and body in headings and not last:
            reasons.add(HEADING_AFTER)
        if pauses.heading_before_ms and not last and blocks[index + 1][0] in headings:
            reasons.add(HEADING_BEFORE)
        if pauses.line_ms:
            parts = _legacy_lines(chunk, body)
            for order, part in enumerate(parts):
                tail = order + 1 == len(parts)
                pieces.append(
                    _LegacyPiece(
                        part, frozenset(reasons) if tail else frozenset({LINE})
                    )
                )
        else:
            pieces.append(_LegacyPiece(chunk, frozenset(reasons)))
    return tuple(piece for piece in pieces if piece.text)


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


# Captured at d72c1e8 before the production planner switched packers. Keeping
# this as plain immutable data makes the Task 7 differential independent of
# whichever compiler implementation is currently imported by the test suite.
LEGACY_PLAN_OBSERVATION = PlanObservation(
    spoken_text=(
        'Chapter Four\n\nHe paid twenty-one percent.\n"Wait," Mira said.\n\nTail.'
    ),
    segment_texts=(
        "Chapter Four\n\n",
        "He paid twenty-one percent.\n",
        '"Wait,"',
        " Mira said.\n\n",
        "Tail.",
    ),
    speakers=(None, None, "mira", None, None),
    voices=("narrator", "narrator", "mira-voice", "narrator", "narrator"),
    canonical_ranges=((0, 12), (12, 25), (25, 32), (32, 45), (45, 50)),
    spoken_ranges=((0, 14), (14, 42), (42, 49), (49, 62), (62, 67)),
    effective_silences_ms=(500, 100, 0, 300, 0),
)


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
        for piece in _legacy_split_structural(text, headings, pauses)
    )


def legacy_chunks(chapter_id: str, text: str) -> tuple[str, ...]:
    """Run the current ``_chunk_span`` policy behind a test-only boundary."""
    chapter = ChapterInspection(chapter_id, 0, "Fixture", len(text), text)
    return planning._chunk_span(chapter, text)  # noqa: SLF001


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
