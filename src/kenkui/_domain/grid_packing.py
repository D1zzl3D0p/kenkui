"""Hierarchical packing over the addressable grid.

The packer is deliberately ignorant of why a canonical edge is mandatory and
of how spoken text was produced.  Its caller supplies already-transformed
regions and sparse source-to-spoken replacement mappings; this module only
projects grid ranges, applies the character ceiling, and reports boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import TYPE_CHECKING

from kenkui._domain.grid import LeafRange, StructuralIndex, Unit

if TYPE_CHECKING:
    from kenkui._domain.paths import Path

_PUNCTUATION_OR_HYPHEN = re.compile(r"[.!?,;:\-\u2010-\u2015][\"'\u2019\u201d)\]]*\s*")
_WHITESPACE = re.compile(r"\s+")
_LEVELS = ("paragraph", "line", "sentence", "phrase")


@dataclass(frozen=True, slots=True)
class SpokenMapping:
    """One replacement inside a spoken region.

    Canonical offsets are absolute chapter offsets.  Spoken offsets are local
    to ``SpokenRegion.text``.  Text between mappings is unchanged and must
    therefore occupy equal-length canonical and spoken ranges.
    """

    canonical_start: int
    canonical_end: int
    spoken_start: int
    spoken_end: int


@dataclass(frozen=True, slots=True)
class SpokenRegion:
    """One contiguous canonical run and the exact text spoken for it."""

    canonical_start: int
    canonical_end: int
    text: str
    mappings: tuple[SpokenMapping, ...] = ()


@dataclass(frozen=True, slots=True)
class PackingInput:
    """Complete, immutable inputs to one deterministic packing operation."""

    leaves: tuple[Unit, ...]
    ranges: StructuralIndex
    spoken_regions: tuple[SpokenRegion, ...]
    mandatory_cuts: frozenset[int]
    character_budget: int


class FallbackCut(StrEnum):
    """Why a within-phrase emergency boundary was selected."""

    PUNCTUATION_OR_HYPHEN = "punctuation_or_hyphen"
    WHITESPACE = "whitespace"
    HARD_TOKEN = "hard_token"  # noqa: S105 - a cut category, not a credential


class _PackingViolation(StrEnum):
    CANONICAL_OFFSET = "canonical offset lies outside spoken regions"
    SPOKEN_OFFSET = "spoken offset lies outside spoken regions"
    INVALID_MAPPING = "invalid spoken mapping"
    MAPPING_COVERAGE = "spoken mapping does not cover unchanged text exactly"
    EMPTY_REGIONS = "spoken regions must be non-empty"
    REGION_ORDER = "spoken regions must be ordered, contiguous, and non-empty"
    EMPTY_SPOKEN = "spoken text must be non-empty"
    FALLBACK_START = "fallback lost its canonical start"
    INVALID_REQUEST = "packing requires non-empty leaves and a positive budget"
    RANGES = "structural ranges do not match leaves"
    REGION_BOUNDS = "spoken regions lie outside the grid"
    CUT_BOUNDS = "mandatory cut lies outside spoken regions"
    CUT_REPLACEMENT = "mandatory cut lies inside a spoken replacement"
    EMPTY_RESULT = "packing produced no output"
    RESULT_START = "packing lost its spoken start"
    RESULT_BOUNDS = "packing produced an empty or over-budget segment"
    RESULT_CANONICAL = "packing produced invalid canonical envelopes"
    RESULT_RECONSTRUCTION = "packing does not reconstruct spoken text exactly"
    RESULT_END = "packing lost its spoken end"
    MANDATORY_CROSSED = "packing crossed a mandatory boundary"
    EMPTY_INTERVAL = "mandatory interval has no spoken text"


class PackingError(ValueError):
    """Raised when typed packing inputs cannot form a valid bounded plan."""

    def __init__(self, violation: _PackingViolation) -> None:
        super().__init__(violation.value)


@dataclass(frozen=True, slots=True)
class PackedRange:
    """One output segment in canonical and concatenated-spoken coordinates.

    ``fallback_cut_after`` is set only when this segment ends at a boundary
    created inside one over-budget phrase.  Canonical ranges are exact for
    grid and mandatory edges.  If an emergency spoken cut falls inside a
    transformed replacement, the range is the stable canonical envelope of
    that replacement and may overlap the next emergency range.
    """

    canonical_start: int
    canonical_end: int
    spoken_start: int
    spoken_end: int
    fallback_cut_after: FallbackCut | None = None


@dataclass(frozen=True, slots=True)
class _Projection:
    """Validated regions with their starts in concatenated spoken text."""

    regions: tuple[SpokenRegion, ...]
    spoken_starts: tuple[int, ...]

    @property
    def canonical_start(self) -> int:
        return self.regions[0].canonical_start

    @property
    def canonical_end(self) -> int:
        return self.regions[-1].canonical_end

    @property
    def spoken_text(self) -> str:
        return "".join(region.text for region in self.regions)

    def to_spoken(self, offset: int) -> int:
        """Project a canonical edge, attaching a split edit to its right side."""
        for index, region in enumerate(self.regions):
            if region.canonical_start <= offset <= region.canonical_end:
                local = _canonical_to_spoken(region, offset)
                return self.spoken_starts[index] + local
        raise PackingError(_PackingViolation.CANONICAL_OFFSET)

    def to_canonical(self, offset: int, *, upper_edge: bool) -> int:
        """Project a spoken edge back to its stable canonical envelope."""
        total = sum(len(region.text) for region in self.regions)
        if not 0 <= offset <= total:
            raise PackingError(_PackingViolation.SPOKEN_OFFSET)
        for index, (region, spoken_start) in enumerate(
            zip(self.regions, self.spoken_starts, strict=True)
        ):
            spoken_end = spoken_start + len(region.text)
            if spoken_start <= offset <= spoken_end:
                return _spoken_to_canonical(
                    region, offset - spoken_start, upper_edge=upper_edge
                )
            if index + 1 == len(self.regions):
                break
        raise PackingError(_PackingViolation.SPOKEN_OFFSET)


@dataclass(frozen=True, slots=True)
class _Piece:
    canonical_start: int
    canonical_end: int
    spoken_start: int
    spoken_end: int
    fallback_cut_after: FallbackCut | None = None


def _packed_range(piece: _Piece) -> PackedRange:
    return PackedRange(
        piece.canonical_start,
        piece.canonical_end,
        piece.spoken_start,
        piece.spoken_end,
        piece.fallback_cut_after,
    )


def _valid_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_mapping(region: SpokenRegion) -> None:
    canonical_cursor = region.canonical_start
    spoken_cursor = 0
    for mapping in region.mappings:
        if (
            not all(
                _valid_int(value)
                for value in (
                    mapping.canonical_start,
                    mapping.canonical_end,
                    mapping.spoken_start,
                    mapping.spoken_end,
                )
            )
            or not canonical_cursor <= mapping.canonical_start < mapping.canonical_end
            or not spoken_cursor <= mapping.spoken_start <= mapping.spoken_end
            or mapping.canonical_end > region.canonical_end
            or mapping.spoken_end > len(region.text)
            or mapping.canonical_start - canonical_cursor
            != mapping.spoken_start - spoken_cursor
        ):
            raise PackingError(_PackingViolation.INVALID_MAPPING)
        canonical_cursor = mapping.canonical_end
        spoken_cursor = mapping.spoken_end
    if region.canonical_end - canonical_cursor != len(region.text) - spoken_cursor:
        raise PackingError(_PackingViolation.MAPPING_COVERAGE)


def _projection(regions: tuple[SpokenRegion, ...]) -> _Projection:
    if not regions:
        raise PackingError(_PackingViolation.EMPTY_REGIONS)
    starts: list[int] = []
    spoken_cursor = 0
    canonical_cursor = regions[0].canonical_start
    for region in regions:
        if (
            not _valid_int(region.canonical_start)
            or not _valid_int(region.canonical_end)
            or region.canonical_start != canonical_cursor
            or region.canonical_end <= region.canonical_start
        ):
            raise PackingError(_PackingViolation.REGION_ORDER)
        _validate_mapping(region)
        starts.append(spoken_cursor)
        spoken_cursor += len(region.text)
        canonical_cursor = region.canonical_end
    if spoken_cursor == 0:
        raise PackingError(_PackingViolation.EMPTY_SPOKEN)
    return _Projection(regions, tuple(starts))


def _canonical_to_spoken(region: SpokenRegion, offset: int) -> int:
    canonical_cursor = region.canonical_start
    spoken_cursor = 0
    for mapping in region.mappings:
        if offset < mapping.canonical_start:
            return spoken_cursor + offset - canonical_cursor
        if offset <= mapping.canonical_end:
            if offset == mapping.canonical_end:
                return mapping.spoken_end
            return mapping.spoken_start
        canonical_cursor = mapping.canonical_end
        spoken_cursor = mapping.spoken_end
    return spoken_cursor + offset - canonical_cursor


def _spoken_to_canonical(region: SpokenRegion, offset: int, *, upper_edge: bool) -> int:
    canonical_cursor = region.canonical_start
    spoken_cursor = 0
    for mapping in region.mappings:
        if offset < mapping.spoken_start:
            return canonical_cursor + offset - spoken_cursor
        if offset <= mapping.spoken_end:
            if offset == mapping.spoken_end:
                return mapping.canonical_end
            return mapping.canonical_end if upper_edge else mapping.canonical_start
        canonical_cursor = mapping.canonical_end
        spoken_cursor = mapping.spoken_end
    return canonical_cursor + offset - spoken_cursor


def _path_depth(path: Path) -> int:
    values = (path.paragraph, path.line, path.sentence, path.phrase)
    return sum(value is not None for value in values)


def _level_ranges(index: StructuralIndex) -> tuple[tuple[LeafRange, ...], ...]:
    levels: list[list[LeafRange]] = [[] for _level in _LEVELS]
    for path, leaf_range in index.items():
        depth = _path_depth(path)
        if 1 <= depth <= len(_LEVELS):
            levels[depth - 1].append(leaf_range)
    return tuple(tuple(level) for level in levels)


def _canonical_range(
    leaves: tuple[Unit, ...], leaf_range: LeafRange
) -> tuple[int, int]:
    return (
        leaves[leaf_range.first].start,
        leaves[leaf_range.past_last - 1].end,
    )


def _children(
    levels: tuple[tuple[LeafRange, ...], ...],
    level: int,
    parent: LeafRange,
) -> tuple[LeafRange, ...]:
    return tuple(
        child
        for child in levels[level + 1]
        if parent.first <= child.first and child.past_last <= parent.past_last
    )


def _last_match_end(pattern: re.Pattern[str], text: str, start: int, stop: int) -> int:
    return max(
        (
            start + match.end()
            for match in pattern.finditer(text[start:stop])
            if match.end()
        ),
        default=0,
    )


def _fallback_pieces(  # noqa: PLR0913, PLR0917 - one bounded leaf projection.
    projection: _Projection,
    canonical_start: int,
    canonical_end: int,
    spoken_start: int,
    spoken_end: int,
    budget: int,
) -> tuple[_Piece, ...]:
    text = projection.spoken_text
    pieces: list[_Piece] = []
    position = spoken_start
    while spoken_end - position > budget:
        stop = position + budget
        cut = _last_match_end(_PUNCTUATION_OR_HYPHEN, text, position, stop)
        reason = FallbackCut.PUNCTUATION_OR_HYPHEN
        if cut <= position:
            cut = _last_match_end(_WHITESPACE, text, position, stop)
            reason = FallbackCut.WHITESPACE
        if cut <= position:
            cut = stop
            reason = FallbackCut.HARD_TOKEN
        pieces.append(
            _Piece(
                projection.to_canonical(position, upper_edge=False),
                projection.to_canonical(cut, upper_edge=True),
                position,
                cut,
                reason,
            )
        )
        position = cut
    pieces.append(
        _Piece(
            projection.to_canonical(position, upper_edge=False),
            canonical_end,
            position,
            spoken_end,
        )
    )
    if pieces[0].canonical_start != canonical_start:
        raise AssertionError(_PackingViolation.FALLBACK_START.value)
    return tuple(pieces)


def _structural_pieces(  # noqa: PLR0913, PLR0917 - recursive traversal state.
    leaves: tuple[Unit, ...],
    levels: tuple[tuple[LeafRange, ...], ...],
    projection: _Projection,
    leaf_range: LeafRange,
    level: int,
    lower: int,
    upper: int,
    budget: int,
) -> tuple[_Piece, ...]:
    range_start, range_end = _canonical_range(leaves, leaf_range)
    canonical_start = max(lower, range_start)
    canonical_end = min(upper, range_end)
    if canonical_start >= canonical_end:
        return ()
    spoken_start = projection.to_spoken(canonical_start)
    spoken_end = projection.to_spoken(canonical_end)
    if spoken_end - spoken_start <= budget:
        return (_Piece(canonical_start, canonical_end, spoken_start, spoken_end),)
    if level + 1 == len(_LEVELS):
        return _fallback_pieces(
            projection,
            canonical_start,
            canonical_end,
            spoken_start,
            spoken_end,
            budget,
        )
    pieces: list[_Piece] = []
    for child in _children(levels, level, leaf_range):
        pieces.extend(
            _structural_pieces(
                leaves,
                levels,
                projection,
                child,
                level + 1,
                lower,
                upper,
                budget,
            )
        )
    return tuple(pieces)


def _pieces_for_interval(  # noqa: PLR0913, PLR0917 - immutable traversal state.
    leaves: tuple[Unit, ...],
    levels: tuple[tuple[LeafRange, ...], ...],
    projection: _Projection,
    lower: int,
    upper: int,
    budget: int,
) -> tuple[_Piece, ...]:
    pieces: list[_Piece] = []
    for paragraph in levels[0]:
        pieces.extend(
            _structural_pieces(
                leaves,
                levels,
                projection,
                paragraph,
                0,
                lower,
                upper,
                budget,
            )
        )
    return tuple(pieces)


def _combine(pieces: tuple[_Piece, ...], budget: int) -> tuple[PackedRange, ...]:
    packed: list[PackedRange] = []
    pending: _Piece | None = None
    for piece in pieces:
        if piece.spoken_start == piece.spoken_end:
            if pending is None:
                pending = piece
            else:
                pending = _Piece(
                    pending.canonical_start,
                    piece.canonical_end,
                    pending.spoken_start,
                    pending.spoken_end,
                    pending.fallback_cut_after,
                )
            continue
        if pending is None:
            pending = piece
        elif (
            pending.fallback_cut_after is None
            and pending.spoken_end == piece.spoken_start
            and piece.spoken_end - pending.spoken_start <= budget
        ):
            pending = _Piece(
                pending.canonical_start,
                piece.canonical_end,
                pending.spoken_start,
                piece.spoken_end,
                piece.fallback_cut_after,
            )
        else:
            if pending.spoken_start != pending.spoken_end:
                packed.append(_packed_range(pending))
            pending = piece
        if pending.fallback_cut_after is not None:
            packed.append(_packed_range(pending))
            pending = None
    if pending is not None and pending.spoken_start != pending.spoken_end:
        packed.append(_packed_range(pending))
    return tuple(packed)


def _validate_request(request: PackingInput) -> _Projection:
    if (
        not _valid_int(request.character_budget)
        or request.character_budget <= 0
        or not request.leaves
    ):
        raise PackingError(_PackingViolation.INVALID_REQUEST)
    expected = StructuralIndex(request.leaves)
    if request.ranges != expected:
        raise PackingError(_PackingViolation.RANGES)
    projection = _projection(request.spoken_regions)
    if (
        projection.canonical_start < request.leaves[0].start
        or projection.canonical_end > request.leaves[-1].end
    ):
        raise PackingError(_PackingViolation.REGION_BOUNDS)
    for cut in request.mandatory_cuts:
        if (
            not _valid_int(cut)
            or not projection.canonical_start <= cut <= projection.canonical_end
        ):
            raise PackingError(_PackingViolation.CUT_BOUNDS)
        if any(
            mapping.canonical_start < cut < mapping.canonical_end
            for region in request.spoken_regions
            for mapping in region.mappings
        ):
            raise PackingError(_PackingViolation.CUT_REPLACEMENT)
    return projection


def _assert_result(
    result: tuple[PackedRange, ...],
    projection: _Projection,
    mandatory: frozenset[int],
    budget: int,
) -> None:
    if not result:
        raise AssertionError(_PackingViolation.EMPTY_RESULT.value)
    if result[0].spoken_start != 0:
        raise AssertionError(_PackingViolation.RESULT_START.value)
    if any(
        item.spoken_start >= item.spoken_end
        or item.spoken_end - item.spoken_start > budget
        for item in result
    ):
        raise AssertionError(_PackingViolation.RESULT_BOUNDS.value)
    if (
        result[0].canonical_start != projection.canonical_start
        or result[-1].canonical_end != projection.canonical_end
        or any(
            item.canonical_start < projection.canonical_start
            or item.canonical_start >= item.canonical_end
            or item.canonical_end > projection.canonical_end
            for item in result
        )
        or any(
            left.canonical_start > right.canonical_start
            or left.canonical_end > right.canonical_end
            for left, right in pairwise(result)
        )
    ):
        raise AssertionError(_PackingViolation.RESULT_CANONICAL.value)
    if any(left.spoken_end != right.spoken_start for left, right in pairwise(result)):
        raise AssertionError(_PackingViolation.RESULT_RECONSTRUCTION.value)
    if result[-1].spoken_end != len(projection.spoken_text):
        raise AssertionError(_PackingViolation.RESULT_END.value)
    boundaries = {item.canonical_end for item in result[:-1]}
    if (
        not mandatory - {projection.canonical_start, projection.canonical_end}
        <= boundaries
    ):
        raise AssertionError(_PackingViolation.MANDATORY_CROSSED.value)


def pack_grid(request: PackingInput) -> tuple[PackedRange, ...]:
    """Pack transformed text by paragraph, line, sentence, then phrase.

    Structural candidates descend only when their complete spoken projection
    exceeds the hard budget.  The resulting adjacent candidates are combined
    greedily, while each mandatory canonical interval is packed independently.
    """
    projection = _validate_request(request)
    levels = _level_ranges(request.ranges)
    boundaries = (
        projection.canonical_start,
        *sorted(
            cut
            for cut in request.mandatory_cuts
            if projection.canonical_start < cut < projection.canonical_end
        ),
        projection.canonical_end,
    )
    result: list[PackedRange] = []
    for lower, upper in pairwise(boundaries):
        spoken_lower = projection.to_spoken(lower)
        spoken_upper = projection.to_spoken(upper)
        if spoken_lower == spoken_upper:
            raise PackingError(_PackingViolation.EMPTY_INTERVAL)
        pieces = _pieces_for_interval(
            request.leaves,
            levels,
            projection,
            lower,
            upper,
            request.character_budget,
        )
        result.extend(_combine(pieces, request.character_budget))
    packed = tuple(result)
    _assert_result(packed, projection, request.mandatory_cuts, request.character_budget)
    return packed
