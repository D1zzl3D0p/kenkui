"""Materialized stable chapter selection semantics."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, TypeVar

from kenkui._domain.grid import build_grid, sibling_counts
from kenkui._domain.operations import Select
from kenkui._domain.paths import Any, matches
from kenkui.errors import ErrorCode, ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Protocol

    from kenkui._domain.grid import Unit
    from kenkui._domain.operations import Operation
    from kenkui._domain.paths import Pattern, SiblingCounts
    from kenkui.inspection import ChapterInspection

    class _Chapter(Protocol):
        @property
        def id(self) -> str:
            """Stable chapter identifier."""
            ...

else:
    _Chapter = object

_ChapterT = TypeVar("_ChapterT", bound=_Chapter)


def selected_patterns(operations: tuple[Operation, ...]) -> tuple[Pattern, ...]:
    """Return the recorded grid union, or no restriction."""
    return next((op.patterns for op in operations if isinstance(op, Select)), ())


def selected_unit(
    unit: Unit, patterns: tuple[Pattern, ...], siblings: SiblingCounts
) -> bool:
    """Match a union against source sibling counts, never selected counts."""
    return not patterns or any(matches(pattern, unit, siblings) for pattern in patterns)


def selected_ranges(
    chapter: ChapterInspection, patterns: tuple[Pattern, ...]
) -> tuple[tuple[int, int], ...]:
    """Return disjoint canonical intervals without filling holes between matches."""
    if not patterns:
        return ((0, len(chapter.text)),)
    if not any(
        pattern.get("chapter", Any()).covers(chapter.id, None) for pattern in patterns
    ):
        return ()
    units = build_grid(chapter)
    siblings = sibling_counts(units)
    ranges: list[tuple[int, int]] = []
    for unit in units:
        if selected_unit(unit, patterns, siblings):
            if ranges and ranges[-1][1] == unit.start:
                ranges[-1] = (ranges[-1][0], unit.end)
            else:
                ranges.append((unit.start, unit.end))
    return tuple(ranges)


def _unique_positions(chapters: Sequence[_ChapterT]) -> dict[str, int]:
    counts = Counter(chapter.id for chapter in chapters)
    if any(count > 1 for count in counts.values()):
        raise ValidationError(ErrorCode.DUPLICATE_CHAPTER_ID)
    return {chapter.id: index for index, chapter in enumerate(chapters)}


def select_chapters(
    chapters: Sequence[_ChapterT], chapter_ids: tuple[str, ...]
) -> tuple[_ChapterT, ...]:
    """Select unique IDs in caller order after authoritative materialization."""
    if not chapter_ids:
        raise ValidationError(ErrorCode.EMPTY_SELECTION)
    if len(set(chapter_ids)) != len(chapter_ids):
        raise ValidationError(ErrorCode.DUPLICATE_CHAPTER_ID)
    positions = _unique_positions(chapters)
    try:
        return tuple(chapters[positions[chapter_id]] for chapter_id in chapter_ids)
    except KeyError as error:
        raise ValidationError(ErrorCode.CHAPTER_NOT_FOUND) from error


def select_range(
    chapters: Sequence[_ChapterT], start_id: str, end_id: str
) -> tuple[_ChapterT, ...]:
    """Select an inclusive ID range in semantic spine order."""
    positions = _unique_positions(chapters)
    try:
        start = positions[start_id]
        end = positions[end_id]
    except KeyError as error:
        raise ValidationError(ErrorCode.CHAPTER_NOT_FOUND) from error
    if start > end:
        raise ValidationError(ErrorCode.REVERSED_CHAPTER_RANGE)
    return tuple(chapters[start : end + 1])
