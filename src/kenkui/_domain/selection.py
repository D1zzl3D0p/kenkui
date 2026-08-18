"""Materialized stable chapter selection semantics."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, TypeVar

from kenkui.errors import ErrorCode, ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Protocol

    class _Chapter(Protocol):
        @property
        def id(self) -> str:
            """Stable chapter identifier."""
            ...

else:
    _Chapter = object

_ChapterT = TypeVar("_ChapterT", bound=_Chapter)


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
