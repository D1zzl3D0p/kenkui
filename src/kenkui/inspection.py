"""Immutable source inspection values."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BookMetadata:
    """Bibliographic metadata materialized from a readable source."""

    title: str | None = None
    author: str | None = None
    cover_available: bool = False


@dataclass(frozen=True, slots=True)
class ChapterInspection:
    """Stable chapter information available without rendering."""

    id: str
    index: int
    title: str
    speech_characters: int | None
    text: str = ""


@dataclass(frozen=True, slots=True)
class BookInspection:
    """Immutable source inspection result."""

    metadata: BookMetadata
    chapters: tuple[ChapterInspection, ...]
