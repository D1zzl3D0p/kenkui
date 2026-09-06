"""Immutable source and resolved-casting inspection values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._characters.models import CharacterRoster
    from ._domain.casting import CharacterProfile, Collision
    from ._domain.planning import SpeakerSpan


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
    # Normalized heading strings in document order, title first. Recorded
    # rather than character offsets: normalization collapses whitespace
    # across the whole chapter, so raw offsets do not survive it.
    headings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CastingInspection:
    """A resolved roster, cast, and attribution without rendering resources.

    Assignments are immutable ``(character_id, voice_id)`` pairs. Collisions
    describe expected voice sharing within chapters when voices are limited.
    Span offsets refer to the normalized chapter text in the same inspection.
    """

    narrator_voice_id: str
    unknown_voice_id: str
    characters: tuple[CharacterProfile, ...] = ()
    assignments: tuple[tuple[str, str], ...] = ()
    spans: tuple[SpeakerSpan, ...] = ()
    collisions: tuple[Collision, ...] = ()


@dataclass(frozen=True, slots=True)
class BookInspection:
    """Source information plus casting when this pipeline has been resolved."""

    metadata: BookMetadata
    chapters: tuple[ChapterInspection, ...]
    casting: CastingInspection | None = None
    roster: CharacterRoster | None = None
