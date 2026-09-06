"""Immutable character, casting, and series values independent of persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from kenkui._domain.casting import CharacterProfile
    from kenkui._domain.planning import SpeakerSpan


@dataclass(frozen=True, slots=True)
class CharacterRoster:
    """Discovered or reviewed characters before quote attribution.

    Chapter IDs describe where a name was found, when known. Speech counts
    are measured later, after attribution. ``narrator_id`` identifies a
    first-person narrating character, independently of the narrator voice.
    """

    characters: tuple[CharacterProfile, ...]
    narrator_id: str | None = None


@dataclass(frozen=True, slots=True)
class AttributionRecord:
    """One book's roster and speaker spans, as derived by one model."""

    attribution_id: str
    book_id: str
    model_id: str
    prompt_version: str
    params: Mapping[str, Any]
    characters: tuple[CharacterProfile, ...]
    spans: tuple[SpeakerSpan, ...]


@dataclass(frozen=True, slots=True)
class CastRecord:
    """One resolved cast. ``assignments`` is (character, voice, pinned)."""

    cast_id: str
    attribution_id: str
    method: str
    narrator_voice_id: str
    unknown_voice_id: str
    assignments: tuple[tuple[str, str, bool], ...]


@dataclass(frozen=True, slots=True)
class SeriesCharacter:
    """One person across a series, and the voice they keep."""

    canonical_id: str
    display_name: str
    gender: str | None
    voice_id: str
    spoken_characters: int
    aliases: tuple[str, ...]
    # Per-volume ledger behind `spoken_characters`, keyed by book digest.
    # Empty for a character no merge has ever named a volume for; see
    # `_characters.series.merged_series`. Continuity preparation subtracts
    # the current volume's contribution before balancing a new cast.
    contributions: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True, slots=True)
class SeriesRecord:
    """A series' cast, ordered by accumulated speech.

    Keyed on the series and the character, never on an attribution: a volume
    re-read by a different model must not re-cast the series.
    """

    series_id: str
    narrator_voice_id: str
    characters: tuple[SeriesCharacter, ...]
