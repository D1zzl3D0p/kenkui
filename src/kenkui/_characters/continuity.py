"""Pure preparation of series constraints for the shared casting solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from kenkui._characters.series import match_roster

if TYPE_CHECKING:
    from collections.abc import Sequence

    from kenkui._characters.models import SeriesRecord
    from kenkui._domain.casting import CharacterProfile
    from kenkui._domain.operations import AssignVoices
    from kenkui.voices.types import Voice


@dataclass(frozen=True, slots=True)
class SeriesPins:
    """Immutable constraints and diagnostics contributed by an earlier cast."""

    explicit: tuple[tuple[str, str], ...]
    prior_load: tuple[tuple[str, int], ...]
    dropped_pins: tuple[str, ...]
    dropped_voice_ids: tuple[tuple[str, str], ...]
    overridden_pins: tuple[str, ...]


def eligible_series_voice_ids(
    voices: Sequence[Voice], narrator_voice_id: str, unknown_voice_id: str
) -> frozenset[str]:
    """Return loaded, language-compatible voices excluding reserved narrators.

    Validation uses the same pool rules as resolution. If the narrator is not
    known yet, its language cannot be checked; voice resolution reports that
    resource error separately.
    """
    reserved = {narrator_voice_id, unknown_voice_id}
    narrator = next((voice for voice in voices if voice.id == narrator_voice_id), None)
    return frozenset(
        voice.id
        for voice in voices
        if voice.state == "loaded"
        and (narrator is None or voice.language == narrator.language)
        and voice.id not in reserved
    )


def prepare_series_cast(
    stored: SeriesRecord | None,
    characters: tuple[CharacterProfile, ...],
    casting: AssignVoices,
    pool: tuple[Voice, ...],
    book_digest: str,
) -> SeriesPins:
    """Derive constraints without reading or updating the series store.

    Explicit caller assignments take precedence. Existing series pins survive
    when their voice is available and does not contradict an inferred gender.
    Prior voice usage excludes this volume so repeated resolution is stable.
    The caller decides when to persist the resulting cast and report changes.
    """
    explicit = dict(casting.cast)
    prior_load: dict[str, int] = {}
    dropped_pins: list[str] = []
    dropped_voice_ids: dict[str, str] = {}
    overridden_pins: list[str] = []
    if stored is not None:
        pool_ids = frozenset(voice.id for voice in pool) - {
            casting.narrator_voice_id,
            casting.unknown_voice_id,
        }
        by_canonical = {c.canonical_id: c for c in stored.characters}
        character_genders = {c.id: c.gender for c in characters}
        voice_genders = {voice.id: voice.perceived_gender for voice in pool}
        for book_id, canonical in match_roster(stored, characters).items():
            known = by_canonical[canonical]
            if book_id in explicit:
                if explicit[book_id] != known.voice_id:
                    overridden_pins.append(book_id)
                continue
            # Continuity is not gender evidence: a later confident inference
            # can correct a voice chosen when the character was still unknown.
            contradicts = (
                casting.method == "gendered"
                and character_genders.get(book_id) is not None
                and voice_genders.get(known.voice_id) is not None
                and character_genders[book_id] != voice_genders[known.voice_id]
            )
            if known.voice_id in pool_ids and not contradicts:
                explicit[book_id] = known.voice_id
            else:
                dropped_pins.append(book_id)
                dropped_voice_ids[book_id] = known.voice_id
        for known in stored.characters:
            already = dict(known.contributions).get(book_digest, 0)
            prior_load[known.voice_id] = (
                prior_load.get(known.voice_id, 0) + known.spoken_characters - already
            )
    return SeriesPins(
        explicit=tuple(sorted(explicit.items())),
        prior_load=tuple(sorted(prior_load.items())),
        dropped_pins=tuple(dropped_pins),
        dropped_voice_ids=tuple(sorted(dropped_voice_ids.items())),
        overridden_pins=tuple(overridden_pins),
    )
