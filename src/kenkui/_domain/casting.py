"""Pure character-to-voice casting. No I/O, no model, no clock, no randomness.

Casting is list colouring on a chapter co-occurrence graph: characters are
vertices, an edge joins two characters who speak in the same chapter, voices
are colours, and a method supplies each character's admissible colours.

Methods answer only eligibility. Colouring, weighting, and tie-breaking belong
to the one shared solver, so adding an LLM-driven or description-and-keyword
method later means writing one filter rather than a second solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

from kenkui.errors import ErrorCode, ValidationError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from kenkui.voices.types import Voice

CastingMethod = Literal["random", "gendered"]
_METHODS: frozenset[str] = frozenset({"random", "gendered"})


@dataclass(frozen=True, slots=True)
class CharacterProfile:
    """One speaking character, measured in normalized speech characters.

    spoken_characters is the prominence weight throughout, matching the unit
    chunking, progress, and billing already use.
    """

    id: str
    display_name: str
    gender: str | None
    spoken_characters: int
    chapter_ids: tuple[str, ...]
    # Every surface form this character was seen under, sorted. The display
    # name alone cannot find them again: a later volume says "Kaladin" where
    # this one recorded "Kaladin Stormblessed", and merge_rosters keeps only
    # the head's name. Defaulted so every existing construction still works.
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Collision:
    """Two characters sharing one voice inside one chapter."""

    chapter_id: str
    first: str
    second: str
    voice_id: str


@dataclass(frozen=True, slots=True)
class CastingRequest:
    """Everything the solver needs, and nothing that requires I/O to obtain."""

    characters: tuple[CharacterProfile, ...]
    pool: tuple[Voice, ...]
    explicit: Mapping[str, str]
    narrator_voice_id: str
    unknown_voice_id: str
    method: CastingMethod
    # Voice usage carried in from outside this book, in spoken characters.
    # A series continues its spread across volumes rather than restarting
    # it; empty is exactly today's behaviour.
    prior_load: Mapping[str, int] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class CastingOutcome:
    """Resolved assignments, plus any same-chapter clash the pool forced."""

    assignments: Mapping[str, str]
    collisions: tuple[Collision, ...]


def validate_method(method: str) -> CastingMethod:
    """Return the method, or raise if it names no known strategy.

    Callable at intent time. Checking only inside ``candidates`` lets a typo
    survive any pipeline that never reaches the solver -- a single-voice run,
    or one where every character is explicitly cast -- and silently render.
    """
    if method not in _METHODS:
        raise ValidationError(ErrorCode.CASTING_METHOD_UNKNOWN)
    return cast("CastingMethod", method)


def candidates(
    method: str, character: CharacterProfile, pool: tuple[Voice, ...]
) -> tuple[Voice, ...]:
    """Return the voices a method admits for one character.

    A character whose gender was never inferred falls back to the whole pool,
    because refusing to cast them would silently drop their speech. The reverse
    does not hold: a voice whose trait is unsourced never joins a gendered
    pool, since a missing trait is an admission of ignorance rather than a
    wildcard.
    """
    validate_method(method)
    if method == "random" or character.gender is None:
        return pool
    matched = tuple(
        voice for voice in pool if voice.perceived_gender == character.gender
    )
    return matched or pool


def ungendered_pool_characters(
    method: str,
    characters: Sequence[CharacterProfile],
    pool: Sequence[Voice],
) -> tuple[str, ...]:
    """Return the ids a gendered cast cannot honour, in the order given.

    `candidates` falls back to the whole pool rather than dropping the
    speech, which is the right thing to do at render time and the wrong
    thing to do quietly: it turns a gendered cast into a random one with no
    signal at all. Reporting is kept separate from casting so the caller can
    warn without changing what gets rendered.

    A character whose gender was never inferred is not reported. The whole
    pool is the designed answer for those, not a degradation.
    """
    if method != "gendered":
        return ()
    return tuple(
        character.id
        for character in characters
        if character.gender is not None
        and not any(voice.perceived_gender == character.gender for voice in pool)
    )


def solve(request: CastingRequest) -> CastingOutcome:
    """Assign voices by deterministic greedy list colouring over co-occurrence.

    Ordering and tie-breaks are total and derived from content alone, so the
    result cannot depend on the order the caller supplied. The plan fingerprint
    depends on that: equal intent must yield an equal plan.
    """
    pool = _castable(request)
    by_id = {character.id: character for character in request.characters}
    for character_id in request.explicit:
        if character_id not in by_id:
            raise ValidationError(ErrorCode.CHARACTER_UNKNOWN)

    neighbours = _neighbours(request.characters)
    assignments: dict[str, str] = dict(request.explicit)
    load: dict[str, int] = {
        voice.id: request.prior_load.get(voice.id, 0) for voice in pool
    }
    for character_id, voice_id in assignments.items():
        load[voice_id] = load.get(voice_id, 0) + by_id[character_id].spoken_characters

    collisions: list[Collision] = _pinned_collisions(
        request.explicit, request.characters, neighbours
    )
    remaining = [c for c in request.characters if c.id not in assignments]
    while remaining:
        character = _most_constrained(remaining, neighbours, assignments)
        remaining.remove(character)
        admissible = candidates(request.method, character, pool)
        taken = {
            assignments[other]
            for other in neighbours[character.id]
            if other in assignments
        }
        free = tuple(voice for voice in admissible if voice.id not in taken)
        chosen = min(
            free or admissible,
            key=lambda voice: (load.get(voice.id, 0), voice.id),
        )
        if not free:
            collisions.extend(
                _collisions(character, chosen.id, neighbours, assignments, by_id)
            )
        assignments[character.id] = chosen.id
        load[chosen.id] = load.get(chosen.id, 0) + character.spoken_characters
    return CastingOutcome(assignments, tuple(collisions))


def _castable(request: CastingRequest) -> tuple[Voice, ...]:
    """Drop the reserved roles from the pool.

    The narrator speaks in every chapter, so as a vertex it is adjacent to
    every character. Excluding its voice is pre-colouring a universally
    adjacent vertex rather than a special case, which is why the exclusion
    applies to every method and not only the gendered one.
    """
    reserved = {request.narrator_voice_id, request.unknown_voice_id}
    pool = tuple(voice for voice in request.pool if voice.id not in reserved)
    if not pool:
        raise ValidationError(ErrorCode.CAST_POOL_EMPTY)
    return pool


def _neighbours(
    characters: tuple[CharacterProfile, ...],
) -> dict[str, frozenset[str]]:
    """Build the co-occurrence graph: one edge per shared chapter."""
    edges: dict[str, set[str]] = {character.id: set() for character in characters}
    for index, first in enumerate(characters):
        for second in characters[index + 1 :]:
            if set(first.chapter_ids) & set(second.chapter_ids):
                edges[first.id].add(second.id)
                edges[second.id].add(first.id)
    return {key: frozenset(value) for key, value in edges.items()}


def _most_constrained(
    remaining: list[CharacterProfile],
    neighbours: dict[str, frozenset[str]],
    assignments: Mapping[str, str],
) -> CharacterProfile:
    """Pick the most saturated character, breaking ties on volume then ID."""
    return min(
        remaining,
        key=lambda character: (
            -len(
                {
                    assignments[other]
                    for other in neighbours[character.id]
                    if other in assignments
                }
            ),
            -character.spoken_characters,
            character.id,
        ),
    )


def _pinned_collisions(
    explicit: Mapping[str, str],
    characters: Sequence[CharacterProfile],
    neighbours: Mapping[str, frozenset[str]],
) -> list[Collision]:
    """Record same-chapter clashes between two characters that were pinned.

    The greedy loop only ever inspects the character it is about to assign,
    and every ``explicit`` entry is already in ``assignments`` before that
    loop starts -- so a clash between two pins was invisible, while the same
    clash between two solved characters was reported. A series manufactures
    exactly this without any caller ``cast=``: volume one pins A to a voice,
    volume two pins B to it while A is absent, and volume three has both in
    one chapter.

    Reported against whichever character the roster lists first, since
    neither pin is the "newly assigned" one the greedy loop's ordering
    assumes. One clash therefore yields one Collision rather than a mirrored
    pair.
    """
    found: list[Collision] = []
    for index, character in enumerate(characters):
        voice_id = explicit.get(character.id)
        if voice_id is None:
            continue
        for other in characters[index + 1 :]:
            if (
                explicit.get(other.id) != voice_id
                or other.id not in neighbours[character.id]
            ):
                continue
            found.extend(
                Collision(chapter_id, character.id, other.id, voice_id)
                for chapter_id in sorted(
                    set(character.chapter_ids) & set(other.chapter_ids)
                )
            )
    return found


def _collisions(
    character: CharacterProfile,
    voice_id: str,
    neighbours: dict[str, frozenset[str]],
    assignments: Mapping[str, str],
    by_id: Mapping[str, CharacterProfile],
) -> list[Collision]:
    """Record every same-chapter clash a forced assignment introduces."""
    found: list[Collision] = []
    for other in sorted(neighbours[character.id]):
        if assignments.get(other) != voice_id:
            continue
        shared = sorted(set(character.chapter_ids) & set(by_id[other].chapter_ids))
        found.extend(
            Collision(chapter_id, character.id, other, voice_id)
            for chapter_id in shared
        )
    return found
