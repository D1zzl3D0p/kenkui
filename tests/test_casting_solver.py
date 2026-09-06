"""Casting is list colouring on a chapter co-occurrence graph.

Characters are vertices, an edge joins two characters who speak in the same
chapter, voices are colours, and a method supplies each character's admissible
colours. The solver is pure: no I/O, no model, no clock, no randomness.
"""

from __future__ import annotations

import pytest

from kenkui._domain.casting import (
    CastingRequest,
    CharacterProfile,
    Collision,
    candidates,
    solve,
    ungendered_pool_characters,
)
from kenkui.errors import ErrorCode, ValidationError
from kenkui.voices.types import PerceivedGender, Voice


def _voice(voice_id: str, gender: PerceivedGender) -> Voice:
    return Voice(
        id=voice_id,
        name=voice_id.title(),
        enabled=True,
        provenance="test",
        license_id="CC-BY-4.0",
        commercial_use_allowed=False,
        language="english",
        state="loaded",
        perceived_gender=gender,
    )


def _character(
    character_id: str,
    gender: str | None,
    spoken: int,
    chapters: tuple[str, ...],
) -> CharacterProfile:
    return CharacterProfile(
        character_id, character_id.title(), gender, spoken, chapters
    )


# Four masculine voices minus one reserved narrator leaves two free, which is
# what the exhaustion cases below rely on.
_FREE_AFTER_RESERVING_MARIUS = 2

POOL = (
    _voice("anna", "feminine"),
    _voice("vera", "feminine"),
    _voice("charles", "masculine"),
    _voice("paul", "masculine"),
    _voice("eponine", "feminine"),
    _voice("marius", "masculine"),
)


def _request(
    characters: tuple[CharacterProfile, ...], **overrides: object
) -> CastingRequest:
    values: dict[str, object] = {
        "characters": characters,
        "pool": POOL,
        "explicit": {},
        "narrator_voice_id": "eponine",
        "unknown_voice_id": "eponine",
        "method": "gendered",
    }
    values.update(overrides)
    return CastingRequest(**values)  # type: ignore[arg-type]


def test_gendered_candidates_filter_by_trait() -> None:
    """Only voices matching the character's gender are admissible."""
    character = _character("darcy", "masculine", 100, ("ch1",))
    assert {v.id for v in candidates("gendered", character, POOL)} == {
        "charles",
        "paul",
        "marius",
    }


def test_unknown_gender_character_falls_back_to_the_whole_pool() -> None:
    """A character whose gender was never inferred must still be castable."""
    character = _character("voice-in-the-dark", None, 10, ("ch1",))
    assert len(candidates("gendered", character, POOL)) == len(POOL)


def test_untraited_voices_do_not_join_a_gendered_pool() -> None:
    """An unsourced trait must not be silently treated as a match."""
    pool = (*POOL, _voice("mystery", None))
    character = _character("darcy", "masculine", 100, ("ch1",))
    assert "mystery" not in {v.id for v in candidates("gendered", character, pool)}


def test_random_method_admits_the_whole_pool() -> None:
    """The unconstrained method filters nothing."""
    character = _character("darcy", "masculine", 100, ("ch1",))
    assert len(candidates("random", character, POOL)) == len(POOL)


def test_unknown_method_is_rejected() -> None:
    """An unrecognised method fails rather than silently defaulting."""
    character = _character("darcy", "masculine", 100, ("ch1",))
    with pytest.raises(ValidationError) as caught:
        candidates("astrology", character, POOL)
    assert caught.value.code is ErrorCode.CASTING_METHOD_UNKNOWN


def test_same_chapter_characters_never_share_a_voice() -> None:
    """The rule the whole feature exists to enforce."""
    outcome = solve(
        _request(
            (
                _character("darcy", "masculine", 400, ("ch1",)),
                _character("bingley", "masculine", 300, ("ch1",)),
            )
        )
    )
    assert outcome.assignments["darcy"] != outcome.assignments["bingley"]
    assert outcome.collisions == ()


def test_voices_spread_before_they_repeat() -> None:
    """Distinct voices are preferred even where sharing would be legal.

    Least-used weighting spreads first and reuses only under pressure, which
    is what keeps a large cast sounding varied.
    """
    outcome = solve(
        _request(
            (
                _character("darcy", "masculine", 400, ("ch1",)),
                _character("wickham", "masculine", 300, ("ch2",)),
            )
        )
    )
    assert outcome.assignments["darcy"] != outcome.assignments["wickham"]
    assert outcome.collisions == ()


def test_non_co_occurring_characters_share_once_the_pool_runs_out() -> None:
    """Sharing is legal across chapters, and is not a collision."""
    characters = tuple(
        _character(f"m{index}", "masculine", 100, (f"ch{index}",)) for index in range(4)
    )
    outcome = solve(_request(characters, narrator_voice_id="marius"))
    assert len(set(outcome.assignments.values())) == _FREE_AFTER_RESERVING_MARIUS
    assert outcome.collisions == ()


def test_least_used_prefers_the_quietest_voice_by_speech_volume() -> None:
    """A lead must not land on the voice a talkative character already holds."""
    outcome = solve(
        _request(
            (
                _character("darcy", "masculine", 5000, ("ch1",)),
                _character("collins", "masculine", 50, ("ch2",)),
                _character("wickham", "masculine", 4000, ("ch3",)),
            )
        )
    )
    assert outcome.assignments["wickham"] != outcome.assignments["darcy"]


def test_explicit_pins_are_constraints_not_suggestions() -> None:
    """A pinned voice is honoured and removed from its neighbours' options."""
    outcome = solve(
        _request(
            (
                _character("darcy", "masculine", 400, ("ch1",)),
                _character("bingley", "masculine", 300, ("ch1",)),
            ),
            explicit={"darcy": "paul"},
        )
    )
    assert outcome.assignments["darcy"] == "paul"
    assert outcome.assignments["bingley"] != "paul"


def test_pinning_an_unknown_character_is_rejected() -> None:
    """A typo in a cast dict must fail loudly, not cast nobody."""
    with pytest.raises(ValidationError) as caught:
        solve(
            _request(
                (_character("darcy", "masculine", 400, ("ch1",)),),
                explicit={"ghost": "paul"},
            )
        )
    assert caught.value.code is ErrorCode.CHARACTER_UNKNOWN


def test_reserved_roles_are_excluded_from_every_method() -> None:
    """The narrator speaks in every chapter, so it is adjacent to everyone."""
    outcome = solve(
        _request(
            (_character("darcy", "masculine", 400, ("ch1",)),),
            narrator_voice_id="charles",
            unknown_voice_id="paul",
            method="random",
        )
    )
    assert outcome.assignments["darcy"] not in {"charles", "paul"}


def test_empty_pool_after_reservation_is_rejected() -> None:
    """Reserving the only voice leaves nothing to cast with."""
    outcome_pool = (_voice("solo", "feminine"),)
    with pytest.raises(ValidationError) as caught:
        solve(
            _request(
                (_character("darcy", None, 400, ("ch1",)),),
                pool=outcome_pool,
                narrator_voice_id="solo",
                unknown_voice_id="solo",
            )
        )
    assert caught.value.code is ErrorCode.CAST_POOL_EMPTY


def test_exhausted_pool_collides_minimally_and_reports_it() -> None:
    """Three masculine speakers in one chapter against two free voices."""
    characters = tuple(
        _character(f"man{index}", "masculine", 500 - index * 100, ("ch1",))
        for index in range(4)
    )
    outcome = solve(_request(characters, narrator_voice_id="marius"))
    assert len(set(outcome.assignments.values())) == _FREE_AFTER_RESERVING_MARIUS
    assert outcome.collisions
    quietest = {"man2", "man3"}
    assert any(c.first in quietest or c.second in quietest for c in outcome.collisions)


def test_solving_is_deterministic_regardless_of_input_order() -> None:
    """The plan fingerprint depends on this: same content, same cast."""
    characters = tuple(
        _character(f"c{index}", "feminine", 500 - index * 10, ("ch1", "ch2"))
        for index in range(3)
    )
    first = solve(_request(characters))
    second = solve(_request(tuple(reversed(characters))))
    assert first.assignments == second.assignments


def test_every_character_is_assigned() -> None:
    """Speech must never silently vanish because casting skipped someone."""
    characters = tuple(
        _character(f"c{index}", None, 100, ("ch1",)) for index in range(3)
    )
    outcome = solve(_request(characters))
    assert set(outcome.assignments) == {c.id for c in characters}


def test_ungendered_pool_names_the_characters_it_cannot_serve() -> None:
    """A gendered cast with no matching voice must say so, not degrade quietly."""
    pool = (_voice("charles", "masculine"),)
    characters = (
        _character("her", "feminine", 100, ("ch1",)),
        _character("him", "masculine", 100, ("ch1",)),
    )
    assert ungendered_pool_characters("gendered", characters, pool) == ("her",)


def test_a_character_with_no_gender_is_not_reported() -> None:
    """Falling back to the whole pool is the designed behaviour for these."""
    pool = (_voice("charles", "masculine"),)
    characters = (_character("who", None, 100, ("ch1",)),)
    assert ungendered_pool_characters("gendered", characters, pool) == ()


def test_random_method_reports_nothing() -> None:
    """The random method never promised a gendered pool."""
    pool = (_voice("charles", "masculine"),)
    characters = (_character("her", "feminine", 100, ("ch1",)),)
    assert ungendered_pool_characters("random", characters, pool) == ()


def test_a_served_pool_reports_nothing() -> None:
    """A pool with a matching voice for every gender reports nothing."""
    pool = (_voice("charles", "masculine"), _voice("anna", "feminine"))
    characters = (_character("her", "feminine", 100, ("ch1",)),)
    assert ungendered_pool_characters("gendered", characters, pool) == ()


def test_prior_load_steers_the_next_volume() -> None:
    """A voice the series has already spent is not the least-used one.

    Without this the second volume restarts the count and hands its first
    character the same voice the first volume did.
    """
    characters = (_character("newcomer", "feminine", 100, ("ch1",)),)
    fresh = solve(_request(characters))
    spent = solve(
        _request(characters, prior_load={fresh.assignments["newcomer"]: 10_000})
    )
    assert spent.assignments["newcomer"] != fresh.assignments["newcomer"]


def test_no_prior_load_is_todays_behaviour() -> None:
    """A pipeline that never mentions a series must cast exactly as before."""
    characters = (_character("solo", "feminine", 100, ("ch1",)),)
    assert (
        solve(_request(characters)).assignments
        == solve(_request(characters, prior_load={})).assignments
    )


def test_two_pinned_characters_sharing_a_chapter_collide() -> None:
    """A clash between two pins is the same clash the solver reports itself.

    `explicit` entries were seeded straight into `assignments` and only the
    greedy loop ever looked for collisions, so two pins landing on one voice
    in one chapter reported nothing at all. A series manufactures exactly
    that without any caller `cast=`: volume one pins A to a voice, volume
    two pins B to the same voice while A is absent, and volume three has
    both of them in one chapter.
    """
    outcome = solve(
        _request(
            (
                _character("darcy", "masculine", 400, ("ch1",)),
                _character("bingley", "masculine", 300, ("ch1",)),
            ),
            explicit={"darcy": "charles", "bingley": "charles"},
        )
    )
    assert outcome.collisions == (Collision("ch1", "darcy", "bingley", "charles"),)


def test_pins_on_distinct_voices_report_no_collision() -> None:
    """The ordinary pinned cast must stay silent."""
    outcome = solve(
        _request(
            (
                _character("darcy", "masculine", 400, ("ch1",)),
                _character("bingley", "masculine", 300, ("ch1",)),
            ),
            explicit={"darcy": "charles", "bingley": "paul"},
        )
    )
    assert outcome.collisions == ()
