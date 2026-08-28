"""One person across volumes, decided from the names alone."""

from __future__ import annotations

from kenkui._characters import store
from kenkui._characters.series import match_roster, merged_series
from kenkui._domain.casting import CharacterProfile


def _known(
    canonical_id: str, display: str, aliases: tuple[str, ...]
) -> store.SeriesCharacter:
    return store.SeriesCharacter(
        canonical_id=canonical_id,
        display_name=display,
        gender="masculine",
        voice_id="alf",
        spoken_characters=500,
        aliases=aliases,
    )


def _profile(character_id: str, display: str, *aliases: str) -> CharacterProfile:
    return CharacterProfile(
        id=character_id,
        display_name=display,
        gender=None,
        spoken_characters=100,
        chapter_ids=("ch1",),
        aliases=tuple(sorted({display, *aliases})),
    )


def test_an_exact_alias_matches() -> None:
    """Volume 3 says "Kaladin"; volume 1 recorded it as an alias."""
    record = store.SeriesRecord(
        "stormlight", "eponine",
        (_known("kaladin-stormblessed", "Kaladin Stormblessed",
                ("Kaladin", "Kaladin Stormblessed")),),
    )
    matched = match_roster(record, (_profile("kaladin", "Kaladin"),))
    assert matched == {"kaladin": "kaladin-stormblessed"}


def test_a_nested_name_matches_without_an_exact_alias() -> None:
    """Identity resolution, not just string equality.

    The series recorded only the full name; this volume uses the short one
    and it was never stored as an alias, so an exact hit cannot find it.
    """
    record = store.SeriesRecord(
        "stormlight", "eponine",
        (_known("dalinar-kholin", "Dalinar Kholin", ("Dalinar Kholin",)),),
    )
    matched = match_roster(record, (_profile("dalinar", "Dalinar"),))
    assert matched == {"dalinar": "dalinar-kholin"}


def test_two_people_sharing_a_surname_do_not_match() -> None:
    """Over-merging gives two people one voice, which is the worse failure."""
    record = store.SeriesRecord(
        "s", "eponine",
        (_known("charles-hayter", "Charles Hayter", ("Charles Hayter",)),),
    )
    matched = match_roster(record, (_profile("charles-musgrove", "Charles Musgrove"),))
    assert matched == {}


def test_an_ambiguous_short_form_matches_nobody() -> None:
    """Two hosts could claim it, so it names neither."""
    record = store.SeriesRecord(
        "s", "eponine",
        (
            _known("charles-hayter", "Charles Hayter", ("Charles Hayter",)),
            _known("charles-musgrove", "Charles Musgrove", ("Charles Musgrove",)),
        ),
    )
    assert match_roster(record, (_profile("charles", "Charles"),)) == {}


def test_no_series_yet_matches_nothing() -> None:
    """The first volume has nothing to match against."""
    assert match_roster(None, (_profile("kaladin", "Kaladin"),)) == {}


def test_merging_accumulates_speech_and_aliases() -> None:
    """A returning character's totals grow; their voice does not change."""
    record = store.SeriesRecord(
        "s", "eponine", (_known("kaladin-stormblessed", "Kaladin Stormblessed",
                                ("Kaladin Stormblessed",)),),
    )
    updated = merged_series(
        record,
        (_profile("kaladin", "Kaladin"),),
        {"kaladin": "alf"},
        "eponine",
        "s",
    )
    kaladin = next(
        c for c in updated.characters if c.canonical_id == "kaladin-stormblessed"
    )
    assert kaladin.voice_id == "alf"
    assert kaladin.spoken_characters == 600  # noqa: PLR2004 - 500 known + 100 this volume
    assert "Kaladin" in kaladin.aliases


def test_merging_adds_a_newcomer() -> None:
    """A character the series has not met joins it with the voice just solved."""
    updated = merged_series(
        None, (_profile("shallan", "Shallan Davar"),), {"shallan": "aoife"},
        "eponine", "s",
    )
    assert updated.series_id == "s"
    assert updated.narrator_voice_id == "eponine"
    assert [c.canonical_id for c in updated.characters] == ["shallan"]
    assert updated.characters[0].voice_id == "aoife"


def test_two_local_characters_claiming_one_series_person_match_neither() -> None:
    """A maid named "Elizabeth" and the protagonist "Lizzy" are not one person.

    Both surface forms the series accumulated for Elizabeth Bennet also reach
    two different characters in this volume. Attaching either would hand one
    of them her voice and discard the one the in-volume solver already gave
    them, so a canonical id claimed by more than one local character is
    refused for all of its claimants -- the same one-person-two-voices bias
    `identity` states, applied across the series boundary too.
    """
    record = store.SeriesRecord(
        "s", "eponine",
        (_known("elizabeth-bennet", "Elizabeth Bennet", ("Elizabeth", "Lizzy")),),
    )
    characters = (_profile("elizabeth-maid", "Elizabeth"), _profile("lizzy", "Lizzy"))
    assert match_roster(record, characters) == {}

    updated = merged_series(
        record, characters, {"elizabeth-maid": "v1", "lizzy": "v2"}, "eponine", "s",
    )
    assert len(updated.characters) == 3  # noqa: PLR2004 - two locals plus the untouched original
    by_id = {c.canonical_id: c.voice_id for c in updated.characters}
    assert by_id["elizabeth-maid"] == "v1"
    assert by_id["lizzy"] == "v2"
    assert by_id["elizabeth-bennet"] == "alf"


def test_unmatched_newcomer_does_not_inherit_an_unrelated_slug_collision() -> None:
    """Two different volumes' characters can slug to the same raw id.

    Volume 1's "Town Guard" and a later volume's "Castle Guard" are different
    people who never matched by name. merged_series must not fall back to the
    raw character id when that id is already claimed by someone else in the
    series -- doing so would hand the second guard the first guard's voice
    and running total with no name comparison ever happening.
    """
    record = merged_series(
        None, (_profile("guard", "Town Guard"),), {"guard": "v1"}, "n", "s",
    )
    updated = merged_series(
        record, (_profile("guard", "Castle Guard"),), {"guard": "v2"}, "n", "s",
    )
    assert len(updated.characters) == 2  # noqa: PLR2004 - two distinct guards
    canonical_ids = {c.canonical_id for c in updated.characters}
    assert len(canonical_ids) == 2  # noqa: PLR2004 - two distinct guards
    by_display = {c.display_name: c for c in updated.characters}
    assert by_display["Town Guard"].voice_id == "v1"
    assert by_display["Town Guard"].spoken_characters == 100  # noqa: PLR2004
    assert by_display["Castle Guard"].voice_id == "v2"
    assert by_display["Castle Guard"].spoken_characters == 100  # noqa: PLR2004
