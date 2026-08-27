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
