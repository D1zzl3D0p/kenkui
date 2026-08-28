"""A series outlives any one book, and any one model that read it."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest

import kenkui as kk
from kenkui._characters import store

if TYPE_CHECKING:
    from pathlib import Path


def _character(
    canonical_id: str, voice_id: str, spoken: int = 100
) -> store.SeriesCharacter:
    return store.SeriesCharacter(
        canonical_id=canonical_id,
        display_name=canonical_id.title(),
        gender="masculine",
        voice_id=voice_id,
        spoken_characters=spoken,
        aliases=(canonical_id.title(),),
    )


def test_a_series_round_trips(tmp_path: Path) -> None:
    """Everything written is everything read back."""
    path = tmp_path / "s.sqlite3"
    record = store.SeriesRecord(
        series_id="stormlight",
        narrator_voice_id="eponine",
        characters=(_character("kaladin", "alf"), _character("shallan", "aoife")),
    )
    store.write_series(record, path)
    assert store.read_series("stormlight", path) == record


def test_characters_are_ordered_by_speech(tmp_path: Path) -> None:
    """The listing reads as the cast in prominence order."""
    path = tmp_path / "s.sqlite3"
    store.write_series(
        store.SeriesRecord(
            "wot",
            "eponine",
            (_character("nynaeve", "aoife", 10), _character("rand", "alf", 900)),
        ),
        path,
    )
    read = store.read_series("wot", path)
    assert read is not None
    assert [c.canonical_id for c in read.characters] == ["rand", "nynaeve"]


def test_writing_again_replaces_the_series(tmp_path: Path) -> None:
    """A volume updates the series; it does not accumulate duplicates."""
    path = tmp_path / "s.sqlite3"
    store.write_series(
        store.SeriesRecord("s", "eponine", (_character("a", "alf"),)), path
    )
    store.write_series(
        store.SeriesRecord("s", "eponine", (_character("a", "alf"),
                                            _character("b", "aoife"))),
        path,
    )
    read = store.read_series("s", path)
    assert read is not None
    assert len(read.characters) == 2  # noqa: PLR2004 - two characters were written


def test_a_missing_series_reads_as_none(tmp_path: Path) -> None:
    """An absent series is a miss, not an error."""
    assert store.read_series("absent", tmp_path / "s.sqlite3") is None


def test_listing_and_removal(tmp_path: Path) -> None:
    """The pair that makes a series something a caller can name and discard."""
    path = tmp_path / "s.sqlite3"
    store.write_series(
        store.SeriesRecord("a", "eponine", (_character("x", "alf"),)), path
    )
    store.write_series(
        store.SeriesRecord("b", "eponine", (_character("y", "aoife"),)), path
    )
    assert [r.series_id for r in store.list_series(path)] == ["a", "b"]
    store.remove_series("a", path)
    assert [r.series_id for r in store.list_series(path)] == ["b"]


def test_an_unusable_store_lists_empty(tmp_path: Path) -> None:
    """Reads degrade rather than raise, matching read_attribution."""
    assert store.list_series(tmp_path / "never-created.sqlite3") == ()


def test_corrupt_database_reads_as_a_miss(tmp_path: Path) -> None:
    """Everything here is recoverable by recomputing, so a read fails soft."""
    path = tmp_path / "s.sqlite3"
    path.write_bytes(b"not a database")
    assert store.read_series("stormlight", path) is None
    assert store.list_series(path) == ()


def test_corrupt_database_fails_loud_on_write(tmp_path: Path) -> None:
    """Silently losing a series would silently re-cast it on the next volume."""
    path = tmp_path / "s.sqlite3"
    path.write_bytes(b"not a database")
    record = store.SeriesRecord(
        "stormlight", "eponine", (_character("kaladin", "alf"),)
    )
    with pytest.raises(OSError, match="could not write series"):
        store.write_series(record, path)


def test_corrupt_database_fails_loud_on_remove(tmp_path: Path) -> None:
    """The remove path added to match its siblings must fail loud too."""
    path = tmp_path / "s.sqlite3"
    path.write_bytes(b"not a database")
    with pytest.raises(OSError, match="could not remove series"):
        store.remove_series("stormlight", path)


def test_remove_series_cascades_to_its_characters_and_aliases(
    tmp_path: Path,
) -> None:
    """Dropping a series must not strand the rows that reference it.

    Only the parent row is visible through the store's own reads once it is
    gone, so this reaches past the module and queries the child tables
    directly - the one way to show ON DELETE CASCADE, and the foreign_keys
    pragma that arms it, actually fired.
    """
    path = tmp_path / "s.sqlite3"
    store.write_series(
        store.SeriesRecord("a", "eponine", (_character("x", "alf"),)), path
    )
    store.remove_series("a", path)
    with sqlite3.connect(path) as connection:
        characters = connection.execute(
            "SELECT * FROM series_characters WHERE series_id=?", ("a",)
        ).fetchall()
        aliases = connection.execute(
            "SELECT * FROM series_aliases WHERE series_id=?", ("a",)
        ).fetchall()
    assert characters == []
    assert aliases == []


def test_a_character_with_several_aliases_round_trips(tmp_path: Path) -> None:
    """Aliases group by canonical_id, not by insertion order, and read back sorted."""
    path = tmp_path / "s.sqlite3"
    kaladin = store.SeriesCharacter(
        canonical_id="kaladin",
        display_name="Kaladin",
        gender="masculine",
        voice_id="alf",
        spoken_characters=900,
        aliases=("Stormblessed", "Kal", "Radiant"),
    )
    shallan = store.SeriesCharacter(
        canonical_id="shallan",
        display_name="Shallan",
        gender="feminine",
        voice_id="aoife",
        spoken_characters=400,
        aliases=("Veil",),
    )
    store.write_series(
        store.SeriesRecord("stormlight", "eponine", (kaladin, shallan)), path
    )
    read = store.read_series("stormlight", path)
    assert read is not None
    by_id = {c.canonical_id: c for c in read.characters}
    assert by_id["kaladin"].aliases == ("Kal", "Radiant", "Stormblessed")
    assert by_id["shallan"].aliases == ("Veil",)


def test_a_character_with_book_contributions_round_trips(tmp_path: Path) -> None:
    """The per-volume ledger behind spoken_characters survives a round trip."""
    path = tmp_path / "s.sqlite3"
    kaladin = store.SeriesCharacter(
        canonical_id="kaladin",
        display_name="Kaladin",
        gender="masculine",
        voice_id="alf",
        spoken_characters=300,
        aliases=(),
        contributions=(("volume-1", 100), ("volume-2", 200)),
    )
    store.write_series(store.SeriesRecord("stormlight", "eponine", (kaladin,)), path)
    read = store.read_series("stormlight", path)
    assert read is not None
    assert read.characters[0].contributions == (("volume-1", 100), ("volume-2", 200))


def test_the_public_pair_is_exported() -> None:
    """Mirrors list_castings / remove_casting: see it, or start over."""
    assert callable(kk.list_series)
    assert callable(kk.remove_series)
    assert kk.SeriesRecord is store.SeriesRecord
    assert kk.SeriesCharacter is store.SeriesCharacter


def test_two_characters_sharing_an_alias_round_trip(tmp_path: Path) -> None:
    """An alias belongs to a surface form, not to one canonical id.

    Two people in a series can be known by the same bare name; the alias key
    made that impossible, so the later row silently stole "Elizabeth" from
    the earlier one and what came back was not what went in. Losing the
    alias also loses the ambiguity, which is the signal `match_roster`
    refuses on -- the surviving owner would then quietly host every later
    "Elizabeth" alone.
    """
    path = tmp_path / "s.sqlite3"
    bennet = store.SeriesCharacter(
        canonical_id="elizabeth-bennet",
        display_name="Elizabeth Bennet",
        gender="feminine",
        voice_id="alf",
        spoken_characters=900,
        aliases=("Elizabeth", "Elizabeth Bennet"),
    )
    gardiner = store.SeriesCharacter(
        canonical_id="elizabeth-gardiner",
        display_name="Elizabeth Gardiner",
        gender="feminine",
        voice_id="aoife",
        spoken_characters=100,
        aliases=("Elizabeth", "Elizabeth Gardiner"),
    )
    record = store.SeriesRecord("s", "eponine", (bennet, gardiner))
    store.write_series(record, path)
    assert store.read_series("s", path) == record
