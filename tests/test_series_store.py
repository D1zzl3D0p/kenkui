"""A series outlives any one book, and any one model that read it."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
