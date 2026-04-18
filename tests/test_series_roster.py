"""Tests for series CharacterRoster persistence and merge logic."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from kenkui.nlp.models import CharacterRecord, CharacterRoster, TitleRecord
from kenkui import series as _series
from kenkui.services import series_service


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rec(slug: str, canonical: str, aliases=None, chapters=None, mentions=10, quotes=0,
         first=None, last=None) -> CharacterRecord:
    return CharacterRecord(
        slug=slug,
        canonical_name=canonical,
        aliases=aliases or [],
        chapters=chapters or [0],
        mention_count=mentions,
        quote_count=quotes,
        first_appearance=first or (slug + "_book1", 0),
        last_appearance=last or (slug + "_book1", 5),
    )


# ---------------------------------------------------------------------------
# load_series_roster / save_series_roster
# ---------------------------------------------------------------------------


def test_load_series_roster_returns_none_when_missing(tmp_path):
    with patch.object(_series, "_series_dir_override", tmp_path):
        result = _series.load_series_roster("nonexistent")
    assert result is None


def test_save_and_load_series_roster_roundtrip(tmp_path):
    roster = CharacterRoster(characters=[_rec("jane_eyre", "Jane Eyre", aliases=["Jane"])])

    with patch.object(_series, "_series_dir_override", tmp_path):
        _series.save_series_roster("jane-eyre-series", roster)
        loaded = _series.load_series_roster("jane-eyre-series")

    assert loaded is not None
    assert len(loaded.characters) == 1
    assert loaded.characters[0].slug == "jane_eyre"
    assert loaded.characters[0].aliases == ["Jane"]


# ---------------------------------------------------------------------------
# merge_into_series_roster
# ---------------------------------------------------------------------------


def test_merge_exact_slug_match_unions_aliases_and_sums_counts():
    existing_rec = _rec("jane_eyre", "Jane Eyre", aliases=["Jane"], mentions=50, quotes=20,
                        first=("book1", 0), last=("book1", 30))
    existing = CharacterRoster(characters=[existing_rec])

    new_rec = _rec("jane_eyre", "Jane Eyre", aliases=["Miss Eyre"], mentions=30, quotes=15,
                   chapters=[1, 2], last=("book2", 10))
    new_roster = CharacterRoster(characters=[new_rec])

    merged = _series.merge_into_series_roster(existing, new_roster, book_slug="book2")

    assert len(merged.characters) == 1
    jane = merged.characters[0]
    assert "Jane" in jane.aliases
    assert "Miss Eyre" in jane.aliases
    assert jane.mention_count == 80
    assert jane.quote_count == 35
    assert jane.last_appearance == ("book2", 10)


def test_merge_alias_intersection_merges_under_existing_slug():
    existing_rec = _rec("elizabeth_bennet", "Elizabeth Bennet", aliases=["Lizzy", "Eliza"])
    existing = CharacterRoster(characters=[existing_rec])

    # New book uses slug "miss_bennet" but has "Lizzy" as an alias — exact match on alias map
    new_rec = _rec("miss_bennet", "Miss Bennet", aliases=["Lizzy", "Elizabeth Bennet"], mentions=5)
    new_roster = CharacterRoster(characters=[new_rec])

    merged = _series.merge_into_series_roster(existing, new_roster, book_slug="book2")

    assert len(merged.characters) == 1
    result = merged.characters[0]
    assert result.slug == "elizabeth_bennet"
    assert "Miss Bennet" in result.aliases


def test_merge_new_character_appended_with_first_appearance():
    existing = CharacterRoster(characters=[_rec("jane_eyre", "Jane Eyre")])
    new_rec = CharacterRecord(
        slug="mr_rochester",
        canonical_name="Mr. Rochester",
        aliases=["Rochester"],
        chapters=[2, 3],
        mention_count=40,
        first_appearance=None,
        last_appearance=("book2", 20),
    )
    new_roster = CharacterRoster(characters=[new_rec])

    merged = _series.merge_into_series_roster(existing, new_roster, book_slug="book2")

    assert len(merged.characters) == 2
    rochester = next(r for r in merged.characters if r.slug == "mr_rochester")
    assert rochester.first_appearance == ("book2", 2)


def test_merge_preserves_existing_first_appearance():
    new_rec = CharacterRecord(
        slug="mr_rochester",
        canonical_name="Mr. Rochester",
        aliases=[],
        chapters=[5],
        mention_count=20,
        first_appearance=("book1", 3),   # already set
        last_appearance=("book2", 15),
    )
    existing = CharacterRoster(characters=[])
    new_roster = CharacterRoster(characters=[new_rec])

    merged = _series.merge_into_series_roster(existing, new_roster, book_slug="book2")

    rochester = merged.characters[0]
    assert rochester.first_appearance == ("book1", 3)


# ---------------------------------------------------------------------------
# series_service.get_roster / update_roster
# ---------------------------------------------------------------------------


def test_get_roster_returns_empty_for_new_series(tmp_path):
    with patch.object(_series, "_series_dir_override", tmp_path):
        roster = series_service.get_roster("brand-new-series")
    assert roster.characters == []


def test_update_roster_persists_new_characters(tmp_path):
    new_roster = CharacterRoster(characters=[_rec("jane_eyre", "Jane Eyre")])

    with patch.object(_series, "_series_dir_override", tmp_path):
        series_service.update_roster("jane-eyre-series", new_roster, book_slug="jane_eyre_v1")
        loaded = series_service.get_roster("jane-eyre-series")

    assert len(loaded.characters) == 1
    assert loaded.characters[0].slug == "jane_eyre"


def test_update_roster_merges_across_calls(tmp_path):
    book1_roster = CharacterRoster(characters=[_rec("jane_eyre", "Jane Eyre", mentions=50)])
    book2_roster = CharacterRoster(characters=[_rec("jane_eyre", "Jane Eyre", mentions=40)])

    with patch.object(_series, "_series_dir_override", tmp_path):
        series_service.update_roster("jane-eyre-series", book1_roster, book_slug="book1")
        series_service.update_roster("jane-eyre-series", book2_roster, book_slug="book2")
        final = series_service.get_roster("jane-eyre-series")

    assert len(final.characters) == 1
    assert final.characters[0].mention_count == 90


# ---------------------------------------------------------------------------
# nlp_service series wiring
# ---------------------------------------------------------------------------


def test_fast_scan_passes_series_roster_to_provider(tmp_path):
    """fast_scan should fetch the series roster and pass it to build_roster."""
    from pathlib import Path
    from unittest.mock import MagicMock
    from kenkui.models import Chapter
    from kenkui.services.nlp_service import fast_scan

    fake_epub = tmp_path / "book.epub"
    fake_epub.write_bytes(b"fake")

    mock_reader = MagicMock()
    mock_reader.get_chapters.return_value = [Chapter(index=0, title="Ch", paragraphs=["t"])]

    existing_series_roster = CharacterRoster(characters=[_rec("jane_eyre", "Jane Eyre")])
    mock_roster_result = CharacterRoster(characters=[_rec("jane_eyre", "Jane Eyre")])

    mock_provider = MagicMock()
    mock_provider.build_roster.return_value = mock_roster_result

    captured_series_roster = []

    def _capture_build_roster(chapters, series_roster=None, progress_callback=None):
        captured_series_roster.append(series_roster)
        return mock_roster_result

    mock_provider.build_roster.side_effect = _capture_build_roster

    with (
        patch("kenkui.services.nlp_service.get_reader", return_value=mock_reader),
        patch("kenkui.services.nlp_service.get_provider", return_value=mock_provider),
        patch("kenkui.services.nlp_service.get_cached_roster", return_value=None),
        patch("kenkui.services.nlp_service.cache_roster"),
        patch("kenkui.services.nlp_service.book_hash", return_value="abc123"),
        patch("kenkui.services.series_service.get_roster", return_value=existing_series_roster) as mock_get,
        patch("kenkui.services.series_service.update_roster") as mock_update,
    ):
        fast_scan(
            str(fake_epub),
            nlp_model="llama3.2",
            series_slug="jane-eyre-series",
            book_slug="jane_eyre_v1",
        )

    # Series roster was fetched and passed to build_roster
    assert len(captured_series_roster) == 1
    assert captured_series_roster[0] is existing_series_roster
