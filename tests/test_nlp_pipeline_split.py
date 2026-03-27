"""Tests for the split NLP pipeline: cache_roster / get_cached_roster."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest


def _make_fast_scan_result():
    from kenkui.models import CharacterInfo, FastScanResult
    from kenkui.nlp.models import AliasGroup, CharacterRoster
    roster = CharacterRoster(characters=[
        AliasGroup(canonical="Alice", aliases=["Alice"], gender="she/her"),
    ])
    characters = [CharacterInfo(character_id="Alice", display_name="Alice", mention_count=50)]
    return FastScanResult(roster=roster, characters=characters, book_hash="testhash")


class TestRosterCache:
    def test_cache_and_retrieve(self, tmp_path):
        from kenkui.nlp import cache_roster, get_cached_roster

        # Make a fake ebook path; book_hash uses path + mtime
        ebook = tmp_path / "book.epub"
        ebook.write_text("fake")

        result = _make_fast_scan_result()

        with patch("kenkui.nlp.CONFIG_DIR", tmp_path):
            with patch("kenkui.config.CONFIG_DIR", tmp_path):
                cache_path = cache_roster(result, ebook)
                assert cache_path.exists()
                assert cache_path.name.endswith("-roster.json")

                restored = get_cached_roster(ebook)
                assert restored is not None
                assert restored.characters[0].mention_count == 50
                assert restored.roster.characters[0].canonical == "Alice"

    def test_get_cached_roster_returns_none_when_missing(self, tmp_path):
        from kenkui.nlp import get_cached_roster

        ebook = tmp_path / "book.epub"
        ebook.write_text("fake")

        with patch("kenkui.nlp.CONFIG_DIR", tmp_path):
            with patch("kenkui.config.CONFIG_DIR", tmp_path):
                result = get_cached_roster(ebook)
                assert result is None

    def test_get_cached_roster_returns_none_on_corrupt_file(self, tmp_path):
        from kenkui.nlp import book_hash, get_cached_roster

        ebook = tmp_path / "book.epub"
        ebook.write_text("fake")

        cache_dir = tmp_path / "nlp_cache"
        cache_dir.mkdir()
        corrupt = cache_dir / f"{book_hash(ebook)}-roster.json"
        corrupt.write_text("not json {{{{")

        with patch("kenkui.nlp.CONFIG_DIR", tmp_path):
            with patch("kenkui.config.CONFIG_DIR", tmp_path):
                result = get_cached_roster(ebook)
                assert result is None
