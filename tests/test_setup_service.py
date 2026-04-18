"""Tests for setup-flow helper functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


def test_parse_fast_scan_result_round_trip():
    from kenkui.models import CharacterInfo, FastScanResult
    from kenkui.nlp.models import CharacterRoster
    from kenkui.services.setup_service import parse_fast_scan_result

    result = FastScanResult(
        roster=CharacterRoster(characters=[]),
        characters=[CharacterInfo(character_id="Rand", display_name="Rand", mention_count=10)],
        book_hash="abc123",
    )

    parsed = parse_fast_scan_result(result.to_dict())
    assert parsed.book_hash == "abc123"
    assert parsed.characters[0].character_id == "Rand"


def test_cache_roster_result_returns_string_path(tmp_path):
    from kenkui.models import CharacterInfo, FastScanResult
    from kenkui.nlp.models import CharacterRoster
    from kenkui.services.setup_service import cache_roster_result

    result = FastScanResult(
        roster=CharacterRoster(characters=[]),
        characters=[CharacterInfo(character_id="Rand", display_name="Rand")],
        book_hash="abc123",
    )

    with patch("kenkui.nlp.cache_roster", return_value=tmp_path / "roster.json"):
        path = cache_roster_result(result, Path("/tmp/book.epub"))
    assert path == str(tmp_path / "roster.json")


def test_cache_roster_result_returns_none_on_failure():
    from kenkui.models import CharacterInfo, FastScanResult
    from kenkui.nlp.models import CharacterRoster
    from kenkui.services.setup_service import cache_roster_result

    result = FastScanResult(
        roster=CharacterRoster(characters=[]),
        characters=[CharacterInfo(character_id="Rand", display_name="Rand")],
        book_hash="abc123",
    )

    with patch("kenkui.nlp.cache_roster", side_effect=RuntimeError("boom")):
        path = cache_roster_result(result, Path("/tmp/book.epub"))
    assert path is None


def test_build_chapter_prompt_items_uses_chapters_key():
    from kenkui.services.setup_service import build_chapter_prompt_items

    chapters = build_chapter_prompt_items(
        {
            "chapters": [
                {"index": 0, "title": "Chapter 1"},
                {"index": 1, "title": "Chapter 2"},
            ]
        }
    )
    assert [ch.title for ch in chapters] == ["Chapter 1", "Chapter 2"]


def test_build_chapter_prompt_items_handles_missing_chapters():
    from kenkui.services.setup_service import build_chapter_prompt_items

    assert build_chapter_prompt_items({}) == []
