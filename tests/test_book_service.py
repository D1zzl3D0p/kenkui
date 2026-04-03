"""Tests for kenkui.services.book_service."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kenkui.chapter_classifier import ChapterTags
from kenkui.models import Chapter, ChapterPreset, ChapterSelection
from kenkui.services.book_cache import BookCache, BookEntry, ChapterSummary
from kenkui.services.book_service import (
    BookParseResult,
    ChapterFilterResult,
    filter_chapters,
    parse_book,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLES_DIR = Path(__file__).parent.parent / "src" / "kenkui" / "samples"
PRIDE_EPUB = SAMPLES_DIR / "Pride and Predjudice.epub"


def _make_summary(
    index: int = 0,
    title: str = "Chapter 1",
    word_count: int = 100,
    tags: ChapterTags | None = None,
) -> ChapterSummary:
    if tags is None:
        tags = ChapterTags(is_chapter=True)
    return ChapterSummary(
        index=index,
        title=title,
        word_count=word_count,
        paragraph_count=5,
        toc_index=index,
        tags=tags,
    )


def _make_entry(summaries: list[ChapterSummary], book_hash: str = "abc123", path: str = "/fake/book.epub") -> BookEntry:
    from kenkui.services.book_cache import BookEntry
    return BookEntry(
        book_hash=book_hash,
        ebook_path=path,
        metadata={"title": "Fake Book", "author": "Author", "language": "en", "publisher": None, "description": None},
        chapter_summaries=summaries,
        parsed_at=1000000.0,
    )


def _mock_cache(entry: BookEntry | None = None) -> MagicMock:
    cache = MagicMock(spec=BookCache)
    cache.get.return_value = entry
    return cache


# ---------------------------------------------------------------------------
# filter_chapters — unit tests (mock cache)
# ---------------------------------------------------------------------------

class TestFilterChapters:

    def test_raises_keyerror_when_not_in_cache(self):
        cache = _mock_cache(entry=None)
        selection = ChapterSelection(preset=ChapterPreset.CONTENT_ONLY)
        with pytest.raises(KeyError, match="parse_book first"):
            filter_chapters("nonexistent_hash", selection, cache)

    def test_content_only_preset_excludes_front_matter(self):
        front = _make_summary(0, "Preface", tags=ChapterTags(is_front_matter=True, is_chapter=False))
        ch1 = _make_summary(1, "Chapter One", tags=ChapterTags(is_chapter=True))
        ch2 = _make_summary(2, "Chapter Two", tags=ChapterTags(is_chapter=True))
        back = _make_summary(3, "Index", tags=ChapterTags(is_back_matter=True, is_chapter=False))

        entry = _make_entry([front, ch1, ch2, back], book_hash="h1")
        cache = _mock_cache(entry)

        result = filter_chapters("h1", ChapterSelection(preset=ChapterPreset.CONTENT_ONLY), cache)

        assert result.included_indices == [1, 2]
        assert result.chapter_count == 2
        assert result.estimated_word_count == ch1.word_count + ch2.word_count
        assert len(result.chapters) == 2

    def test_manual_preset_uses_included_list(self):
        summaries = [_make_summary(i, f"Chapter {i}") for i in range(5)]
        entry = _make_entry(summaries, book_hash="h2")
        cache = _mock_cache(entry)

        selection = ChapterSelection(preset=ChapterPreset.MANUAL, included=[1, 3])
        result = filter_chapters("h2", selection, cache)

        assert result.included_indices == [1, 3]
        assert result.chapter_count == 2

    def test_manual_preset_empty_included_falls_back_to_content_only(self):
        summaries = [_make_summary(i, f"Chapter {i}", tags=ChapterTags(is_chapter=True)) for i in range(3)]
        entry = _make_entry(summaries, book_hash="h3")
        cache = _mock_cache(entry)

        selection = ChapterSelection(preset=ChapterPreset.MANUAL, included=[])
        result = filter_chapters("h3", selection, cache)

        # content-only includes all chapters (none flagged as front/back matter)
        assert result.chapter_count == 3

    def test_custom_preset_uses_included_list(self):
        summaries = [_make_summary(i, f"Chapter {i}") for i in range(4)]
        entry = _make_entry(summaries, book_hash="h4")
        cache = _mock_cache(entry)

        selection = ChapterSelection(preset=ChapterPreset.CUSTOM, included=[0, 2])
        result = filter_chapters("h4", selection, cache)

        assert result.included_indices == [0, 2]

    def test_excluded_indices_are_removed_after_preset(self):
        summaries = [_make_summary(i, f"Chapter {i}", tags=ChapterTags(is_chapter=True)) for i in range(5)]
        entry = _make_entry(summaries, book_hash="h5")
        cache = _mock_cache(entry)

        # content-only would include all 5, but exclude [2, 4]
        selection = ChapterSelection(preset=ChapterPreset.CONTENT_ONLY, excluded=[2, 4])
        result = filter_chapters("h5", selection, cache)

        assert 2 not in result.included_indices
        assert 4 not in result.included_indices
        assert result.chapter_count == 3

    def test_chapters_only_preset(self):
        part = _make_summary(0, "Part I", tags=ChapterTags(is_chapter=True, is_part_divider=True))
        ch1 = _make_summary(1, "Chapter One", tags=ChapterTags(is_chapter=True))
        ch2 = _make_summary(2, "Chapter Two", tags=ChapterTags(is_chapter=True))

        entry = _make_entry([part, ch1, ch2], book_hash="h6")
        cache = _mock_cache(entry)

        result = filter_chapters("h6", ChapterSelection(preset=ChapterPreset.CHAPTERS_ONLY), cache)

        # part divider excluded, chapters included
        assert result.included_indices == [1, 2]

    def test_with_parts_preset_includes_part_dividers(self):
        part = _make_summary(0, "Part I", tags=ChapterTags(is_chapter=True, is_part_divider=True))
        ch1 = _make_summary(1, "Chapter One", tags=ChapterTags(is_chapter=True))

        entry = _make_entry([part, ch1], book_hash="h7")
        cache = _mock_cache(entry)

        result = filter_chapters("h7", ChapterSelection(preset=ChapterPreset.WITH_PARTS), cache)

        assert 0 in result.included_indices
        assert 1 in result.included_indices

    def test_estimated_word_count_is_sum_of_included(self):
        summaries = [
            _make_summary(0, "Ch0", word_count=200, tags=ChapterTags(is_chapter=True)),
            _make_summary(1, "Ch1", word_count=300, tags=ChapterTags(is_chapter=True)),
        ]
        entry = _make_entry(summaries, book_hash="h8")
        cache = _mock_cache(entry)

        result = filter_chapters("h8", ChapterSelection(preset=ChapterPreset.CONTENT_ONLY), cache)

        assert result.estimated_word_count == 500


# ---------------------------------------------------------------------------
# parse_book — integration tests (real epub, real cache)
# ---------------------------------------------------------------------------

class TestParseBook:

    @pytest.fixture()
    def cache(self, tmp_path: Path) -> BookCache:
        return BookCache(cache_path=tmp_path / "book_cache.json")

    def test_raises_file_not_found_for_missing_file(self, cache: BookCache):
        with pytest.raises(FileNotFoundError):
            parse_book("/nonexistent/path/to/book.epub", cache)

    @pytest.mark.skipif(
        not PRIDE_EPUB.exists(),
        reason="Sample epub not present",
    )
    def test_parse_real_epub_returns_result(self, cache: BookCache):
        result = parse_book(str(PRIDE_EPUB), cache)

        assert isinstance(result, BookParseResult)
        assert result.book_hash  # non-empty hash
        assert result.total_chapters > 0
        assert result.total_word_count > 0
        assert isinstance(result.metadata, dict)
        assert "title" in result.metadata
        assert len(result.chapters) == result.total_chapters

    @pytest.mark.skipif(
        not PRIDE_EPUB.exists(),
        reason="Sample epub not present",
    )
    def test_parse_book_stores_in_cache(self, cache: BookCache):
        result = parse_book(str(PRIDE_EPUB), cache)

        entry = cache.get(result.book_hash)
        assert entry is not None
        assert entry.book_hash == result.book_hash
        assert entry.ebook_path == str(PRIDE_EPUB.resolve())

    @pytest.mark.skipif(
        not PRIDE_EPUB.exists(),
        reason="Sample epub not present",
    )
    def test_parse_book_second_call_uses_cache(self, cache: BookCache):
        result1 = parse_book(str(PRIDE_EPUB), cache)

        # Patch get_reader to confirm it is NOT called on second parse.
        with patch("kenkui.services.book_service.get_reader") as mock_reader:
            result2 = parse_book(str(PRIDE_EPUB), cache)
            mock_reader.assert_not_called()

        assert result2.book_hash == result1.book_hash
        assert result2.total_chapters == result1.total_chapters

    @pytest.mark.skipif(
        not PRIDE_EPUB.exists(),
        reason="Sample epub not present",
    )
    def test_parse_book_chapter_summaries_have_word_counts(self, cache: BookCache):
        result = parse_book(str(PRIDE_EPUB), cache)

        assert all(s.word_count >= 0 for s in result.chapters)
        # At least some chapters should have actual content
        assert any(s.word_count > 0 for s in result.chapters)
