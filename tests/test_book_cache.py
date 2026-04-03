"""Tests for kenkui.services.book_cache."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kenkui.chapter_classifier import ChapterTags
from kenkui.models import Chapter
from kenkui.readers import EbookMetadata
from kenkui.services.book_cache import BookCache, BookEntry, ChapterSummary


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_chapter(
    index: int = 0,
    title: str = "Chapter 1",
    paragraphs: list[str] | None = None,
    toc_index: int = 0,
    tags: ChapterTags | None = None,
) -> Chapter:
    if paragraphs is None:
        paragraphs = ["Hello world foo bar.", "Another paragraph with words."]
    if tags is None:
        tags = ChapterTags(is_chapter=True)
    return Chapter(
        index=index,
        title=title,
        paragraphs=paragraphs,
        toc_index=toc_index,
        tags=tags,
    )


def _make_metadata(title: str = "Test Book") -> EbookMetadata:
    return EbookMetadata(
        title=title,
        author="Test Author",
        language="en",
        publisher="Test Publisher",
        description="A test book.",
        cover_image=b"\x89PNG fake image bytes",  # should NOT end up in cache
    )


@pytest.fixture()
def cache_path(tmp_path: Path) -> Path:
    return tmp_path / "book_cache.json"


@pytest.fixture()
def cache(cache_path: Path) -> BookCache:
    return BookCache(cache_path=cache_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPut:
    def test_put_returns_book_entry(self, cache: BookCache) -> None:
        ch = _make_chapter()
        meta = _make_metadata()
        entry = cache.put("hash123", "/some/book.epub", meta, [ch])
        assert isinstance(entry, BookEntry)
        assert entry.book_hash == "hash123"
        assert entry.ebook_path == "/some/book.epub"

    def test_put_creates_chapter_summaries(self, cache: BookCache) -> None:
        ch = _make_chapter(index=2, title="Intro", toc_index=3)
        meta = _make_metadata()
        entry = cache.put("hash456", "/book.epub", meta, [ch])
        assert len(entry.chapter_summaries) == 1
        s = entry.chapter_summaries[0]
        assert isinstance(s, ChapterSummary)
        assert s.index == 2
        assert s.title == "Intro"
        assert s.toc_index == 3
        assert s.paragraph_count == len(ch.paragraphs)

    def test_put_sets_parsed_at(self, cache: BookCache) -> None:
        import time
        before = time.time()
        entry = cache.put("h1", "/b.epub", _make_metadata(), [_make_chapter()])
        after = time.time()
        assert before <= entry.parsed_at <= after

    def test_put_excludes_cover_image(self, cache: BookCache) -> None:
        meta = _make_metadata()
        entry = cache.put("h1", "/b.epub", meta, [_make_chapter()])
        assert "cover_image" not in entry.metadata
        assert "cover_mime_type" not in entry.metadata

    def test_put_stores_metadata_fields(self, cache: BookCache) -> None:
        meta = _make_metadata(title="My Novel")
        entry = cache.put("h1", "/b.epub", meta, [_make_chapter()])
        assert entry.metadata["title"] == "My Novel"
        assert entry.metadata["author"] == "Test Author"
        assert entry.metadata["language"] == "en"
        assert entry.metadata["publisher"] == "Test Publisher"
        assert entry.metadata["description"] == "A test book."

    def test_put_stores_tags(self, cache: BookCache) -> None:
        tags = ChapterTags(is_front_matter=True, is_chapter=False)
        ch = _make_chapter(tags=tags)
        entry = cache.put("h1", "/b.epub", _make_metadata(), [ch])
        stored_tags = entry.chapter_summaries[0].tags
        assert stored_tags.is_front_matter is True
        assert stored_tags.is_chapter is False


class TestWordCount:
    def test_word_count_calculated_correctly(self, cache: BookCache) -> None:
        # "Hello world" = 2 words, "foo bar baz" = 3 words => total 5
        ch = _make_chapter(paragraphs=["Hello world", "foo bar baz"])
        entry = cache.put("h1", "/b.epub", _make_metadata(), [ch])
        assert entry.chapter_summaries[0].word_count == 5

    def test_word_count_empty_paragraphs(self, cache: BookCache) -> None:
        ch = _make_chapter(paragraphs=[])
        entry = cache.put("h1", "/b.epub", _make_metadata(), [ch])
        assert entry.chapter_summaries[0].word_count == 0
        assert entry.chapter_summaries[0].paragraph_count == 0

    def test_paragraph_count_matches_list_length(self, cache: BookCache) -> None:
        ch = _make_chapter(paragraphs=["A", "B", "C", "D"])
        entry = cache.put("h1", "/b.epub", _make_metadata(), [ch])
        assert entry.chapter_summaries[0].paragraph_count == 4


class TestGet:
    def test_get_returns_stored_entry(self, cache: BookCache) -> None:
        cache.put("h1", "/b.epub", _make_metadata(), [_make_chapter()])
        entry = cache.get("h1")
        assert entry is not None
        assert entry.book_hash == "h1"

    def test_get_returns_none_for_missing(self, cache: BookCache) -> None:
        assert cache.get("nonexistent") is None

    def test_get_after_multiple_puts(self, cache: BookCache) -> None:
        cache.put("a", "/a.epub", _make_metadata("A"), [_make_chapter(title="A")])
        cache.put("b", "/b.epub", _make_metadata("B"), [_make_chapter(title="B")])
        assert cache.get("a").metadata["title"] == "A"
        assert cache.get("b").metadata["title"] == "B"


class TestEvict:
    def test_evict_removes_entry(self, cache: BookCache) -> None:
        cache.put("h1", "/b.epub", _make_metadata(), [_make_chapter()])
        cache.evict("h1")
        assert cache.get("h1") is None

    def test_evict_nonexistent_does_not_raise(self, cache: BookCache) -> None:
        cache.evict("does_not_exist")  # should not raise

    def test_evict_only_removes_target(self, cache: BookCache) -> None:
        cache.put("a", "/a.epub", _make_metadata(), [_make_chapter()])
        cache.put("b", "/b.epub", _make_metadata(), [_make_chapter()])
        cache.evict("a")
        assert cache.get("a") is None
        assert cache.get("b") is not None


class TestPersistence:
    def test_reload_recovers_entry(self, cache_path: Path) -> None:
        c1 = BookCache(cache_path=cache_path)
        c1.put("h1", "/b.epub", _make_metadata("Reload Test"), [_make_chapter()])

        c2 = BookCache(cache_path=cache_path)
        entry = c2.get("h1")
        assert entry is not None
        assert entry.metadata["title"] == "Reload Test"

    def test_reload_chapter_summaries(self, cache_path: Path) -> None:
        ch = _make_chapter(
            index=5,
            title="Epilogue",
            paragraphs=["one two three", "four five"],
            toc_index=7,
            tags=ChapterTags(is_back_matter=True, is_chapter=False),
        )
        c1 = BookCache(cache_path=cache_path)
        c1.put("h2", "/b.epub", _make_metadata(), [ch])

        c2 = BookCache(cache_path=cache_path)
        entry = c2.get("h2")
        s = entry.chapter_summaries[0]
        assert s.index == 5
        assert s.title == "Epilogue"
        assert s.word_count == 5  # "one two three" (3) + "four five" (2)
        assert s.paragraph_count == 2
        assert s.toc_index == 7
        assert s.tags.is_back_matter is True
        assert s.tags.is_chapter is False

    def test_reload_after_evict(self, cache_path: Path) -> None:
        c1 = BookCache(cache_path=cache_path)
        c1.put("a", "/a.epub", _make_metadata(), [_make_chapter()])
        c1.put("b", "/b.epub", _make_metadata(), [_make_chapter()])
        c1.evict("a")

        c2 = BookCache(cache_path=cache_path)
        assert c2.get("a") is None
        assert c2.get("b") is not None

    def test_json_format(self, cache_path: Path) -> None:
        """Verify the on-disk JSON structure matches the spec."""
        c = BookCache(cache_path=cache_path)
        c.put("h1", "/b.epub", _make_metadata(), [_make_chapter()])

        raw = json.loads(cache_path.read_text())
        assert "entries" in raw
        assert "h1" in raw["entries"]
        e = raw["entries"]["h1"]
        assert "book_hash" in e
        assert "ebook_path" in e
        assert "parsed_at" in e
        assert "metadata" in e
        assert "chapter_summaries" in e
        assert "cover_image" not in e["metadata"]
        s = e["chapter_summaries"][0]
        assert "tags" in s
        assert isinstance(s["tags"], dict)
        assert "is_chapter" in s["tags"]


class TestMissingCacheFile:
    def test_load_missing_file_no_error(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent" / "book_cache.json"
        # Should not raise even though file and directory don't exist yet.
        cache = BookCache(cache_path=path)
        assert cache.get("anything") is None

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "dir" / "book_cache.json"
        cache = BookCache(cache_path=path)
        cache.put("h1", "/b.epub", _make_metadata(), [_make_chapter()])
        assert path.exists()

    def test_corrupted_cache_file_handled_gracefully(self, tmp_path: Path) -> None:
        path = tmp_path / "book_cache.json"
        path.write_text("NOT VALID JSON", encoding="utf-8")
        # Should not raise; starts with empty cache.
        cache = BookCache(cache_path=path)
        assert cache.get("anything") is None
