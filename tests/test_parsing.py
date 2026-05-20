"""Tests for kenkui parsing functionality."""

from pathlib import Path

import pytest

from kenkui.models import Chapter
from kenkui.readers.epub import EpubReader

TEST_EPUB = Path("src/kenkui/samples/Les Miserables - Victor Hugo.epub")


class TestEpubReader:
    """Tests for the EpubReader class."""

    @pytest.fixture
    def reader(self):
        """Create an EpubReader instance for testing."""
        return EpubReader(TEST_EPUB)

    def test_reader_initialization(self, reader):
        """Test that EpubReader can be initialized."""
        assert reader is not None
        assert reader.filepath == TEST_EPUB

    def test_get_book_title(self, reader):
        """Test extracting book title from EPUB."""
        title = reader.get_metadata().title
        assert isinstance(title, str)
        assert len(title) > 0
        # Title should be sanitized (no special chars)
        assert "<" not in title
        assert ">" not in title

    def test_extract_chapters_returns_list(self, reader):
        """Test that get_chapters returns a list."""
        chapters = reader.get_chapters()
        assert isinstance(chapters, list)

    def test_extract_chapters_returns_chapter_objects(self, reader):
        """Test that get_chapters returns Chapter objects."""
        chapters = reader.get_chapters()
        if chapters:  # Only test if chapters were found
            for chapter in chapters:
                assert isinstance(chapter, Chapter)
                assert hasattr(chapter, "index")
                assert hasattr(chapter, "title")
                assert hasattr(chapter, "paragraphs")
                assert isinstance(chapter.index, int)
                assert isinstance(chapter.title, str)
                assert isinstance(chapter.paragraphs, list)


class TestChapterDataclass:
    """Tests for the Chapter dataclass."""

    def test_chapter_creation(self):
        """Test creating a Chapter instance."""
        chapter = Chapter(
            index=1, title="Test Chapter", paragraphs=["Paragraph 1", "Paragraph 2"]
        )
        assert chapter.index == 1
        assert chapter.title == "Test Chapter"
        assert len(chapter.paragraphs) == 2

    def test_chapter_with_empty_paragraphs(self):
        """Test creating a Chapter with empty paragraphs."""
        chapter = Chapter(index=2, title="Empty Chapter", paragraphs=[])
        assert chapter.index == 2
        assert chapter.paragraphs == []


class TestLoadAnnotatedChaptersSpeakerSlugNormalization:
    """Test that _load_annotated_chapters normalizes speaker strings to slugs at read time."""

    def _make_cache(self, tmp_path: Path) -> Path:
        import json

        cache_data = {
            "chapters": [
                {
                    "index": 0,
                    "title": "Chapter 1",
                    "paragraphs": [],
                    "toc_index": 0,
                    "segments": [
                        {"text": "He said something.", "speaker": "Darrow", "index": 0, "is_scene_break": False},
                        {"text": "She replied.", "speaker": "Rhonna", "index": 1, "is_scene_break": False},
                        {"text": "Narration.", "speaker": "NARRATOR", "index": 2, "is_scene_break": False},
                        {"text": "Unknown speaker.", "speaker": "Unknown", "index": 3, "is_scene_break": False},
                        {"text": "Mixed Case Name.", "speaker": "Sevro Au Barca", "index": 4, "is_scene_break": False},
                    ],
                }
            ]
        }
        cache_file = tmp_path / "nlp_cache.json"
        cache_file.write_text(json.dumps(cache_data), encoding="utf-8")
        return cache_file

    def test_slug_normalization_lowercases_non_sentinel_speakers(self, tmp_path):
        """Non-sentinel speakers are converted to lowercase slugs at cache read time."""
        from kenkui.parsing import _load_annotated_chapters

        cache_path = self._make_cache(tmp_path)
        chapters = _load_annotated_chapters(cache_path, [])

        assert len(chapters) == 1
        segs = {s.index: s for s in chapters[0].segments}

        # Non-sentinels must be slugified
        assert segs[0].speaker == "darrow"
        assert segs[1].speaker == "rhonna"
        assert segs[4].speaker == "sevro_au_barca"

    def test_slug_normalization_preserves_sentinel_speakers(self, tmp_path):
        """NARRATOR and Unknown sentinels are left unchanged."""
        from kenkui.parsing import _load_annotated_chapters

        cache_path = self._make_cache(tmp_path)
        chapters = _load_annotated_chapters(cache_path, [])

        segs = {s.index: s for s in chapters[0].segments}

        assert segs[2].speaker == "NARRATOR"
        assert segs[3].speaker == "Unknown"
