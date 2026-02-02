"""Tests for kenkui parsing functionality."""

import pytest
from pathlib import Path
from kenkui.parsing import EpubReader
from kenkui.helpers import Chapter

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
        title = reader.get_book_title()
        assert isinstance(title, str)
        assert len(title) > 0
        # Title should be sanitized (no special chars)
        assert "<" not in title
        assert ">" not in title

    def test_extract_chapters_returns_list(self, reader):
        """Test that extract_chapters returns a list."""
        chapters = reader.extract_chapters()
        assert isinstance(chapters, list)

    def test_extract_chapters_returns_chapter_objects(self, reader):
        """Test that extract_chapters returns Chapter objects."""
        chapters = reader.extract_chapters()
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
