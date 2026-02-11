"""Tests for the chapter filtering system."""

import pytest
from kenkui.chapter_filter import ChapterFilter, FilterOperation, FilterPreset
from kenkui.chapter_classifier import ChapterTags
from kenkui.helpers import Chapter


@pytest.fixture
def sample_chapters():
    """Create a list of sample chapters for testing."""
    return [
        Chapter(
            index=0,
            title="Title Page",
            paragraphs=["content"],
            tags=ChapterTags(is_title_page=True, is_chapter=False),
        ),
        Chapter(
            index=1,
            title="Copyright",
            paragraphs=["content"],
            tags=ChapterTags(is_title_page=True, is_chapter=False),
        ),
        Chapter(
            index=2,
            title="Introduction",
            paragraphs=["content"],
            tags=ChapterTags(is_front_matter=True, is_chapter=True),
        ),
        Chapter(
            index=3,
            title="Chapter 1: The Beginning",
            paragraphs=["content"],
            tags=ChapterTags(is_chapter=True),
        ),
        Chapter(
            index=4,
            title="Chapter 2: The Journey",
            paragraphs=["content"],
            tags=ChapterTags(is_chapter=True),
        ),
        Chapter(
            index=5,
            title="Part II",
            paragraphs=["content"],
            tags=ChapterTags(is_part_divider=True, is_chapter=True),
        ),
        Chapter(
            index=6,
            title="Chapter 3: The Climax",
            paragraphs=["content"],
            tags=ChapterTags(is_chapter=True),
        ),
        Chapter(
            index=7,
            title="Appendix A",
            paragraphs=["content"],
            tags=ChapterTags(is_back_matter=True, is_chapter=True),
        ),
        Chapter(
            index=8,
            title="References",
            paragraphs=["content"],
            tags=ChapterTags(is_back_matter=True, is_chapter=True),
        ),
    ]


class TestFilterPresets:
    """Tests for the built-in filter presets."""

    def test_preset_all_includes_everything(self, sample_chapters):
        """Test that 'all' preset includes all chapters."""
        operations = [FilterOperation("preset", "all")]
        filter_chain = ChapterFilter(operations)
        result = filter_chain.apply(sample_chapters)
        assert len(result) == len(sample_chapters)

    def test_preset_content_only_excludes_front_back_matter(self, sample_chapters):
        """Test that 'content-only' excludes front/back matter and title pages."""
        operations = [FilterOperation("preset", "content-only")]
        filter_chain = ChapterFilter(operations)
        result = filter_chain.apply(sample_chapters)

        # Should exclude: Title Page, Copyright, Introduction, Appendix, References
        # Should include: Chapter 1, Chapter 2, Part II, Chapter 3
        assert len(result) == 4
        titles = [ch.title for ch in result]
        assert "Title Page" not in titles
        assert "Copyright" not in titles
        assert "Introduction" not in titles
        assert "Appendix A" not in titles
        assert "References" not in titles

    def test_preset_chapters_only_excludes_dividers(self, sample_chapters):
        """Test that 'chapters-only' excludes part dividers."""
        operations = [FilterOperation("preset", "chapters-only")]
        filter_chain = ChapterFilter(operations)
        result = filter_chain.apply(sample_chapters)

        titles = [ch.title for ch in result]
        assert "Part II" not in titles
        assert "Chapter 1: The Beginning" in titles

    def test_preset_with_parts_includes_part_dividers(self, sample_chapters):
        """Test that 'with-parts' includes part dividers."""
        operations = [FilterOperation("preset", "with-parts")]
        filter_chain = ChapterFilter(operations)
        result = filter_chain.apply(sample_chapters)

        titles = [ch.title for ch in result]
        assert "Part II" in titles


class TestRegexFiltering:
    """Tests for regex-based include/exclude filtering."""

    def test_include_pattern_starts_empty(self, sample_chapters):
        """Test that include without preset starts from empty selection."""
        operations = [FilterOperation("include", "Chapter.*")]
        filter_chain = ChapterFilter(operations)
        result = filter_chain.apply(sample_chapters)

        # Should only include chapters matching "Chapter.*"
        assert len(result) == 3
        titles = [ch.title for ch in result]
        assert all("Chapter" in t for t in titles)

    def test_exclude_pattern_starts_with_all(self, sample_chapters):
        """Test that exclude without preset starts from all chapters."""
        operations = [FilterOperation("exclude", "Chapter.*")]
        filter_chain = ChapterFilter(operations)
        result = filter_chain.apply(sample_chapters)

        # Should exclude chapters matching "Chapter.*"
        assert len(result) == 6
        titles = [ch.title for ch in result]
        assert all("Chapter" not in t for t in titles)

    def test_multiple_include_patterns(self, sample_chapters):
        """Test that multiple include patterns are additive."""
        operations = [
            FilterOperation("include", "Chapter 1.*"),
            FilterOperation("include", "Appendix.*"),
        ]
        filter_chain = ChapterFilter(operations)
        result = filter_chain.apply(sample_chapters)

        assert len(result) == 2
        titles = [ch.title for ch in result]
        assert "Chapter 1: The Beginning" in titles
        assert "Appendix A" in titles

    def test_preset_then_include(self, sample_chapters):
        """Test preset followed by include adds to the selection."""
        operations = [
            FilterOperation("preset", "chapters-only"),
            FilterOperation("include", "Part.*"),
        ]
        filter_chain = ChapterFilter(operations)
        result = filter_chain.apply(sample_chapters)

        # chapters-only excludes Part II, then we include it back
        titles = [ch.title for ch in result]
        assert "Part II" in titles

    def test_preset_then_exclude(self, sample_chapters):
        """Test preset followed by exclude removes from the selection."""
        operations = [
            FilterOperation("preset", "all"),
            FilterOperation("exclude", "Chapter.*"),
        ]
        filter_chain = ChapterFilter(operations)
        result = filter_chain.apply(sample_chapters)

        titles = [ch.title for ch in result]
        assert all("Chapter" not in t for t in titles)


class TestOperationPrecedence:
    """Tests for operation ordering and precedence."""

    def test_later_preset_overrides_earlier(self, sample_chapters):
        """Test that later presets override earlier ones."""
        operations = [
            FilterOperation("preset", "all"),
            FilterOperation("preset", "chapters-only"),
        ]
        filter_chain = ChapterFilter(operations)
        result = filter_chain.apply(sample_chapters)

        # Should use chapters-only preset
        assert len(result) == 6

    def test_complex_filter_chain(self, sample_chapters):
        """Test a complex filter chain with multiple operations."""
        operations = [
            FilterOperation("preset", "content-only"),
            FilterOperation("include", "Introduction"),
            FilterOperation("exclude", "Part.*"),
            FilterOperation("exclude", "Chapter 1.*"),
        ]
        filter_chain = ChapterFilter(operations)
        result = filter_chain.apply(sample_chapters)

        titles = [ch.title for ch in result]
        assert "Introduction" in titles  # Added by include
        assert "Part II" not in titles  # Excluded by exclude
        assert "Chapter 1: The Beginning" not in titles  # Excluded by exclude
        assert "Chapter 2: The Journey" in titles  # Still included from preset


class TestValidation:
    """Tests for input validation."""

    def test_invalid_preset_raises_error(self, sample_chapters):
        """Test that invalid preset name raises ValueError."""
        operations = [FilterOperation("preset", "invalid-preset")]
        with pytest.raises(ValueError, match="Unknown preset"):
            ChapterFilter(operations)

    def test_invalid_regex_raises_error(self, sample_chapters):
        """Test that invalid regex pattern raises ValueError."""
        operations = [FilterOperation("include", "[invalid")]
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            ChapterFilter(operations)

    def test_case_sensitive_regex(self, sample_chapters):
        """Test that regex matching is case-sensitive by default."""
        operations = [FilterOperation("include", "chapter.*")]
        filter_chain = ChapterFilter(operations)
        result = filter_chain.apply(sample_chapters)

        # Should not match "Chapter" (capital C)
        assert len(result) == 0

    def test_case_insensitive_regex_pattern(self, sample_chapters):
        """Test that (?i) flag makes regex case-insensitive."""
        operations = [FilterOperation("include", "(?i)chapter.*")]
        filter_chain = ChapterFilter(operations)
        result = filter_chain.apply(sample_chapters)

        # Should match "Chapter" (capital C) with case-insensitive flag
        assert len(result) == 3


class TestEmptyAndEdgeCases:
    """Tests for edge cases."""

    def test_no_operations_returns_all_chapters(self, sample_chapters):
        """Test that empty operations list returns all chapters."""
        filter_chain = ChapterFilter([])
        result = filter_chain.apply(sample_chapters)
        assert len(result) == len(sample_chapters)

    def test_no_matching_chapters_returns_empty(self, sample_chapters):
        """Test that no matching chapters returns empty list."""
        operations = [FilterOperation("include", "NonExistentPattern")]
        filter_chain = ChapterFilter(operations)
        result = filter_chain.apply(sample_chapters)
        assert len(result) == 0

    def test_empty_chapter_list(self):
        """Test that empty chapter list returns empty list."""
        operations = [FilterOperation("preset", "all")]
        filter_chain = ChapterFilter(operations)
        result = filter_chain.apply([])
        assert len(result) == 0


class TestPresetRegistration:
    """Tests for custom preset registration."""

    def test_register_custom_preset(self):
        """Test that custom presets can be registered."""
        custom_preset = FilterPreset(
            name="Custom Filter",
            description="Only title pages",
            filter_fn=lambda t: t.is_title_page,
        )

        ChapterFilter.register_preset("custom", custom_preset)
        assert "custom" in ChapterFilter.PRESETS

        # Clean up
        del ChapterFilter.PRESETS["custom"]

    def test_registered_preset_can_be_used(self, sample_chapters):
        """Test that registered custom preset works in filtering."""
        custom_preset = FilterPreset(
            name="Custom Filter",
            description="Only chapters with numbers",
            filter_fn=lambda t: True,  # Include all
        )

        ChapterFilter.register_preset("custom", custom_preset)
        try:
            operations = [FilterOperation("preset", "custom")]
            filter_chain = ChapterFilter(operations)
            result = filter_chain.apply(sample_chapters)
            assert len(result) == len(sample_chapters)
        finally:
            # Clean up
            del ChapterFilter.PRESETS["custom"]
