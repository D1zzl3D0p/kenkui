"""Tests for kenkui.chapter_classifier.ChapterClassifier."""

from __future__ import annotations

import pytest

from kenkui.chapter_classifier import ChapterClassifier


class TestRegularChapter:
    def test_numbered_chapter(self):
        tags = ChapterClassifier.classify("Chapter 1: The Beginning")
        assert tags.is_chapter is True
        assert tags.is_front_matter is False
        assert tags.is_back_matter is False
        assert tags.is_title_page is False

    def test_simple_named_chapter(self):
        tags = ChapterClassifier.classify("The Dark Forest")
        assert tags.is_chapter is True

    def test_roman_numeral_chapter(self):
        tags = ChapterClassifier.classify("IV")
        assert tags.is_chapter is True

    def test_empty_string(self):
        tags = ChapterClassifier.classify("")
        assert tags.is_chapter is False

    def test_none_title(self):
        tags = ChapterClassifier.classify(None)
        assert tags.is_chapter is False


class TestFrontMatter:
    @pytest.mark.parametrize(
        "title",
        [
            "Acknowledgments",
            "Preface",
            "Introduction",
            "Contents",
            "Copyright Notice",
            "Dedication",
            "Foreword",
        ],
    )
    def test_front_matter_detected(self, title):
        tags = ChapterClassifier.classify(title)
        assert tags.is_front_matter is True
        assert tags.is_chapter is False


class TestBackMatter:
    @pytest.mark.parametrize(
        "title",
        [
            "References",
            "Index",
            "Bibliography",
            "Appendix A",
            "Notes",
            "Glossary",
        ],
    )
    def test_back_matter_detected(self, title):
        tags = ChapterClassifier.classify(title)
        assert tags.is_back_matter is True
        assert tags.is_chapter is False


class TestTitlePage:
    @pytest.mark.parametrize(
        "title",
        [
            "Title Page",
            "Cover",
            "Colophon",
        ],
    )
    def test_title_page_detected(self, title):
        tags = ChapterClassifier.classify(title)
        assert tags.is_title_page is True
        assert tags.is_chapter is False


class TestPartDividers:
    @pytest.mark.parametrize(
        "title",
        [
            "Part I",
            "Part II",
            "Part One",
            "Part Three",
            "Book First",
            "Book 2",
            "Volume IV",
        ],
    )
    def test_part_divider_detected(self, title):
        tags = ChapterClassifier.classify(title)
        assert tags.is_part_divider is True

    def test_part_divider_also_chapter(self):
        tags = ChapterClassifier.classify("Part I")
        # Part dividers are marked as both part_divider and chapter
        assert tags.is_part_divider is True
        assert tags.is_chapter is True


class TestAppliedTags:
    def test_get_applied_tags_chapter(self):
        tags = ChapterClassifier.classify("Chapter 5")
        applied = tags.get_applied_tags()
        assert "chapter" in applied

    def test_get_applied_tags_front_matter(self):
        tags = ChapterClassifier.classify("Acknowledgments")
        applied = tags.get_applied_tags()
        assert "front-matter" in applied
        assert "chapter" not in applied


class TestCustomPatterns:
    def setup_method(self):
        ChapterClassifier.reset_patterns()

    def teardown_method(self):
        ChapterClassifier.reset_patterns()

    def test_add_custom_front_matter_pattern(self):
        ChapterClassifier.add_custom_pattern("front_matter", r"(?i)^prologue", "prologue")
        tags = ChapterClassifier.classify("Prologue")
        assert tags.is_front_matter is True

    def test_add_custom_back_matter_pattern(self):
        ChapterClassifier.add_custom_pattern("back_matter", r"(?i)^epilogue", "epilogue")
        tags = ChapterClassifier.classify("Epilogue")
        assert tags.is_back_matter is True

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError):
            ChapterClassifier.add_custom_pattern("invalid", r".*", "desc")

    def test_reset_removes_custom_patterns(self):
        ChapterClassifier.add_custom_pattern("front_matter", r"(?i)^prologue", "prologue")
        ChapterClassifier.reset_patterns()
        # After reset, prologue should be treated as a regular chapter
        tags = ChapterClassifier.classify("Prologue")
        assert tags.is_front_matter is False
