"""
Chapter classification system - replaces skip_patterns with tag-based filtering.
Tags allow users to understand WHY a chapter was filtered and give them control.
"""

from dataclasses import dataclass
from typing import Optional
import re


@dataclass
class ChapterTags:
    """Tags describing chapter type for intelligent filtering."""

    is_front_matter: bool = False  # Acknowledgments, Preface, Introduction
    is_back_matter: bool = False  # References, Index, Bibliography
    is_title_page: bool = False  # Title, Copyright pages
    is_part_divider: bool = False  # "Part I", "Book One"
    is_chapter: bool = True  # Main content chapter

    def get_applied_tags(self) -> list[str]:
        """Return list of human-readable tags."""
        tags = []
        if self.is_front_matter:
            tags.append("front-matter")
        if self.is_back_matter:
            tags.append("back-matter")
        if self.is_title_page:
            tags.append("title-page")
        if self.is_part_divider:
            tags.append("part-divider")
        if self.is_chapter:
            tags.append("chapter")
        return tags


class ChapterClassifier:
    """Classifies chapters based on title patterns."""

    # Class-level patterns (extensible via config)
    FRONT_MATTER_PATTERNS = [
        (r"(?i)^acknowledgments", "acknowledgments"),
        (r"(?i)^preface", "preface"),
        (
            r"(?i)introduction",
            "introduction",
        ),  # Match anywhere (e.g., "1. Introduction")
        (r"(?i)^contents", "table of contents"),
        (r"(?i)^copyright", "copyright"),
        (r"(?i)^dedication", "dedication"),
        (r"(?i)^foreword", "foreword"),
    ]

    BACK_MATTER_PATTERNS = [
        (r"(?i)^references", "references"),
        (r"(?i)^index", "index"),
        (r"(?i)^bibliography", "bibliography"),
        (r"(?i)^appendix", "appendix"),
        (r"(?i)^notes", "notes"),
        (r"(?i)^glossary", "glossary"),
    ]

    TITLE_PAGE_PATTERNS = [
        (r"(?i)^title", "title page"),
        (r"(?i)^cover", "cover page"),
        (r"(?i)^colophon", "colophon"),
    ]

    PART_DIVIDER_PATTERNS = [
        (r"(?i)^part\s+[\divxlc]+", "part divider"),  # Part I, Part II, Part 1, etc.
        (
            r"(?i)^part\s+(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|one|two|three|four|five)",
            "part divider",
        ),
        (
            r"(?i)^book\s+(first|second|third|fourth|fifth|sixth|"
            r"seventh|eighth|ninth|tenth|one|two|three|four|five|\d+|[ivx]+)",
            "book divider",
        ),
        (r"(?i)^volume\s+[\divxlc]+", "volume divider"),
        (
            r"(?i)^volume\s+(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|one|two|three|four|five)",
            "volume divider",
        ),
    ]

    @classmethod
    def classify(cls, title: Optional[str]) -> ChapterTags:
        """Classify a chapter based on its title."""
        if not title:
            return ChapterTags(is_chapter=False)

        title_lower = title.lower().strip()
        tags = ChapterTags()

        # Check front matter
        for pattern, _ in cls.FRONT_MATTER_PATTERNS:
            if re.search(pattern, title_lower):
                tags.is_front_matter = True
                tags.is_chapter = False
                return tags

        # Check back matter
        for pattern, _ in cls.BACK_MATTER_PATTERNS:
            if re.search(pattern, title_lower):
                tags.is_back_matter = True
                tags.is_chapter = False
                return tags

        # Check title pages
        for pattern, _ in cls.TITLE_PAGE_PATTERNS:
            if re.search(pattern, title_lower):
                tags.is_title_page = True
                tags.is_chapter = False
                return tags

        # Check part/book/volume dividers
        for pattern, _ in cls.PART_DIVIDER_PATTERNS:
            if re.search(pattern, title_lower):
                tags.is_part_divider = True
                # Part dividers can also be chapters (for navigation)
                return tags

        # Default: it's a regular chapter
        return tags

    @classmethod
    def add_custom_pattern(cls, category: str, pattern: str, description: str = ""):
        """Add a custom pattern to the classifier.

        Args:
            category: One of 'front_matter', 'back_matter', 'title_page', 'part_divider'
            pattern: Regex pattern string
            description: Optional description of the pattern
        """
        tuple_val = (pattern, description or pattern)

        if category == "front_matter":
            cls.FRONT_MATTER_PATTERNS.append(tuple_val)
        elif category == "back_matter":
            cls.BACK_MATTER_PATTERNS.append(tuple_val)
        elif category == "title_page":
            cls.TITLE_PAGE_PATTERNS.append(tuple_val)
        elif category == "part_divider":
            cls.PART_DIVIDER_PATTERNS.append(tuple_val)
        else:
            raise ValueError(f"Unknown category: {category}")

    @classmethod
    def reset_patterns(cls):
        """Reset all patterns to defaults (useful for testing)."""
        cls.FRONT_MATTER_PATTERNS = [
            (r"(?i)^acknowledgments", "acknowledgments"),
            (r"(?i)^preface", "preface"),
            (r"(?i)introduction", "introduction"),
            (r"(?i)^contents", "table of contents"),
            (r"(?i)^copyright", "copyright"),
            (r"(?i)^dedication", "dedication"),
            (r"(?i)^foreword", "foreword"),
        ]

        cls.BACK_MATTER_PATTERNS = [
            (r"(?i)^references", "references"),
            (r"(?i)^index", "index"),
            (r"(?i)^bibliography", "bibliography"),
            (r"(?i)^appendix", "appendix"),
            (r"(?i)^notes", "notes"),
            (r"(?i)^glossary", "glossary"),
        ]

        cls.TITLE_PAGE_PATTERNS = [
            (r"(?i)^title", "title page"),
            (r"(?i)^cover", "cover page"),
            (r"(?i)^colophon", "colophon"),
        ]

        cls.PART_DIVIDER_PATTERNS = [
            (r"(?i)^part\s+[\divxlc]+", "part divider"),
            (
                r"(?i)^part\s+(first|second|third|fourth|fifth|sixth|"
                r"seventh|eighth|ninth|tenth|one|two|three|four|five)",
                "part divider",
            ),
            (
                r"(?i)^book\s+(first|second|third|fourth|fifth|sixth|"
                r"seventh|eighth|ninth|tenth|one|two|three|four|five|\d+|[ivx]+)",
                "book divider",
            ),
            (r"(?i)^volume\s+[\divxlc]+", "volume divider"),
            (
                r"(?i)^volume\s+(first|second|third|fourth|fifth|sixth|"
                r"seventh|eighth|ninth|tenth|one|two|three|four|five)",
                "volume divider",
            ),
        ]


__all__ = ["ChapterTags", "ChapterClassifier"]
