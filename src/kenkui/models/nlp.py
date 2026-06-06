from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .book import CharacterInfo

if TYPE_CHECKING:
    from ..chapter_classifier import ChapterTags
    from ..nlp.models import CharacterRoster
@dataclass
class Segment:
    """A unit of attributed speech within a chapter.

    Populated by the NLP pipeline during multi-voice tagging.
    When ``Chapter.segments`` is ``None`` the single-voice
    paragraph rendering path is used unchanged.
    """

    text: str
    speaker: str = "NARRATOR"  # voice name, character id, or "NARRATOR"
    index: int = 0  # original position in the chapter
    is_scene_break: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "speaker": self.speaker,
            "index": self.index,
            "is_scene_break": self.is_scene_break,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Segment:
        return cls(
            text=data["text"],
            speaker=data.get("speaker", "NARRATOR"),
            index=data.get("index", 0),
            is_scene_break=data.get("is_scene_break", False),
        )


def _default_chapter_tags() -> ChapterTags:
    from ..chapter_classifier import ChapterTags

    return ChapterTags(is_chapter=True)


@dataclass
class Chapter:
    index: int
    title: str
    paragraphs: list[str]
    tags: ChapterTags = field(default_factory=_default_chapter_tags)
    toc_index: int = 0
    segments: list[Segment] | None = None  # Populated by NLP pipeline for multi-voice

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "index": self.index,
            "title": self.title,
            "paragraphs": self.paragraphs,
            "toc_index": self.toc_index,
        }
        if self.segments is not None:
            d["segments"] = [s.to_dict() for s in self.segments]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chapter:
        segments: list[Segment] | None = None
        if "segments" in data and data["segments"] is not None:
            segments = [Segment.from_dict(s) for s in data["segments"]]
        return cls(
            index=data["index"],
            title=data["title"],
            paragraphs=data.get("paragraphs", []),
            toc_index=data.get("toc_index", 0),
            segments=segments,
        )


@dataclass
class NLPResult:
    """Output of an NLP analysis run (speaker attribution pipeline).

    Replaces the old BookNLPResult.  The ``chapters`` list has
    ``Chapter.segments`` populated for every chapter.  The cache JSON
    written to disk has the shape ``{"characters": [...], "chapters": [...],
    "book_hash": "..."}`` so that ``_load_annotated_chapters`` in
    ``parsing.py`` can reconstruct the chapters via ``Chapter.from_dict``.
    """

    characters: list[CharacterInfo]
    chapters: list[Chapter]
    book_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "characters": [c.to_dict() for c in self.characters],
            "chapters": [ch.to_dict() for ch in self.chapters],
            "book_hash": self.book_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NLPResult:
        return cls(
            characters=[CharacterInfo.from_dict(c) for c in data.get("characters", [])],
            chapters=[Chapter.from_dict(ch) for ch in data.get("chapters", [])],
            book_hash=data.get("book_hash", ""),
        )


@dataclass
class FastScanResult:
    """Result of the Stage 1-2 fast scan (no LLM attribution).

    Contains the character roster with name-mention counts. Used by the wizard
    for voice assignment and passed to the worker via ``roster_cache_path`` so
    Stage 3-4 attribution can use the same canonical names.
    """

    roster: CharacterRoster  # Pydantic model from nlp.models
    characters: list[CharacterInfo]  # Sorted by mention_count descending
    book_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "book_hash": self.book_hash,
            "roster": self.roster.model_dump(),
            "characters": [c.to_dict() for c in self.characters],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FastScanResult:
        from ..nlp.models import CharacterRoster, slugify

        roster_data = data.get("roster", {"characters": []})

        # Schema migration: handle old cache format (pre-slug generation)
        # Old format: {"characters": [{"canonical": "...", "aliases": [...], "gender": "..."}]}
        # New format: {"characters": [{"slug": "...", "canonical_name": "...", ...}]}
        if roster_data.get("characters"):
            first_char = roster_data["characters"][0]
            if "canonical" in first_char and "slug" not in first_char:
                # Legacy format detected - migrate to new schema
                for char in roster_data["characters"]:
                    # Generate slug from canonical name
                    char["slug"] = slugify(char.get("canonical", ""))
                    # Rename canonical -> canonical_name
                    char["canonical_name"] = char.pop("canonical")
                    # Set default values for missing fields
                    char.setdefault("role", "")
                    char.setdefault("description", "")
                    char.setdefault("titles", [])
                    char.setdefault("chapters", [])
                    char.setdefault("first_appearance", None)
                    char.setdefault("last_appearance", None)
                    char.setdefault("mention_count", 0)
                    char.setdefault("quote_count", 0)

        return cls(
            book_hash=data.get("book_hash", ""),
            roster=CharacterRoster.model_validate(roster_data),
            characters=[CharacterInfo.from_dict(c) for c in data.get("characters", [])],
        )


