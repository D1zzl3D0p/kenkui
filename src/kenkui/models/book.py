from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .common import ChapterPreset

if TYPE_CHECKING:
    from ..nlp.models import TitleRecord
@dataclass
class CharacterInfo:
    """Metadata for a character identified by the NLP pipeline.

    Used for per-character voice assignment in multi-voice mode.
    Not persisted directly in JobConfig — only the resulting
    ``speaker_voices`` mapping is stored.
    """

    character_id: str  # Canonical name, e.g. "Elizabeth Bennet"
    display_name: str  # Human-readable label shown in the UI
    quote_count: int = 0
    mention_count: int = 0  # Name occurrences in full text; populated by fast scan
    gender_pronoun: str = ""  # "he", "she", "they", etc. (optional)

    @property
    def prominence(self) -> int:
        """Best available count for sorting — prefer mention_count, fall back to quote_count."""
        return self.mention_count or self.quote_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "character_id": self.character_id,
            "display_name": self.display_name,
            "quote_count": self.quote_count,
            "mention_count": self.mention_count,
            "gender_pronoun": self.gender_pronoun,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CharacterInfo:
        return cls(
            character_id=data["character_id"],
            display_name=data.get("display_name", data["character_id"]),
            quote_count=data.get("quote_count", 0),
            mention_count=data.get("mention_count", 0),
            gender_pronoun=data.get("gender_pronoun", ""),
        )


@dataclass
class CharacterRecord:
    """Rich app-layer character representation. Built from NLP pipeline output.

    The ``slug`` is the canonical lookup key used in ``JobConfig.speaker_voices``.
    ``CharacterInfo`` is the slimmer UI-facing view, built via ``to_character_info()``.
    """

    slug: str
    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    titles: list[TitleRecord] = field(default_factory=list)
    gender: str = ""
    role: str = ""
    description: str = ""
    chapters: list[int] = field(default_factory=list)
    first_appearance: tuple[str, int] | None = None   # (book_slug, chapter_index)
    last_appearance: tuple[str, int] | None = None
    mention_count: int = 0
    quote_count: int = 0

    @classmethod
    def from_nlp(cls, record: Any) -> CharacterRecord:
        """Build from a kenkui.nlp.models.CharacterRecord Pydantic object."""
        return cls(
            slug=record.slug,
            canonical_name=record.canonical_name,
            aliases=list(record.aliases),
            titles=list(record.titles),
            gender=record.gender,
            role=record.role,
            description=record.description,
            chapters=list(record.chapters),
            first_appearance=record.first_appearance,
            last_appearance=record.last_appearance,
            mention_count=record.mention_count,
            quote_count=record.quote_count,
        )

    def to_character_info(self) -> CharacterInfo:
        """Build a CharacterInfo for the UI / voice assignment layer."""
        pronoun = self.gender.split("/")[0] if self.gender else ""
        return CharacterInfo(
            character_id=self.slug,
            display_name=self.canonical_name,
            quote_count=self.quote_count,
            mention_count=self.mention_count,
            gender_pronoun=pronoun,
        )


@dataclass
class BookInfo:
    """Lightweight book info for display in selection UI."""

    path: Path
    title: str
    author: str | None = None
    chapter_count: int = 0
    format: str = ""

    @property
    def display_name(self) -> str:
        return self.title or self.path.stem

    @property
    def display_author(self) -> str:
        return self.author or "Unknown"


@dataclass
class ChapterSelection:
    preset: ChapterPreset = ChapterPreset.CONTENT_ONLY
    included: list[int] = field(default_factory=list)
    excluded: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset": self.preset.value,
            "included": self.included,
            "excluded": self.excluded,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChapterSelection:
        return cls(
            preset=ChapterPreset(data.get("preset", "content-only")),
            included=data.get("included", []),
            excluded=data.get("excluded", []),
        )


