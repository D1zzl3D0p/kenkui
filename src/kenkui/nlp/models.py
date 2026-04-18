"""Pydantic schemas for LLM structured-output contracts.

These are the *wire* types passed to and from the LLM.  They are intentionally
separate from the core ``kenkui.models`` dataclasses so that the LLM layer
stays self-contained.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Slug utility
# ---------------------------------------------------------------------------


def slugify(name: str) -> str:
    """Convert a character name to a stable, lowercase, underscore-separated slug.

    Examples:
        "Elizabeth Bennet" -> "elizabeth_bennet"
        "Rand al'Thor"     -> "rand_althor"
        "Mr. Darcy"        -> "mr_darcy"
    """
    s = name.lower()
    s = re.sub(r"['\u2018\u2019]", "", s)   # remove apostrophes
    s = re.sub(r"[^a-z0-9]+", "_", s)        # non-alphanumeric → underscore
    return s.strip("_")


# ---------------------------------------------------------------------------
# Stage 1 — Quote extraction (pure Python, no LLM)
# ---------------------------------------------------------------------------


class Quote(BaseModel):
    """A single dialogue quote or italic span extracted by the regex pass."""

    id: int
    text: str  # Includes quote marks for dialogue; plain content (no markers) for italic
    para_index: int  # Which paragraph (0-based) this quote lives in
    char_offset: int  # Byte offset within the full chapter text
    kind: str = "dialogue"  # "dialogue" | "italic"


# ---------------------------------------------------------------------------
# Stage 2 — Character roster
# ---------------------------------------------------------------------------


class TitleRecord(BaseModel):
    """A title or honorific held by a character, with chapter/book scope.

    Titles can transfer between characters across a series (e.g. "Queen of Andor"),
    so chapter ranges and book scope are stored per-character.
    """

    title: str = Field(description="Title or honorific, e.g. 'Queen of Andor', 'Mr.', 'The Dark Lord'")
    chapters: list[int] = Field(default_factory=list, description="Chapter indices where this character holds this title")
    book_slug: str | None = Field(default=None, description="Book scope slug; None = current book only")


class CharacterRecord(BaseModel):
    """One character with full rich metadata. Replaces AliasGroup."""

    slug: str = Field(description="Stable underscore slug derived from canonical_name, e.g. 'elizabeth_bennet'")
    canonical_name: str = Field(description="Most complete / formal name form")
    aliases: list[str] = Field(default_factory=list, description="All name variants that refer to this character")
    titles: list[TitleRecord] = Field(default_factory=list, description="Positional titles with chapter and book scope")
    gender: str = Field(default="", description="Pronouns, e.g. 'he/him', 'she/her', 'they/them'")
    role: str = Field(default="", description="One of: protagonist, antagonist, supporting, minor")
    description: str = Field(default="", description="One-line LLM-generated character summary")
    chapters: list[int] = Field(default_factory=list, description="Chapter indices this character appears in")
    first_appearance: tuple[str, int] | None = Field(default=None, description="(book_slug, chapter_index)")
    last_appearance: tuple[str, int] | None = Field(default=None, description="(book_slug, chapter_index)")
    mention_count: int = 0
    quote_count: int = 0


# Backward-compat alias — existing code that imports AliasGroup continues to work.
# New code should import CharacterRecord directly.
class AliasGroup(BaseModel):
    """Deprecated — use CharacterRecord. Kept for backward compatibility."""
    canonical: str = Field(description="Most complete / formal name form")
    aliases: list[str] = Field(description="All name variants that refer to this character")
    gender: str = ""


class CharacterRoster(BaseModel):
    """Result of character extraction for a whole book."""

    characters: list[CharacterRecord] = Field(default_factory=list)

    def by_slug(self, slug: str) -> CharacterRecord | None:
        """Return the character with the given slug, or None."""
        return next((c for c in self.characters if c.slug == slug), None)

    def all_slugs(self) -> list[str]:
        """Return all character slugs in roster order."""
        return [c.slug for c in self.characters]


# ---------------------------------------------------------------------------
# Stage 4 — Speaker attribution
# ---------------------------------------------------------------------------


class AttributionItem(BaseModel):
    """Attribution for a single regex-extracted quote."""

    quote_id: int
    speaker: str = Field(
        description=(
            "Character slug from the roster (e.g. 'elizabeth_bennet'), "
            "or 'NARRATOR', or 'Unknown'"
        )
    )
    emotion: str = Field(
        description=(
            "One of: neutral, happy, sad, angry, fearful, surprised, disgusted"
        )
    )
    confidence: int = Field(
        default=3,
        description="Confidence 1-5: 1=very uncertain, 5=very confident"
    )


class AttributionResult(BaseModel):
    """The LLM's attributions for all quotes in one chunk."""

    attributions: list[AttributionItem]


class CanonicalMergeEntry(BaseModel):
    canonical: str
    duplicates: list[str]

class CanonicalMergeResult(BaseModel):
    merges: list[CanonicalMergeEntry]

class EpithetMapping(BaseModel):
    epithet: str
    canonical_name: str

class EpithetResolutionResult(BaseModel):
    mappings: list[EpithetMapping]

class NameNormalizationEntry(BaseModel):
    original: str
    simplified: str

class NameNormalizationResult(BaseModel):
    names: list[NameNormalizationEntry]
