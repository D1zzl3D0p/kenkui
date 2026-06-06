"""Pydantic schemas for LLM structured-output contracts.

Two layers:
  - Storage types (CharacterRecord, AttributionItem, …): full rich models used
    for caching, series tracking, and the app's internal data model.
  - Wire types (*Wire suffix): slim schemas sent to the LLM.  They omit fields
    that are server-computed (mention_count, quote_count), server-derived from
    context (chapters, first/last_appearance in compact mode), or intentionally
    excluded to save output tokens (emotion, char_start/end).  After each LLM
    call the wire result is converted back to the full storage type.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

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


# Speaker sentinel values that must never be slugified, remapped, or included
# in attribution counts.  A single canonical definition shared by all modules.
_SPEAKER_SENTINELS: frozenset[str] = frozenset({"NARRATOR", "Unknown"})


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


# Backward-compat alias — AliasGroup instances can still be constructed and imported.
# However, they cannot be added to CharacterRoster.characters (which requires CharacterRecord).
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
        default="neutral",
        description=(
            "One of: neutral, happy, sad, angry, fearful, surprised, disgusted"
        )
    )
    confidence: int = Field(
        default=3,
        description="Confidence 1-5: 1=very uncertain, 5=very confident"
    )
    char_start: int | None = None  # LLM echoes back position for verification
    char_end: int | None = None


class AttributionResult(BaseModel):
    """The LLM's attributions for all quotes in one chunk."""

    attributions: list[AttributionItem]


class StrictLLMWireModel(BaseModel):
    """Base for strict structured-output schemas sent to LLM providers."""

    model_config = ConfigDict(extra="forbid")


class CanonicalMergeEntry(StrictLLMWireModel):
    canonical: str
    duplicates: list[str]


class CanonicalMergeResult(StrictLLMWireModel):
    merges: list[CanonicalMergeEntry]


class EpithetMapping(StrictLLMWireModel):
    epithet: str
    canonical_name: str


class EpithetResolutionResult(StrictLLMWireModel):
    mappings: list[EpithetMapping]


class NameNormalizationEntry(StrictLLMWireModel):
    original: str
    simplified: str


class NameNormalizationResult(StrictLLMWireModel):
    names: list[NameNormalizationEntry]


# ---------------------------------------------------------------------------
# Wire types — slim schemas sent to the LLM
# ---------------------------------------------------------------------------


class TitleWire(StrictLLMWireModel):
    """Slim title type for LLM extraction.

    The full TitleRecord stores chapters/book_slug for series tracking; the LLM
    only needs to identify the title string itself.  Chapter scope is back-filled
    from context after extraction.
    """
    title: str = Field(description="Title or honorific, e.g. 'Queen of Andor', 'Mr.', 'The Dark One'")


class CharacterRecordWire(StrictLLMWireModel):
    """Wire schema for LLM roster extraction (compact mode — default).

    Omits: chapters, first/last_appearance (server-derived), mention_count,
    quote_count (server-computed).  TitleRecord is simplified to TitleWire
    (title string only; chapter scope back-filled from context).
    """
    slug: str = Field(description="Stable underscore slug, e.g. 'elizabeth_bennet'")
    canonical_name: str = Field(description="Most complete / formal name form")
    aliases: list[str] = Field(default_factory=list, description="All name variants for this character")
    titles: list[TitleWire] = Field(default_factory=list, description="Positional titles or honorifics")
    gender: str = Field(default="", description="Pronouns, e.g. 'he/him', 'she/her', 'they/them'")
    role: str = Field(default="", description="protagonist, antagonist, supporting, or minor")
    description: str = Field(default="", description="One-line summary (omit for minor/supporting)")

    @field_validator("titles", mode="before")
    @classmethod
    def _coerce_titles(cls, v: object) -> object:
        """Coerce bare title strings to TitleWire-compatible dicts.

        LLMs occasionally return ``titles: ["Mr.", "DCI"]`` despite the prompt
        requesting objects.  This validator normalises bare strings so a format
        deviation doesn't crash the entire roster extraction.
        """
        if isinstance(v, list):
            return [{"title": item} if isinstance(item, str) else item for item in v]
        return v


class CharacterRosterWire(StrictLLMWireModel):
    """Wire container for compact roster extraction."""
    characters: list[CharacterRecordWire] = Field(default_factory=list)

    @field_validator("characters", mode="before")
    @classmethod
    def _coerce_character_titles(cls, v: object) -> object:
        """Coerce bare title strings to TitleWire objects in each character.

        Fallback coercion in case the validator on CharacterRecordWire is not applied
        in all scenarios (e.g., when using instructor's JSON parsing path).
        """
        if isinstance(v, list):
            result = []
            for char in v:
                if isinstance(char, dict) and "titles" in char:
                    char["titles"] = [
                        {"title": t} if isinstance(t, str) else t
                        for t in char["titles"]
                    ]
                result.append(char)
            return result
        return v


class CharacterRecordFullWire(StrictLLMWireModel):
    """Wire schema for LLM roster extraction (full mode — non-default).

    Includes chapter position fields but uses plain lists (not tuples) to avoid
    JSON parse errors.  Still omits mention_count/quote_count (server-computed).
    """
    slug: str = Field(description="Stable underscore slug, e.g. 'elizabeth_bennet'")
    canonical_name: str = Field(description="Most complete / formal name form")
    aliases: list[str] = Field(default_factory=list, description="All name variants for this character")
    titles: list[TitleWire] = Field(default_factory=list, description="Positional titles or honorifics")
    gender: str = Field(default="", description="Pronouns, e.g. 'he/him', 'she/her', 'they/them'")
    role: str = Field(default="", description="protagonist, antagonist, supporting, or minor")
    description: str = Field(default="", description="One-line summary (omit for minor/supporting)")
    chapters: list[int] = Field(default_factory=list, description="Chapter indices this character appears in")
    first_appearance: list | None = Field(default=None, description="[null, chapter_index] for this book")
    last_appearance: list | None = Field(default=None, description="[null, chapter_index] for this book")

    @field_validator("titles", mode="before")
    @classmethod
    def _coerce_titles(cls, v: object) -> object:
        if isinstance(v, list):
            return [{"title": item} if isinstance(item, str) else item for item in v]
        return v


class CharacterRosterFullWire(StrictLLMWireModel):
    """Wire container for full (non-compact) roster extraction."""
    characters: list[CharacterRecordFullWire] = Field(default_factory=list)

    @field_validator("characters", mode="before")
    @classmethod
    def _coerce_character_titles(cls, v: object) -> object:
        """Coerce bare title strings to TitleWire objects in each character."""
        if isinstance(v, list):
            result = []
            for char in v:
                if isinstance(char, dict) and "titles" in char:
                    char["titles"] = [
                        {"title": t} if isinstance(t, str) else t
                        for t in char["titles"]
                    ]
                result.append(char)
            return result
        return v


class AttributionItemWire(StrictLLMWireModel):
    """Minimal attribution wire — short field names minimise LLM output token count.

    Fields intentionally abbreviated: 'q' = quote_id, 's' = speaker slug.
    Confidence and emotion are omitted — not used downstream and cost tokens.
    """
    q: int = Field(description="The N from [QUOTE:N]")
    s: str = Field(description="Character slug, 'NARRATOR', or 'Unknown'")


class AttributionResultWire(StrictLLMWireModel):
    """Wire container for slim attribution results. Field 'a' is abbreviated to save tokens."""
    a: list[AttributionItemWire]


# ---------------------------------------------------------------------------
# Wire → storage type conversions
# ---------------------------------------------------------------------------


def _title_wire_to_record(t: TitleWire) -> TitleRecord:
    return TitleRecord(title=t.title)


def _char_wire_to_record(w: CharacterRecordWire) -> CharacterRecord:
    return CharacterRecord(
        slug=w.slug,
        canonical_name=w.canonical_name,
        aliases=w.aliases,
        titles=[_title_wire_to_record(t) for t in w.titles],
        gender=w.gender,
        role=w.role,
        description=w.description,
    )


def _char_full_wire_to_record(w: CharacterRecordFullWire) -> CharacterRecord:
    first = None
    if w.first_appearance and len(w.first_appearance) == 2:
        first = (w.first_appearance[0], int(w.first_appearance[1]))
    last = None
    if w.last_appearance and len(w.last_appearance) == 2:
        last = (w.last_appearance[0], int(w.last_appearance[1]))
    return CharacterRecord(
        slug=w.slug,
        canonical_name=w.canonical_name,
        aliases=w.aliases,
        titles=[_title_wire_to_record(t) for t in w.titles],
        gender=w.gender,
        role=w.role,
        description=w.description,
        chapters=w.chapters,
        first_appearance=first,
        last_appearance=last,
    )


def roster_wire_to_full(wire: CharacterRosterWire | CharacterRosterFullWire) -> CharacterRoster:
    """Convert a wire roster response to a full CharacterRoster."""
    if isinstance(wire, CharacterRosterFullWire):
        return CharacterRoster(characters=[_char_full_wire_to_record(c) for c in wire.characters])
    return CharacterRoster(characters=[_char_wire_to_record(c) for c in wire.characters])


def attribution_wire_to_full(wire: AttributionResultWire) -> AttributionResult:
    """Convert a slim wire attribution to a full AttributionResult."""
    return AttributionResult(attributions=[
        AttributionItem(quote_id=w.q, speaker=w.s)
        for w in wire.a
    ])
