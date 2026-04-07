"""Pydantic schemas for LLM structured-output contracts.

These are the *wire* types passed to and from the local LLM via Ollama's
``format`` parameter.  They are intentionally separate from the core
``kenkui.models`` dataclasses so that the LLM layer stays self-contained.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


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
# Stage 2 — Alias clustering
# ---------------------------------------------------------------------------


class AliasGroup(BaseModel):
    """One character with all their aliases grouped under a canonical name."""

    canonical: str = Field(description="Most complete / formal name form")
    aliases: list[str] = Field(description="All name variants that refer to this character")
    gender: str = ""


class CharacterRoster(BaseModel):
    """Result of the alias-clustering LLM call for a whole book."""

    characters: list[AliasGroup]


# ---------------------------------------------------------------------------
# Stage 4 — Speaker attribution
# ---------------------------------------------------------------------------


class AttributionItem(BaseModel):
    """Attribution for a single regex-extracted quote."""

    quote_id: int
    speaker: str = Field(
        description=(
            "Canonical character name from the roster, or 'NARRATOR', or 'Unknown'"
        )
    )
    emotion: str = Field(
        description=(
            "One of: neutral, happy, sad, angry, fearful, surprised, disgusted"
        )
    )
    confidence: int = Field(
        default=3,
        description=(
            "Confidence 1-5: 1=very uncertain, 5=very confident"
        )
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
