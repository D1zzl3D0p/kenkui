"""Structural boundaries and the gap model that turns them into silence.

Between any two adjacent pieces there is exactly one gap, and a gap may have
several reasons. Its duration is the maximum of them, never the sum, so a
chapter boundary meeting a chapter title's leading pause cannot compound into
one long hole. Modelling gaps rather than per-piece durations is what makes
that impossible by construction instead of by a rule someone must remember.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

STRUCTURE_SCHEMA_VERSION = "epub-structure-v1"

CHAPTER = "chapter"
HEADING_BEFORE = "heading_before"
HEADING_AFTER = "heading_after"
PARAGRAPH = "paragraph"
LINE = "line"

_BLOCK = re.compile(r"\n{2,}")


class PauseSpec(Protocol):
    """The five independently variable pause durations, in milliseconds."""

    @property
    def chapter_ms(self) -> int:
        """Silence between chapters."""
        ...

    @property
    def heading_before_ms(self) -> int:
        """Silence before a heading."""
        ...

    @property
    def heading_after_ms(self) -> int:
        """Silence after a heading."""
        ...

    @property
    def paragraph_ms(self) -> int:
        """Silence at a block boundary."""
        ...

    @property
    def line_ms(self) -> int:
        """Silence at a single line break."""
        ...


@dataclass(frozen=True, slots=True)
class Piece:
    """One structural run of canonical text plus the reasons for the gap after it."""

    text: str
    reasons: frozenset[str]


def break_tiers(pauses: PauseSpec) -> tuple[str, ...]:
    """Return which tiers force a chunk break, derived from non-zero durations.

    ``chapter_ms`` is deliberately absent: segment compilation already iterates
    chapter by chapter, so a chapter edge is inherently a segment boundary.
    Including it would engage the v3 chunker, changing every segment identity
    and invalidating every cache entry while changing no segment text at all.
    """
    tiers: list[str] = []
    if pauses.heading_before_ms or pauses.heading_after_ms:
        tiers.append("heading")
    if pauses.paragraph_ms:
        tiers.append(PARAGRAPH)
    if pauses.line_ms:
        tiers.append(LINE)
    return tuple(sorted(tiers))


def gap_ms(reasons: frozenset[str], pauses: PauseSpec) -> int:
    """Return one gap's duration: the maximum of its reasons, never the sum."""
    durations = {
        CHAPTER: pauses.chapter_ms,
        HEADING_BEFORE: pauses.heading_before_ms,
        HEADING_AFTER: pauses.heading_after_ms,
        PARAGRAPH: pauses.paragraph_ms,
        LINE: pauses.line_ms,
    }
    return max((durations[reason] for reason in reasons), default=0)


def _blocks(text: str) -> list[tuple[str, str]]:
    """Return (body, body-plus-separator) pairs so joining stays exact."""
    out: list[tuple[str, str]] = []
    position = 0
    for match in _BLOCK.finditer(text):
        out.append((text[position : match.start()], text[position : match.end()]))
        position = match.end()
    if position < len(text) or not out:
        out.append((text[position:], text[position:]))
    return out


def _lines(chunk: str, body: str) -> list[str]:
    """Split a block on single newlines, keeping its trailing separator last."""
    separator = chunk[len(body) :]
    parts: list[str] = []
    position = 0
    for match in re.finditer(r"\n", body):
        parts.append(body[position : match.end()])
        position = match.end()
    parts.append(body[position:])
    parts = [part for part in parts if part] or [""]
    parts[-1] = f"{parts[-1]}{separator}"
    return parts


def split_structural(
    text: str, headings: frozenset[str], pauses: PauseSpec
) -> tuple[Piece, ...]:
    """Split canonical text at the boundaries that carry a non-zero pause.

    Runs in canonical coordinates, before spoken form, so no offset map is
    needed: spoken form is applied to each piece afterwards and cannot move a
    boundary that has already been decided.
    """
    if not break_tiers(pauses) or not text:
        return (Piece(text, frozenset()),)
    blocks = _blocks(text)
    pieces: list[Piece] = []
    for index, (body, chunk) in enumerate(blocks):
        last = index + 1 == len(blocks)
        reasons: set[str] = set()
        if pauses.paragraph_ms and not last:
            reasons.add(PARAGRAPH)
        if pauses.heading_after_ms and body in headings and not last:
            reasons.add(HEADING_AFTER)
        if pauses.heading_before_ms and not last and blocks[index + 1][0] in headings:
            reasons.add(HEADING_BEFORE)
        if pauses.line_ms:
            parts = _lines(chunk, body)
            for order, part in enumerate(parts):
                tail = order + 1 == len(parts)
                pieces.append(
                    Piece(part, frozenset(reasons) if tail else frozenset({LINE}))
                )
        else:
            pieces.append(Piece(chunk, frozenset(reasons)))
    return tuple(piece for piece in pieces if piece.text)
