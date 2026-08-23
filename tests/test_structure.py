"""Structural split exactness, break derivation, and the gap model."""

from __future__ import annotations

import random
from dataclasses import dataclass

import pytest

from kenkui._domain.structure import (
    CHAPTER,
    HEADING_AFTER,
    HEADING_BEFORE,
    LINE,
    break_tiers,
    gap_ms,
    split_structural,
)


@dataclass(frozen=True, slots=True)
class Spec:
    """Stand-in for the Pauses operation record."""

    chapter_ms: int = 0
    heading_before_ms: int = 0
    heading_after_ms: int = 0
    paragraph_ms: int = 0
    line_ms: int = 0


HEADINGS = frozenset({"Chapter One", "A Section"})
CHAPTER_MS = 1500
HEADING_MS = 400


def test_no_tier_yields_exactly_one_piece() -> None:
    """With nothing enabled the split is a no-op, which is what keeps v2 alive."""
    text = "Chapter One\n\nHe woke.\n\nShe slept."
    pieces = split_structural(text, HEADINGS, Spec())
    assert len(pieces) == 1
    assert pieces[0].text == text
    assert pieces[0].reasons == frozenset()


def test_chapter_ms_alone_does_not_split() -> None:
    """Chapter edges are already segment boundaries, so v3 must not engage."""
    text = "Chapter One\n\nHe woke."
    assert break_tiers(Spec(chapter_ms=1500)) == ()
    assert len(split_structural(text, HEADINGS, Spec(chapter_ms=1500))) == 1


def test_heading_after_marks_the_heading_piece() -> None:
    """The gap after a heading rides the piece containing it."""
    pieces = split_structural(
        "Chapter One\n\nHe woke.\n\nShe slept.",
        HEADINGS,
        Spec(heading_after_ms=300),
    )
    assert pieces[0].text == "Chapter One\n\n"
    assert HEADING_AFTER in pieces[0].reasons


def test_heading_before_marks_the_preceding_piece() -> None:
    """A pause before a heading attaches to the piece that ends before it."""
    pieces = split_structural(
        "Intro line.\n\nChapter One\n\nBody.",
        HEADINGS,
        Spec(heading_before_ms=400),
    )
    assert pieces[0].text == "Intro line.\n\n"
    assert HEADING_BEFORE in pieces[0].reasons


def test_line_tier_splits_inside_a_block() -> None:
    """Verse breathes only if single newlines become boundaries."""
    pieces = split_structural("one\ntwo\nthree", frozenset(), Spec(line_ms=100))
    assert [piece.text for piece in pieces] == ["one\n", "two\n", "three"]
    assert pieces[0].reasons == frozenset({LINE})
    assert pieces[-1].reasons == frozenset()


def test_final_piece_carries_no_internal_reason() -> None:
    """The last piece's gap belongs to the chapter tier, not the paragraph tier."""
    pieces = split_structural("A.\n\nB.", frozenset(), Spec(paragraph_ms=200))
    assert pieces[-1].reasons == frozenset()


def test_gap_takes_the_maximum_not_the_sum() -> None:
    """A chapter end meeting a heading-before pause must not stack."""
    spec = Spec(chapter_ms=CHAPTER_MS, heading_before_ms=HEADING_MS)
    assert gap_ms(frozenset({CHAPTER, HEADING_BEFORE}), spec) == CHAPTER_MS
    assert gap_ms(frozenset({CHAPTER}), spec) == CHAPTER_MS
    assert gap_ms(frozenset({HEADING_BEFORE}), spec) == HEADING_MS
    assert gap_ms(frozenset(), spec) == 0


def test_break_tiers_is_derived_from_nonzero_durations() -> None:
    """Turning a tier to zero removes its chunking cost entirely."""
    assert break_tiers(Spec()) == ()
    assert break_tiers(Spec(paragraph_ms=1)) == ("paragraph",)
    assert break_tiers(Spec(line_ms=1)) == ("line",)
    assert break_tiers(Spec(heading_before_ms=1)) == ("heading",)
    assert break_tiers(Spec(heading_after_ms=1)) == ("heading",)
    assert break_tiers(Spec(paragraph_ms=1, line_ms=1)) == ("line", "paragraph")


@pytest.mark.parametrize("seed", range(20))
def test_split_is_exact_over_random_text(seed: int) -> None:
    """Concatenating every piece must reproduce the input character for character.

    A gap here silently drops audio and an overlap silently duplicates it, so
    this property is checked by construction rather than by example.
    """
    rng = random.Random(seed)  # noqa: S311 - deterministic fixture, not crypto
    alphabet = ["a", "b", " ", "\n", "\n\n", "\n\n\n", ".", "Q"]
    for _ in range(500):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 40)))
        spec = Spec(*[rng.choice([0, 300]) for _ in range(5)])
        pieces = split_structural(text, frozenset({"Q", "a"}), spec)
        assert "".join(piece.text for piece in pieces) == text
        assert all(piece.text for piece in pieces)
