"""Derived structural range-index tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import pytest

from kenkui._domain.grid import (
    GapReason,
    LeafRange,
    StructuralIndex,
    Unit,
    build_grid,
    build_structure_index,
)
from kenkui._domain.paths import Path, contains
from kenkui.inspection import ChapterInspection


def chapter(text: str, *, headings: tuple[str, ...] = ()) -> ChapterInspection:
    """Build a minimal chapter inspection for index tests."""
    return ChapterInspection(
        id="ch01",
        index=0,
        title="One",
        speech_characters=None,
        text=text,
        emphasis=(),
        headings=headings,
    )


def unit(  # noqa: PLR0913
    *,
    chapter_id: str = "ch01",
    paragraph: int = 1,
    line: int = 1,
    sentence: int = 1,
    phrase: int = 1,
    start: int = 0,
    end: int = 1,
) -> Unit:
    """Build one structural leaf for validation tests."""
    return Unit(
        chapter_id=chapter_id,
        paragraph=paragraph,
        line=line,
        sentence=sentence,
        phrase=phrase,
        start=start,
        end=end,
        is_dialogue=False,
        is_emphasised=False,
    )


def test_index_maps_every_prefix_to_a_half_open_leaf_range() -> None:
    """Every hierarchy level receives one exact contiguous leaf range."""
    units = build_grid(chapter("One, two.\nLine.\n\nEnd."))
    index = build_structure_index(units)

    assert isinstance(index, Mapping)
    assert index[Path(chapter="ch01")] == LeafRange(0, 4)
    assert index[Path(chapter="ch01", paragraph=1)] == LeafRange(0, 3)
    assert index[Path(chapter="ch01", paragraph=1, line=1)] == LeafRange(0, 2)
    assert index[Path(chapter="ch01", paragraph=1, line=1, sentence=1)] == LeafRange(
        0, 2
    )
    assert index[
        Path(chapter="ch01", paragraph=1, line=1, sentence=1, phrase=2)
    ] == LeafRange(1, 2)
    assert index[Path(chapter="ch01", paragraph=2)] == LeafRange(3, 4)


def test_ranges_are_nested_and_cover_only_matching_leaves() -> None:
    """Child ranges nest in parents and never include a sibling leaf."""
    units = build_grid(chapter("One, two.\nLine.\n\nEnd."))
    index = build_structure_index(units)

    for path, leaf_range in index.items():
        selected = units[leaf_range.first : leaf_range.past_last]
        assert selected
        assert all(
            contains(
                path,
                Path(
                    leaf.chapter_id,
                    leaf.paragraph,
                    leaf.line,
                    leaf.sentence,
                    leaf.phrase,
                ),
            )
            for leaf in selected
        )
        if leaf_range.first:
            previous = units[leaf_range.first - 1]
            assert not contains(
                path,
                Path(
                    previous.chapter_id,
                    previous.paragraph,
                    previous.line,
                    previous.sentence,
                    previous.phrase,
                ),
            )
        if leaf_range.past_last < len(units):
            following = units[leaf_range.past_last]
            assert not contains(
                path,
                Path(
                    following.chapter_id,
                    following.paragraph,
                    following.line,
                    following.sentence,
                    following.phrase,
                ),
            )


def test_gap_metadata_combines_every_reason_that_closes() -> None:
    """One compact bit set records coincident hierarchy boundaries."""
    units = build_grid(chapter("One, two.\nLine.\n\nEnd."))

    assert build_structure_index(units).gaps == (
        GapReason.PHRASE,
        GapReason.PHRASE | GapReason.SENTENCE | GapReason.LINE,
        GapReason.PHRASE | GapReason.SENTENCE | GapReason.LINE | GapReason.PARAGRAPH,
        GapReason.PHRASE
        | GapReason.SENTENCE
        | GapReason.LINE
        | GapReason.PARAGRAPH
        | GapReason.CHAPTER,
    )


def test_paragraph_boundary_closes_reset_sentence_coordinate() -> None:
    """Parent changes close child ranges even when child coordinates both equal one."""
    units = build_grid(chapter("One.\n\nTwo."))

    assert build_structure_index(units).gaps[0] == (
        GapReason.PHRASE | GapReason.SENTENCE | GapReason.LINE | GapReason.PARAGRAPH
    )


def test_heading_reasons_share_their_canonical_paragraph_gaps() -> None:
    """Heading pauses are metadata on existing gaps, never extra partitions."""
    text = "Intro.\n\nChapter One\n\nBody."
    units = build_grid(chapter(text, headings=("Chapter One",)))
    gaps = build_structure_index(units).gaps

    first_paragraph = next(
        index for index, unit in enumerate(units) if unit.paragraph == 1
    )
    heading = next(index for index, unit in enumerate(units) if unit.is_heading)
    assert GapReason.HEADING_BEFORE in gaps[first_paragraph]
    assert GapReason.HEADING_AFTER in gaps[heading]
    assert GapReason.PARAGRAPH in gaps[first_paragraph]
    assert GapReason.PARAGRAPH in gaps[heading]


def test_index_is_immutable_deterministic_and_owns_no_canonical_data() -> None:
    """Equality uses only paths, leaf indices, and compact gap reasons."""
    units = (
        unit(),
        unit(phrase=2, start=1, end=2),
    )
    same_structure_different_canonical_data = (
        replace(units[0], end=10, is_dialogue=True),
        replace(units[1], start=10, end=30, is_emphasised=True),
    )

    index = StructuralIndex(units)
    duplicate = StructuralIndex(tuple(units))
    assert index == duplicate
    assert hash(index) == hash(duplicate)
    assert index == StructuralIndex(same_structure_different_canonical_data)
    assert isinstance(index.gaps, tuple)
    assert all(isinstance(value, LeafRange) for value in index.values())
    assert index != StructuralIndex(
        tuple(replace(leaf, chapter_id="another") for leaf in units)
    )


def test_empty_leaf_tuple_has_an_empty_index() -> None:
    """An empty chapter has neither prefixes nor trailing gaps."""
    index = build_structure_index(())
    assert dict(index) == {}
    assert index.gaps == ()


@pytest.mark.parametrize(
    ("units", "message"),
    [
        ((unit(chapter_id=""),), "chapter IDs must be non-empty"),
        (
            (unit(), unit(chapter_id="ch02", phrase=2, start=1, end=2)),
            "chapter ID changed",
        ),
        ((unit(paragraph=0),), "coordinates must be one-based"),
        ((unit(start=1, end=2),), "must begin at zero"),
        ((unit(end=0),), "must be non-empty and ordered"),
        (
            (unit(), unit(phrase=2, start=2, end=3)),
            "must be contiguous",
        ),
        (
            (unit(), unit(phrase=3, start=1, end=2)),
            "increase without gaps",
        ),
        (
            (unit(), unit(paragraph=2, line=2, start=1, end=2)),
            "must reset",
        ),
        (
            (unit(), unit(start=1, end=2)),
            "paths must be unique",
        ),
    ],
)
def test_malformed_leaf_sequences_are_rejected(
    units: tuple[Unit, ...], message: str
) -> None:
    """The derived index rejects unstable or non-hierarchical leaf input."""
    with pytest.raises(ValueError, match=message):
        build_structure_index(units)
