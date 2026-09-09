"""Tests for set-valued path patterns and their partial ordering."""

# The brief supplies descriptive test names; repeating them in docstrings adds noise.
# ruff: noqa: D103

import pytest

from kenkui._domain.grid import Unit
from kenkui._domain.paths import (
    Any,
    Exact,
    Last,
    OneOf,
    Span,
    matches,
    parse_pattern,
    subset,
)
from kenkui.errors import ErrorCode, ValidationError


def test_wildcard_parses() -> None:
    assert parse_pattern({"chapter": "*", "paragraph": 1}) is not None


def test_list_and_range_and_last_parse() -> None:
    assert parse_pattern({"chapter": "ch08", "paragraph": [2, 4, 7]}) is not None
    assert parse_pattern({"chapter": "ch08", "paragraph": "2..4"}) is not None
    assert parse_pattern({"chapter": "ch08", "paragraph": -1}) is not None


def test_chapter_range_is_refused() -> None:
    # Stable spine IDs are unordered; ranges belong to select_chapter_range.
    with pytest.raises(ValidationError):
        parse_pattern({"chapter": "2..4"})


def test_concrete_is_a_strict_subset_of_wildcard() -> None:
    narrow = parse_pattern({"chapter": "ch08", "paragraph": 3})
    wide = parse_pattern({"chapter": "ch08", "paragraph": "*"})
    assert subset(narrow, wide) is True
    assert subset(wide, narrow) is False


def test_deeper_path_is_a_subset_of_its_ancestor() -> None:
    deep = parse_pattern({"chapter": "ch08", "paragraph": 3, "sentence": 2})
    shallow = parse_pattern({"chapter": "ch08", "paragraph": 3})
    assert subset(deep, shallow) is True


def test_crossing_patterns_are_incomparable() -> None:
    # The Irulan case: neither match set contains the other.
    a = parse_pattern({"chapter": "ch08", "paragraph": "*"})
    b = parse_pattern({"chapter": "*", "paragraph": 1})
    assert subset(a, b) is None
    assert subset(b, a) is None


def test_last_is_incomparable_with_a_concrete_index() -> None:
    # Resolving -1 needs a sibling count, so this cannot be decided statically.
    last = parse_pattern({"chapter": "ch08", "paragraph": -1})
    third = parse_pattern({"chapter": "ch08", "paragraph": 3})
    assert subset(last, third) is None


def test_last_is_a_subset_of_wildcard() -> None:
    last = parse_pattern({"chapter": "ch08", "paragraph": -1})
    wide = parse_pattern({"chapter": "ch08", "paragraph": "*"})
    assert subset(last, wide) is True


def test_list_subset_of_wider_list() -> None:
    narrow = parse_pattern({"chapter": "ch08", "paragraph": [2, 4]})
    wide = parse_pattern({"chapter": "ch08", "paragraph": [2, 4, 7]})
    assert subset(narrow, wide) is True


def test_pattern_is_an_immutable_mapping() -> None:
    pattern = parse_pattern({"sentence": 2, "chapter": "ch08"})
    assert list(pattern) == ["chapter", "sentence"]
    assert pattern["chapter"] == Exact("ch08")
    assert dict(pattern) == {"chapter": Exact("ch08"), "sentence": Exact(2)}
    assert parse_pattern({}).is_whole_book()
    assert parse_pattern({"chapter": "*"}).is_whole_book()


def test_matching_resolves_last_from_parent_path() -> None:
    unit = Unit(
        "ch08",
        3,
        1,
        2,
        1,
        0,
        10,
        is_dialogue=False,
        is_emphasised=False,
    )
    pattern = parse_pattern({"paragraph": -1, "sentence": "2..4"})
    assert matches(pattern, unit, {("ch08",): 3})
    assert not matches(pattern, unit, {("ch08",): 4})


def test_selector_coverage() -> None:
    assert Any().covers("chapter", None)
    assert Exact(2).covers(2, None)
    assert OneOf(frozenset({2, 4})).covers(4, None)
    assert Span(2, 4).covers(3, None)
    assert Last().covers(3, 3)
    assert not Last().covers(3, None)


@pytest.mark.parametrize(
    "mapping",
    [
        {"stanza": 1},
        {"chapter": [1, 2]},
        {"chapter": -1},
        {"paragraph": True},
        {"paragraph": 0},
        {"paragraph": []},
        {"paragraph": [1, False]},
        {"paragraph": "4..2"},
        {"paragraph": "first"},
    ],
)
def test_invalid_patterns_are_refused(mapping: dict[str, object]) -> None:
    with pytest.raises(ValidationError) as caught:
        parse_pattern(mapping)
    assert caught.value.code is ErrorCode.INVALID_PATTERN


def test_overlapping_sets_have_partial_subset_ordering() -> None:
    left = parse_pattern({"paragraph": [2, 4]})
    right = parse_pattern({"paragraph": [4, 7]})
    assert subset(left, right) is None
    assert subset(parse_pattern({"paragraph": "2..4"}), left) is False
