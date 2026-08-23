"""Golden tables for deterministic en-US number-to-words conversion."""

from __future__ import annotations

import pytest

from kenkui._domain.spoken.numbers import (
    cardinal_words,
    decimal_words,
    ordinal_words,
    roman_value,
    year_words,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "zero"),
        (7, "seven"),
        (13, "thirteen"),
        (20, "twenty"),
        (21, "twenty-one"),
        (99, "ninety-nine"),
        (100, "one hundred"),
        (101, "one hundred one"),
        (342, "three hundred forty-two"),
        (1000, "one thousand"),
        (100_000, "one hundred thousand"),
        (1_000_000, "one million"),
        (
            1_234_567,
            "one million two hundred thirty-four thousand five hundred sixty-seven",
        ),
        (-5, "minus five"),
    ],
)
def test_cardinal_words(value: int, expected: str) -> None:
    """Cardinals read in en-US form with no connecting "and"."""
    assert cardinal_words(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, "first"),
        (2, "second"),
        (3, "third"),
        (5, "fifth"),
        (8, "eighth"),
        (9, "ninth"),
        (12, "twelfth"),
        (20, "twentieth"),
        (21, "twenty-first"),
        (40, "fortieth"),
        (100, "one hundredth"),
        (1000, "one thousandth"),
    ],
)
def test_ordinal_words(value: int, expected: str) -> None:
    """Ordinals apply irregular stems to the final word only."""
    assert ordinal_words(value) == expected


def test_decimal_words_reads_fraction_digit_by_digit() -> None:
    """The integer part is a number; the fraction is individual digits."""
    assert decimal_words("3", "14") == "three point one four"
    assert decimal_words("0", "5") == "zero point five"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1000, "one thousand"),
        (1900, "nineteen hundred"),
        (1905, "nineteen oh five"),
        (1984, "nineteen eighty-four"),
        (2000, "two thousand"),
        (2005, "two thousand five"),
        (2010, "twenty ten"),
        (2015, "twenty fifteen"),
    ],
)
def test_year_words(value: int, expected: str) -> None:
    """Years read as century pairs except the two-thousands run."""
    assert year_words(value) == expected


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("IV", 4),
        ("VIII", 8),
        ("XIV", 14),
        ("MCMLXXXIV", 1984),
        ("I", 1),
        ("", None),
        ("IIII", None),
        ("ABC", None),
        ("iv", None),
    ],
)
def test_roman_value(token: str, expected: int | None) -> None:
    """Only canonical uppercase numerals convert; anything else declines."""
    assert roman_value(token) == expected
