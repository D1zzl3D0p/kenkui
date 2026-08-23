"""Golden tables for deterministic en-US number-to-words conversion."""

from __future__ import annotations

import pytest

from kenkui._domain.spoken.numbers import (
    Rule,
    cardinal_words,
    conservative_rules,
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


def apply_rules(rules: tuple[Rule, ...], text: str) -> str:
    """Drive ordered rules across text the way the Task 5 matcher will.

    Deliberately duplicated here rather than imported: these tests must fail
    when the rules break, not when the matcher does.
    """
    out: list[str] = []
    position = 0
    while position < len(text):
        for pattern, handler in rules:
            match = pattern.match(text, position)
            if match is None:
                continue
            replacement = handler(match)
            if replacement is None:
                continue
            out.append(replacement)
            position = match.end()
            break
        else:
            out.append(text[position])
            position += 1
    return "".join(out)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("100,000", "one hundred thousand"),
        ("He had 3 apples.", "He had three apples."),
        ("3.14", "three point one four"),
        ("-5 degrees", "minus five degrees"),
        ("the 3rd time", "the third time"),
        ("the 21st time", "the twenty-first time"),
        ("the 11th time", "the eleventh time"),
        ("40%", "forty percent"),
        ("$1.50", "one dollar fifty"),
        ("$2", "two dollars"),
        ("5 km", "five kilometers"),
        ("1 km", "one kilometer"),
    ],
)
def test_conservative_tier_converts(source: str, expected: str) -> None:
    """Unambiguous numeric forms convert under the conservative tier."""
    assert apply_rules(conservative_rules(), source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "the 3th time",
        "COVID19",
        "3D printing",
        "1234567890123456789",
        "St. Mary",
        "Dr. Who",
    ],
)
def test_conservative_tier_declines(source: str) -> None:
    """Ambiguous or out-of-range forms are left exactly as written."""
    assert apply_rules(conservative_rules(), source) == source
