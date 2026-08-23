"""Deterministic en-US number-to-words conversion.

Pure and offline: the same input always produces the same words, on any
machine and in any process, which is what lets the result take part in
segment identity without making a cache entry machine-specific.
"""

from __future__ import annotations

import re

_ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = (
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)
_SCALES = (
    (10**12, "trillion"),
    (10**9, "billion"),
    (10**6, "million"),
    (10**3, "thousand"),
)
# Above this a digit run is far more likely an identifier than a quantity, so
# the scanners decline rather than reading out forty digits.
MAX_CARDINAL = 10**15 - 1

_IRREGULAR_ORDINALS = {
    "one": "first",
    "two": "second",
    "three": "third",
    "five": "fifth",
    "eight": "eighth",
    "nine": "ninth",
    "twelve": "twelfth",
}
_TENS_ORDINALS = {
    "twenty": "twentieth",
    "thirty": "thirtieth",
    "forty": "fortieth",
    "fifty": "fiftieth",
    "sixty": "sixtieth",
    "seventy": "seventieth",
    "eighty": "eightieth",
    "ninety": "ninetieth",
}

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
# Canonical subtractive form only. "IIII" is a clock face, not prose.
_ROMAN_CANONICAL = re.compile(
    r"M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"
)

_CENTURY = 100
_THOUSAND = 1000
_TWO_THOUSANDS_START = 2000
_TWO_THOUSANDS_END = 2010
_TEN = 10


def cardinal_words(value: int) -> str:
    """Return en-US words for an integer, with no connecting "and"."""
    if value < 0:
        return f"minus {cardinal_words(-value)}"
    if value < len(_ONES):
        return _ONES[value]
    if value < _CENTURY:
        tens, ones = divmod(value, _TEN)
        return _TENS[tens] if ones == 0 else f"{_TENS[tens]}-{_ONES[ones]}"
    if value < _THOUSAND:
        hundreds, rest = divmod(value, _CENTURY)
        head = f"{_ONES[hundreds]} hundred"
        return head if rest == 0 else f"{head} {cardinal_words(rest)}"
    for scale, name in _SCALES:
        if value >= scale:
            count, rest = divmod(value, scale)
            head = f"{cardinal_words(count)} {name}"
            return head if rest == 0 else f"{head} {cardinal_words(rest)}"
    message = "value exceeds MAX_CARDINAL"
    raise ValueError(message)


def ordinal_words(value: int) -> str:
    """Return en-US ordinal words, applying the irregular stem to the last word."""
    head, _, tail = cardinal_words(value).rpartition(" ")
    stem, hyphen, last = tail.rpartition("-")
    if last in _IRREGULAR_ORDINALS:
        last = _IRREGULAR_ORDINALS[last]
    elif last in _TENS_ORDINALS:
        last = _TENS_ORDINALS[last]
    elif last.endswith("y"):
        last = f"{last[:-1]}ieth"
    else:
        last = f"{last}th"
    tail = f"{stem}{hyphen}{last}"
    return f"{head} {tail}" if head else tail


def decimal_words(whole: str, fraction: str) -> str:
    """Read the integer part as a number and the fraction digit by digit."""
    digits = " ".join(_ONES[int(digit)] for digit in fraction)
    return f"{cardinal_words(int(whole))} point {digits}"


def year_words(value: int) -> str:
    """Read a year as a century pair, except the two-thousands run."""
    if value % _THOUSAND == 0:
        return cardinal_words(value)
    century, rest = divmod(value, _CENTURY)
    if _TWO_THOUSANDS_START <= value < _TWO_THOUSANDS_END:
        return f"two thousand {_ONES[rest]}"
    if rest == 0:
        return f"{cardinal_words(century)} hundred"
    if rest < _TEN:
        return f"{cardinal_words(century)} oh {_ONES[rest]}"
    return f"{cardinal_words(century)} {cardinal_words(rest)}"


def roman_value(token: str) -> int | None:
    """Return the value of a canonical uppercase Roman numeral, else ``None``."""
    if not token or _ROMAN_CANONICAL.fullmatch(token) is None:
        return None
    total = 0
    highest = 0
    for char in reversed(token):
        value = _ROMAN_VALUES[char]
        total = total - value if value < highest else total + value
        highest = max(highest, value)
    return total or None
