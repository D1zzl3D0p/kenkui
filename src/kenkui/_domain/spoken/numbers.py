"""Deterministic en-US number-to-words conversion.

Pure and offline: the same input always produces the same words, on any
machine and in any process, which is what lets the result take part in
segment identity without making a cache entry machine-specific.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Literal

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


Handler = Callable[[re.Match[str]], str | None]
Rule = tuple[re.Pattern[str], Handler]
NumberTier = Literal["off", "conservative", "standard", "aggressive"]

# A number may neither begin nor end inside a word: "COVID19" and "3D" are
# tokens, not quantities, and reading them aloud is worse than leaving them.
# The left guard also stops an over-long digit run from being re-matched one
# character in, which would otherwise read its 15-digit tail aloud.
_LB = r"(?<![0-9A-Za-z])"
_RB = r"(?![0-9A-Za-z])"
_INT = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
_MAX_DIGITS = 15

_CURRENCY = {
    "$": ("dollar", "dollars"),
    "£": ("pound", "pounds"),
    "€": ("euro", "euros"),
}
# Deliberately excludes "m", "g", and "in": each collides with an ordinary
# English word often enough that expanding it is a net loss.
_UNITS = {
    "km": ("kilometer", "kilometers"),
    "cm": ("centimeter", "centimeters"),
    "mm": ("millimeter", "millimeters"),
    "kg": ("kilogram", "kilograms"),
    "mg": ("milligram", "milligrams"),
    "lb": ("pound", "pounds"),
    "oz": ("ounce", "ounces"),
    "ft": ("feet", "feet"),
    "mph": ("mile per hour", "miles per hour"),
    "kW": ("kilowatt", "kilowatts"),
}
_TEEN_ORDINALS = (11, 12, 13)
_ORDINAL_SUFFIXES = {1: "st", 2: "nd", 3: "rd"}


def _plain(digits: str) -> int | None:
    """Strip group separators and decline anything past the cardinal bound."""
    stripped = digits.replace(",", "")
    if len(stripped) > _MAX_DIGITS:
        return None
    return int(stripped)


def _ordinal_suffix(value: int) -> str:
    """Return the English ordinal suffix a value actually takes."""
    if value % _CENTURY in _TEEN_ORDINALS:
        return "th"
    return _ORDINAL_SUFFIXES.get(value % 10, "th")


def _currency(match: re.Match[str]) -> str | None:
    """Read a leading currency symbol with optional cents."""
    singular, plural = _CURRENCY[match.group(1)]
    whole = _plain(match.group(2))
    if whole is None:
        return None
    unit = singular if whole == 1 else plural
    cents = match.group(3)
    if cents is None or int(cents) == 0:
        return f"{cardinal_words(whole)} {unit}"
    return f"{cardinal_words(whole)} {unit} {cardinal_words(int(cents))}"


def _percent(match: re.Match[str]) -> str | None:
    """Read a percentage, whole or fractional."""
    whole = _plain(match.group(1))
    if whole is None:
        return None
    fraction = match.group(2)
    head = (
        cardinal_words(whole)
        if fraction is None
        else decimal_words(str(whole), fraction)
    )
    return f"{head} percent"


def _ordinal(match: re.Match[str]) -> str | None:
    """Read an ordinal, declining when the suffix disagrees with the number."""
    value = _plain(match.group(1))
    if value is None or match.group(2) != _ordinal_suffix(value):
        return None
    return ordinal_words(value)


def _unit(match: re.Match[str]) -> str | None:
    """Read a number followed by a recognized unit abbreviation."""
    whole = _plain(match.group(2))
    if whole is None:
        return None
    singular, plural = _UNITS[match.group(4)]
    fraction = match.group(3)
    if fraction is None:
        head = cardinal_words(whole)
        unit = singular if whole == 1 else plural
    else:
        head = decimal_words(str(whole), fraction)
        unit = plural
    sign = "minus " if match.group(1) else ""
    return f"{sign}{head} {unit}"


def _decimal(match: re.Match[str]) -> str | None:
    """Read a decimal number."""
    whole = _plain(match.group(2))
    if whole is None:
        return None
    sign = "minus " if match.group(1) else ""
    return f"{sign}{decimal_words(str(whole), match.group(3))}"


def _integer(match: re.Match[str]) -> str | None:
    """Read a plain integer, grouped or not."""
    value = _plain(match.group(2))
    if value is None:
        return None
    sign = "minus " if match.group(1) else ""
    return f"{sign}{cardinal_words(value)}"


def conservative_rules() -> tuple[Rule, ...]:
    """Return forms that are unambiguous under any reading.

    Order is significant: the first rule whose handler accepts wins, so more
    specific forms precede the plain integer that would otherwise swallow
    their leading digits.
    """
    units = "|".join(sorted(_UNITS, key=len, reverse=True))
    return (
        (re.compile(rf"{_LB}([$£€])({_INT})(?:\.(\d{{2}}))?{_RB}"), _currency),
        (re.compile(rf"{_LB}({_INT})(?:\.(\d+))?%"), _percent),
        (re.compile(rf"{_LB}({_INT})(st|nd|rd|th){_RB}"), _ordinal),
        (re.compile(rf"{_LB}(-)?({_INT})(?:\.(\d+))?[  ]?({units}){_RB}"), _unit),
        (re.compile(rf"{_LB}(-)?({_INT})\.(\d+){_RB}"), _decimal),
        (re.compile(rf"{_LB}(-)?({_INT}){_RB}"), _integer),
    )
