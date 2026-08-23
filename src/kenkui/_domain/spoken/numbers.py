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
    "ft": ("foot", "feet"),
    "mph": ("mile per hour", "miles per hour"),
    "kW": ("kilowatt", "kilowatts"),
}
# Two digits after the point is a cents amount. Any other length is an
# ordinary decimal, which is read as one number rather than as cents.
_CENTS_DIGITS = 2
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
    """Read a leading currency symbol with optional cents, decimal, or scale.

    The symbol precedes its amount in writing but follows it in speech, so a
    scale word has to be pulled inside the match: bound to the integer alone,
    "$3 million" reads as "three dollars million".
    """
    singular, plural = _CURRENCY[match.group(1)]
    whole = _plain(match.group(2))
    if whole is None:
        return None
    fraction, scale = match.group(3), match.group(4)
    if scale is not None:
        head = (
            decimal_words(str(whole), fraction) if fraction else cardinal_words(whole)
        )
        return f"{head} {scale} {plural}"
    if fraction is None:
        return f"{cardinal_words(whole)} {singular if whole == 1 else plural}"
    if len(fraction) != _CENTS_DIGITS:
        return f"{decimal_words(str(whole), fraction)} {plural}"
    unit = singular if whole == 1 else plural
    cents = int(fraction)
    if cents == 0:
        return f"{cardinal_words(whole)} {unit}"
    return f"{cardinal_words(whole)} {unit} {cardinal_words(cents)}"


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
    scales = "|".join(name for _, name in _SCALES)
    return (
        (
            re.compile(rf"{_LB}([$£€])({_INT})(?:\.(\d+))?(?:[ ]({scales}))?{_RB}"),
            _currency,
        ),
        (re.compile(rf"{_LB}({_INT})(?:\.(\d+))?%"), _percent),
        (re.compile(rf"{_LB}({_INT})(st|nd|rd|th){_RB}"), _ordinal),
        (re.compile(rf"{_LB}(-)?({_INT})(?:\.(\d+))?[  ]?({units}){_RB}"), _unit),
        (re.compile(rf"{_LB}(-)?({_INT})\.(\d+){_RB}"), _decimal),
        (re.compile(rf"{_LB}(-)?({_INT}){_RB}"), _integer),
    )


_YEAR = r"(?:1[0-9]{3}|20[0-9]{2})"
_TITLE_WORDS = "Chapter|Part|Book|Act|Scene|Volume|Section|Appendix"
# The dash family, escaped: en and em dashes are visually
# indistinguishable from a hyphen in source.
_DASHES = r"[-\u2013\u2014]"
# Valid canonical numerals that are also ordinary English words or common
# abbreviations. Without this, "MIX" and "CIVIC" read as numbers.
_ROMAN_STOPLIST = frozenset(
    {
        "MIX",
        "DID",
        "CIVIC",
        "MILD",
        "DIM",
        "LID",
        "MI",
        "DI",
        "CD",
        "MM",
        "LI",
        "MC",
        "ID",
        "MD",
        "CI",
    }
)
_DENOMINATORS = {
    2: ("half", "halves"),
    3: ("third", "thirds"),
    4: ("quarter", "quarters"),
}
# conservative_rules() ends with the decimal and integer catch-alls. Higher
# tiers splice their more specific forms in front of those two so a bare year
# is not swallowed as a plain integer.
_GENERIC_RULE_COUNT = 2


def _year_range(match: re.Match[str]) -> str | None:
    """Read a hyphenated or dashed span of two years."""
    return f"{year_words(int(match.group(1)))} to {year_words(int(match.group(2)))}"


def _clock(match: re.Match[str]) -> str | None:
    """Read a clock time the way it is spoken."""
    hour = int(match.group(1))
    minute = int(match.group(2))
    if minute == 0:
        return f"{cardinal_words(hour)} o'clock"
    if minute < _TEN:
        return f"{cardinal_words(hour)} oh {_ONES[minute]}"
    return f"{cardinal_words(hour)} {cardinal_words(minute)}"


def _year(match: re.Match[str]) -> str | None:
    """Read a bare four-digit year as a century pair."""
    return year_words(int(match.group(1)))


def _title_roman(match: re.Match[str]) -> str | None:
    """Read a Roman numeral that follows a structural title word."""
    value = roman_value(match.group(3))
    if value is None:
        return None
    return f"{match.group(1)}{match.group(2)}{cardinal_words(value).capitalize()}"


def _regnal(match: re.Match[str]) -> str | None:
    """Read a Roman numeral that follows a capitalized personal name."""
    value = roman_value(match.group(3))
    if value is None:
        return None
    return f"{match.group(1)}{match.group(2)}the {ordinal_words(value).capitalize()}"


def _bare_roman(match: re.Match[str]) -> str | None:
    """Read a standalone Roman numeral that is not also an English word."""
    token = match.group(1)
    if token in _ROMAN_STOPLIST:
        return None
    value = roman_value(token)
    if value is None:
        return None
    return cardinal_words(value).capitalize()


def _numbered(match: re.Match[str]) -> str | None:
    """Read the "No." abbreviation as the word Number."""
    value = _plain(match.group(2))
    if value is None:
        return None
    return f"Number {cardinal_words(value)}"


def _fraction(match: re.Match[str]) -> str | None:
    """Read a slash fraction, using the ordinary names for small denominators."""
    numerator = _plain(match.group(1))
    denominator = _plain(match.group(2))
    if numerator is None or denominator is None or denominator == 0:
        return None
    if denominator in _DENOMINATORS:
        singular, plural = _DENOMINATORS[denominator]
        return f"{cardinal_words(numerator)} {singular if numerator == 1 else plural}"
    tail = ordinal_words(denominator)
    return f"{cardinal_words(numerator)} {tail}{'' if numerator == 1 else 's'}"


def _standard_rules() -> tuple[Rule, ...]:
    """Return forms that are usually right but require context."""
    return (
        (re.compile(rf"{_LB}({_YEAR})\s*{_DASHES}\s*({_YEAR}){_RB}"), _year_range),
        (re.compile(rf"{_LB}([01]?[0-9]|2[0-3]):([0-5][0-9]){_RB}"), _clock),
        (re.compile(rf"{_LB}({_YEAR}){_RB}"), _year),
        (re.compile(rf"{_LB}({_TITLE_WORDS})(\s+)([IVXLCDM]+){_RB}"), _title_roman),
        # Two or more numeral characters: a lone "I" is far more often the
        # pronoun, and "said I" must never become "said the First".
        (re.compile(rf"{_LB}([A-Z][a-z]+)(\s+)([IVXLCDM]{{2,}}){_RB}"), _regnal),
    )


def _aggressive_rules() -> tuple[Rule, ...]:
    """Return forms that require guessing and must be opted into explicitly."""
    return (
        (re.compile(rf"{_LB}([A-Z][a-z]+)(\s+)([IVXLCDM]+){_RB}"), _regnal),
        (re.compile(rf"No\.(\s*)({_INT}){_RB}"), _numbered),
        (re.compile(rf"{_LB}({_INT})/({_INT}){_RB}"), _fraction),
        (re.compile(rf"{_LB}([IVXLCDM]{{2,}}){_RB}"), _bare_roman),
    )


def number_rules(tier: NumberTier) -> tuple[Rule, ...]:
    """Return the ordered rules for one tier, most specific form first."""
    if tier == "off":
        return ()
    base = conservative_rules()
    if tier == "conservative":
        return base
    extra = _standard_rules()
    if tier == "aggressive":
        extra = (*extra, *_aggressive_rules())
    head = base[:-_GENERIC_RULE_COUNT]
    generic = base[-_GENERIC_RULE_COUNT:]
    return (*head, *extra, *generic)
