# Spoken Form, Structural Pauses, and Custom Cover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add numeric normalization for speech, configurable silence around structural boundaries, a pronunciation lexicon, and caller-supplied cover art to the Kenkui core pipeline.

**Architecture:** One canonical normalized text remains the authority for billing, inspection, chapter identity, and attribution offsets. A new pure `spoken-form-v1` stage, preceded by a structural split, derives a separate spoken text that only the synthesis engine sees. Silence is generated at spill time as padding on `SegmentAudio.frame_count`, never reaching a worker or the cache.

**Tech Stack:** Python 3.11-3.13, uv, pytest + pytest-cov, ruff (`select = ALL`), mypy strict. No new runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-08-23-spoken-form-pauses-cover-design.md`

## Global Constraints

Copied verbatim from the spec and the repo toolchain. Every task's requirements implicitly include this section.

- **D6 is the highest-priority invariant:** a pipeline calling neither `.pronounce()` nor `.pauses()` MUST produce byte-identical segment IDs and an identical plan fingerprint to the pre-change implementation. Task 7 installs the guard test; every later task must keep it green.
- New segment-identity fields are added **only when the corresponding feature is enabled**, mirroring the existing precedent at `src/kenkui/_domain/planning.py:494-498` where `speaker_id`/`voice_id` appear only for attributed speech.
- `src/kenkui/_domain/text.py` is **not modified**. `normalize_text()` and `nfc-space-newline-v1` stay frozen.
- Domain modules must not import `Pipeline` (core spec §24). `_domain/spoken/` and `_domain/structure.py` import from `kenkui.errors` and `kenkui.inspection` only.
- Pure functions do not open files, mutate caches, configure log handlers, or load models (core spec §6). The built-in lexicon data file is read once at import as package data.
- English only. Number words use en-US conventions (no "one hundred *and* one"). A non-English narrator voice disables the spoken-form stage entirely.
- Gap 0 (before the book's first segment) is always zero.
- The inter-chapter gap folds into the **preceding** chapter's last segment.
- `chapter_ms` alone must NOT engage `tts-chunks-v3`.
- Python floor is 3.11; `target-version = "py311"`. No `match` on types requiring 3.12+, no PEP-695 generics.
- ruff runs with `select = ["ALL"]`. Every module, class, function, and test needs a docstring. Every signature needs full annotations. Line length 88.
- mypy runs `strict = true` over `src` and `tests`.
- Coverage gate is `--cov-fail-under=90` with `--cov-branch`. Targeted test runs during a task use `--no-cov`; the final run of each task uses the full suite.
- Every commit uses `git commit --signoff` (DCO required by `CONTRIBUTING.md`).

## Verification Commands

```console
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
```

Targeted single-test run during a task:

```console
uv run pytest tests/test_spoken_numbers.py::test_name -v --no-cov
```

## File Structure

**Created:**

| Path | Responsibility |
|---|---|
| `src/kenkui/_domain/spoken/__init__.py` | `to_spoken()` — the combined single-pass ranked matcher |
| `src/kenkui/_domain/spoken/numbers.py` | Number-to-words conversion and per-tier scanning rules |
| `src/kenkui/_domain/spoken/lexicon.py` | Phrase matching, capitalization shape, caller validation |
| `src/kenkui/_domain/spoken/data/lexicon-v1.json` | Versioned built-in pronunciation entries |
| `src/kenkui/_domain/structure.py` | Forced-break derivation, structural split, gap table |
| `tests/test_spoken_numbers.py` | Number tier golden tables |
| `tests/test_spoken_lexicon.py` | Lexicon matching and validation |
| `tests/test_spoken_form.py` | `to_spoken()` composition and rank order |
| `tests/test_structure.py` | Split exactness, break derivation, gap model |
| `tests/test_identity_stability.py` | The D6 opt-out guard |
| `tests/test_pauses_render.py` | Silence padding vs. part size and markers |
| `tests/test_cover_file.py` | Cover accept/reject matrix |

**Modified:**

| Path | Change |
|---|---|
| `src/kenkui/_domain/operations.py` | Add `SpokenForm`, `Pauses` records; extend `Operation` union |
| `src/kenkui/_domain/planning.py` | Split+spoken stages, v3 chunking, gap table, conditional identity, billing total |
| `src/kenkui/_epub/parser.py` | Record heading offsets |
| `src/kenkui/inspection.py` | `ChapterInspection.headings` |
| `src/kenkui/pipeline.py` | `.pronounce()`, `.pauses()`, widened `.metadata(cover=)` |
| `src/kenkui/_execution/coordinator.py` | Silence padding and spill |
| `src/kenkui/_audio/cover.py` | Caller-supplied cover validation |
| `src/kenkui/errors.py` | Three new codes |
| `pyproject.toml` | Package the lexicon data file |

---
# Phase 1 — Spoken Form

## Task 1: Number-to-words primitives

**Files:**
- Create: `src/kenkui/_domain/spoken/__init__.py` (empty package marker for now)
- Create: `src/kenkui/_domain/spoken/numbers.py`
- Test: `tests/test_spoken_numbers.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `cardinal_words(value: int) -> str`, `ordinal_words(value: int) -> str`, `decimal_words(whole: str, fraction: str) -> str`, `year_words(value: int) -> str`, `roman_value(token: str) -> int | None`, and the constant `MAX_CARDINAL: int`. Task 2 and Task 3 call all of these.

- [ ] **Step 1: Write the failing test**

Create `tests/test_spoken_numbers.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_spoken_numbers.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'kenkui._domain.spoken'`

- [ ] **Step 3: Create the package marker**

Create `src/kenkui/_domain/spoken/__init__.py`:

```python
"""Deterministic speech-form text transformation."""
```

- [ ] **Step 4: Write the implementation**

Create `src/kenkui/_domain/spoken/numbers.py`:

```python
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

_TWO_THOUSANDS_END = 2010
_CENTURY = 100


def cardinal_words(value: int) -> str:
    """Return en-US words for an integer, with no connecting "and"."""
    if value < 0:
        return f"minus {cardinal_words(-value)}"
    if value < len(_ONES):
        return _ONES[value]
    if value < _CENTURY:
        tens, ones = divmod(value, 10)
        return _TENS[tens] if ones == 0 else f"{_TENS[tens]}-{_ONES[ones]}"
    if value < 1000:
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
    if value % 1000 == 0:
        return cardinal_words(value)
    century, rest = divmod(value, _CENTURY)
    if 2000 <= value < _TWO_THOUSANDS_END:
        return f"two thousand {_ONES[rest]}"
    if rest == 0:
        return f"{cardinal_words(century)} hundred"
    if rest < 10:
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_spoken_numbers.py -v --no-cov`
Expected: PASS, all parametrized cases.

- [ ] **Step 6: Run lint, types, and the full suite**

```console
uv run ruff format .
uv run ruff check .
uv run mypy
uv run pytest
```

Expected: clean. If ruff objects to the `ValueError` message being inline (rule `EM101`/`TRY003`), the message is already bound to a local first — keep that shape.

- [ ] **Step 7: Commit**

```bash
git add src/kenkui/_domain/spoken tests/test_spoken_numbers.py
git commit --signoff -m "feat: add deterministic en-US number-to-words primitives"
```

---
## Task 2: Conservative-tier number rules

The scanner exposes ordered `(pattern, handler)` rules rather than doing its own
sweep, so Task 5's single-pass matcher can interleave them with lexicon entries
at the same position. A handler returning `None` **declines** and the matcher
falls through to the next rule — that is how `3th` and over-long digit runs are
rejected without a second scan.

**Files:**
- Modify: `src/kenkui/_domain/spoken/numbers.py`
- Test: `tests/test_spoken_numbers.py`

**Interfaces:**
- Consumes: `cardinal_words`, `ordinal_words`, `decimal_words` from Task 1.
- Produces: `Handler = Callable[[re.Match[str]], str | None]`, `Rule = tuple[re.Pattern[str], Handler]`, `NumberTier = Literal["off", "conservative", "standard", "aggressive"]`, `conservative_rules() -> tuple[Rule, ...]`. Task 3 consumes `Rule`, `NumberTier`, `conservative_rules()`; Task 5 consumes `Rule` only.

- [ ] **Step 1: Write the failing test**

Replace the import block at the top of `tests/test_spoken_numbers.py` with:

```python
from kenkui._domain.spoken.numbers import (
    Rule,
    cardinal_words,
    conservative_rules,
    decimal_words,
    ordinal_words,
    roman_value,
    year_words,
)
```

Append to `tests/test_spoken_numbers.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_spoken_numbers.py -v --no-cov`
Expected: FAIL — `ImportError: cannot import name 'conservative_rules'`

- [ ] **Step 3: Extend the module imports**

At the top of `src/kenkui/_domain/spoken/numbers.py`, below `import re`, add:

```python
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable
```

- [ ] **Step 4: Write the implementation**

Append to `src/kenkui/_domain/spoken/numbers.py`:

```python
Handler = Callable[["re.Match[str]"], str | None]
Rule = tuple["re.Pattern[str]", Handler]
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
    value = _plain(match.group(1))
    if value is None or match.group(2) != _ordinal_suffix(value):
        return None
    return ordinal_words(value)


def _unit(match: re.Match[str]) -> str | None:
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
    whole = _plain(match.group(2))
    if whole is None:
        return None
    sign = "minus " if match.group(1) else ""
    return f"{sign}{decimal_words(str(whole), match.group(3))}"


def _integer(match: re.Match[str]) -> str | None:
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_spoken_numbers.py -v --no-cov`
Expected: PASS — 12 conversion cases, 6 decline cases.

If `-5 degrees` fails, check the `_LB` lookbehind on the minus: it must reject a hyphen following a digit or letter so `1914-1918` never reads as a negative.

- [ ] **Step 6: Run lint, types, and the full suite**

```console
uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest
```

- [ ] **Step 7: Commit**

```bash
git add src/kenkui/_domain/spoken/numbers.py tests/test_spoken_numbers.py
git commit --signoff -m "feat: add conservative-tier number scanning rules"
```

---
## Task 3: Standard and aggressive tiers

**Spec refinement — read before implementing.** The spec places "roman numerals
after a regnal name" in the *standard* tier. Implemented literally that would
convert `said I` into `said the First`, which is common in older prose and was
the exact risk raised during design. The rule is therefore split: **standard**
requires two or more numeral characters (so `Henry VIII` converts and
`Elizabeth I` does not), and **aggressive** lifts that to single-character
numerals. This is a narrowing of the spec, not an addition.

**Files:**
- Modify: `src/kenkui/_domain/spoken/numbers.py`
- Test: `tests/test_spoken_numbers.py`

**Interfaces:**
- Consumes: `Rule`, `NumberTier`, `conservative_rules()` from Task 2; `cardinal_words`, `ordinal_words`, `year_words`, `roman_value` from Task 1.
- Produces: `number_rules(tier: NumberTier) -> tuple[Rule, ...]`. Task 5 calls exactly this and nothing else from the module.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_spoken_numbers.py`:

```python
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("In 1984 he left.", "In nineteen eighty-four he left."),
        ("1914-1918", "nineteen fourteen to nineteen eighteen"),
        ("at 3:45", "at three forty-five"),
        ("at 3:00", "at three o'clock"),
        ("at 3:05", "at three oh five"),
        ("Chapter IV", "Chapter Four"),
        ("Part VIII", "Part Eight"),
        ("Henry VIII", "Henry the Eighth"),
    ],
)
def test_standard_tier_converts(source: str, expected: str) -> None:
    """Heuristic-but-usually-right forms convert under the standard tier."""
    assert apply_rules(number_rules("standard"), source) == expected


@pytest.mark.parametrize(
    "source",
    [
        '"I know," said I.',
        "Elizabeth I",
    ],
)
def test_standard_tier_leaves_single_i(source: str) -> None:
    """A one-character numeral is left alone: "I" is far more often a pronoun."""
    assert apply_rules(number_rules("standard"), source) == source


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Elizabeth I", "Elizabeth the First"),
        ("No. 5", "Number five"),
        ("1/2", "one half"),
        ("3/4", "three quarters"),
        ("1/7", "one seventh"),
        ("XIV", "Fourteen"),
    ],
)
def test_aggressive_tier_converts(source: str, expected: str) -> None:
    """The aggressive tier accepts forms that require guessing."""
    assert apply_rules(number_rules("aggressive"), source) == expected


@pytest.mark.parametrize("source", ["MIX", "CIVIC", "DID"])
def test_aggressive_tier_respects_roman_stoplist(source: str) -> None:
    """Words that are also valid numerals stay words."""
    assert apply_rules(number_rules("aggressive"), source) == source


def test_off_tier_has_no_rules() -> None:
    """The off tier converts nothing at all."""
    assert number_rules("off") == ()
    assert apply_rules(number_rules("off"), "100,000") == "100,000"


def test_tiers_are_cumulative() -> None:
    """Every tier still converts everything the conservative tier does."""
    for tier in ("conservative", "standard", "aggressive"):
        assert apply_rules(number_rules(tier), "100,000") == "one hundred thousand"
```

Add `number_rules` to the import block at the top of the file.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_spoken_numbers.py -v --no-cov`
Expected: FAIL — `ImportError: cannot import name 'number_rules'`

- [ ] **Step 3: Write the implementation**

Append to `src/kenkui/_domain/spoken/numbers.py`:

```python
_YEAR = r"(?:1[0-9]{3}|20[0-9]{2})"
_TITLE_WORDS = "Chapter|Part|Book|Act|Scene|Volume|Section|Appendix"
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
    return f"{year_words(int(match.group(1)))} to {year_words(int(match.group(2)))}"


def _clock(match: re.Match[str]) -> str | None:
    hour = int(match.group(1))
    minute = int(match.group(2))
    if minute == 0:
        return f"{cardinal_words(hour)} o'clock"
    if minute < 10:
        return f"{cardinal_words(hour)} oh {_ONES[minute]}"
    return f"{cardinal_words(hour)} {cardinal_words(minute)}"


def _year(match: re.Match[str]) -> str | None:
    return year_words(int(match.group(1)))


def _title_roman(match: re.Match[str]) -> str | None:
    value = roman_value(match.group(3))
    if value is None:
        return None
    return f"{match.group(1)}{match.group(2)}{cardinal_words(value).capitalize()}"


def _regnal(match: re.Match[str]) -> str | None:
    value = roman_value(match.group(3))
    if value is None:
        return None
    return f"{match.group(1)}{match.group(2)}the {ordinal_words(value).capitalize()}"


def _bare_roman(match: re.Match[str]) -> str | None:
    token = match.group(1)
    if token in _ROMAN_STOPLIST:
        return None
    value = roman_value(token)
    if value is None:
        return None
    return cardinal_words(value).capitalize()


def _numbered(match: re.Match[str]) -> str | None:
    value = _plain(match.group(2))
    if value is None:
        return None
    return f"Number {cardinal_words(value)}"


def _fraction(match: re.Match[str]) -> str | None:
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
        (re.compile(rf"{_LB}({_YEAR})\s*[-–—]\s*({_YEAR}){_RB}"), _year_range),
        (re.compile(rf"{_LB}([01]?[0-9]|2[0-3]):([0-5][0-9]){_RB}"), _clock),
        (re.compile(rf"{_LB}({_YEAR}){_RB}"), _year),
        (
            re.compile(rf"{_LB}({_TITLE_WORDS})(\s+)([IVXLCDM]+){_RB}"),
            _title_roman,
        ),
        # Two or more numeral characters: see the spec refinement note.
        (
            re.compile(rf"{_LB}([A-Z][a-z]+)(\s+)([IVXLCDM]{{2,}}){_RB}"),
            _regnal,
        ),
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_spoken_numbers.py -v --no-cov`
Expected: PASS — all conservative, standard, and aggressive cases.

- [ ] **Step 5: Run lint, types, and the full suite**

```console
uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest
```

- [ ] **Step 6: Commit**

```bash
git add src/kenkui/_domain/spoken/numbers.py tests/test_spoken_numbers.py
git commit --signoff -m "feat: add standard and aggressive number tiers"
```

---
## Task 4: Pronunciation lexicon

The built-in list is deliberately **small and high-confidence**. Pocket-TTS
already handles a great deal of English correctly, and a wrong "correction" is
more audible than the original error. Every added entry is a `LEXICON_VERSION`
bump, which invalidates caches — that friction is the point.

**Files:**
- Create: `src/kenkui/_domain/spoken/lexicon.py`
- Create: `src/kenkui/_domain/spoken/data/lexicon-v1.json`
- Modify: `src/kenkui/errors.py`
- Modify: `pyproject.toml`
- Test: `tests/test_spoken_lexicon.py`

**Interfaces:**
- Consumes: `Rule` from Task 2.
- Produces: `LEXICON_VERSION: str`, `MAX_LEXICON_ENTRIES: int`, `MAX_PHRASE_CHARACTERS: int`, `validate_entries(mapping: Mapping[str, str]) -> tuple[tuple[str, str], ...]`, `builtin_entries() -> tuple[tuple[str, str], ...]`, `lexicon_rules(caller: tuple[tuple[str, str], ...], *, builtin: bool) -> tuple[Rule, ...]`. Task 5 calls `lexicon_rules` and `validate_entries`; Task 6 calls `validate_entries`.

- [ ] **Step 1: Add the error code**

In `src/kenkui/errors.py`, add to the `ErrorCode` enum immediately after `INVALID_METADATA`:

```python
    INVALID_PRONUNCIATION = "invalid_pronunciation"
    INVALID_PAUSE = "invalid_pause"
```

and to `_DEFAULT_MESSAGES` immediately after the `INVALID_METADATA` entry:

```python
    ErrorCode.INVALID_PRONUNCIATION: "The pronunciation entry is invalid.",
    ErrorCode.INVALID_PAUSE: "The pause duration is invalid.",
```

(`INVALID_PAUSE` is unused until Task 12; adding both codes now keeps the enum
edited once.)

- [ ] **Step 2: Write the failing test**

Create `tests/test_spoken_lexicon.py`:

```python
"""Whole-word pronunciation matching, capitalization shape, and validation."""

from __future__ import annotations

import pytest

from kenkui._domain.spoken.lexicon import (
    MAX_LEXICON_ENTRIES,
    builtin_entries,
    lexicon_rules,
    validate_entries,
)
from kenkui._domain.spoken.numbers import Rule
from kenkui.errors import ErrorCode, ValidationError


def apply_rules(rules: tuple[Rule, ...], text: str) -> str:
    """Drive ordered rules across text the way the Task 5 matcher will."""
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


def caller(**entries: str) -> tuple[Rule, ...]:
    """Build rules from caller entries only, with the built-in list disabled."""
    return lexicon_rules(validate_entries(entries), builtin=False)


def test_matches_whole_words_only() -> None:
    """A phrase inside a longer word is not a match."""
    rules = caller(cat="kat")
    assert apply_rules(rules, "the cat sat") == "the kat sat"
    assert apply_rules(rules, "concatenate") == "concatenate"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("cello", "chello"),
        ("Cello", "Chello"),
        ("CELLO", "CHELLO"),
    ],
)
def test_replacement_inherits_capitalization_shape(source: str, expected: str) -> None:
    """The replacement takes the source's shape, not the entry's."""
    assert apply_rules(caller(cello="chello"), source) == expected


def test_longest_phrase_wins() -> None:
    """A multi-word entry beats a shorter entry that starts at the same place."""
    rules = lexicon_rules(
        validate_entries({"coup": "koo", "coup de grace": "coo de grahss"}),
        builtin=False,
    )
    assert apply_rules(rules, "a coup de grace") == "a coo de grahss"


def test_builtin_matches_diacritic_insensitively() -> None:
    """Built-in entries catch both the accented and unaccented spelling."""
    rules = lexicon_rules((), builtin=True)
    assert apply_rules(rules, "coup de grâce") == apply_rules(rules, "coup de grace")
    assert apply_rules(rules, "coup de grace") != "coup de grace"


def test_caller_entry_overrides_builtin() -> None:
    """A caller entry wins over a built-in for the same phrase."""
    rules = lexicon_rules(validate_entries({"cello": "SELLO"}), builtin=True)
    assert apply_rules(rules, "cello") == "SELLO"


def test_output_is_never_rescanned() -> None:
    """A replacement containing another entry's key does not cascade."""
    rules = lexicon_rules(
        validate_entries({"alpha": "beta", "beta": "gamma"}), builtin=False
    )
    assert apply_rules(rules, "alpha") == "beta"


def test_builtin_entries_are_sorted_and_nonempty() -> None:
    """The shipped data file loads and is deterministically ordered."""
    entries = builtin_entries()
    assert entries
    assert list(entries) == sorted(entries)


@pytest.mark.parametrize(
    "entries",
    [
        {"": "x"},
        {"x": ""},
        {"   ": "x"},
        {"x": "   "},
        {"a" * 200: "x"},
        {"x": "a" * 200},
    ],
)
def test_validate_rejects_malformed_entries(entries: dict[str, str]) -> None:
    """Empty, blank, or over-long entries are refused."""
    with pytest.raises(ValidationError) as error:
        validate_entries(entries)
    assert error.value.code is ErrorCode.INVALID_PRONUNCIATION


def test_validate_rejects_too_many_entries() -> None:
    """The entry count is bounded because a server accepts this untrusted."""
    entries = {f"word{index}": "x" for index in range(MAX_LEXICON_ENTRIES + 1)}
    with pytest.raises(ValidationError) as error:
        validate_entries(entries)
    assert error.value.code is ErrorCode.INVALID_PRONUNCIATION


def test_validate_returns_sorted_pairs() -> None:
    """Order is canonical so identity does not depend on caller dict order."""
    assert validate_entries({"b": "2", "a": "1"}) == (("a", "1"), ("b", "2"))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_spoken_lexicon.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'kenkui._domain.spoken.lexicon'`

- [ ] **Step 4: Create the data file**

Create `src/kenkui/_domain/spoken/data/lexicon-v1.json`:

```json
{
  "version": "lexicon-v1",
  "entries": {
    "Arkansas": "Arkansaw",
    "Gloucester": "Gloster",
    "Leicester": "Lester",
    "Ouija": "wee-jah",
    "Thames": "Tems",
    "Worcestershire": "wooster-sher",
    "Yosemite": "yo-sem-it-ee",
    "cello": "chello",
    "cellos": "chellos",
    "colonel": "kernel",
    "coup de grâce": "coo de grahss",
    "epitome": "eh-pit-oh-mee",
    "façade": "fuh-sahd",
    "hors d'oeuvres": "or-derv",
    "hyperbole": "high-per-boh-lee",
    "quay": "key",
    "rendezvous": "ron-day-voo",
    "segue": "seg-way"
  }
}
```

- [ ] **Step 5: Package the data file**

In `pyproject.toml`, under `[tool.hatch.build.targets.wheel]`, add the
`force-include` so the JSON ships in the wheel:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/kenkui"]
artifacts = ["src/kenkui/_domain/spoken/data/*.json"]
```

- [ ] **Step 6: Write the implementation**

Create `src/kenkui/_domain/spoken/lexicon.py`:

```python
"""Whole-word pronunciation replacement with a versioned built-in lexicon.

Matching is deliberately single-shot: a replacement is emitted and never
re-examined, so entries cannot chain or loop. That makes the transformation a
pure function of its inputs, which is what lets it take part in segment
identity.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, cast

from kenkui.errors import ErrorCode, ValidationError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from kenkui._domain.spoken.numbers import Rule

LEXICON_VERSION = "lexicon-v1"
MAX_LEXICON_ENTRIES = 512
MAX_PHRASE_CHARACTERS = 128

_DATA = Path(__file__).parent / "data" / f"{LEXICON_VERSION}.json"
_LB = r"(?<![0-9A-Za-z])"
_RB = r"(?![0-9A-Za-z])"


def _fold(value: str) -> str:
    """Return a case- and diacritic-insensitive lookup key."""
    decomposed = unicodedata.normalize("NFD", value)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return stripped.casefold()


def _shaped(source: str, replacement: str) -> str:
    """Give the replacement the source's capitalization shape."""
    if len(source) > 1 and source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return f"{replacement[:1].upper()}{replacement[1:]}"
    return replacement


@lru_cache(maxsize=1)
def builtin_entries() -> tuple[tuple[str, str], ...]:
    """Load the shipped lexicon once, in canonical sorted order."""
    payload = cast("dict[str, object]", json.loads(_DATA.read_text("utf-8")))
    entries = cast("dict[str, str]", payload["entries"])
    return tuple(sorted(entries.items()))


def validate_entries(mapping: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    """Bound and canonicalize caller entries, refusing malformed input."""
    items = tuple(sorted(mapping.items()))
    if len(items) > MAX_LEXICON_ENTRIES:
        raise ValidationError(ErrorCode.INVALID_PRONUNCIATION)
    for key, value in items:
        if (
            not isinstance(key, str)
            or not isinstance(value, str)
            or not key.strip()
            or not value.strip()
            or len(key) > MAX_PHRASE_CHARACTERS
            or len(value) > MAX_PHRASE_CHARACTERS
        ):
            raise ValidationError(ErrorCode.INVALID_PRONUNCIATION)
    return items


def _rule(entries: tuple[tuple[str, str], ...], *, folded: bool) -> Rule | None:
    """Compile one alternation, longest phrase first so it wins the position."""
    if not entries:
        return None
    table: dict[str, str] = {}
    alternatives: list[str] = []
    for key, value in entries:
        forms = {key, _fold(key)} if folded else {key}
        for form in forms:
            table[_fold(form)] = value
            alternatives.append(re.escape(form))
    alternatives.sort(key=len, reverse=True)
    pattern = re.compile(rf"{_LB}(?:{'|'.join(alternatives)}){_RB}", re.IGNORECASE)

    def handler(match: re.Match[str]) -> str | None:
        source = match.group(0)
        replacement = table.get(_fold(source))
        if replacement is None:
            return None
        return _shaped(source, replacement)

    return (pattern, handler)


def lexicon_rules(
    caller: tuple[tuple[str, str], ...], *, builtin: bool
) -> tuple[Rule, ...]:
    """Return caller rules ahead of built-in rules, so the caller always wins."""
    rules = [_rule(caller, folded=False)]
    if builtin:
        rules.append(_rule(builtin_entries(), folded=True))
    return tuple(rule for rule in rules if rule is not None)
```

- [ ] **Step 7: Run test to verify it passes**

Run: `uv run pytest tests/test_spoken_lexicon.py -v --no-cov`
Expected: PASS.

If `test_builtin_matches_diacritic_insensitively` fails, confirm `_fold` is
applied to both the table key and the matched source, and that the accented and
stripped spellings are both in `alternatives`.

- [ ] **Step 8: Run lint, types, and the full suite**

```console
uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest
```

- [ ] **Step 9: Commit**

```bash
git add src/kenkui/_domain/spoken/lexicon.py src/kenkui/_domain/spoken/data \
        src/kenkui/errors.py pyproject.toml tests/test_spoken_lexicon.py
git commit --signoff -m "feat: add pronunciation lexicon with caller overrides"
```

---

## Task 5: The single-pass spoken-form matcher

**Files:**
- Modify: `src/kenkui/_domain/spoken/__init__.py`
- Test: `tests/test_spoken_form.py`

**Interfaces:**
- Consumes: `number_rules`, `NumberTier`, `Rule` from Tasks 2-3; `lexicon_rules`, `LEXICON_VERSION` from Task 4.
- Produces: `to_spoken(text: str, *, numbers: NumberTier, lexicon: tuple[tuple[str, str], ...], builtin: bool) -> str` and `spoken_identity(*, numbers: NumberTier, lexicon: tuple[tuple[str, str], ...], builtin: bool) -> dict[str, object]` and `SPOKEN_FORM_VERSION: str`. Task 7 calls both.

- [ ] **Step 1: Write the failing test**

Create `tests/test_spoken_form.py`:

```python
"""Composition and rank order of the combined spoken-form matcher."""

from __future__ import annotations

from kenkui._domain.spoken import SPOKEN_FORM_VERSION, spoken_identity, to_spoken


def spoken(text: str, **kwargs: object) -> str:
    """Call to_spoken with conservative defaults."""
    numbers = kwargs.get("numbers", "conservative")
    lexicon = kwargs.get("lexicon", ())
    builtin = kwargs.get("builtin", True)
    return to_spoken(
        text,
        numbers=numbers,  # type: ignore[arg-type]
        lexicon=lexicon,  # type: ignore[arg-type]
        builtin=builtin,  # type: ignore[arg-type]
    )


def test_numbers_and_lexicon_apply_in_one_pass() -> None:
    """Both rule families act on the same text without a second sweep."""
    assert spoken("100,000 cellos") == "one hundred thousand chellos"


def test_caller_lexicon_outranks_numbers() -> None:
    """A caller entry matching at a position beats a number rule there."""
    result = to_spoken(
        "Room 101",
        numbers="conservative",
        lexicon=(("101", "one oh one"),),
        builtin=False,
    )
    assert result == "Room one oh one"


def test_everything_off_is_the_identity_function() -> None:
    """With no rules the text is returned unchanged, character for character."""
    source = "100,000 cellos in 1984."
    assert to_spoken(source, numbers="off", lexicon=(), builtin=False) == source


def test_text_without_matches_is_unchanged() -> None:
    """Ordinary prose survives the pass byte for byte."""
    source = "The quick brown fox jumps over the lazy dog."
    assert spoken(source) == source


def test_identity_changes_with_each_input() -> None:
    """Every knob that can change output also changes the identity payload."""
    base = spoken_identity(numbers="conservative", lexicon=(), builtin=True)
    assert base["spoken_form_schema"] == SPOKEN_FORM_VERSION
    assert base != spoken_identity(numbers="standard", lexicon=(), builtin=True)
    assert base != spoken_identity(numbers="conservative", lexicon=(), builtin=False)
    assert base != spoken_identity(
        numbers="conservative", lexicon=(("a", "b"),), builtin=True
    )


def test_identity_is_stable_across_calls() -> None:
    """The payload is a pure function of its inputs."""
    first = spoken_identity(numbers="standard", lexicon=(("a", "b"),), builtin=True)
    second = spoken_identity(numbers="standard", lexicon=(("a", "b"),), builtin=True)
    assert first == second
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_spoken_form.py -v --no-cov`
Expected: FAIL — `ImportError: cannot import name 'to_spoken'`

- [ ] **Step 3: Write the implementation**

Replace `src/kenkui/_domain/spoken/__init__.py` with:

```python
"""Deterministic speech-form text transformation.

The canonical normalized text stays the authority for billing, inspection,
chapter identity, and attribution offsets. This stage derives the separate
string the synthesis engine actually speaks, and nothing else consumes it.
"""

from __future__ import annotations

import hashlib
import json

from kenkui._domain.spoken.lexicon import LEXICON_VERSION, lexicon_rules
from kenkui._domain.spoken.numbers import NumberTier, Rule, number_rules

SPOKEN_FORM_VERSION = "spoken-form-v1"


def _rules(
    numbers: NumberTier, lexicon: tuple[tuple[str, str], ...], *, builtin: bool
) -> tuple[Rule, ...]:
    """Rank caller entries, then built-in entries, then number rules."""
    return (*lexicon_rules(lexicon, builtin=builtin), *number_rules(numbers))


def to_spoken(
    text: str,
    *,
    numbers: NumberTier,
    lexicon: tuple[tuple[str, str], ...],
    builtin: bool,
) -> str:
    """Return the string the engine should speak for this canonical text.

    One left-to-right pass. At each position the first accepting rule wins and
    its output is emitted verbatim; emitted output is never re-examined, so
    rules can neither cascade nor loop.
    """
    rules = _rules(numbers, lexicon, builtin=builtin)
    if not rules:
        return text
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


def spoken_identity(
    *,
    numbers: NumberTier,
    lexicon: tuple[tuple[str, str], ...],
    builtin: bool,
) -> dict[str, object]:
    """Return the identity fields this configuration contributes to a segment.

    Every input that can change ``to_spoken`` output appears here, so a segment
    rendered under one configuration can never collide with another in the
    cache.
    """
    payload = json.dumps(
        {"builtin": builtin, "entries": [list(pair) for pair in lexicon]},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "spoken_form_schema": SPOKEN_FORM_VERSION,
        "numbers_tier": numbers,
        "lexicon_identity": (
            f"{LEXICON_VERSION if builtin else 'none'}:"
            f"{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"
        ),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_spoken_form.py -v --no-cov`
Expected: PASS.

- [ ] **Step 5: Run lint, types, and the full suite**

```console
uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest
```

- [ ] **Step 6: Commit**

```bash
git add src/kenkui/_domain/spoken/__init__.py tests/test_spoken_form.py
git commit --signoff -m "feat: add single-pass spoken-form matcher"
```

---
## Task 6: The `SpokenForm` operation and `Pipeline.pronounce()`

**Files:**
- Modify: `src/kenkui/_domain/operations.py`
- Modify: `src/kenkui/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `validate_entries` from Task 4.
- Produces: `SpokenForm` dataclass with fields `numbers: str`, `builtin_lexicon: bool`, `lexicon: tuple[tuple[str, str], ...]`; `Pipeline.pronounce(...)`. Task 7 reads the record out of `pipeline.operations`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:

```python
def test_pronounce_records_intent_without_effects() -> None:
    """The operation captures configuration as an immutable value."""
    pipeline = kk.epub("book.epub").pronounce({"Cthulhu": "kuh-THOO-loo"})
    recorded = pipeline.operations[0]
    assert isinstance(recorded, SpokenForm)
    assert recorded.numbers == "conservative"
    assert recorded.builtin_lexicon is True
    assert recorded.lexicon == (("Cthulhu", "kuh-THOO-loo"),)


def test_pronounce_is_branchable_and_absent_by_default() -> None:
    """A pipeline that never calls pronounce records no spoken-form intent."""
    root = kk.epub("book.epub")
    branch = root.pronounce()
    assert root.operations == ()
    assert any(isinstance(item, SpokenForm) for item in branch.operations)


def test_pronounce_rejects_an_unknown_tier() -> None:
    """Only the four defined tiers are accepted."""
    with pytest.raises(kk.ValidationError) as error:
        kk.epub("book.epub").pronounce(numbers="wild")
    assert error.value.code is kk.ErrorCode.INVALID_PRONUNCIATION


def test_pronounce_rejects_a_malformed_entry() -> None:
    """Caller entries are validated at the Pipeline boundary, not at render."""
    with pytest.raises(kk.ValidationError) as error:
        kk.epub("book.epub").pronounce({"": "x"})
    assert error.value.code is kk.ErrorCode.INVALID_PRONUNCIATION


def test_pronounce_cannot_be_requested_twice() -> None:
    """Duplicate operations are refused by the existing append rule."""
    with pytest.raises(kk.ValidationError) as error:
        kk.epub("book.epub").pronounce().pronounce()
    assert error.value.code is kk.ErrorCode.DUPLICATE_OPERATION
```

Add to the imports at the top of `tests/test_pipeline.py`:

```python
from kenkui._domain.operations import SpokenForm
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -k pronounce -v --no-cov`
Expected: FAIL — `ImportError: cannot import name 'SpokenForm'`

- [ ] **Step 3: Add the operation record**

In `src/kenkui/_domain/operations.py`, add after `NormalizeText`:

```python
@dataclass(frozen=True, slots=True)
class SpokenForm:
    """How canonical text becomes the string the engine actually speaks.

    Never affects the canonical text, and therefore never affects billing,
    inspection, chapter identity, or attribution offsets.
    """

    numbers: str = "conservative"
    builtin_lexicon: bool = True
    # Sorted pairs rather than a mapping: an operation record must be hashable
    # and compare equal regardless of how the caller ordered it.
    lexicon: tuple[tuple[str, str], ...] = ()
```

Extend the `Operation` union with `| SpokenForm`.

- [ ] **Step 4: Add the Pipeline method**

In `src/kenkui/pipeline.py`, add `SpokenForm` to the `._domain.operations` import
block, and add this method immediately after `normalize_text`:

```python
    def pronounce(
        self,
        lexicon: Mapping[str, str] | None = None,
        *,
        numbers: str = "conservative",
        builtin: bool = True,
    ) -> Pipeline:
        """Return a branch controlling how text is spoken rather than counted.

        Off unless called. Canonical text, and therefore the billable
        character count, is unaffected either way.
        """
        from ._domain.spoken.lexicon import validate_entries  # noqa: PLC0415

        if numbers not in _NUMBER_TIERS:
            raise ValidationError(ErrorCode.INVALID_PRONUNCIATION)
        return self._append(
            SpokenForm(
                numbers=numbers,
                builtin_lexicon=builtin,
                lexicon=validate_entries(lexicon or {}),
            ),
            before_tts=True,
        )
```

and add the module constant beside `_HASH_CHUNK_BYTES`:

```python
_NUMBER_TIERS = frozenset({"off", "conservative", "standard", "aggressive"})
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_pipeline.py -k pronounce -v --no-cov`
Expected: PASS.

- [ ] **Step 6: Run lint, types, and the full suite**

```console
uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest
```

`tests/test_package.py` may assert the exported operation count; if it fails,
update the expected count and note that `SpokenForm` is intentionally not
re-exported from `kenkui/__init__.py` (only `MetadataIntent` is public today).

- [ ] **Step 7: Commit**

```bash
git add src/kenkui/_domain/operations.py src/kenkui/pipeline.py tests/test_pipeline.py
git commit --signoff -m "feat: add SpokenForm operation and Pipeline.pronounce"
```

---

## Task 7: Wire spoken form into planning, with the D6 guard

This is the task that can silently break every existing cache. The guard test is
written and its golden values captured **before** any planner edit, so the
regression is caught the moment it appears.

**Files:**
- Create: `tests/test_identity_stability.py`
- Create: `tests/data/identity-golden.json`
- Modify: `src/kenkui/_domain/planning.py`
- Test: `tests/test_planning.py`

**Interfaces:**
- Consumes: `to_spoken`, `spoken_identity` from Task 5; `SpokenForm` from Task 6.
- Produces: `compile_execution_plan` gains spoken-form behaviour; `SchemaVersions` gains `spoken_form: str | None`. Task 10 adds `structure` to the same dataclass.

- [ ] **Step 1: Write the guard test**

Create `tests/test_identity_stability.py`:

```python
"""D6: a pipeline requesting nothing new must render byte-identically.

Segment identities are cache keys. If they shift for a pipeline that asked for
neither pronunciation nor pauses, every user silently re-synthesizes an entire
book. The golden file is captured from the pre-change planner; this test is the
only thing standing between a refactor and that outcome.
"""

from __future__ import annotations

import json
from pathlib import Path

import kenkui as kk
from kenkui._domain.planning import compile_execution_plan

GOLDEN = Path(__file__).parent / "data" / "identity-golden.json"
SOURCE_HASH = "1" * 64
MODEL_REVISION = "pocket-tts/model@0123456789abcdef"
TEXT = (
    "It was 100,000 to one. The cello sounded in 1984, and the 3rd movement "
    "began.\n\nHe waited by the door for a long while, thinking of nothing at "
    "all, and then he left without speaking to anyone."
)


def voice() -> kk.Voice:
    """Build the resolved voice the planner converts into a VoicePlan."""
    return kk.Voice(
        id="eponine",
        name="Eponine",
        enabled=True,
        provenance="Project-owned recording by Test Speaker",
        license_id="CC0-1.0",
        commercial_use_allowed=True,
        language="en-US",
        content_fingerprint="2" * 64,
        compatible_model_revisions=(MODEL_REVISION,),
        state="loaded",
    )


def inspection() -> kk.BookInspection:
    """Build a one-chapter inspection with prose the new stages would change."""
    chapter = kk.ChapterInspection("ch-v1-one", 0, "Chapter One", len(TEXT), TEXT)
    return kk.BookInspection(
        kk.BookMetadata("Source Title", "Source Author", cover_available=True),
        (chapter,),
    )


def plain_plan() -> object:
    """Compile the plan for a pipeline that requests nothing new."""
    pipeline = kk.epub("book.epub").assign_voice("eponine").tts()
    return compile_execution_plan(
        pipeline,
        inspection(),
        source_bytes_hash=SOURCE_HASH,
        resolved_voice=voice(),
        model_revision=MODEL_REVISION,
    )


def snapshot() -> dict[str, object]:
    """Reduce a plan to the values that must never drift."""
    plan = plain_plan()
    return {
        "fingerprint": plan.semantic_fingerprint,  # type: ignore[attr-defined]
        "segment_ids": [s.id for s in plan.segments],  # type: ignore[attr-defined]
        "segment_texts": [s.text for s in plan.segments],  # type: ignore[attr-defined]
        "total": plan.total_speech_characters,  # type: ignore[attr-defined]
    }


def test_plain_pipeline_identity_is_unchanged() -> None:
    """Segment IDs, texts, fingerprint, and billable total all hold steady."""
    expected = json.loads(GOLDEN.read_text("utf-8"))
    assert snapshot() == expected


def test_billable_total_equals_canonical_text_length() -> None:
    """normalized_speech_characters describes the source, not the spoken form."""
    plan = plain_plan()
    assert plan.total_speech_characters == len(TEXT)  # type: ignore[attr-defined]
```

- [ ] **Step 2: Capture the golden file from the UNMODIFIED planner**

Make no planner change yet. Run:

```console
mkdir -p tests/data
uv run python -c "
import json, pathlib, sys
sys.path.insert(0, 'tests')
from test_identity_stability import snapshot
pathlib.Path('tests/data/identity-golden.json').write_text(
    json.dumps(snapshot(), indent=2, sort_keys=True) + chr(10), encoding='utf-8')
print('captured')
"
```

- [ ] **Step 3: Verify the guard passes against unmodified code**

Run: `uv run pytest tests/test_identity_stability.py -v --no-cov`
Expected: PASS — both tests. If `test_billable_total_equals_canonical_text_length`
fails here, the pre-existing `total` already disagrees with canonical length and
that must be understood before continuing.

- [ ] **Step 4: Commit the guard before touching the planner**

```bash
git add tests/test_identity_stability.py tests/data/identity-golden.json
git commit --signoff -m "test: pin plain-pipeline segment identity before spoken form"
```

- [ ] **Step 5: Write the failing test for the new behaviour**

Append to `tests/test_identity_stability.py`:

```python
def spoken_plan(**kwargs: object) -> object:
    """Compile the same book with a pronounce() request attached."""
    pipeline = (
        kk.epub("book.epub")
        .pronounce(**kwargs)  # type: ignore[arg-type]
        .assign_voice("eponine")
        .tts()
    )
    return compile_execution_plan(
        pipeline,
        inspection(),
        source_bytes_hash=SOURCE_HASH,
        resolved_voice=voice(),
        model_revision=MODEL_REVISION,
    )


def test_spoken_form_changes_segment_text_but_not_the_bill() -> None:
    """The engine hears words; the meter still counts the source characters."""
    plan = spoken_plan()
    spoken_text = "".join(s.text for s in plan.segments)  # type: ignore[attr-defined]
    assert "one hundred thousand" in spoken_text
    assert "chello" in spoken_text
    assert "100,000" not in spoken_text
    assert plan.total_speech_characters == len(TEXT)  # type: ignore[attr-defined]


def test_spoken_form_changes_segment_identity() -> None:
    """Different spoken output must never reuse a plain pipeline's cache entry."""
    plain = {s.id for s in plain_plan().segments}  # type: ignore[attr-defined]
    spoken = {s.id for s in spoken_plan().segments}  # type: ignore[attr-defined]
    assert plain.isdisjoint(spoken)


def test_non_english_narrator_disables_the_stage() -> None:
    """A voice that cannot speak English number-words leaves text alone."""
    french = kk.Voice(
        id="eponine",
        name="Eponine",
        enabled=True,
        provenance="Project-owned recording by Test Speaker",
        license_id="CC0-1.0",
        commercial_use_allowed=True,
        language="fr-FR",
        content_fingerprint="2" * 64,
        compatible_model_revisions=(MODEL_REVISION,),
        state="loaded",
    )
    pipeline = kk.epub("book.epub").pronounce().assign_voice("eponine").tts()
    plan = compile_execution_plan(
        pipeline,
        inspection(),
        source_bytes_hash=SOURCE_HASH,
        resolved_voice=french,
        model_revision=MODEL_REVISION,
    )
    assert "100,000" in "".join(s.text for s in plan.segments)
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/test_identity_stability.py -v --no-cov`
Expected: the three new tests FAIL (spoken form is not wired in yet); the two
Step 1 tests still PASS.

- [ ] **Step 7: Wire the planner**

In `src/kenkui/_domain/planning.py`:

Add to the operations import: `SpokenForm`. Do **not** redeclare the version
string — import the single definition from Task 5 so the two can never drift:

```python
from kenkui._domain.spoken import (
    SPOKEN_FORM_VERSION,
    spoken_identity,
    to_spoken,
)
```

Add `spoken_form: str | None = None` as the last field of `SchemaVersions`.

Replace the `total` computation in `compile_execution_plan` with:

```python
    spoken = _one_operation(pipeline.operations, SpokenForm)
    if spoken is not None and not narrator.language.lower().startswith("en"):
        # The number words and lexicon are English. Mangling a French book is
        # worse than leaving it, so the stage disables itself rather than
        # asking the caller to know this.
        spoken = None

    segments = _compile_segments(inspection.chapters, spans, cast, spoken)
    if not segments:
        raise ValidationError(ErrorCode.EMPTY_SPEECH)
    # Canonical characters, not spoken characters: this is the bill, and it
    # must describe the book the caller supplied.
    total = sum(len(chapter.text) for chapter in inspection.chapters)
```

and build `schemas` with:

```python
    schemas = SchemaVersions(
        parser=PARSER_SCHEMA_VERSION,
        normalization=NORMALIZATION_SCHEMA_VERSION,
        planning=PLANNING_SCHEMA_VERSION,
        render=RENDER_SCHEMA_VERSION,
        spoken_form=SPOKEN_FORM_VERSION if spoken is not None else None,
    )
```

Change `_compile_segments` to accept and apply the config:

```python
def _compile_segments(
    chapters: tuple[ChapterInspection, ...],
    spans: tuple[SpeakerSpan, ...],
    cast_plan: CastPlan,
    spoken: SpokenForm | None = None,
) -> tuple[SpeechSegment, ...]:
    """Split each speaker span while assigning one global plan-order ordinal.

    Spans partition the canonical chapter, and the frozen chunker runs inside
    each one. When spoken form is active the chunker runs over the spoken
    string instead, so chunks join to ``to_spoken(span)`` rather than to the
    span -- exactness one level down, not exactness lost.
    """
    result: list[SpeechSegment] = []
    for chapter in chapters:
        for span in _spans_for(chapter, spans):
            voice = cast_plan.voice_for(span.character_id)
            text = chapter.text[span.start : span.end]
            if spoken is not None:
                text = to_spoken(
                    text,
                    numbers=cast("NumberTier", spoken.numbers),
                    lexicon=spoken.lexicon,
                    builtin=spoken.builtin_lexicon,
                )
            for chunk_index, chunk in enumerate(_chunk_span(chapter, text)):
                result.append(
                    _segment(
                        chapter,
                        len(result),
                        chunk_index,
                        chunk,
                        speaker_id=span.character_id,
                        voice_id=voice.id,
                        spoken=spoken,
                    )
                )
    return tuple(result)
```

`_chunk_span` currently validates `chapter.speech_characters != len(chapter.text)`
against the text it is chunking. Leave that chapter-level validation exactly as
written — it still refers to `chapter.text`, which is canonical and unchanged —
and only the `not text` guard now refers to the spoken string.

Extend `_segment` with the new keyword and identity fields:

```python
def _segment(  # noqa: PLR0913 - each field is part of a distinct identity.
    chapter: ChapterInspection,
    ordinal: int,
    chunk_index: int,
    text: str,
    *,
    speaker_id: str | None = None,
    voice_id: str = "",
    spoken: SpokenForm | None = None,
) -> SpeechSegment:
```

and inside, after the existing `speaker_id` block:

```python
    if spoken is not None:
        # Added only when the stage is active, so a plain pipeline's identities
        # -- and therefore every existing cache entry -- stay byte-identical.
        fields.update(
            spoken_identity(
                numbers=cast("NumberTier", spoken.numbers),
                lexicon=spoken.lexicon,
                builtin=spoken.builtin_lexicon,
            )
        )
```

Add to the `TYPE_CHECKING` block:

```python
    from kenkui._domain.spoken.numbers import NumberTier
```

Finally, add `spoken_form` to the `_fingerprint` payload's `schema_versions`
block so a plan compiled with the stage active never collides with one without:

```python
            "spoken_form": schemas.spoken_form,
```

- [ ] **Step 8: Run the guard and the new tests**

Run: `uv run pytest tests/test_identity_stability.py -v --no-cov`
Expected: ALL PASS. **If `test_plain_pipeline_identity_is_unchanged` fails, stop.**
The golden file is correct by construction; a failure means a new identity field
or the `total` change leaked into the no-request path. Do not regenerate the
golden file to make it pass — that defeats its only purpose.

- [ ] **Step 9: Run lint, types, and the full suite**

```console
uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest
```

`tests/test_planning.py` may assert `SchemaVersions` field count or the exact
fingerprint payload; update those assertions to include `spoken_form=None` for
plain pipelines.

- [ ] **Step 10: Commit**

```bash
git add src/kenkui/_domain/planning.py tests/test_identity_stability.py tests/test_planning.py
git commit --signoff -m "feat: apply spoken form during segment compilation"
```

---
# Phase 2 — Structural Pauses

## Task 8: Record heading text on chapters

**Spec refinement — read before implementing.** The spec says the parser records
heading *character offsets*. Implemented literally that fails: `_chapter_text`
runs `normalize_text()` over the whole emitted string, which collapses
whitespace runs and shifts every raw offset. Recovering them would need exactly
the offset map that §4.2 of the spec was written to avoid.

Instead the parser records the **normalized heading strings**, which
`_visible_headings` already computes for the chapter title. Task 9 then locates
headings by matching whole blocks of normalized text. A prose paragraph that is
byte-identical to a heading would gain one spurious pause; that is a far smaller
cost than an offset map, and it cannot corrupt text or identity.

**Files:**
- Modify: `src/kenkui/inspection.py`
- Modify: `src/kenkui/_epub/parser.py`
- Test: `tests/test_epub.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ChapterInspection.headings: tuple[str, ...]` (defaulted to `()`), populated in spine order by the parser. Task 9 and Task 11 read it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_epub.py`:

```python
def test_chapter_records_every_visible_heading() -> None:
    """Headings are captured as normalized strings, title first."""
    book = build_epub(
        {
            "chapter1.xhtml": (
                "<h1>Chapter One</h1><p>He woke.</p><h2>A Section</h2><p>She slept.</p>"
            )
        }
    )
    chapter = inspect_epub(book).chapters[0]
    assert chapter.headings == ("Chapter One", "A Section")
    assert chapter.title == "Chapter One"


def test_chapter_without_headings_records_none() -> None:
    """A chapter with no h1-h6 carries an empty heading tuple."""
    book = build_epub({"chapter1.xhtml": "<p>Just prose.</p>"})
    assert inspect_epub(book).chapters[0].headings == ()
```

Reuse whatever EPUB-building helper `tests/test_epub.py` already defines; if it
is named differently from `build_epub`, use that name rather than adding one.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_epub.py -k heading -v --no-cov`
Expected: FAIL — `AttributeError: 'ChapterInspection' object has no attribute 'headings'`

- [ ] **Step 3: Add the field**

In `src/kenkui/inspection.py`, add as the last field of `ChapterInspection`:

```python
    headings: tuple[str, ...] = ()
```

It must be last and defaulted: `pipeline.inspect()` reconstructs
`BookInspection` positionally, and several tests build `ChapterInspection`
with five positional arguments.

- [ ] **Step 4: Populate it in the parser**

In `src/kenkui/_epub/parser.py`, change `_chapter_text` to return a 3-tuple.
Its signature becomes `-> tuple[str, str, tuple[str, ...]]`, the empty return
becomes `return "", "", ()`, and the final return becomes:

```python
    return title, text, tuple(headings)
```

At the spine call site, change the unpacking and the construction:

```python
        title, text, headings = material
```

```python
        chapters.append(
            ChapterInspection(
                chapter_id(member, occurrence, fragment),
                index,
                title or f"Chapter {index + 1}",
                len(text),
                text,
                headings,
            )
        )
```

`material_cache` needs no change — it stores whatever tuple `_chapter_text`
returns.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_epub.py -v --no-cov`
Expected: PASS.

- [ ] **Step 6: Run lint, types, and the full suite**

```console
uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest
```

`tests/test_identity_stability.py` must still pass — headings are metadata only
and touch neither text nor identity.

- [ ] **Step 7: Commit**

```bash
git add src/kenkui/inspection.py src/kenkui/_epub/parser.py tests/test_epub.py
git commit --signoff -m "feat: record visible heading text on chapter inspection"
```

---

## Task 9: Structural split and the gap model

**Files:**
- Create: `src/kenkui/_domain/structure.py`
- Test: `tests/test_structure.py`

**Interfaces:**
- Consumes: `Pauses` is defined in Task 10, so this task defines the split against a **structural protocol** instead, and Task 10's dataclass satisfies it.
- Produces: `STRUCTURE_SCHEMA_VERSION: str`, `Piece` dataclass with `text: str` and `reasons: frozenset[str]`, `PauseSpec` protocol, `break_tiers(pauses: PauseSpec) -> tuple[str, ...]`, `split_structural(text: str, headings: frozenset[str], pauses: PauseSpec) -> tuple[Piece, ...]`, `gap_ms(reasons: frozenset[str], pauses: PauseSpec) -> int`, and the reason constants `CHAPTER`, `HEADING_BEFORE`, `HEADING_AFTER`, `PARAGRAPH`, `LINE`. Tasks 11 and 12 consume all of these.

- [ ] **Step 1: Write the failing test**

Create `tests/test_structure.py`:

```python
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
    PARAGRAPH,
    break_tiers,
    gap_ms,
    split_structural,
)


@dataclass(frozen=True, slots=True)
class Spec:
    """Stand-in for the Task 10 operation record."""

    chapter_ms: int = 0
    heading_before_ms: int = 0
    heading_after_ms: int = 0
    paragraph_ms: int = 0
    line_ms: int = 0


HEADINGS = frozenset({"Chapter One", "A Section"})


def test_no_tier_yields_exactly_one_piece() -> None:
    """With nothing enabled the split is a no-op, which is what keeps v2 alive."""
    text = "Chapter One\n\nHe woke.\n\nShe slept."
    assert split_structural(text, HEADINGS, Spec()) == (
        split_structural(text, HEADINGS, Spec())[0],
    )
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
    spec = Spec(chapter_ms=1500, heading_before_ms=400)
    assert gap_ms(frozenset({CHAPTER, HEADING_BEFORE}), spec) == 1500
    assert gap_ms(frozenset({CHAPTER}), spec) == 1500
    assert gap_ms(frozenset({HEADING_BEFORE}), spec) == 400
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
    rng = random.Random(seed)
    alphabet = ["a", "b", " ", "\n", "\n\n", "\n\n\n", ".", "Q"]
    for _ in range(500):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 40)))
        spec = Spec(*[rng.choice([0, 300]) for _ in range(5)])
        pieces = split_structural(text, frozenset({"Q", "a"}), spec)
        assert "".join(piece.text for piece in pieces) == text
        assert all(piece.text for piece in pieces)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_structure.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'kenkui._domain.structure'`

- [ ] **Step 3: Write the implementation**

Create `src/kenkui/_domain/structure.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_structure.py -v --no-cov`
Expected: PASS, including all 20 random-seed exactness cases.

- [ ] **Step 5: Run lint, types, and the full suite**

```console
uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest
```

- [ ] **Step 6: Commit**

```bash
git add src/kenkui/_domain/structure.py tests/test_structure.py
git commit --signoff -m "feat: add structural split and gap model"
```

---
## Task 10: The `Pauses` operation and `Pipeline.pauses()`

**Files:**
- Modify: `src/kenkui/_domain/operations.py`
- Modify: `src/kenkui/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `INVALID_PAUSE` added in Task 4.
- Produces: `Pauses` dataclass with `chapter_ms: int = 0`, `heading_before_ms: int = 0`, `heading_after_ms: int = 0`, `paragraph_ms: int = 0`, `line_ms: int = 0` — structurally satisfying Task 9's `PauseSpec`; plus `Pipeline.pauses(...)`. Task 11 reads the record from `pipeline.operations`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:

```python
def test_pauses_records_five_independent_durations() -> None:
    """Each boundary kind is separately variable."""
    pipeline = kk.epub("book.epub").pauses(
        chapter_ms=1500, heading_after_ms=600, paragraph_ms=250
    )
    recorded = pipeline.operations[0]
    assert isinstance(recorded, Pauses)
    assert recorded.chapter_ms == 1500
    assert recorded.heading_after_ms == 600
    assert recorded.paragraph_ms == 250
    assert recorded.heading_before_ms == 0
    assert recorded.line_ms == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chapter_ms": -1},
        {"line_ms": -100},
        {"paragraph_ms": 60_001},
    ],
)
def test_pauses_rejects_out_of_range_durations(kwargs: dict[str, int]) -> None:
    """Negative and absurd durations are refused at the Pipeline boundary."""
    with pytest.raises(kk.ValidationError) as error:
        kk.epub("book.epub").pauses(**kwargs)
    assert error.value.code is kk.ErrorCode.INVALID_PAUSE


def test_pauses_defaults_to_silence_free() -> None:
    """Calling pauses with no argument enables nothing."""
    recorded = kk.epub("book.epub").pauses().operations[0]
    assert isinstance(recorded, Pauses)
    assert recorded == Pauses()
```

Add `Pauses` to the `kenkui._domain.operations` import in the test file.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -k pauses -v --no-cov`
Expected: FAIL — `ImportError: cannot import name 'Pauses'`

- [ ] **Step 3: Add the operation record**

In `src/kenkui/_domain/operations.py`, add after `SpokenForm`:

```python
@dataclass(frozen=True, slots=True)
class Pauses:
    """Silence durations for each structural boundary, in milliseconds.

    Zero disables a tier completely, including the chunk-break cost it would
    otherwise impose. Structurally satisfies the domain ``PauseSpec`` protocol.
    """

    chapter_ms: int = 0
    heading_before_ms: int = 0
    heading_after_ms: int = 0
    paragraph_ms: int = 0
    line_ms: int = 0
```

Extend the `Operation` union with `| Pauses`.

- [ ] **Step 4: Add the Pipeline method**

In `src/kenkui/pipeline.py`, add `Pauses` to the operations import, add the
bound beside `_NUMBER_TIERS`:

```python
_MAX_PAUSE_MS = 60_000
```

and add the method immediately after `pronounce`:

```python
    def pauses(
        self,
        *,
        chapter_ms: int = 0,
        heading_before_ms: int = 0,
        heading_after_ms: int = 0,
        paragraph_ms: int = 0,
        line_ms: int = 0,
    ) -> Pipeline:
        """Return a branch requesting silence at structural boundaries.

        Off unless called. Durations are retunable without re-synthesis: only
        turning a tier on or off changes segment identity, because only that
        changes where a segment ends.
        """
        requested = (
            chapter_ms,
            heading_before_ms,
            heading_after_ms,
            paragraph_ms,
            line_ms,
        )
        for duration in requested:
            if (
                isinstance(duration, bool)
                or not isinstance(duration, int)
                or duration < 0
                or duration > _MAX_PAUSE_MS
            ):
                raise ValidationError(ErrorCode.INVALID_PAUSE)
        return self._append(Pauses(*requested), before_tts=True)
```

- [ ] **Step 5: Run test, lint, types, full suite**

```console
uv run pytest tests/test_pipeline.py -k pauses -v --no-cov
uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest
```

- [ ] **Step 6: Commit**

```bash
git add src/kenkui/_domain/operations.py src/kenkui/pipeline.py tests/test_pipeline.py
git commit --signoff -m "feat: add Pauses operation and Pipeline.pauses"
```

---

## Task 11: Chunking v3 and the plan's silence table

**Files:**
- Modify: `src/kenkui/_domain/planning.py`
- Test: `tests/test_identity_stability.py`, `tests/test_planning.py`

**Interfaces:**
- Consumes: `split_structural`, `break_tiers`, `gap_ms`, `CHAPTER`, `STRUCTURE_SCHEMA_VERSION` from Task 9; `Pauses` from Task 10.
- Produces: `ExecutionPlan.trailing_silence_ms: tuple[int, ...]` — one entry per segment, the gap **after** that segment; `CHUNKING_V3_SCHEMA_VERSION: str`; `SchemaVersions.structure: str | None`. Task 12 reads `trailing_silence_ms`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_identity_stability.py`:

```python
def paused_plan(**kwargs: object) -> object:
    """Compile the same book with a pauses() request attached."""
    pipeline = (
        kk.epub("book.epub")
        .pauses(**kwargs)  # type: ignore[arg-type]
        .assign_voice("eponine")
        .tts()
    )
    return compile_execution_plan(
        pipeline,
        inspection(),
        source_bytes_hash=SOURCE_HASH,
        resolved_voice=voice(),
        model_revision=MODEL_REVISION,
    )


def test_chapter_pause_alone_keeps_v2_identity() -> None:
    """chapter_ms forces no chunk break, so every cache entry stays valid."""
    plain = [s.id for s in plain_plan().segments]  # type: ignore[attr-defined]
    paused = paused_plan(chapter_ms=1500)
    assert [s.id for s in paused.segments] == plain  # type: ignore[attr-defined]


def test_paragraph_pause_changes_identity_and_adds_silence() -> None:
    """Enabling a break tier re-chunks and records a gap on the right segment."""
    plain = {s.id for s in plain_plan().segments}  # type: ignore[attr-defined]
    paused = paused_plan(paragraph_ms=250)
    assert {s.id for s in paused.segments}.isdisjoint(plain)  # type: ignore[attr-defined]
    silence = paused.trailing_silence_ms  # type: ignore[attr-defined]
    assert len(silence) == len(paused.segments)  # type: ignore[attr-defined]
    assert 250 in silence


def test_silence_table_is_all_zero_without_pauses() -> None:
    """A plain pipeline records no silence anywhere."""
    plan = plain_plan()
    assert set(plan.trailing_silence_ms) == {0}  # type: ignore[attr-defined]


def test_final_segment_has_no_trailing_silence() -> None:
    """A book must not end on dead air."""
    plan = paused_plan(chapter_ms=1500, paragraph_ms=250)
    assert plan.trailing_silence_ms[-1] == 0  # type: ignore[attr-defined]


def test_retuning_a_duration_does_not_change_identity() -> None:
    """Changing 250ms to 600ms must cost no re-synthesis."""
    first = [s.id for s in paused_plan(paragraph_ms=250).segments]  # type: ignore[attr-defined]
    second = [s.id for s in paused_plan(paragraph_ms=600).segments]  # type: ignore[attr-defined]
    assert first == second
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_identity_stability.py -v --no-cov`
Expected: the five new tests FAIL; the earlier tests still PASS.

- [ ] **Step 3: Write the implementation**

In `src/kenkui/_domain/planning.py`:

Add imports:

```python
from kenkui._domain.operations import Pauses
from kenkui._domain.structure import (
    CHAPTER,
    STRUCTURE_SCHEMA_VERSION,
    break_tiers,
    gap_ms,
    split_structural,
)
```

Add the constant and the neutral default beside the other schema versions:

```python
CHUNKING_V3_SCHEMA_VERSION = "tts-chunks-v3"
_NO_PAUSES = Pauses()
```

Add `structure: str | None = None` to `SchemaVersions` and
`trailing_silence_ms: tuple[int, ...] = ()` as the last field of `ExecutionPlan`.

Rewrite `_compile_segments` to return both tables:

```python
def _compile_segments(
    chapters: tuple[ChapterInspection, ...],
    spans: tuple[SpeakerSpan, ...],
    cast_plan: CastPlan,
    spoken: SpokenForm | None = None,
    pauses: Pauses = _NO_PAUSES,
) -> tuple[tuple[SpeechSegment, ...], tuple[int, ...]]:
    """Split spans structurally, speak them, chunk them, and record the gaps.

    Order is split -> spoken -> chunk. Splitting first, in canonical
    coordinates, means spoken form cannot move a boundary that has already
    been decided, which is what removes the need for an offset map.
    """
    tiers = break_tiers(pauses)
    result: list[SpeechSegment] = []
    silence: list[int] = []
    for chapter_index, chapter in enumerate(chapters):
        headings = frozenset(chapter.headings)
        for span in _spans_for(chapter, spans):
            voice = cast_plan.voice_for(span.character_id)
            span_text = chapter.text[span.start : span.end]
            for piece in split_structural(span_text, headings, pauses):
                text = piece.text
                if spoken is not None:
                    text = to_spoken(
                        text,
                        numbers=cast("NumberTier", spoken.numbers),
                        lexicon=spoken.lexicon,
                        builtin=spoken.builtin_lexicon,
                    )
                for chunk_index, chunk in enumerate(_chunk_span(chapter, text)):
                    result.append(
                        _segment(
                            chapter,
                            len(result),
                            chunk_index,
                            chunk,
                            speaker_id=span.character_id,
                            voice_id=voice.id,
                            spoken=spoken,
                            tiers=tiers,
                        )
                    )
                    silence.append(0)
                if silence:
                    silence[-1] = gap_ms(piece.reasons, pauses)
        # The inter-chapter gap folds into this chapter's last segment, so
        # chapter N+1 begins exactly on its first spoken word. Max, not sum:
        # a chapter end meeting a heading-before pause is one gap.
        if silence and chapter_index + 1 < len(chapters):
            silence[-1] = max(silence[-1], gap_ms(frozenset({CHAPTER}), pauses))
    if silence:
        silence[-1] = 0  # A book must not end on dead air.
    return tuple(result), tuple(silence)
```

Extend `_segment` with `tiers: tuple[str, ...] = ()` and, inside, replace the
fixed chunking schema field and add the structural identity:

```python
        "chunking_schema": (
            CHUNKING_V3_SCHEMA_VERSION if tiers else CHUNKING_SCHEMA_VERSION
        ),
```

```python
    if tiers:
        # Added only when a break tier is active, so a plain pipeline's
        # identities stay byte-identical.
        fields["structure_schema"] = STRUCTURE_SCHEMA_VERSION
        fields["break_tiers"] = list(tiers)
```

In `compile_execution_plan`, resolve the record and thread it through:

```python
    pauses = _one_operation(pipeline.operations, Pauses) or _NO_PAUSES
    segments, trailing_silence = _compile_segments(
        inspection.chapters, spans, cast, spoken, pauses
    )
```

Set `structure=STRUCTURE_SCHEMA_VERSION if break_tiers(pauses) else None` in
`SchemaVersions`, pass `trailing_silence_ms=trailing_silence` into
`ExecutionPlan`, and add to the `_fingerprint` payload — inside
`schema_versions` add `"structure": schemas.structure`, and at the top level
add:

```python
        "trailing_silence_ms": list(material.trailing_silence),
```

with a matching `trailing_silence: tuple[int, ...]` field on `_PlanMaterial`
so retuning a duration still changes the plan fingerprint even though it does
not change any segment identity.

- [ ] **Step 4: Implement `tts-chunks-v3` as v2, restricted**

No change to `_chunk_span` is required. The v3 behaviour *is* `split_structural`
feeding the unmodified v2 chunker one piece at a time — concatenation-exactness
is inherited rather than re-proved, and `MIN_BREAK_FILL` and
`MAX_SEPARATOR_FREE_CHARACTERS` are not forked. Confirm by reading
`_chunk_span`: it must remain byte-for-byte as Task 7 left it.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_identity_stability.py tests/test_structure.py -v --no-cov`
Expected: ALL PASS. **`test_plain_pipeline_identity_is_unchanged` must still pass.**

- [ ] **Step 6: Run lint, types, and the full suite**

```console
uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest
```

Callers of `_compile_segments` in `tests/test_planning*.py` now receive a
2-tuple; update those unpackings.

- [ ] **Step 7: Commit**

```bash
git add src/kenkui/_domain/planning.py tests/test_identity_stability.py tests/test_planning.py
git commit --signoff -m "feat: add tts-chunks-v3 and the plan silence table"
```

---

## Task 12: Render silence into the chapter parts

`SegmentAudio.byte_count` is a **derived** property of `frame_count`
(`src/kenkui/_tts/protocols.py:53-56`). Padding `frame_count` after validation
therefore corrects the part-size check in `_validate_pcm_parts`, the chapter
markers from `chapter_frame_boundaries_ms`, and the reported duration all at
once. **No change is needed in `_audio/production.py` or `_audio/m4b.py`.**

**Files:**
- Modify: `src/kenkui/_execution/coordinator.py`
- Test: `tests/test_pauses_render.py`

**Interfaces:**
- Consumes: `ExecutionPlan.trailing_silence_ms` from Task 11.
- Produces: no new public names; `_render` writes silence frames into the spilled chapter parts.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pauses_render.py`:

```python
"""Silence padding must agree with part size, chapter markers, and duration."""

from __future__ import annotations

from dataclasses import replace

from kenkui._audio.m4b import chapter_frame_boundaries_ms
from kenkui._tts.protocols import SegmentAudio

SAMPLE_RATE = 24_000


def audio(frames: int, chapter: str = "ch-1", segment: str = "seg-1") -> SegmentAudio:
    """Build one segment's post-render metadata."""
    return SegmentAudio(
        segment, chapter, SAMPLE_RATE, 1, frames, frames * 1000 // SAMPLE_RATE
    )


def padded(item: SegmentAudio, silence_ms: int) -> SegmentAudio:
    """Apply the same padding the coordinator applies, for arithmetic checks."""
    frames = item.frame_count + silence_ms * item.sample_rate_hz // 1000
    return replace(
        item,
        frame_count=frames,
        duration_ms=frames * 1000 // item.sample_rate_hz,
    )


def test_padding_extends_byte_count_consistently() -> None:
    """byte_count is derived, so one padded field corrects the part-size check."""
    item = padded(audio(SAMPLE_RATE), 500)
    assert item.frame_count == SAMPLE_RATE + SAMPLE_RATE // 2
    assert item.byte_count == item.frame_count * item.channels * 2
    assert item.duration_ms == 1500


def test_zero_silence_leaves_metadata_untouched() -> None:
    """A plain pipeline's audio metadata is bit-for-bit what the worker produced."""
    item = audio(SAMPLE_RATE)
    assert padded(item, 0) == item


MODEL_REVISION = "pocket-tts/model@0123456789abcdef"


def two_chapter_plan(chapter_ms: int) -> object:
    """Compile a real two-chapter plan so the arithmetic is not stubbed."""
    voice = kk.Voice(
        id="eponine",
        name="Eponine",
        enabled=True,
        provenance="Project-owned recording by Test Speaker",
        license_id="CC0-1.0",
        commercial_use_allowed=True,
        language="en-US",
        content_fingerprint="2" * 64,
        compatible_model_revisions=(MODEL_REVISION,),
        state="loaded",
    )
    chapters = tuple(
        kk.ChapterInspection(f"ch-v1-{n}", i, f"Chapter {n}", len(t), t)
        for i, (n, t) in enumerate(
            (("one", "He woke early.\n\nShe slept on."), ("two", "They left."))
        )
    )
    pipeline = (
        kk.epub("book.epub").pauses(chapter_ms=chapter_ms).assign_voice("eponine").tts()
    )
    return compile_execution_plan(
        pipeline,
        kk.BookInspection(kk.BookMetadata("T", "A", cover_available=True), chapters),
        source_bytes_hash="1" * 64,
        resolved_voice=voice,
        model_revision=MODEL_REVISION,
    )


def test_inter_chapter_gap_lands_in_the_preceding_chapter() -> None:
    """Skipping to a chapter must land on speech, not on silence."""
    plan = two_chapter_plan(1000)
    segments = plan.segments  # type: ignore[attr-defined]
    silence = plan.trailing_silence_ms  # type: ignore[attr-defined]
    last_of_first = max(
        i for i, s in enumerate(segments) if s.chapter_id == "ch-v1-one"
    )
    assert silence[last_of_first] == 1000
    assert silence[-1] == 0
```

Add to the test module imports:

```python
import kenkui as kk
from kenkui._domain.planning import compile_execution_plan
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pauses_render.py -v --no-cov`
Expected: FAIL until the padding helper exists in the coordinator.

- [ ] **Step 3: Write the implementation**

In `src/kenkui/_execution/coordinator.py`, add `replace` to the existing
`dataclasses` import, and add this helper beside `_spill_chapter`:

```python
def _padded(item: SegmentAudio, silence_ms: int) -> tuple[SegmentAudio, bytes]:
    """Extend one segment's metadata and produce its trailing silence bytes.

    Applied only after the raw worker output has been validated, so every
    existing audio invariant still governs what the worker actually sent.
    byte_count is derived from frame_count, so this single adjustment keeps
    the part-size check, the chapter markers, and the duration in agreement.
    """
    if silence_ms <= 0:
        return item, b""
    frames = silence_ms * item.sample_rate_hz // 1000
    if frames <= 0:
        return item, b""
    total = item.frame_count + frames
    extended = replace(
        item,
        frame_count=total,
        duration_ms=total * 1000 // item.sample_rate_hz,
    )
    return extended, bytes(frames * item.channels * 2)
```

In `_render`, replace the three lines beginning `rendered.append(segment_audio(item))`
with:

```python
entry, padding = _padded(segment_audio(item), plan.trailing_silence_ms[index])
# Silence occupies real bytes and must be charged against the budgets.
item_bytes += len(padding)
chapter_bytes += len(padding)
total_bytes += len(padding)
if chapter_bytes > MAX_CHAPTER_PCM_BYTES or total_bytes > MAX_TOTAL_PCM_BYTES:
    raise RenderError(ErrorCode.INVALID_AUDIO)
rendered.append(entry)
pending.append(item.pcm_s16le)
if padding:
    pending.append(padding)
```

and delete the now-duplicated budget check that preceded it, so the budget is
tested exactly once per segment against the padded size.

`plan.trailing_silence_ms` defaults to `()` on plans built by older tests;
guard the index with:

```python
    silence = plan.trailing_silence_ms or (0,) * len(plan.segments)
```

declared once before the loop, and index `silence[index]`.

- [ ] **Step 4: Cover the two remaining spec §13 integration requirements**

Append to `tests/test_pauses_render.py`:

```python
def test_stats_separate_the_bill_from_the_work() -> None:
    """Spoken form is exactly what makes the two statistics diverge.

    normalized_speech_characters is what a server charges for and must keep
    describing the source book; synthesized_characters describes the work the
    engine actually did.
    """
    text = "It cost 100,000 exactly."
    chapter = kk.ChapterInspection("ch-v1-one", 0, "One", len(text), text)
    inspection = kk.BookInspection(
        kk.BookMetadata("T", "A", cover_available=True), (chapter,)
    )
    voice = kk.Voice(
        id="eponine",
        name="Eponine",
        enabled=True,
        provenance="Project-owned recording by Test Speaker",
        license_id="CC0-1.0",
        commercial_use_allowed=True,
        language="en-US",
        content_fingerprint="2" * 64,
        compatible_model_revisions=(MODEL_REVISION,),
        state="loaded",
    )
    pipeline = kk.epub("book.epub").pronounce().assign_voice("eponine").tts()
    plan = compile_execution_plan(
        pipeline,
        inspection,
        source_bytes_hash="1" * 64,
        resolved_voice=voice,
        model_revision=MODEL_REVISION,
    )
    synthesized = sum(segment.character_count for segment in plan.segments)
    assert plan.total_speech_characters == len(text)
    assert synthesized > plan.total_speech_characters
```

Then add a cache-equivalence case to `tests/test_cache.py`, mirroring the
existing hit/miss test in that file but building the pipeline with
`.pronounce()` attached. It must assert that a second render of the same
spoken-form pipeline reuses cached PCM, and that a render with a *different*
`numbers` tier does not — spoken configuration is part of segment identity, so
the two must never share an entry.

Run: `uv run pytest tests/test_pauses_render.py tests/test_cache.py -v --no-cov`
Expected: PASS.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_pauses_render.py tests/test_execution.py -v --no-cov`
Expected: PASS.

- [ ] **Step 6: Run lint, types, and the full suite**

```console
uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest
```

- [ ] **Step 7: Run the native acceptance path**

```console
KENKUI_RUN_NATIVE=1 uv run pytest --no-cov -m native tests/test_native_ffmpeg.py
```

This is the only check that proves FFmpeg accepts the padded PCM and that the
chapter markers in a real M4B land where the plan says.

- [ ] **Step 8: Commit**

```bash
git add src/kenkui/_execution/coordinator.py tests/test_pauses_render.py tests/test_cache.py
git commit --signoff -m "feat: render structural silence into chapter parts"
```

---
# Phase 3 — Custom Cover

## Task 13: Validate a caller-supplied cover file

**Files:**
- Modify: `src/kenkui/_audio/cover.py`
- Modify: `src/kenkui/errors.py`
- Test: `tests/test_cover_file.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `MAX_COVER_BYTES: int`, `read_cover(path: Path) -> tuple[bytes, str]` returning validated bytes and their SHA-256 hex digest, and `materialize_file_cover(payload: bytes, destination: Path) -> None`. Task 14 calls both.

- [ ] **Step 1: Add the error code**

In `src/kenkui/errors.py`, add to `ErrorCode` after `COVER_FAILED`:

```python
    COVER_INVALID = "cover_invalid"
```

and to `_DEFAULT_MESSAGES`:

```python
    ErrorCode.COVER_INVALID: "The cover image is unreadable or unsupported.",
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_cover_file.py`:

```python
"""Accept/reject matrix for a caller-supplied cover image."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from kenkui._audio.cover import MAX_COVER_BYTES, read_cover
from kenkui.errors import EncodingError, ErrorCode

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def write(tmp_path: Path, name: str, payload: bytes) -> Path:
    """Write a candidate cover file and return its path."""
    path = tmp_path / name
    path.write_bytes(payload)
    return path


@pytest.mark.parametrize(("name", "payload"), [("c.jpg", JPEG), ("c.png", PNG)])
def test_accepts_jpeg_and_png(tmp_path: Path, name: str, payload: bytes) -> None:
    """Both supported formats return their bytes and content digest."""
    data, digest = read_cover(write(tmp_path, name, payload))
    assert data == payload
    assert digest == hashlib.sha256(payload).hexdigest()


def test_rejects_an_unsupported_format(tmp_path: Path) -> None:
    """A GIF is a valid image and still not something we hand to FFmpeg."""
    with pytest.raises(EncodingError) as error:
        read_cover(write(tmp_path, "c.gif", b"GIF89a" + b"\x00" * 64))
    assert error.value.code is ErrorCode.COVER_INVALID


def test_rejects_a_missing_file(tmp_path: Path) -> None:
    """An absent path fails loudly rather than falling back to the source."""
    with pytest.raises(EncodingError) as error:
        read_cover(tmp_path / "absent.jpg")
    assert error.value.code is ErrorCode.COVER_INVALID


def test_rejects_a_directory(tmp_path: Path) -> None:
    """Only regular files are accepted."""
    with pytest.raises(EncodingError) as error:
        read_cover(tmp_path)
    assert error.value.code is ErrorCode.COVER_INVALID


def test_rejects_a_symlink(tmp_path: Path) -> None:
    """O_NOFOLLOW: a link could redirect the read outside the caller's intent."""
    target = write(tmp_path, "real.jpg", JPEG)
    link = tmp_path / "link.jpg"
    link.symlink_to(target)
    with pytest.raises(EncodingError) as error:
        read_cover(link)
    assert error.value.code is ErrorCode.COVER_INVALID


def test_rejects_an_oversize_file(tmp_path: Path) -> None:
    """The size bound is enforced before the bytes are read into memory."""
    payload = JPEG + b"\x00" * MAX_COVER_BYTES
    with pytest.raises(EncodingError) as error:
        read_cover(write(tmp_path, "big.jpg", payload))
    assert error.value.code is ErrorCode.COVER_INVALID


def test_rejects_an_empty_file(tmp_path: Path) -> None:
    """An empty file has no magic bytes to sniff."""
    with pytest.raises(EncodingError) as error:
        read_cover(write(tmp_path, "empty.jpg", b""))
    assert error.value.code is ErrorCode.COVER_INVALID
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_cover_file.py -v --no-cov`
Expected: FAIL — `ImportError: cannot import name 'read_cover'`

- [ ] **Step 4: Write the implementation**

Append to `src/kenkui/_audio/cover.py`:

```python
MAX_COVER_BYTES = 8 * 1024 * 1024
# Magic bytes rather than the file extension: FFmpeg is handed this stream, and
# an extension is a claim while a signature is evidence.
_SIGNATURES = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n")


def read_cover(path: Path) -> tuple[bytes, str]:
    """Validate a caller-supplied cover and return its bytes and digest.

    Fails loudly. A caller who names a specific image does not want a silent
    fallback to whatever the EPUB happened to contain.
    """
    descriptor = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size <= 0
            or info.st_size > MAX_COVER_BYTES
        ):
            raise EncodingError(ErrorCode.COVER_INVALID)  # noqa: TRY301
        payload = os.read(descriptor, info.st_size)
        if len(payload) != info.st_size or not payload.startswith(_SIGNATURES):
            raise EncodingError(ErrorCode.COVER_INVALID)  # noqa: TRY301
    except (OSError, ValueError):
        raise EncodingError(ErrorCode.COVER_INVALID) from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return payload, hashlib.sha256(payload).hexdigest()


def materialize_file_cover(payload: bytes, destination: Path) -> None:
    """Write already-validated cover bytes to one fixed exclusive private path."""
    descriptor = -1
    failed = False
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(destination, flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        failed = True
    if failed:
        if descriptor >= 0:
            os.close(descriptor)
        with suppress(OSError):
            destination.unlink(missing_ok=True)
        raise EncodingError(ErrorCode.COVER_FAILED) from None
```

Add `hashlib` and `stat` to the module imports, and change the `TYPE_CHECKING`
import of `Path` to a real import (it is now used at runtime by the signature).

Note `payload.startswith(_SIGNATURES)` — `bytes.startswith` accepts a tuple, so
this is one check across both formats.

- [ ] **Step 5: Run test, lint, types, full suite**

```console
uv run pytest tests/test_cover_file.py -v --no-cov
uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest
```

If `test_rejects_a_symlink` passes on a platform lacking `O_NOFOLLOW`, the
`getattr` fallback resolved to `0`; add an explicit `path.is_symlink()` check
before the open so the guarantee does not depend on the platform.

- [ ] **Step 6: Commit**

```bash
git add src/kenkui/_audio/cover.py src/kenkui/errors.py tests/test_cover_file.py
git commit --signoff -m "feat: validate caller-supplied cover images"
```

---

## Task 14: Wire the custom cover through metadata, plan, and assembly

**Files:**
- Modify: `src/kenkui/_domain/operations.py`
- Modify: `src/kenkui/pipeline.py`
- Modify: `src/kenkui/_domain/planning.py`
- Modify: `src/kenkui/_audio/m4b.py`
- Modify: `src/kenkui/_audio/production.py`
- Modify: `src/kenkui/_execution/coordinator.py`
- Test: `tests/test_cover_file.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `read_cover`, `materialize_file_cover` from Task 13.
- Produces: `CoverIntent.FILE`; `OutputMetadata.cover_content_hash: str | None`; `compile_execution_plan(..., cover_content_hash: str | None = None)`; `AssemblyRequest.cover_file: Path | None`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:

```python
def test_metadata_accepts_a_cover_path(tmp_path: Path) -> None:
    """A caller-supplied cover is recorded as intent, not read eagerly."""
    cover = tmp_path / "cover.jpg"
    recorded = kk.epub("book.epub").metadata(cover=cover).operations[0]
    assert isinstance(recorded, kk.MetadataIntent)
    assert recorded.cover == cover


def test_metadata_still_rejects_an_unknown_cover_token() -> None:
    """Only "source", None, or a path are valid."""
    with pytest.raises(kk.ValidationError) as error:
        kk.epub("book.epub").metadata(cover="banana")  # type: ignore[arg-type]
    assert error.value.code is kk.ErrorCode.INVALID_METADATA
```

Append to `tests/test_cover_file.py`:

```python
def test_plan_records_cover_content_hash_not_path() -> None:
    """Two identical images at different paths must produce the same plan."""
    from kenkui._domain.planning import CoverIntent, _output_metadata  # noqa: PLC0415

    intent = kk.MetadataIntent(None, None, Path("/somewhere/cover.jpg"))
    metadata = _output_metadata(_inspection(), intent, cover_content_hash="ab" * 32)
    assert metadata.cover is CoverIntent.FILE
    assert metadata.cover_content_hash == "ab" * 32
```

Add whatever `_inspection()` helper the file needs, mirroring the one in
`tests/test_planning_multi_voice.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -k cover tests/test_cover_file.py -v --no-cov`
Expected: FAIL — `metadata()` rejects a `Path`.

- [ ] **Step 3: Widen the operation and the Pipeline method**

In `src/kenkui/_domain/operations.py`, change `MetadataIntent.cover` to:

```python
    cover: Literal["source"] | Path | None = "source"
```

and add `from pathlib import Path` to the module imports.

In `src/kenkui/pipeline.py`, change the `metadata` signature's `cover` parameter
to `Literal["source"] | os.PathLike[str] | None = "source"` and replace the
validation line with:

```python
        if isinstance(cover, str) and cover != "source":
            raise ValidationError(ErrorCode.INVALID_METADATA)
        if cover is not None and cover != "source":
            cover = Path(cover)
```

`os` is currently imported only under `TYPE_CHECKING` in `pipeline.py`; `Path`
is already imported at runtime, so no import change is needed.

- [ ] **Step 4: Extend the plan**

In `src/kenkui/_domain/planning.py`:

Add `FILE = "file"` to `CoverIntent`. Add `cover_content_hash: str | None = None`
as the last field of `OutputMetadata`. Change `_output_metadata` to accept and
apply the hash:

```python
def _output_metadata(
    inspection: BookInspection,
    intent: MetadataIntent | None,
    cover_content_hash: str | None = None,
) -> OutputMetadata:
```

and inside, replace the cover resolution with:

```python
        if intent.cover == "source":
            cover = CoverIntent.SOURCE
        elif intent.cover is None:
            cover = CoverIntent.NONE
        else:
            cover = CoverIntent.FILE
```

passing `cover_content_hash=cover_content_hash` into the returned
`OutputMetadata`.

Add `cover_content_hash: str | None = None` as a keyword parameter of
`compile_execution_plan` and forward it to `_output_metadata`.

In `_fingerprint`, add to the `output` block:

```python
            "cover_content_hash": output.cover_content_hash,
```

The path itself is deliberately absent: core §14 requires that where an output
lives cannot change what it means.

- [ ] **Step 5: Extend assembly**

In `src/kenkui/_audio/m4b.py`, add to `AssemblyRequest`:

```python
    cover_file: Path | None = None
```

In `src/kenkui/_audio/production.py`, change the cover branch in `assemble`:

```python
            expect_cover = (
                request.plan.output.cover is CoverIntent.SOURCE
                and request.plan.output.source_cover_available
            ) or request.plan.output.cover is CoverIntent.FILE
```

and where it currently calls `materialize_source_cover(request.source_epub, cover)`:

```python
            if expect_cover:
                if request.plan.output.cover is CoverIntent.FILE:
                    payload, _ = read_cover(cast("Path", request.cover_file))
                    materialize_file_cover(payload, cover)
                else:
                    materialize_source_cover(cast("Path", request.source_epub), cover)
```

importing `materialize_file_cover` and `read_cover` alongside the existing
`materialize_source_cover` import.

- [ ] **Step 6: Wire the coordinator**

In `src/kenkui/_execution/coordinator.py`, resolve the cover before planning so
an invalid image fails before any worker spawns:

```python
    cover_file: Path | None = None
    cover_content_hash: str | None = None
    if metadata_intent is not None and isinstance(metadata_intent.cover, Path):
        cover_file = metadata_intent.cover
        _, cover_content_hash = read_cover(cover_file)
```

placed immediately after `metadata_intent = pipeline.metadata_intent`, and
change the preflight call to expect a cover for the file case too:

```python
    _preflight_assembler(
        bindings.assembler,
        expect_cover=metadata_intent is None
        or metadata_intent.cover == "source"
        or cover_file is not None,
    )
```

Pass `cover_content_hash=cover_content_hash` into `compile_execution_plan`, and
thread `cover_file` into the `AssemblyRequest` built inside `_assemble` (add a
`cover_file` parameter to `_assemble` and pass it from `execute_sequential`).

- [ ] **Step 7: Run tests, lint, types, full suite**

```console
uv run pytest tests/test_cover_file.py tests/test_pipeline.py -v --no-cov
uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest
KENKUI_RUN_NATIVE=1 uv run pytest --no-cov -m native tests/test_native_ffmpeg.py
```

The native run is what proves FFmpeg actually embeds the supplied image.

- [ ] **Step 8: Commit**

```bash
git add src/kenkui/_domain/operations.py src/kenkui/pipeline.py \
        src/kenkui/_domain/planning.py src/kenkui/_audio/m4b.py \
        src/kenkui/_audio/production.py src/kenkui/_execution/coordinator.py \
        tests/test_cover_file.py tests/test_pipeline.py
git commit --signoff -m "feat: embed a caller-supplied cover image"
```

---

## Task 15: Documentation

**Files:**
- Modify: `kenkui/docs/usage.md`
- Modify: `kenkui/docs/architecture.md`

- [ ] **Step 1: Document the three new capabilities in `docs/usage.md`**

Add a section showing the opt-in nature explicitly:

````markdown
## Speech shaping

Nothing below is applied unless you ask for it. A pipeline that calls neither
`pronounce()` nor `pauses()` renders exactly as it did before these existed.

```python
pipeline = (
    kk.epub("book.epub")
    .pronounce({"Cthulhu": "kuh-THOO-loo"}, numbers="standard")
    .pauses(chapter_ms=1500, heading_after_ms=600, paragraph_ms=250)
    .assign_voice("alba")
    .tts()
    .metadata(cover=Path("cover.jpg"))
)
```

`pronounce()` changes what the engine says, never what you are billed for:
`ExecutionStats.normalized_speech_characters` still counts the source text,
while `synthesized_characters` reflects the expansion.

Pause durations are retunable for free — only turning a tier on or off
re-chunks and re-synthesizes. `chapter_ms` never re-chunks at all.
````

- [ ] **Step 2: Update `docs/architecture.md`**

Add a short section after "Span-then-chunk segmentation" describing the
split-then-speak-then-chunk order, the three-level exactness invariant, and the
gap model. Match the existing prose register — dense, explanatory, no bullets.

- [ ] **Step 3: Verify docs build**

```console
uv run mkdocs build --strict
```

- [ ] **Step 4: Commit**

```bash
git add docs/
git commit --signoff -m "docs: describe spoken form, pauses, and custom cover"
```

---
