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

from kenkui.errors import ErrorCode, SourceError, ValidationError

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


def read_entries(path: Path) -> dict[str, str]:
    """Read a caller's pronunciation table from a JSON file.

    Two shapes are accepted: a plain object of word to replacement, and the
    shape the shipped table uses, so copying `lexicon-v1.json` and editing it
    works without rewriting the wrapper away. Entries are validated exactly
    as a literal would be -- a file must not be able to smuggle in input a
    caller could not have written inline.
    """
    if not path.exists():
        raise SourceError(ErrorCode.SOURCE_NOT_FOUND)
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError) as error:
        raise ValidationError(ErrorCode.INVALID_PRONUNCIATION) from error
    if not isinstance(payload, dict):
        raise ValidationError(ErrorCode.INVALID_PRONUNCIATION)
    entries = payload.get("entries", payload)
    if not isinstance(entries, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in entries.items()
    ):
        raise ValidationError(ErrorCode.INVALID_PRONUNCIATION)
    return dict(validate_entries(entries))


def validate_entries(mapping: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    """Bound and canonicalize caller entries, refusing malformed input.

    Two keys differing only in case are refused rather than resolved. Matching
    is case-insensitive, so they name one position with two replacements, and
    silently keeping either one drops an entry the caller asked for.
    """
    items = tuple(sorted(mapping.items()))
    if len(items) > MAX_LEXICON_ENTRIES:
        raise ValidationError(ErrorCode.INVALID_PRONUNCIATION)
    seen: set[str] = set()
    for key, value in items:
        if (
            not key.strip()
            or not value.strip()
            or len(key) > MAX_PHRASE_CHARACTERS
            or len(value) > MAX_PHRASE_CHARACTERS
            or key.casefold() in seen
        ):
            raise ValidationError(ErrorCode.INVALID_PRONUNCIATION)
        seen.add(key.casefold())
    return items


def _rule(entries: tuple[tuple[str, str], ...], *, folded: bool) -> Rule | None:
    """Compile one alternation, longest phrase first so it wins the position.

    ``folded`` decides how wide a key reaches. Built-in entries also answer to
    their unaccented spelling, so they are indexed by the fully folded form.
    Caller entries are not: "grace" and "grâce" are two entries the caller
    wrote deliberately, and indexing them by a form that strips the accent
    collapses them into one and loses whichever sorts first.
    """
    if not entries:
        return None
    index = _fold if folded else str.casefold
    table: dict[str, str] = {}
    alternatives: list[str] = []
    for key, value in entries:
        forms = {key, _fold(key)} if folded else {key}
        for form in forms:
            table[index(form)] = value
            alternatives.append(re.escape(form))
    alternatives.sort(key=len, reverse=True)
    pattern = re.compile(rf"{_LB}(?:{'|'.join(alternatives)}){_RB}", re.IGNORECASE)

    def handler(match: re.Match[str]) -> str | None:
        source = match.group(0)
        replacement = table.get(index(source))
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
