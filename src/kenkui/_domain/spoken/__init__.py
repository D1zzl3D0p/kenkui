"""Deterministic speech-form text transformation.

The canonical normalized text stays the authority for billing, inspection,
chapter identity, and attribution offsets. This stage derives the separate
string the synthesis engine actually speaks, and nothing else consumes it.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

from kenkui._domain.spoken.lexicon import LEXICON_VERSION, lexicon_rules
from kenkui._domain.spoken.numbers import NumberTier, Rule, number_rules

SPOKEN_FORM_VERSION = "spoken-form-v1"


def _rules(
    numbers: NumberTier,
    lexicon: tuple[tuple[str, str], ...],
    *,
    builtin: bool,
    features: Mapping[str, bool] | None = None,
) -> tuple[Rule, ...]:
    """Rank caller entries, then built-in entries, then number rules."""
    return (
        *lexicon_rules(lexicon, builtin=builtin),
        *number_rules(numbers, features),
    )


def to_spoken(
    text: str,
    *,
    numbers: NumberTier,
    lexicon: tuple[tuple[str, str], ...],
    builtin: bool,
    features: Mapping[str, bool] | None = None,
) -> str:
    """Return the string the engine should speak for this canonical text.

    One left-to-right pass. At each position the first accepting rule wins and
    its output is emitted verbatim; emitted output is never re-examined, so
    rules can neither cascade nor loop.
    """
    rules = _rules(numbers, lexicon, builtin=builtin, features=features)
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
    features: Mapping[str, bool] | None = None,
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
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    identity: dict[str, object] = {
        "spoken_form_schema": SPOKEN_FORM_VERSION,
        "numbers_tier": numbers,
        "lexicon_identity": f"{LEXICON_VERSION if builtin else 'none'}:{digest}",
    }
    # Absent when nothing was overridden, so a caller who never touched a
    # feature keeps the identity -- and therefore the cached audio -- they
    # had before features existed.
    if features:
        identity["number_features"] = ",".join(
            f"{name}={int(value)}" for name, value in sorted(features.items())
        )
    return identity
