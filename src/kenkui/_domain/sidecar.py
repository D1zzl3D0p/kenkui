"""Versioned, hand-editable persistence for ordered human tuning rules."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, cast

from kenkui._domain.grid import build_grid, sibling_counts, unit_digest
from kenkui._domain.operations import (
    Annotations,
    Attributions,
    Operation,
    Pronunciations,
    Silences,
)
from kenkui._domain.paths import (
    LEVELS,
    Any,
    Exact,
    Last,
    OneOf,
    Pattern,
    Selector,
    matches,
    parse_pattern,
)
from kenkui._domain.spoken.lexicon import validate_entries
from kenkui._domain.tiers import tier_of
from kenkui._domain.tuning import Rule
from kenkui.errors import ErrorCode, ValidationError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from kenkui._domain.grid import Unit
    from kenkui.inspection import ChapterInspection

SIDECAR_VERSION = 1
_MAX_SILENCE_MS = 60_000
_DIGEST = re.compile(r"sha256:[0-9a-f]{16}")
RuleOperation: TypeAlias = Attributions | Silences | Pronunciations
_KINDS: dict[str, tuple[type[RuleOperation], str]] = {
    "attributions": (Attributions, "character"),
    "silences": (Silences, "ms"),
    "pronunciations": (Pronunciations, "words"),
}


def sidecar_path(epub: Path) -> Path:
    """Return the sidecar beside the source, retaining its complete stem."""
    return epub.with_suffix(".kenkui.json")


def serialize(operations: Iterable[Operation]) -> dict[str, object]:
    """Serialize tuning in declaration order, excluding the loader itself."""
    groups: dict[str, list[dict[str, object]]] = {}
    for operation in operations:
        if tier_of(type(operation)) != "tuning" or isinstance(operation, Annotations):
            continue
        for name, (kind, value_key) in _KINDS.items():
            if isinstance(operation, kind):
                groups.setdefault(name, []).extend(
                    _rule_to(rule, value_key) for rule in operation.rules
                )
                break
    payload: dict[str, object] = {"kenkui_sidecar": SIDECAR_VERSION, **groups}
    # A programmatically constructed operation must not produce an unreadable file.
    deserialize(payload)
    return payload


def _selector_to(selector: Selector) -> object:
    if isinstance(selector, Exact):
        return selector.value
    if isinstance(selector, Any):
        return "*"
    if isinstance(selector, Last):
        return -1
    if isinstance(selector, OneOf):
        return sorted(selector.values)
    return f"{selector.lo}..{selector.hi}"


def _rule_to(rule: Rule, value_key: str) -> dict[str, object]:
    value = rule.value
    if value_key == "words":
        try:
            value = dict(cast("tuple[tuple[str, str], ...]", value))
        except (TypeError, ValueError):
            raise ValidationError(ErrorCode.INVALID_SIDECAR) from None
    result = {
        "where": {
            level: _selector_to(selector) for level, selector in rule.where.items()
        },
        value_key: value,
    }
    if rule.digest is not None:
        result["digest"] = rule.digest
    if rule.matched is not None:
        result["matched"] = rule.matched
    return result


def deserialize(payload: object) -> tuple[Operation, ...]:
    """Validate v1 JSON and restore immutable operations and rule tuples."""
    if (
        not isinstance(payload, dict)
        or type(payload.get("kenkui_sidecar")) is not int
        or payload["kenkui_sidecar"] != SIDECAR_VERSION
        or payload.keys() - {"kenkui_sidecar", *_KINDS}
    ):
        raise ValidationError(ErrorCode.INVALID_SIDECAR)
    operations: list[Operation] = []
    # JSON object order preserves the order of operation families we authored;
    # precedence within a family is always the rule array's declaration order.
    for name, entries in payload.items():
        if name == "kenkui_sidecar":
            continue
        kind, value_key = _KINDS[name]
        if not isinstance(entries, list):
            raise ValidationError(ErrorCode.INVALID_SIDECAR)
        operations.append(
            kind(
                tuple(
                    _rule_from(entry, value_key, i) for i, entry in enumerate(entries)
                )
            )
        )
    return tuple(operations)


def _rule_from(entry: object, value_key: str, index: int) -> Rule:
    if (
        not isinstance(entry, dict)
        or not {"where", value_key} <= entry.keys()
        or entry.keys() - {"where", value_key, "digest", "matched"}
        or not isinstance(entry["where"], dict)
    ):
        raise ValidationError(ErrorCode.INVALID_SIDECAR)
    digest = entry.get("digest")
    matched = entry.get("matched")
    if (
        (
            "digest" in entry
            and (not isinstance(digest, str) or not _DIGEST.fullmatch(digest))
        )
        or ("matched" in entry and (type(matched) is not int or matched < 0))
        or ("digest" in entry and "matched" in entry)
    ):
        raise ValidationError(ErrorCode.INVALID_SIDECAR)
    try:
        where = parse_pattern(entry["where"])
        value = _value_from(entry[value_key], value_key)
    except ValidationError:
        raise ValidationError(ErrorCode.INVALID_SIDECAR) from None
    return Rule(where, value, index, digest, matched)


def _value_from(value: object, value_key: str) -> object:
    if value_key == "character":
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(ErrorCode.INVALID_SIDECAR)
        return value
    if value_key == "ms":
        if type(value) is not int or not 0 <= value <= _MAX_SILENCE_MS:
            raise ValidationError(ErrorCode.INVALID_SIDECAR)
        return value
    if not isinstance(value, dict) or not all(
        isinstance(word, str) and isinstance(spoken, str)
        for word, spoken in value.items()
    ):
        raise ValidationError(ErrorCode.INVALID_SIDECAR)
    return validate_entries(value)


def authoring_snapshot(
    operations: tuple[Operation, ...], chapters: Iterable[ChapterInspection]
) -> tuple[Operation, ...]:
    """Attach current grid digests/counts without mutating pipeline intent.

    Exact selectors in an exact chapter can address one contiguous subtree,
    even with an elided single line. Hash that whole span. Set-valued or
    disjoint targets record leaf counts; a missing target records zero.
    """
    tuning = tuple(
        operation
        for operation in operations
        if tier_of(type(operation)) == "tuning"
        and isinstance(operation, (Attributions, Silences, Pronunciations))
    )
    rules = tuple(rule for operation in tuning for rule in operation.rules)
    selected = tuple(
        chapter
        for chapter in chapters
        if any(
            rule.where.get("chapter", Any()).covers(chapter.id, None) for rule in rules
        )
    )
    units = tuple(unit for chapter in selected for unit in build_grid(chapter))
    texts = {chapter.id: chapter.text for chapter in selected}
    siblings = sibling_counts(units)
    snapshots: dict[Pattern, tuple[str | None, int | None]] = {}
    for rule in rules:
        if rule.where not in snapshots:
            matched = tuple(
                unit for unit in units if matches(rule.where, unit, siblings)
            )
            snapshots[rule.where] = _anchor(rule.where, matched, texts)
    return tuple(
        replace(
            operation,
            rules=tuple(
                replace(
                    rule,
                    digest=snapshots[rule.where][0],
                    matched=snapshots[rule.where][1],
                )
                for rule in operation.rules
            ),
        )
        for operation in tuning
    )


def _anchor(
    pattern: Pattern, units: tuple[Unit, ...], texts: Mapping[str, str]
) -> tuple[str | None, int | None]:
    concrete = isinstance(pattern.get("chapter"), Exact) and all(
        isinstance(selector, Exact) for selector in pattern.values()
    )
    contiguous = all(
        left.chapter_id == right.chapter_id and left.end == right.start
        for left, right in pairwise(units)
    )
    depth = max((LEVELS.index(level) + 1 for level in pattern), default=0)
    subtrees = {
        (unit.chapter_id, unit.paragraph, unit.line, unit.sentence, unit.phrase)[:depth]
        for unit in units
    }
    if concrete and units and contiguous and len(subtrees) == 1:
        span = replace(units[0], end=units[-1].end)
        return unit_digest(span, texts[span.chapter_id]), None
    return None, len(units)


def write_sidecar(path: Path, payload: dict[str, object]) -> None:
    """Publish complete UTF-8 JSON atomically, cleaning up failed writes."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except (OSError, ValueError):
        raise ValidationError(ErrorCode.INVALID_SIDECAR) from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
