"""Rule precedence and overlap diagnostics for tuned book settings."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Literal, TypeAlias

from kenkui._domain.paths import (
    LEVELS,
    Any,
    Exact,
    Last,
    OneOf,
    Pattern,
    Selector,
    Span,
    matches,
    subset,
)

if TYPE_CHECKING:
    from kenkui._domain.grid import Unit
    from kenkui._domain.paths import SiblingCounts


@dataclass(frozen=True, slots=True)
class Rule:
    """One declared value attached to a set-valued book path."""

    where: Pattern
    value: object
    index: int
    digest: str | None = None
    matched: int | None = None


Provenance: TypeAlias = Literal["default", "unresolved", "machine", "rule"]


@dataclass(frozen=True, slots=True)
class Decision:
    """A resolved value and the layer that supplied it."""

    value: object | None
    provenance: Provenance
    rule_index: int | None


def resolve_rules(
    unit: Unit,
    machine_value: object | None,
    rules: tuple[Rule, ...],
    siblings: SiblingCounts,
) -> Decision:
    """Resolve one unit through provenance and rule-specificity layers.

    Human rules are a layer above machine attribution. Within that layer,
    strict subsets beat broader patterns; declaration order breaks ties among
    equal or incomparable maximal patterns.
    """
    candidates = tuple(rule for rule in rules if matches(rule.where, unit, siblings))
    if not candidates:
        if machine_value is None:
            return Decision(None, "default", None)
        return Decision(machine_value, "machine", None)

    maximal = tuple(
        candidate
        for candidate in candidates
        if not any(
            _strict_subset(other.where, candidate.where)
            for other in candidates
            if other is not candidate
        )
    )
    winner = maximal[-1]
    return Decision(winner.value, "rule", winner.index)


def _strict_subset(left: Pattern, right: Pattern) -> bool:
    """Return whether ``left`` is a known proper subset of ``right``."""
    return subset(left, right) is True and subset(right, left) is False


def overlap_warnings(rules: tuple[Rule, ...]) -> tuple[tuple[int, int], ...]:
    """Return rule-index pairs whose patterns overlap without nesting."""
    return tuple(
        (left.index, right.index)
        for left, right in combinations(rules, 2)
        if subset(left.where, right.where) is None
        and subset(right.where, left.where) is None
        and _patterns_overlap(left.where, right.where)
    )


def _patterns_overlap(left: Pattern, right: Pattern) -> bool:
    """Return whether both patterns could match at least one valid unit."""
    return all(
        _selectors_overlap(
            left.get(level, Any()), right.get(level, Any()), chapter=level == "chapter"
        )
        for level in LEVELS
    )


def _selectors_overlap(  # noqa: PLR0911
    left: Selector, right: Selector, *, chapter: bool
) -> bool:
    """Return whether two selectors share a possible coordinate."""
    if chapter:
        return _chapter_selectors_overlap(left, right)

    if isinstance(left, Any) or isinstance(right, Any):
        return True
    if isinstance(left, Last) or isinstance(right, Last):
        return True
    if isinstance(left, Exact):
        return _selector_covers_index(right, left.value)
    if isinstance(right, Exact):
        return _selector_covers_index(left, right.value)
    if isinstance(left, OneOf) and isinstance(right, OneOf):
        return not left.values.isdisjoint(right.values)
    if isinstance(left, OneOf) and isinstance(right, Span):
        return any(right.lo <= value <= right.hi for value in left.values)
    if isinstance(left, Span) and isinstance(right, OneOf):
        return any(left.lo <= value <= left.hi for value in right.values)
    if isinstance(left, Span) and isinstance(right, Span):
        return max(left.lo, right.lo) <= min(left.hi, right.hi)
    return False


def _chapter_selectors_overlap(left: Selector, right: Selector) -> bool:
    """Return whether two selectors share a possible chapter ID."""
    if isinstance(left, Any):
        return isinstance(right, Any) or (
            isinstance(right, Exact) and isinstance(right.value, str)
        )
    if isinstance(right, Any):
        return isinstance(left, Exact) and isinstance(left.value, str)
    return (
        isinstance(left, Exact)
        and isinstance(left.value, str)
        and isinstance(right, Exact)
        and left.value == right.value
    )


def _selector_covers_index(selector: Selector, value: str | int) -> bool:
    """Return whether a context-free selector covers a concrete index."""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and selector.covers(value, None)
    )
