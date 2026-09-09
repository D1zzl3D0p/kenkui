"""Immutable labeled paths for addressing book subtrees."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias, cast

from kenkui.errors import ErrorCode, ValidationError

if TYPE_CHECKING:
    from kenkui._domain.grid import Unit


LEVELS: tuple[str, ...] = ("chapter", "paragraph", "line", "sentence", "phrase")
SiblingCounts: TypeAlias = Mapping[tuple[str, ...], int]
_RANGE = re.compile(r"([1-9]\d*)\.\.([1-9]\d*)")


@dataclass(frozen=True, slots=True)
class Path:
    """A possibly partial path through the chapter/paragraph grid."""

    chapter: str | None = None
    paragraph: int | None = None
    line: int | None = None
    sentence: int | None = None
    phrase: int | None = None

    def __post_init__(self) -> None:
        """Validate field types and positive coordinates."""
        if self.chapter is not None and not isinstance(self.chapter, str):
            raise _invalid_path()
        values = (self.paragraph, self.line, self.sentence, self.phrase)
        for value in values:
            if value is None:
                continue
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise _invalid_path()


@dataclass(frozen=True, slots=True)
class Exact:
    """A selector matching one coordinate."""

    value: str | int

    def covers(self, value: str | int, sibling_count: int | None) -> bool:
        """Return whether this selector contains ``value``."""
        del sibling_count
        return self.value == value

    def within(self, other: Selector) -> bool | None:
        """Compare this singleton with another selector."""
        if isinstance(other, Any):
            return True
        if isinstance(other, Exact):
            return True if self.value == other.value else None
        if isinstance(other, OneOf):
            return True if self.value in other.values else None
        if isinstance(other, Span):
            if isinstance(self.value, int) and other.lo <= self.value <= other.hi:
                return True
            return None
        return None


@dataclass(frozen=True, slots=True)
class Any:
    """A selector matching every coordinate."""

    def covers(self, value: str | int, sibling_count: int | None) -> bool:
        """Return true for every coordinate."""
        del value, sibling_count
        return True

    def within(self, other: Selector) -> bool | None:
        """Compare the universal selector with another selector."""
        return isinstance(other, Any)


@dataclass(frozen=True, slots=True)
class OneOf:
    """A selector matching a finite set of positive indices."""

    values: frozenset[int]

    def __post_init__(self) -> None:
        """Validate and defensively freeze the selected indices."""
        values = frozenset(self.values)
        if not values or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in values
        ):
            raise _invalid_pattern()
        object.__setattr__(self, "values", values)

    def covers(self, value: str | int, sibling_count: int | None) -> bool:
        """Return whether ``value`` is one of the selected indices."""
        del sibling_count
        return isinstance(value, int) and value in self.values

    def within(self, other: Selector) -> bool | None:  # noqa: PLR0911
        """Compare this finite set with another selector."""
        if isinstance(other, Any):
            return True
        if isinstance(other, Exact):
            if self.values == {other.value}:
                return True
            return False if other.value in self.values else None
        if isinstance(other, OneOf):
            return _set_relation(self.values, other.values)
        if isinstance(other, Span):
            if all(other.lo <= value <= other.hi for value in self.values):
                return True
            width = other.hi - other.lo + 1
            if len(self.values) >= width and all(
                value in self.values for value in range(other.lo, other.hi + 1)
            ):
                return False
        return None


@dataclass(frozen=True, slots=True)
class Span:
    """A selector matching an inclusive range of positive indices."""

    lo: int
    hi: int

    def __post_init__(self) -> None:
        """Validate an ordered, positive integer span."""
        if (
            not isinstance(self.lo, int)
            or isinstance(self.lo, bool)
            or not isinstance(self.hi, int)
            or isinstance(self.hi, bool)
            or self.lo <= 0
            or self.lo > self.hi
        ):
            raise _invalid_pattern()

    def covers(self, value: str | int, sibling_count: int | None) -> bool:
        """Return whether ``value`` lies inside the inclusive range."""
        del sibling_count
        return isinstance(value, int) and self.lo <= value <= self.hi

    def within(self, other: Selector) -> bool | None:  # noqa: C901, PLR0911
        """Compare this inclusive range with another selector."""
        if isinstance(other, Any):
            return True
        if isinstance(other, Exact):
            if self.lo == self.hi == other.value:
                return True
            if isinstance(other.value, int) and self.covers(other.value, None):
                return False
            return None
        if isinstance(other, OneOf):
            width = self.hi - self.lo + 1
            if len(other.values) >= width and all(
                value in other.values for value in range(self.lo, self.hi + 1)
            ):
                return True
            if all(self.lo <= value <= self.hi for value in other.values):
                return False
            return None
        if isinstance(other, Span):
            if other.lo <= self.lo and self.hi <= other.hi:
                return True
            if self.lo <= other.lo and other.hi <= self.hi:
                return False
        return None


@dataclass(frozen=True, slots=True)
class Last:
    """A selector matching the final child under a concrete parent."""

    def covers(self, value: str | int, sibling_count: int | None) -> bool:
        """Resolve the final index using the supplied sibling count."""
        return sibling_count is not None and value == sibling_count

    def within(self, other: Selector) -> bool | None:
        """Compare this context-dependent selector with another selector."""
        if isinstance(other, (Any, Last)):
            return True
        return None


Selector: TypeAlias = Exact | Any | OneOf | Span | Last


@dataclass(frozen=True, slots=True, init=False)
class Pattern(Mapping[str, Selector]):
    """An immutable, sparse mapping from path levels to selectors."""

    _items: tuple[tuple[str, Selector], ...] = ()

    def __init__(self, mapping: Mapping[str, Selector] | None = None) -> None:
        """Freeze a selector mapping in canonical path-level order."""
        source = {} if mapping is None else mapping
        if any(level not in LEVELS for level in source):
            raise _invalid_pattern()
        items = tuple((level, source[level]) for level in LEVELS if level in source)
        if any(
            not isinstance(selector, (Exact, Any, OneOf, Span, Last))
            for _, selector in items
        ):
            raise _invalid_pattern()
        object.__setattr__(self, "_items", items)

    def __getitem__(self, key: str) -> Selector:
        for level, selector in self._items:
            if level == key:
                return selector
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (level for level, _selector in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def is_whole_book(self) -> bool:
        """Return whether this pattern matches every unit in the book."""
        return all(isinstance(selector, Any) for selector in self.values())


def _set_relation(left: frozenset[int], right: frozenset[int]) -> bool | None:
    if left <= right:
        return True
    if right < left:
        return False
    return None


def _invalid_path() -> ValidationError:
    return ValidationError(ErrorCode.INVALID_PATH)


def _invalid_pattern() -> ValidationError:
    return ValidationError(ErrorCode.INVALID_PATTERN)


def parse_path(mapping: Mapping[str, object]) -> Path:
    """Parse a sparse mapping into a validated path."""
    if any(key not in LEVELS for key in mapping):
        raise _invalid_path()
    values: dict[str, object | None] = dict.fromkeys(LEVELS)
    values.update(mapping)
    return Path(
        chapter=cast("str | None", values["chapter"]),
        paragraph=cast("int | None", values["paragraph"]),
        line=cast("int | None", values["line"]),
        sentence=cast("int | None", values["sentence"]),
        phrase=cast("int | None", values["phrase"]),
    )


def parse_pattern(mapping: Mapping[str, object]) -> Pattern:  # noqa: C901, PLR0912
    """Parse set-valued coordinates into an immutable sparse pattern."""
    if any(key not in LEVELS for key in mapping):
        raise _invalid_pattern()
    items: list[tuple[str, Selector]] = []
    for level in LEVELS:
        if level not in mapping:
            continue
        value = mapping[level]
        selector: Selector
        if value == "*":
            selector = Any()
        elif level == "chapter":
            if not isinstance(value, str) or _RANGE.fullmatch(value):
                raise _invalid_pattern()
            selector = Exact(value)
        elif value == -1 and not isinstance(value, bool):
            selector = Last()
        elif isinstance(value, int) and not isinstance(value, bool) and value > 0:
            selector = Exact(value)
        elif isinstance(value, list):
            if not value or any(
                not isinstance(item, int)
                or isinstance(item, bool)
                or item <= 0
                for item in value
            ):
                raise _invalid_pattern()
            selector = OneOf(frozenset(value))
        elif isinstance(value, str) and (match := _RANGE.fullmatch(value)):
            lo, hi = (int(part) for part in match.groups())
            if lo > hi:
                raise _invalid_pattern()
            selector = Span(lo, hi)
        else:
            raise _invalid_pattern()
        items.append((level, selector))
    return Pattern(dict(items))


def matches(pattern: Pattern, unit: Unit, siblings: SiblingCounts) -> bool:
    """Return whether a grid unit is covered by a pattern."""
    values: tuple[str | int, ...] = (
        unit.chapter_id,
        unit.paragraph,
        unit.line,
        unit.sentence,
        unit.phrase,
    )
    parent: list[str] = []
    for level, value in zip(LEVELS, values, strict=True):
        selector = pattern.get(level, Any())
        if not selector.covers(value, siblings.get(tuple(parent))):
            return False
        parent.append(str(value))
    return True


def subset(a: Pattern, b: Pattern) -> bool | None:
    """Return the partial subset relation between two patterns."""
    narrower = False
    wider = False
    for level in LEVELS:
        left = a.get(level, Any())
        right = b.get(level, Any())
        forward = left.within(right)
        reverse = right.within(left)
        if forward is None or reverse is None:
            return None
        narrower = narrower or (forward and not reverse)
        wider = wider or (reverse and not forward)
    if narrower and wider:
        return None
    return not wider


def path_of(unit: Unit) -> Path:
    """Return the complete leaf path represented by a grid unit."""
    return Path(unit.chapter_id, unit.paragraph, unit.line, unit.sentence, unit.phrase)


def contains(outer: Path, inner: Path) -> bool:
    """Return whether ``outer`` addresses a subtree containing ``inner``."""
    return all(
        outer_value is None or outer_value == inner_value
        for outer_value, inner_value in zip(
            (outer.chapter, outer.paragraph, outer.line, outer.sentence, outer.phrase),
            (inner.chapter, inner.paragraph, inner.line, inner.sentence, inner.phrase),
            strict=True,
        )
    )


def render_path(path: Path) -> str:
    """Render a path for display, eliding line one as a degenerate level."""
    components: list[str] = []
    if path.chapter is not None:
        components.append(path.chapter)
    if path.paragraph is not None:
        components.append(f"¶{path.paragraph}")
    if path.line is not None and path.line != 1:
        components.append(f"l{path.line}")
    if path.sentence is not None:
        components.append(f"s{path.sentence}")
    if path.phrase is not None:
        components.append(f"p{path.phrase}")
    return "  ".join(components) if components else "whole book"
