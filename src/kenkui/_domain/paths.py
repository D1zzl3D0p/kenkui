"""Immutable labeled paths for addressing book subtrees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from kenkui.errors import ErrorCode, ValidationError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from kenkui._domain.grid import Unit


LEVELS: tuple[str, ...] = ("chapter", "paragraph", "line", "sentence", "phrase")


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


def _invalid_path() -> ValidationError:
    return ValidationError(ErrorCode.INVALID_PATH)


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
