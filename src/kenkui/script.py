"""Lazy, immutable review rows over the canonical grid and effective tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from kenkui._domain.grid import build_grid, sibling_counts, unit_text
from kenkui._domain.operations import (
    Attributions,
    Pauses,
    Pronunciations,
    Select,
    SelectChapterRange,
    SelectChapters,
    Silences,
)
from kenkui._domain.paths import (
    Any,
    Exact,
    Path,
    Pattern,
    contains,
    matches,
    parse_path,
    parse_pattern,
    path_of,
    render_path,
)

# Share the planner's machine lookup and normalized gaps, and the sidecar's
# subtree hash definition. Review must describe those exact interpretations.
from kenkui._domain.planning import (
    _machine_lookup,
    grid_silences,
)
from kenkui._domain.selection import selected_patterns, selected_unit
from kenkui._domain.sidecar import _anchor
from kenkui._domain.tuning import Provenance, resolve_rules
from kenkui.api import ValidationIssue
from kenkui.errors import ErrorCode

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from kenkui._domain.grid import Unit
    from kenkui._domain.paths import SiblingCounts
    from kenkui._domain.tuning import Rule
    from kenkui.inspection import BookInspection, ChapterInspection
    from kenkui.pipeline import Pipeline


@dataclass(frozen=True, slots=True)
class ScriptRow:
    """One canonical unit and the effective speaker and following silence.

    Whitespace-only units retain their text and path but carry zero silence;
    their gaps settle onto the preceding speech-bearing row, as in planning.
    """

    path: Path
    text: str
    character: str | None
    provenance: Provenance
    rule_index: int | None
    silence_after_ms: int
    is_dialogue: bool
    is_emphasised: bool


@dataclass(frozen=True, slots=True)
class _ChapterRows:
    units: tuple[Unit, ...]
    siblings: SiblingCounts
    rows: tuple[ScriptRow, ...]


@dataclass(frozen=True, slots=True)
class Script:
    """A chapter-lazy read model whose iteration yields rows in source order.

    ``script[path]`` returns a single row (a partial path must be unambiguous).
    ``at(pattern)`` yields matching rows. This is mapping-like, rather than a
    ``Mapping`` subclass, because iteration yields rows instead of keys.

    Construction inspects source text without resolution or model calls.
    Grids are built only on access, once per chapter. ``materialized`` and
    ``warnings`` are cheap immutable snapshots; neither triggers more work.
    """

    _pipeline: Pipeline = field(repr=False)
    _inspection: BookInspection = field(init=False, repr=False)
    _cache: dict[str, _ChapterRows] = field(
        default_factory=dict, init=False, repr=False
    )
    _findings: dict[int, ValidationIssue] = field(
        default_factory=dict, init=False, repr=False
    )
    _checked: set[int] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        """Snapshot inspected source and resolved attribution without building grids."""
        object.__setattr__(self, "_inspection", self._pipeline.inspect())

    @property
    def materialized(self) -> tuple[str, ...]:
        """Return cached chapter IDs in their first-access order."""
        return tuple(self._cache)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        """Return drift findings discovered so far, in declaration order."""
        return tuple(self._findings[index] for index in sorted(self._findings))

    def __iter__(self) -> Iterator[ScriptRow]:
        """Yield the selected book's rows in source order."""
        return self.at(Pattern())

    def __getitem__(self, path: Path | Mapping[str, object]) -> ScriptRow:
        """Return the unique row at a path, raising ``KeyError`` otherwise."""
        key = path if isinstance(path, Path) else parse_path(path)
        for chapter in self._inspection.chapters:
            if chapter.id == key.chapter:
                rows = tuple(
                    row
                    for row in self.at({"chapter": chapter.id})
                    if contains(key, row.path)
                )
                if len(rows) == 1:
                    return rows[0]
                break
        raise KeyError(key)

    def at(self, pattern: Pattern | Mapping[str, object]) -> Iterator[ScriptRow]:
        """Yield matches while materializing only chapters the pattern selects."""
        where = pattern if isinstance(pattern, Pattern) else parse_pattern(pattern)
        chapter_selector = where.get("chapter", Any())
        for chapter in self._inspection.chapters:
            if chapter_selector.covers(chapter.id, None):
                cached = self._materialize(chapter)
                for unit, row in zip(cached.units, cached.rows, strict=True):
                    if matches(where, unit, cached.siblings) and selected_unit(
                        unit,
                        selected_patterns(self._pipeline.operations),
                        cached.siblings,
                    ):
                        yield row
        self._update_warnings()

    def _materialize(self, chapter: ChapterInspection) -> _ChapterRows:
        if chapter.id not in self._cache:
            units = build_grid(chapter)
            siblings = sibling_counts(units)
            self._cache[chapter.id] = _ChapterRows(
                units, MappingProxyType(siblings), self._rows(chapter, units, siblings)
            )
            self._update_warnings()
        return self._cache[chapter.id]

    def _rows(
        self,
        chapter: ChapterInspection,
        units: tuple[Unit, ...],
        siblings: SiblingCounts,
    ) -> tuple[ScriptRow, ...]:
        rules = tuple(
            rule
            for operation in self._pipeline.operations
            if isinstance(operation, Attributions)
            for rule in operation.rules
        )
        casting = self._inspection.casting
        speaker = _machine_lookup(chapter.id, () if casting is None else casting.spans)
        silences = self._silences(chapter, units)
        rows: list[ScriptRow] = []
        for index, unit in enumerate(units):
            decision = resolve_rules(unit, speaker(unit), rules, siblings)
            provenance = decision.provenance
            if casting is None and provenance != "rule":
                provenance = "unresolved"
            rows.append(
                ScriptRow(
                    path_of(unit),
                    unit_text(unit, chapter.text),
                    cast("str | None", decision.value),
                    provenance,
                    decision.rule_index,
                    silences.get(index, 0),
                    unit.is_dialogue,
                    unit.is_emphasised,
                )
            )
        return tuple(rows)

    def _silences(
        self, chapter: ChapterInspection, units: tuple[Unit, ...]
    ) -> dict[int, int]:
        operations = self._pipeline.operations
        pauses = next((op for op in operations if isinstance(op, Pauses)), Pauses())
        silence = grid_silences(chapter, units, operations)
        if silence:
            # Match planning: inter-chapter silence is a floor; the book's
            # trailing silence is always zero, including manual declarations.
            last = next(reversed(silence))
            silence[last] = (
                0
                if chapter.id == self._inspection.chapters[-1].id
                else max(silence[last], pauses.chapter_ms)
            )
        return silence

    def _update_warnings(self) -> None:
        rules = tuple(
            (type(operation).__name__, rule)
            for operation in self._pipeline.operations
            if isinstance(operation, (Attributions, Silences, Pronunciations))
            for rule in operation.rules
        )
        narrowed = any(
            isinstance(op, (Select, SelectChapters, SelectChapterRange))
            for op in self._pipeline.operations
        )
        for ordinal, (kind, rule) in enumerate(rules):
            if ordinal in self._checked or (
                rule.digest is None and rule.matched is None
            ):
                continue
            selector = rule.where.get("chapter", Any())
            chapters = tuple(
                chapter
                for chapter in self._inspection.chapters
                if selector.covers(chapter.id, None)
            )
            # Sidecar counts describe the source book, not a render selection.
            # A selected script cannot know cross-chapter totals or verify an
            # anchor in a chapter outside its selection, and must not parse it.
            if narrowed and (not isinstance(selector, Exact) or not chapters):
                continue
            if any(chapter.id not in self._cache for chapter in chapters):
                continue
            units = tuple(
                unit
                for chapter in chapters
                for unit in self._cache[chapter.id].units
                if matches(rule.where, unit, self._cache[chapter.id].siblings)
            )
            digest, _count = _anchor(
                rule.where, units, {ch.id: ch.text for ch in chapters}
            )
            warning = _drift_warning(kind, rule, digest, len(units))
            if warning is not None:
                self._findings[ordinal] = warning
            self._checked.add(ordinal)


def _drift_warning(
    kind: str, rule: Rule, digest: str | None, count: int
) -> ValidationIssue | None:
    """Describe a changed anchor without changing the rule's coordinates."""
    path = parse_path(
        {
            level: selector.value
            for level, selector in rule.where.items()
            if isinstance(selector, Exact)
        }
    )
    location = f"{kind} rule[{rule.index}] at {render_path(path)}"
    if rule.digest is not None and rule.digest != digest:
        return ValidationIssue(
            ErrorCode.ANCHOR_DIGEST_MISMATCH,
            f"{location}: anchor digest mismatch "
            f"(saved {rule.digest}, current {digest or 'missing'}).",
            "warning",
        )
    if rule.matched is not None and rule.matched != count:
        return ValidationIssue(
            ErrorCode.PATTERN_MATCH_COUNT_DRIFT,
            f"{location}: pattern match-count drift "
            f"(saved {rule.matched}, current {count}).",
            "warning",
        )
    return None
