"""The addressable grid: a partition of chapter text that depends on nothing else.

Every other partition in this codebase takes a setting, a budget, or a model.
This one takes only the text, which is what lets an annotation anchored to it
survive a change to pause tiers, chunk budgets, or the attribution provider.

Biased to split. Over-splitting costs one extra row in a reviewer's view and
nothing in the audio, because a grid boundary becomes a segment boundary only
when an annotation attaches to it. Under-splitting traps two speakers in one
addressable unit and makes the correction inexpressible. The two failures are
not symmetric, so the guard lists below are allowed to be short.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum, IntFlag, auto
from itertools import pairwise
from types import MappingProxyType
from typing import TYPE_CHECKING

from kenkui._domain.paths import Path
from kenkui._domain.quotes import extract_spans
from kenkui._domain.structure import block_ranges, line_ranges
from kenkui._domain.titles import PREFIX_TITLES

if TYPE_CHECKING:
    from kenkui.inspection import ChapterInspection

# A terminator, any closing brackets or quotes, then whitespace. The lookahead
# keeps the whitespace with the preceding part so joining stays exact.
# Includes each ASCII and typographic closer exactly once.
_SENTENCE: re.Pattern[str] = re.compile(r"""[.!?…]+["'’”)\x5d]*\s+""")  # noqa: RUF001
_PHRASE: re.Pattern[str] = re.compile(r"""[,;:]["'’”)\x5d]*\s+""")  # noqa: RUF001

# A single capital before the period is an initial ("J. R. Smith"), not a
# sentence end.
_INITIAL: re.Pattern[str] = re.compile(r"(?:^|\s)[A-Z]\.$")
# The word a candidate terminator ends, which decides whether it is a title.
_TRAILING_WORD: re.Pattern[str] = re.compile(r"([A-Za-z]+)\.$")


def _is_abbreviation(prefix: str) -> bool:
    """Whether a candidate sentence end is really an abbreviation."""
    if _INITIAL.search(prefix):
        return True
    trailing = _TRAILING_WORD.search(prefix)
    return trailing is not None and trailing.group(1).casefold() in PREFIX_TITLES


def _split(text: str, pattern: re.Pattern[str], *, guard: bool) -> tuple[str, ...]:
    if not text:
        return ()
    parts: list[str] = []
    position = 0
    for match in pattern.finditer(text):
        if guard and _is_abbreviation(text[position : match.start() + 1]):
            continue
        parts.append(text[position : match.end()])
        position = match.end()
    if position < len(text):
        parts.append(text[position:])
    return tuple(parts) if parts else (text,)


def split_sentences(text: str) -> tuple[str, ...]:
    """Split on sentence-terminal punctuation, guarding abbreviations."""
    return _split(text, _SENTENCE, guard=True)


def split_phrases(text: str) -> tuple[str, ...]:
    """Split on clause punctuation. No guard: commas do not abbreviate."""
    return _split(text, _PHRASE, guard=False)


@dataclass(frozen=True, slots=True)
class Unit:
    """One addressable run of a chapter's canonical text."""

    chapter_id: str
    paragraph: int
    line: int
    sentence: int
    phrase: int
    start: int
    end: int
    is_dialogue: bool
    is_emphasised: bool
    # One-based identity of the source quotation covering this leaf. Adjacent
    # quotations can share an edge, so the dialogue flag alone cannot preserve
    # their mandatory attribution boundary.
    dialogue_run: int | None = None
    # Parser-provided heading identity is structural input, like emphasis. It
    # lets the derived gaps carry heading reasons without rescanning text.
    is_heading: bool = False


@dataclass(frozen=True, slots=True)
class DialogueRange:
    """One contiguous canonical range covered by dialogue-marked leaves."""

    chapter_id: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class LeafRange:
    """One contiguous half-open range of leaf indices."""

    first: int
    past_last: int


class GapReason(IntFlag):
    """Structural ranges that close after one leaf, stored as a bit set."""

    NONE = 0
    PHRASE = auto()
    SENTENCE = auto()
    LINE = auto()
    PARAGRAPH = auto()
    HEADING_BEFORE = auto()
    HEADING_AFTER = auto()
    CHAPTER = auto()


@dataclass(frozen=True, slots=True, init=False, eq=False)
class StructuralIndex(Mapping[Path, LeafRange]):
    """Immutable prefix ranges and structural gaps derived from grid leaves.

    Ranges contain leaf indices, never canonical offsets or copied text. The
    caller's leaf tuple therefore remains the sole source of canonical data.
    """

    _ranges: Mapping[Path, LeafRange]
    gaps: tuple[GapReason, ...]

    def __init__(self, units: Iterable[Unit]) -> None:
        """Validate ordered leaves and derive their structural metadata."""
        leaves = tuple(units)
        _validate_structure(leaves)
        mutable_ranges: dict[Path, LeafRange] = {}
        for leaf_index, unit in enumerate(leaves):
            for prefix in _path_prefixes(unit):
                previous = mutable_ranges.get(prefix)
                first = leaf_index if previous is None else previous.first
                mutable_ranges[prefix] = LeafRange(first, leaf_index + 1)
        object.__setattr__(self, "_ranges", MappingProxyType(mutable_ranges))
        object.__setattr__(self, "gaps", _gap_reasons(leaves))

    def __getitem__(self, path: Path) -> LeafRange:
        """Return the contiguous leaf range addressed by ``path``."""
        return self._ranges[path]

    def __iter__(self) -> Iterator[Path]:
        """Iterate prefixes in canonical hierarchy order."""
        return iter(self._ranges)

    def __len__(self) -> int:
        """Return the number of indexed path prefixes."""
        return len(self._ranges)

    def __eq__(self, other: object) -> bool:
        """Compare only deterministic derived values."""
        if not isinstance(other, StructuralIndex):
            return NotImplemented
        return self._ranges == other._ranges and self.gaps == other.gaps

    def __hash__(self) -> int:
        """Hash the same deterministic immutable values used for equality."""
        return hash((tuple(self._ranges.items()), self.gaps))


def _coordinates(unit: Unit) -> tuple[int, int, int, int]:
    return (unit.paragraph, unit.line, unit.sentence, unit.phrase)


def _path_prefixes(unit: Unit) -> tuple[Path, ...]:
    chapter = Path(chapter=unit.chapter_id)
    paragraph = Path(chapter=unit.chapter_id, paragraph=unit.paragraph)
    line = Path(
        chapter=unit.chapter_id,
        paragraph=unit.paragraph,
        line=unit.line,
    )
    sentence = Path(
        chapter=unit.chapter_id,
        paragraph=unit.paragraph,
        line=unit.line,
        sentence=unit.sentence,
    )
    phrase = Path(
        chapter=unit.chapter_id,
        paragraph=unit.paragraph,
        line=unit.line,
        sentence=unit.sentence,
        phrase=unit.phrase,
    )
    return (chapter, paragraph, line, sentence, phrase)


class _GridViolation(Enum):
    EMPTY_CHAPTER_ID = "chapter IDs must be non-empty"
    CHAPTER_ID_CHANGED = "chapter ID changed within one leaf sequence"
    INVALID_COORDINATE = "coordinates must be one-based integers"
    INVALID_RANGE = "canonical ranges must be non-empty and ordered"
    NONZERO_START = "the first canonical range must begin at zero"
    INVALID_FIRST_PATH = "the first path must begin at one at every level"
    DISCONTIGUOUS_RANGE = "canonical ranges must be contiguous"
    DUPLICATE_PATH = "leaf paths must be unique"
    SKIPPED_SIBLING = "sibling coordinates must increase without gaps"
    UNRESET_CHILD = "child coordinates must reset at parent boundaries"


class _GridStructureError(ValueError):
    def __init__(self, violation: _GridViolation) -> None:
        super().__init__(f"invalid grid structure: {violation.value}")


def _validate_unit(unit: Unit, chapter_id: str) -> None:
    coordinates = _coordinates(unit)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in coordinates
    ):
        raise _GridStructureError(_GridViolation.INVALID_COORDINATE)
    if unit.chapter_id != chapter_id:
        raise _GridStructureError(_GridViolation.CHAPTER_ID_CHANGED)
    if (
        not isinstance(unit.start, int)
        or isinstance(unit.start, bool)
        or not isinstance(unit.end, int)
        or isinstance(unit.end, bool)
        or unit.start < 0
        or unit.end <= unit.start
    ):
        raise _GridStructureError(_GridViolation.INVALID_RANGE)


def _validate_first(unit: Unit) -> None:
    if unit.start != 0:
        raise _GridStructureError(_GridViolation.NONZERO_START)
    if _coordinates(unit) != (1, 1, 1, 1):
        raise _GridStructureError(_GridViolation.INVALID_FIRST_PATH)


def _validate_transition(previous: Unit, unit: Unit) -> None:
    if unit.start != previous.end:
        raise _GridStructureError(_GridViolation.DISCONTIGUOUS_RANGE)
    previous_coordinates = _coordinates(previous)
    coordinates = _coordinates(unit)
    changed_level = next(
        (
            index
            for index, (before, after) in enumerate(
                zip(previous_coordinates, coordinates, strict=True)
            )
            if before != after
        ),
        None,
    )
    if changed_level is None:
        raise _GridStructureError(_GridViolation.DUPLICATE_PATH)
    if coordinates[changed_level] != previous_coordinates[changed_level] + 1:
        raise _GridStructureError(_GridViolation.SKIPPED_SIBLING)
    if any(value != 1 for value in coordinates[changed_level + 1 :]):
        raise _GridStructureError(_GridViolation.UNRESET_CHILD)


def _validate_structure(units: tuple[Unit, ...]) -> None:
    if not units:
        return
    chapter_id = units[0].chapter_id
    if not chapter_id:
        raise _GridStructureError(_GridViolation.EMPTY_CHAPTER_ID)
    _validate_unit(units[0], chapter_id)
    _validate_first(units[0])
    for previous, unit in pairwise(units):
        _validate_unit(unit, chapter_id)
        _validate_transition(previous, unit)


def _gap_reasons(units: tuple[Unit, ...]) -> tuple[GapReason, ...]:
    gaps: list[GapReason] = []
    for index, unit in enumerate(units):
        reasons = GapReason.PHRASE
        if index + 1 == len(units):
            reasons |= (
                GapReason.SENTENCE
                | GapReason.LINE
                | GapReason.PARAGRAPH
                | GapReason.CHAPTER
            )
        else:
            following = units[index + 1]
            if (
                following.sentence != unit.sentence
                or following.line != unit.line
                or following.paragraph != unit.paragraph
            ):
                reasons |= GapReason.SENTENCE
            if following.line != unit.line or following.paragraph != unit.paragraph:
                reasons |= GapReason.LINE
            if following.paragraph != unit.paragraph:
                reasons |= GapReason.PARAGRAPH
                if unit.is_heading:
                    reasons |= GapReason.HEADING_AFTER
                if following.is_heading:
                    reasons |= GapReason.HEADING_BEFORE
        gaps.append(reasons)
    return tuple(gaps)


def build_structure_index(units: Iterable[Unit]) -> StructuralIndex:
    """Build immutable tree-like traversal metadata from ordered leaves."""
    return StructuralIndex(units)


def dialogue_ranges(units: Iterable[Unit]) -> tuple[DialogueRange, ...]:
    """Coalesce dialogue leaves only within the same source quotation."""
    ranges: list[DialogueRange] = []
    previous_run: int | None = None
    for unit in units:
        if not unit.is_dialogue:
            continue
        if (
            ranges
            and unit.dialogue_run is not None
            and unit.dialogue_run == previous_run
            and ranges[-1].chapter_id == unit.chapter_id
            and ranges[-1].end == unit.start
        ):
            previous = ranges[-1]
            ranges[-1] = DialogueRange(previous.chapter_id, previous.start, unit.end)
        else:
            ranges.append(DialogueRange(unit.chapter_id, unit.start, unit.end))
        previous_run = unit.dialogue_run
    return tuple(ranges)


def unit_text(unit: Unit, chapter_text: str) -> str:
    """Return a unit's canonical text."""
    return chapter_text[unit.start : unit.end]


def unit_digest(unit: Unit, chapter_text: str) -> str:
    """Return a short content hash for anchor verification."""
    payload = unit_text(unit, chapter_text).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()[:16]}"


def sibling_counts(units: Iterable[Unit]) -> dict[tuple[str, ...], int]:
    """Return how many children each addressed parent has, keyed by its path.

    ``Last()`` is the only selector whose meaning depends on context, and this
    is that context. Keys are string coordinates so they compare equal to the
    parent path ``matches`` builds while walking a unit.
    """
    counts: dict[tuple[str, ...], int] = {}
    for unit in units:
        parent: tuple[str, ...] = (unit.chapter_id,)
        for coordinate in (unit.paragraph, unit.line, unit.sentence, unit.phrase):
            counts[parent] = max(counts.get(parent, 0), coordinate)
            parent = (*parent, str(coordinate))
    return counts


def _cut_points(text: str, offset: int, edges: frozenset[int]) -> list[int]:
    """Return offsets where a phrase must be broken by a quote edge."""
    return sorted(edge for edge in edges if offset < edge < offset + len(text))


def _apply_cuts(text: str, offset: int, edges: frozenset[int]) -> Iterator[str]:
    """Yield text pieces split at every quote edge, in canonical order."""
    position = offset
    for edge in _cut_points(text, offset, edges):
        yield text[position - offset : edge - offset]
        position = edge
    yield text[position - offset :]


def _covers(start: int, length: int, ranges: Iterable[tuple[int, int]]) -> bool:
    """Return whether the unit midpoint falls inside any half-open range."""
    midpoint = start + length / 2
    return any(range_start <= midpoint < range_end for range_start, range_end in ranges)


def _covering_run(
    start: int, length: int, ranges: Iterable[tuple[int, int]]
) -> int | None:
    """Return the one-based source-range identity covering a leaf midpoint."""
    midpoint = start + length / 2
    return next(
        (
            index
            for index, (range_start, range_end) in enumerate(ranges, start=1)
            if range_start <= midpoint < range_end
        ),
        None,
    )


def build_grid(chapter: ChapterInspection) -> tuple[Unit, ...]:
    """Partition one chapter into addressable units.

    Depends only on the chapter's canonical text, its recorded emphasis, and
    quote extraction -- which is itself a pure function of the text. No pause
    setting, chunk budget, or model can change the result.
    """
    text = chapter.text
    if not text:
        return ()
    spans = extract_spans(chapter.id, text)
    edges = frozenset({span.start for span in spans} | {span.end for span in spans})
    dialogue = tuple((span.start, span.end) for span in spans if span.is_dialogue)
    units: list[Unit] = []
    offset = 0
    headings = frozenset(chapter.headings)
    for p_index, block in enumerate(block_ranges(text), start=1):
        is_heading = text[block.start : block.body_end] in headings
        for l_index, line_range in enumerate(line_ranges(text, block), start=1):
            line = text[line_range.start : line_range.end]
            for s_index, sentence in enumerate(split_sentences(line), start=1):
                ph_index = 0
                for phrase in split_phrases(sentence):
                    for piece in _apply_cuts(phrase, offset, edges):
                        ph_index += 1
                        dialogue_run = _covering_run(offset, len(piece), dialogue)
                        units.append(
                            Unit(
                                chapter.id,
                                p_index,
                                l_index,
                                s_index,
                                ph_index,
                                offset,
                                offset + len(piece),
                                is_dialogue=dialogue_run is not None,
                                is_emphasised=_covers(
                                    offset, len(piece), chapter.emphasis
                                ),
                                dialogue_run=dialogue_run,
                                is_heading=is_heading,
                            )
                        )
                        offset += len(piece)
    return tuple(units)
