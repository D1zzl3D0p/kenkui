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
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kenkui._characters.identity import PREFIX_TITLES
from kenkui._characters.quotes import extract_spans
from kenkui._domain.structure import _blocks, _lines

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

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
    for p_index, (body, chunk) in enumerate(_blocks(text), start=1):
        for l_index, line in enumerate(_lines(chunk, body), start=1):
            for s_index, sentence in enumerate(split_sentences(line), start=1):
                ph_index = 0
                for phrase in split_phrases(sentence):
                    for piece in _apply_cuts(phrase, offset, edges):
                        ph_index += 1
                        units.append(
                            Unit(
                                chapter.id,
                                p_index,
                                l_index,
                                s_index,
                                ph_index,
                                offset,
                                offset + len(piece),
                                is_dialogue=_covers(offset, len(piece), dialogue),
                                is_emphasised=_covers(
                                    offset, len(piece), chapter.emphasis
                                ),
                            )
                        )
                        offset += len(piece)
    return tuple(units)
