"""Pure discovery of canonical block and line ranges.

Structure is a property of canonical chapter text. Pause policy is applied
later by planning to the gap reasons derived from these ranges; it must never
decide which ranges exist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_BLOCK = re.compile(r"\n{2,}")
_NEWLINE = re.compile(r"\n")


@dataclass(frozen=True, slots=True)
class BlockRange:
    """One exact canonical block range and its separator-free body edge."""

    start: int
    body_end: int
    end: int


@dataclass(frozen=True, slots=True)
class LineRange:
    """One exact canonical line range, including its trailing separator."""

    start: int
    end: int


def block_ranges(text: str) -> tuple[BlockRange, ...]:
    """Return exact canonical block ranges, retaining every separator."""
    ranges: list[BlockRange] = []
    position = 0
    for match in _BLOCK.finditer(text):
        ranges.append(BlockRange(position, match.start(), match.end()))
        position = match.end()
    if position < len(text) or not ranges:
        ranges.append(BlockRange(position, len(text), len(text)))
    return tuple(ranges)


def line_ranges(text: str, block: BlockRange) -> tuple[LineRange, ...]:
    """Return exact canonical lines within ``block``.

    Single newlines stay with the preceding line. The block separator stays
    with the final line, matching the canonical grid's exact-tiling contract.
    """
    ranges: list[LineRange] = []
    position = block.start
    for match in _NEWLINE.finditer(text, block.start, block.body_end):
        ranges.append(LineRange(position, match.end()))
        position = match.end()
    if position < block.body_end or not ranges:
        ranges.append(LineRange(position, block.body_end))
    final = ranges[-1]
    ranges[-1] = LineRange(final.start, block.end)
    return tuple(ranges)
