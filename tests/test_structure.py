"""Pure canonical structural-range discovery."""

from __future__ import annotations

import random
from itertools import pairwise

import pytest

from kenkui._domain.structure import BlockRange, LineRange, block_ranges, line_ranges


def test_blocks_retain_repeated_blank_separators() -> None:
    """Block offsets retain repeated separators and tile the source."""
    text = "Chapter One\n\n\n\nBody.\n\nTail."
    assert block_ranges(text) == (
        BlockRange(0, 11, 15),
        BlockRange(15, 20, 22),
        BlockRange(22, 27, 27),
    )
    assert "".join(text[item.start : item.end] for item in block_ranges(text)) == text


def test_lines_retain_single_newlines_and_the_block_separator() -> None:
    """Line offsets retain single newlines and their parent separator."""
    text = "one\ntwo\nthree\n\nTail."
    first = block_ranges(text)[0]
    assert line_ranges(text, first) == (
        LineRange(0, 4),
        LineRange(4, 8),
        LineRange(8, 15),
    )
    assert (
        "".join(text[item.start : item.end] for item in line_ranges(text, first))
        == (text[first.start : first.end])
    )


@pytest.mark.parametrize("text", ["", "\n\n", "\n\nA", "A\n\n", "A\n\n\nB"])
def test_ranges_cover_edge_case_text_exactly(text: str) -> None:
    """Leading, trailing, and separator-only blocks retain exact coverage."""
    blocks = block_ranges(text)
    assert blocks[0].start == 0
    assert all(left.end == right.start for left, right in pairwise(blocks))
    assert blocks[-1].end == len(text)
    assert "".join(text[item.start : item.end] for item in blocks) == text
    for block in blocks:
        lines = line_ranges(text, block)
        assert lines[0].start == block.start
        assert all(left.end == right.start for left, right in pairwise(lines))
        assert lines[-1].end == block.end


@pytest.mark.parametrize("seed", range(20))
def test_discovery_is_exact_over_random_text(seed: int) -> None:
    """Nested block and line ranges tile randomized canonical text exactly."""
    rng = random.Random(seed)  # noqa: S311 - deterministic fixture, not crypto
    alphabet = ["a", "b", " ", "\n", "\n\n", "\n\n\n", ".", "Q"]
    for _ in range(500):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 40)))
        blocks = block_ranges(text)
        assert "".join(text[item.start : item.end] for item in blocks) == text
        assert all(
            "".join(text[line.start : line.end] for line in line_ranges(text, block))
            == text[block.start : block.end]
            for block in blocks
        )
