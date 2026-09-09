"""Addressable-grid unit tests."""

from kenkui._domain.grid import build_grid, unit_digest, unit_text
from kenkui.inspection import ChapterInspection


def chapter(
    text: str, *, emphasis: tuple[tuple[int, int], ...] = ()
) -> ChapterInspection:
    """Build a minimal chapter inspection for grid tests."""
    return ChapterInspection(
        id="ch01",
        index=0,
        title="One",
        speech_characters=None,
        text=text,
        emphasis=emphasis,
    )


def test_grid_is_exact() -> None:
    """Grid units reconstruct the canonical chapter text exactly."""
    text = 'Alpha one. Alpha two.\n\n"Beta," she said. "Gamma," he answered.'
    units = build_grid(chapter(text))
    assert "".join(unit_text(unit, text) for unit in units) == text


def test_levels_are_one_based_and_reset_under_their_parent() -> None:
    """Hierarchy indices begin at one and reset below each parent."""
    units = build_grid(chapter("One. Two.\n\nThree."))
    assert (units[0].paragraph, units[0].sentence) == (1, 1)
    assert (units[1].paragraph, units[1].sentence) == (1, 2)
    assert (units[2].paragraph, units[2].sentence) == (2, 1)


def test_quote_edges_force_boundaries() -> None:
    """Dialogue quote edges become addressable unit boundaries."""
    text = '"Yes," she said. "No," he answered.'
    units = build_grid(chapter(text))
    spoken = [unit_text(u, text) for u in units if u.is_dialogue]
    assert '"Yes,"' in "".join(spoken)
    assert '"No,"' in "".join(spoken)


def test_two_speakers_are_never_trapped_in_one_unit() -> None:
    """Separate quoted speakers never share an addressable unit."""
    text = '"Yes," she said. "No," he answered.'
    units = build_grid(chapter(text))
    for unit in units:
        rendered = unit_text(unit, text)
        assert not ("Yes" in rendered and "No" in rendered)


def test_line_level_separates_verse() -> None:
    """Single newlines preserve distinct line indices within a block."""
    text = "Line one\nLine two\n\nProse."
    units = build_grid(chapter(text))
    assert {unit.line for unit in units if unit.paragraph == 1} == {1, 2}


def test_emphasis_marks_units() -> None:
    """A unit whose midpoint is emphasised carries that annotation."""
    text = "He thought I must not fear and stopped."
    units = build_grid(chapter(text, emphasis=((11, 26),)))
    assert any(unit.is_emphasised for unit in units)


def test_empty_chapter_yields_no_units() -> None:
    """Empty canonical text has no addressable units."""
    assert build_grid(chapter("")) == ()


def test_digest_is_stable_and_content_derived() -> None:
    """Unit digests are stable and distinguish different content."""
    text = "One. Two."
    units = build_grid(chapter(text))
    assert unit_digest(units[0], text) == unit_digest(units[0], text)
    assert unit_digest(units[0], text) != unit_digest(units[1], text)
    assert unit_digest(units[0], text).startswith("sha256:")
