"""Tier introspection: identity, tuning, and style at a glance."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import kenkui as kk

if TYPE_CHECKING:
    from pathlib import Path

_RULE_COUNT = 50
_IDENTITY_OPERATION_COUNT = 2


def test_tuning_lists_rules_in_declaration_order(epub_path: Path) -> None:
    """Later declarations must render after earlier ones, never sorted.

    Declaration order is the precedence tiebreaker, so a display that sorts
    misreports which rule wins. Character IDs that appear nowhere else in the
    rendering -- not in the group header, not in a path -- are what make the
    two positions unambiguous.
    """
    book = kk.book(epub_path).attribute("zeta").attribute("kappa")
    rendered = repr(book.tuning)
    assert rendered.count("zeta") == 1
    assert rendered.count("kappa") == 1
    assert rendered.index("zeta") < rendered.index("kappa")


def test_tuning_marks_unsaved_rules(epub_path: Path, tmp_path: Path) -> None:
    """A rule declared after loading annotations is not yet on disk."""
    saved = kk.book(epub_path).attribute("a").write_annotations(tmp_path / "s.json")
    book = kk.book(epub_path).annotations(saved).attribute("b")
    assert "unsaved" in repr(book.tuning)
    assert repr(book.tuning).count("unsaved") == 1


def test_display_elides_degenerate_line_level(epub_path: Path) -> None:
    """Line one is degenerate at display; the stored path still keeps it."""
    book = kk.book(epub_path).attribute(
        "jessica", where={"chapter": "ch08", "paragraph": 3, "line": 1, "sentence": 2}
    )
    assert "¶3  s2" in repr(book.tuning)
    assert "l1" not in repr(book.tuning)


def test_long_rule_lists_are_truncated_but_iterable(epub_path: Path) -> None:
    """Display truncates per kind; iteration still yields every rule."""
    book = kk.book(epub_path)
    for index in range(_RULE_COUNT):
        book = book.attribute(f"c{index}", where={"chapter": f"ch{index:02d}"})
    assert "more" in repr(book.tuning)
    assert len(list(book.tuning)) == _RULE_COUNT


def test_style_shows_effective_values_after_replacement(epub_path: Path) -> None:
    """Style settings replace rather than accumulate; only the latest shows."""
    book = kk.book(epub_path).pauses(paragraph_ms=400).pauses(paragraph_ms=500)
    assert "500ms" in repr(book.style)
    assert "400ms" not in repr(book.style)


def test_introspection_does_not_parse(
    epub_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """These are properties, so they must not reach the filesystem.

    The real symbol lives at ``kenkui.pipeline.inspect_epub`` (imported by
    name into that module), not at ``kenkui._epub.inspect_epub`` -- the
    ``_epub`` package re-exports nothing, so patching that path would raise
    ``AttributeError`` before the pipeline code ever ran.
    """

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("parsed")

    monkeypatch.setattr("kenkui.pipeline.inspect_epub", forbidden)
    repr(kk.book(epub_path).attribute("a").tuning)


def test_identity_summary_lists_operations_and_iterates(epub_path: Path) -> None:
    """Metadata and series are identity-tier and both show up, in order."""
    book = (
        kk.book(epub_path)
        .metadata(title="Dune", author="Frank Herbert")
        .series("dune", book=1, allow_recast=True, allow_narrator_change=True)
    )
    rendered = repr(book.identity)
    assert "Dune" in rendered
    assert "allow_recast=True" in rendered
    assert len(list(book.identity)) == _IDENTITY_OPERATION_COUNT


def test_identity_summary_reports_nothing_set(epub_path: Path) -> None:
    """A book with no identity intent says so plainly, not an empty block."""
    assert "nothing set" in repr(kk.book(epub_path).identity)


def test_identity_summary_covers_chapter_selection_and_explicit_cover(
    epub_path: Path, tmp_path: Path
) -> None:
    """A chapter range and an explicit cover path both render in identity."""
    cover = tmp_path / "cover.jpg"
    cover.write_bytes(b"\x00")
    book = kk.book(epub_path).metadata(cover=cover).select_chapter_range("ch08", "ch09")
    rendered = repr(book.identity)
    assert "select_chapter_range" in rendered
    assert str(cover) in rendered


def test_identity_summary_covers_explicit_chapters_and_patterns(
    epub_path: Path,
) -> None:
    """Explicit chapter IDs and grid patterns each have their own rendering."""
    assert "select_chapters" in repr(
        kk.book(epub_path).select_chapters("ch08", "ch09").identity
    )
    assert "select(" in repr(kk.book(epub_path).select({"chapter": "ch08"}).identity)


def test_tuning_summary_reports_no_rules(epub_path: Path) -> None:
    """A book with no tuning intent says so plainly, not an empty block."""
    assert "no rules" in repr(kk.book(epub_path).tuning)


def test_tuning_groups_every_kind_separately(epub_path: Path) -> None:
    """Silences and pronunciations render as their own groups, not folded in."""
    book = (
        kk.book(epub_path)
        .silence(500, where={"chapter": "ch08"})
        .pronounce({"Muad'Dib": "Moo-ahd-Deeb"})
    )
    rendered = repr(book.tuning)
    assert "silences" in rendered
    assert "pronunciations" in rendered


def test_tuning_renders_set_valued_selectors(epub_path: Path) -> None:
    """Wildcards, ranges, and 'last' render as extra components, not silently."""
    book = kk.book(epub_path).silence(
        200,
        where=(
            {"chapter": "ch08", "paragraph": [1, 3]},
            {"chapter": "ch08", "paragraph": "2..4"},
            {"chapter": "ch08", "paragraph": -1},
        ),
    )
    rendered = repr(book.tuning)
    assert "{1,3}" in rendered
    assert "2..4" in rendered
    assert "last" in rendered


def test_style_summary_covers_casting_and_pronunciation_settings(
    epub_path: Path,
) -> None:
    """Style formatting covers the models, casting, and pronunciation presets."""
    book = (
        kk.book(epub_path)
        .pronounce(numbers="standard", builtin=False, currency=True)
        .infer_characters("spacy")
        .attribute_quotes("some-model")
        .assign_voices(narrator="ivy", unknown="michael", cast={"Paul": "sam"})
    )
    rendered = repr(book.style)
    assert "infer_characters" in rendered
    assert "attribute_quotes" in rendered
    assert "assign_voices" in rendered
    assert "builtin=False" in rendered
    assert "currency=True" in rendered


def test_style_summary_shows_pauses_off_and_synthesis_intent(
    epub_path: Path,
) -> None:
    """A zeroed pauses() call and a declared tts() both render their state."""
    assert "pauses(off)" in repr(kk.book(epub_path).pauses().style)
    book = (
        kk.book(epub_path)
        .infer_characters("spacy")
        .attribute_quotes("some-model")
        .assign_voices(narrator="ivy")
        .tts()
    )
    assert "tts()" in repr(book.style)
