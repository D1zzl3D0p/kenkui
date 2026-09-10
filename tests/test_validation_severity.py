"""Warning-severity validation findings that do not block a render."""

from __future__ import annotations

from typing import TYPE_CHECKING

import kenkui as kk

if TYPE_CHECKING:
    from pathlib import Path


def test_warning_does_not_invalidate(epub_path: Path) -> None:
    """Incomparable rule overlap warns but leaves the pipeline valid."""
    book = (
        kk.book(epub_path)
        .assign_voice("ivy")
        .attribute("paul", where={"chapter": "ch08", "paragraph": "*"})
        .attribute("irulan", where={"chapter": "*", "paragraph": 1})
        .tts()
    )
    result = book.validate()
    assert result.warnings
    assert result.is_valid


def test_an_overlap_warning_names_both_rules(epub_path: Path) -> None:
    """The remedy is to move one of two lines, which needs both named.

    Two overlapping pairs among three rules produce two warnings, and a
    reader who cannot tell them apart has no way to act on either.
    """
    book = (
        kk.book(epub_path)
        .assign_voice("ivy")
        .attribute("paul", where={"chapter": "ch08", "paragraph": "*"})
        .attribute("irulan", where={"chapter": "*", "paragraph": 1})
        .attribute("jessica", where={"chapter": "*", "paragraph": 2})
        .silence(500, where={"chapter": "ch08", "paragraph": "*"})
        .silence(0, where={"chapter": "*", "paragraph": 1})
        .tts()
    )
    messages = [warning.message for warning in book.validate().warnings]
    assert all(
        warning.code is kk.ErrorCode.RULE_OVERLAP
        for warning in book.validate().warnings
    )
    assert len(messages) == len(set(messages))
    assert "Attributions rule[0] and rule[1]" in messages[0]
    assert "Attributions rule[0] and rule[2]" in messages[1]
    assert "Silences rule[0] and rule[1]" in messages[2]
    assert all("declaration order breaks the tie" in message for message in messages)


def test_error_still_invalidates(tmp_path: Path) -> None:
    """A missing source is still an error that invalidates the pipeline."""
    result = kk.book(tmp_path / "missing.epub").validate()
    assert not result.is_valid
    assert result.errors


def test_existing_issues_default_to_error() -> None:
    """Constructing a `ValidationIssue` without severity keeps its old meaning."""
    assert (
        kk.ValidationIssue(code=kk.ErrorCode.INVALID_OUTPUT, message="x").severity
        == "error"
    )
