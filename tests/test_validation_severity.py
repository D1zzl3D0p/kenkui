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
