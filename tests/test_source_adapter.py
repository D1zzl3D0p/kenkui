"""Source adaptation preserves the EPUB parser and snapshot identity contract."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

from helpers import make_epub, xhtml
from kenkui._epub.parser import inspect_epub
from kenkui._source_adapter import inspect_document, snapshot_document
from kenkui.cancellation import CancellationToken
from kenkui.errors import CancelledError, ErrorCode, SourceError

if TYPE_CHECKING:
    from pathlib import Path


def test_adapter_preserves_complete_epub_inspection(tmp_path: Path) -> None:
    """Keep metadata, full spine order, emphasis and scene/heading ranges."""
    path = make_epub(
        tmp_path / "book.epub",
        chapters={
            "one": xhtml(
                "<h1>First</h1><p>A <em>quiet</em> voice.</p><hr/><p>After.</p>"
            ),
            "two": xhtml("<h1>Second</h1><p>Another chapter.</p>"),
        },
        spine=("one", "two"),
        cover=True,
    )
    assert inspect_document(path, "epub") == inspect_epub(path)


def test_snapshot_hash_and_inspection_use_the_same_bytes(tmp_path: Path) -> None:
    """Changes to the original after copying cannot change the inspected book."""
    path = make_epub(
        tmp_path / "book.epub",
        chapters={"one": xhtml("<p>Original text.</p>")},
        spine=("one",),
    )
    expected = inspect_epub(path)
    original = path.read_bytes()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    snapshot = snapshot_document(path, "epub", workspace, None)
    path.write_bytes(b"changed input")
    assert snapshot.path.read_bytes() == original
    assert snapshot.source_hash == hashlib.sha256(original).hexdigest()
    assert inspect_document(snapshot.path, snapshot.format) == expected


def test_unknown_format_is_rejected_before_copying(tmp_path: Path) -> None:
    """A missing path must not obscure an unsupported-format failure."""
    path = tmp_path / "unknown.bin"
    with pytest.raises(SourceError) as failure:
        snapshot_document(path, "../unknown", tmp_path, None)
    assert failure.value.code == ErrorCode.UNSUPPORTED_FORMAT
    with pytest.raises(SourceError) as failure:
        inspect_document(path, "unknown")
    assert failure.value.code == ErrorCode.UNSUPPORTED_FORMAT
    assert not tuple(tmp_path.iterdir())


def test_cancelled_snapshot_does_not_create_a_file(tmp_path: Path) -> None:
    """Cancellation is checked before opening the source or creating output."""
    token = CancellationToken()
    token.cancel()
    with pytest.raises(CancelledError):
        snapshot_document(tmp_path / "missing.epub", "epub", tmp_path, token)
    assert not tuple(tmp_path.iterdir())
