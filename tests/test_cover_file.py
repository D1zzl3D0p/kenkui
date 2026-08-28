"""Accept/reject matrix for a caller-supplied cover image."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

import kenkui as kk
from kenkui._audio.cover import MAX_COVER_BYTES, read_cover
from kenkui._domain.planning import CoverIntent, compile_execution_plan
from kenkui.errors import EncodingError, ErrorCode

if TYPE_CHECKING:
    from pathlib import Path

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
MODEL_REVISION = "pocket-tts/model@0123456789abcdef"
TEXT = "He woke early and left without speaking."


def _voice() -> kk.Voice:
    """Build the resolved voice the planner converts into a VoicePlan."""
    return kk.Voice(
        id="eponine",
        name="Eponine",
        enabled=True,
        provenance="Project-owned recording by Test Speaker",
        license_id="CC0-1.0",
        commercial_use_allowed=True,
        language="en-US",
        content_fingerprint="2" * 64,
        compatible_model_revisions=(MODEL_REVISION,),
        state="loaded",
    )


def _inspection() -> kk.BookInspection:
    """Build a one-chapter inspection for cover planning tests."""
    chapter = kk.ChapterInspection("ch-v1-one", 0, "Chapter One", len(TEXT), TEXT)
    return kk.BookInspection(
        kk.BookMetadata("T", "A", cover_available=True), (chapter,)
    )


def write(tmp_path: Path, name: str, payload: bytes) -> Path:
    """Write a candidate cover file and return its path."""
    path = tmp_path / name
    path.write_bytes(payload)
    return path


@pytest.mark.parametrize(("name", "payload"), [("c.jpg", JPEG), ("c.png", PNG)])
def test_accepts_jpeg_and_png(tmp_path: Path, name: str, payload: bytes) -> None:
    """Both supported formats return their bytes and content digest."""
    data, digest = read_cover(write(tmp_path, name, payload))
    assert data == payload
    assert digest == hashlib.sha256(payload).hexdigest()


def test_rejects_an_unsupported_format(tmp_path: Path) -> None:
    """A GIF is a valid image and still not something we hand to FFmpeg."""
    with pytest.raises(EncodingError) as error:
        read_cover(write(tmp_path, "c.gif", b"GIF89a" + b"\x00" * 64))
    assert error.value.code is ErrorCode.COVER_INVALID


def test_rejects_a_missing_file(tmp_path: Path) -> None:
    """An absent path fails loudly rather than falling back to the source."""
    with pytest.raises(EncodingError) as error:
        read_cover(tmp_path / "absent.jpg")
    assert error.value.code is ErrorCode.COVER_INVALID


def test_rejects_a_directory(tmp_path: Path) -> None:
    """Only regular files are accepted."""
    with pytest.raises(EncodingError) as error:
        read_cover(tmp_path)
    assert error.value.code is ErrorCode.COVER_INVALID


def test_rejects_a_symlink_without_reading_its_target(tmp_path: Path) -> None:
    """A link could redirect the read outside the caller's intent."""
    target = write(tmp_path, "real.jpg", JPEG)
    link = tmp_path / "link.jpg"
    link.symlink_to(target)
    with pytest.raises(EncodingError) as error:
        read_cover(link)
    assert error.value.code is ErrorCode.COVER_INVALID


def test_rejects_a_hardlinked_file(tmp_path: Path) -> None:
    """A second link means another party can swap the bytes after validation."""
    target = write(tmp_path, "real.jpg", JPEG)
    (tmp_path / "second.jpg").hardlink_to(target)
    with pytest.raises(EncodingError) as error:
        read_cover(target)
    assert error.value.code is ErrorCode.COVER_INVALID


def test_rejects_an_oversize_file(tmp_path: Path) -> None:
    """The size bound is enforced from the stat, before the bytes are read."""
    payload = JPEG + b"\x00" * MAX_COVER_BYTES
    with pytest.raises(EncodingError) as error:
        read_cover(write(tmp_path, "big.jpg", payload))
    assert error.value.code is ErrorCode.COVER_INVALID


def test_rejects_an_empty_file(tmp_path: Path) -> None:
    """An empty file has no magic bytes to sniff."""
    with pytest.raises(EncodingError) as error:
        read_cover(write(tmp_path, "empty.jpg", b""))
    assert error.value.code is ErrorCode.COVER_INVALID


def test_plan_records_cover_content_hash_not_path(tmp_path: Path) -> None:
    """Two identical images at different paths must produce the same plan."""
    first = write(tmp_path, "a.jpg", JPEG)
    nested = tmp_path / "nested"
    nested.mkdir()
    second = write(nested, "b.jpg", JPEG)

    plans = [
        compile_execution_plan(
            kk.epub("book.epub").metadata(cover=path).assign_voice("eponine").tts(),
            _inspection(),
            source_bytes_hash="1" * 64,
            resolved_voice=_voice(),
            model_revision=MODEL_REVISION,
            cover_content_hash=read_cover(path)[1],
        )
        for path in (first, second)
    ]
    assert plans[0].output.cover is CoverIntent.FILE
    assert plans[0].output.cover_content_hash == hashlib.sha256(JPEG).hexdigest()
    assert plans[0].semantic_fingerprint == plans[1].semantic_fingerprint


def test_a_different_image_changes_the_plan(tmp_path: Path) -> None:
    """Swapping the cover art must not silently reuse the previous artifact."""
    jpeg = write(tmp_path, "a.jpg", JPEG)
    png = write(tmp_path, "b.png", PNG)

    def plan_for(path: Path) -> object:
        return compile_execution_plan(
            kk.epub("book.epub").metadata(cover=path).assign_voice("eponine").tts(),
            _inspection(),
            source_bytes_hash="1" * 64,
            resolved_voice=_voice(),
            model_revision=MODEL_REVISION,
            cover_content_hash=read_cover(path)[1],
        )

    assert plan_for(jpeg).semantic_fingerprint != plan_for(png).semantic_fingerprint  # type: ignore[attr-defined]
