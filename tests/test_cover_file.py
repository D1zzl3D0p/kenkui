"""Accept/reject matrix for a caller-supplied cover image."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

from kenkui._audio.cover import MAX_COVER_BYTES, read_cover
from kenkui.errors import EncodingError, ErrorCode

if TYPE_CHECKING:
    from pathlib import Path

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


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
