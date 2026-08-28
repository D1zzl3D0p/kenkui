"""Secure cover selection and materialization for the encoded artifact."""

from __future__ import annotations

import hashlib
import os
import stat
from contextlib import suppress
from typing import TYPE_CHECKING
from zipfile import BadZipFile, LargeZipFile, ZipFile

from kenkui._epub.archive import validate_archive
from kenkui._epub.parser import _local, _member_map, _package_path, _read, _xml
from kenkui._epub.paths import resolve_member
from kenkui.errors import EncodingError, ErrorCode, SourceError

if TYPE_CHECKING:
    from pathlib import Path

MAX_COVER_BYTES = 8 * 1024 * 1024
# Magic bytes rather than the file extension: FFmpeg is handed this stream, and
# an extension is a claim while a signature is evidence.
_SIGNATURES = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n")


def materialize_source_cover(source: Path, destination: Path) -> None:
    """Write the declared source cover to one fixed exclusive private path."""
    descriptor = -1
    failed = False
    try:
        with ZipFile(source, "r") as archive:
            validate_archive(archive)
            members = _member_map(archive)
            package_path = _package_path(
                _xml(_read(archive, members, "META-INF/container.xml"))
            )
            package = _xml(_read(archive, members, package_path))
            cover_ids = {
                element.attrib.get("content", "")
                for element in package.iter()
                if _local(element.tag) == "meta"
                and element.attrib.get("name", "").lower() == "cover"
            }
            candidates: list[str] = []
            for item in package.iter():
                if _local(item.tag) != "item":
                    continue
                item_id = item.attrib.get("id", "")
                properties = item.attrib.get("properties", "").split()
                if "cover-image" not in properties and item_id not in cover_ids:
                    continue
                member, fragment = resolve_member(
                    package_path, item.attrib.get("href", "")
                )
                if fragment or member not in members:
                    raise SourceError(  # noqa: TRY301
                        ErrorCode.MALFORMED_EPUB
                    )
                candidates.append(member)
            if not candidates:
                raise SourceError(ErrorCode.MALFORMED_EPUB)  # noqa: TRY301
            payload = _read(archive, members, candidates[0])
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(destination, flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except (SourceError, BadZipFile, LargeZipFile, OSError, ValueError):
        failed = True
    if failed:
        if descriptor >= 0:
            os.close(descriptor)
        with suppress(OSError):
            destination.unlink(missing_ok=True)
        raise EncodingError(ErrorCode.COVER_FAILED) from None


def read_cover(path: Path) -> tuple[bytes, str]:
    """Validate a caller-supplied cover and return its bytes and digest.

    Fails loudly. A caller who names a specific image does not want a silent
    fallback to whatever the EPUB happened to contain.
    """
    descriptor = -1
    try:
        if path.is_symlink():
            raise EncodingError(ErrorCode.COVER_INVALID)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size <= 0
            or info.st_size > MAX_COVER_BYTES
        ):
            raise EncodingError(ErrorCode.COVER_INVALID)
        payload = os.read(descriptor, info.st_size)
        if len(payload) != info.st_size or not payload.startswith(_SIGNATURES):
            raise EncodingError(ErrorCode.COVER_INVALID)
    except (OSError, ValueError):
        raise EncodingError(ErrorCode.COVER_INVALID) from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return payload, hashlib.sha256(payload).hexdigest()


def materialize_file_cover(payload: bytes, destination: Path) -> None:
    """Write already-validated cover bytes to one fixed exclusive private path."""
    descriptor = -1
    failed = False
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(destination, flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        failed = True
    if failed:
        if descriptor >= 0:
            os.close(descriptor)
        with suppress(OSError):
            destination.unlink(missing_ok=True)
        raise EncodingError(ErrorCode.COVER_FAILED) from None
