"""Secure source-cover selection and materialization from an EPUB snapshot."""

from __future__ import annotations

import os
from contextlib import suppress
from typing import TYPE_CHECKING
from zipfile import BadZipFile, LargeZipFile, ZipFile

from kenkui._epub.archive import validate_archive
from kenkui._epub.parser import _local, _member_map, _package_path, _read, _xml
from kenkui._epub.paths import resolve_member
from kenkui.errors import EncodingError, ErrorCode, SourceError

if TYPE_CHECKING:
    from pathlib import Path


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
