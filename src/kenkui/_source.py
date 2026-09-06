"""Bounded source hashing and private snapshots shared by resolution and rendering."""

from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING

from .errors import ErrorCode, KenkuiError, SourceError

if TYPE_CHECKING:
    from pathlib import Path

    from .cancellation import CancellationToken

_READ_CHUNK_BYTES = 64 * 1024
_HASH_CHUNK_BYTES = 1024 * 1024
MAX_COMPRESSED_SOURCE_BYTES = 256 * 1024 * 1024


def snapshot_source(
    source: Path, snapshot: Path, cancel: CancellationToken | None
) -> str:
    """Copy and hash the same bounded source bytes, rejecting concurrent writes."""
    digest = hashlib.sha256()
    failed = False
    too_large = False
    changed = False
    try:
        with source.open("rb") as source_stream, snapshot.open("xb") as target:
            before = os.fstat(source_stream.fileno())
            copied = 0
            while chunk := source_stream.read(_READ_CHUNK_BYTES):
                copied += len(chunk)
                if copied > MAX_COMPRESSED_SOURCE_BYTES:
                    too_large = True
                    break
                target.write(chunk)
                digest.update(chunk)
                if cancel is not None:
                    cancel.raise_if_cancelled()
            after = os.fstat(source_stream.fileno())
            changed = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
    except KenkuiError:
        raise
    except OSError:
        failed = True
    if too_large:
        raise SourceError(ErrorCode.ARCHIVE_LIMIT)
    if failed or changed:
        raise SourceError(ErrorCode.SOURCE_NOT_READABLE)
    return digest.hexdigest()


def source_digest(path: Path) -> str:
    """Hash the source bytes in bounded chunks.

    The same identity the plan uses, so attribution stored for a book is found
    again on a later render of that same book and not of an edited copy.
    """
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
    except FileNotFoundError:
        raise SourceError(ErrorCode.SOURCE_NOT_FOUND) from None
    except OSError:
        raise SourceError(ErrorCode.SOURCE_NOT_READABLE) from None
    return digest.hexdigest()
