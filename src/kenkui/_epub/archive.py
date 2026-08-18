"""Pre-read ZIP validation for untrusted EPUB archives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from kenkui.errors import ErrorCode, SourceError

from .paths import canonical_member

if TYPE_CHECKING:
    from zipfile import ZipFile


@dataclass(frozen=True, slots=True)
class ArchiveLimits:
    """Hard pre-read EPUB ZIP limits."""

    max_members: int = 10_000
    max_member_size: int = 64 * 1024 * 1024
    max_total_size: int = 512 * 1024 * 1024
    max_ratio: float = 100.0


def validate_archive(archive: ZipFile, limits: ArchiveLimits | None = None) -> None:
    """Validate every central-directory entry before any content is read."""
    limits = limits or ArchiveLimits()
    members = archive.infolist()
    if len(members) > limits.max_members:
        raise SourceError(ErrorCode.ARCHIVE_LIMIT)
    total = 0
    names: set[str] = set()
    for info in members:
        raw_name = info.filename.rstrip("/")
        if ".." in raw_name.split("/"):
            raise SourceError(ErrorCode.UNSAFE_ARCHIVE_PATH)
        name = canonical_member(raw_name)
        if name in names:
            raise SourceError(ErrorCode.MALFORMED_EPUB)
        names.add(name)
        if info.flag_bits & 0x1:
            raise SourceError(ErrorCode.MALFORMED_EPUB)
        if info.file_size < 0 or info.compress_size < 0:
            raise SourceError(ErrorCode.MALFORMED_EPUB)
        total += info.file_size
        if info.file_size > limits.max_member_size or total > limits.max_total_size:
            raise SourceError(ErrorCode.ARCHIVE_LIMIT)
        if info.file_size and (
            info.compress_size == 0
            or info.file_size / info.compress_size > limits.max_ratio
        ):
            raise SourceError(ErrorCode.ARCHIVE_LIMIT)
