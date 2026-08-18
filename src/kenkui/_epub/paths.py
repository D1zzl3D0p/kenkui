"""Canonical POSIX-only EPUB member resolution."""

from __future__ import annotations

import posixpath
from urllib.parse import unquote, urlsplit

from kenkui.errors import ErrorCode, SourceError


def canonical_member(path: str) -> str:
    """Canonicalize one archive-root-relative member or reject it."""
    decoded = unquote(path)
    if not decoded or "\x00" in decoded or "\\" in decoded or decoded.startswith("/"):
        raise SourceError(ErrorCode.UNSAFE_ARCHIVE_PATH)
    parts = decoded.split("/")
    depth = 0
    for part in parts:
        if part in ("", "."):
            continue
        if part == "..":
            depth -= 1
            if depth < 0:
                raise SourceError(ErrorCode.UNSAFE_ARCHIVE_PATH)
        else:
            depth += 1
    canonical = posixpath.normpath(decoded)
    if canonical in ("", ".", "..") or canonical.startswith("../"):
        raise SourceError(ErrorCode.UNSAFE_ARCHIVE_PATH)
    return canonical


def resolve_member(base_member: str, href: str) -> tuple[str, str]:
    """Resolve a relative href beside a member and preserve decoded fragment."""
    parsed = urlsplit(href)
    if parsed.scheme or parsed.netloc or parsed.query:
        raise SourceError(ErrorCode.UNSAFE_ARCHIVE_PATH)
    if "\\" in href:
        raise SourceError(ErrorCode.UNSAFE_ARCHIVE_PATH)
    joined = posixpath.join(posixpath.dirname(base_member), unquote(parsed.path))
    return canonical_member(joined), unquote(parsed.fragment)
