from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _count_chapters_with_reader(book_path: Path) -> int | None:
    """Count chapters using the reader interface (supports all formats)."""
    from .readers import get_reader

    try:
        reader = get_reader(book_path, verbose=False)
        return reader.count_chapters()
    except Exception:
        return None


def quick_count_chapters(ebook_path: Path) -> int | None:
    """Quickly count chapters in an ebook by parsing TOC without full extraction.

    This lightweight version uses the reader interface to count TOC entries
    without extracting full text content.

    Supports all ebook formats: EPUB, MOBI, AZW, AZW3, AZW4

    Returns None if counting fails.
    """
    logger.debug(f"Counting chapters in {ebook_path}")
    return _count_chapters_with_reader(ebook_path)


def get_bundled_voices() -> list[str]:
    """Return the names of all available bundled voices (compiled + uncompiled).

    Backward-compatible shim over :func:`~kenkui.voice_registry.get_registry`.
    Returns voice *names* sorted alphabetically.
    """
    from .voice_registry import get_registry

    return sorted(
        v.name
        for v in get_registry().voices
        if v.source in ("compiled", "uncompiled")
    )


__all__ = [
    "get_bundled_voices",
    "quick_count_chapters",
]
