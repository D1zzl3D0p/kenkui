from __future__ import annotations

import importlib.resources
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


def get_bundled_voices():
    """Scans the 'voices' directory inside the package for custom voice files.

    Returns a list of filenames.
    """
    custom_voices = []
    try:
        pkg_name = __package__

        if pkg_name:
            voices_path = importlib.resources.files(pkg_name) / "voices"
            if voices_path.is_dir():
                for entry in voices_path.iterdir():
                    if entry.is_file() and not entry.name.startswith("__"):
                        custom_voices.append(entry.name)
        else:
            local_voices_path = Path(__file__).parent / "voices"
            if local_voices_path.exists():
                custom_voices = [
                    f.name
                    for f in local_voices_path.iterdir()
                    if f.is_file() and not f.name.startswith("__")
                ]

    except Exception:
        pass

    return sorted(custom_voices)


__all__ = [
    "get_bundled_voices",
    "quick_count_chapters",
]
