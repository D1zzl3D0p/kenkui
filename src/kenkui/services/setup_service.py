"""setup_service — shared helpers for local setup flows."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def parse_fast_scan_result(scan_raw: dict):
    """Convert an API fast-scan payload into a FastScanResult."""
    from kenkui.models import FastScanResult

    return FastScanResult.from_dict(scan_raw)


def cache_roster_result(scan_result, book_path: Path) -> str | None:
    """Persist a fast-scan roster cache and return the cache path as a string."""
    from kenkui.nlp import cache_roster

    try:
        cached = cache_roster(scan_result, book_path)
    except Exception:
        return None
    return str(cached)


def build_chapter_prompt_items(parsed_book: dict) -> list[SimpleNamespace]:
    """Convert parsed book chapter payloads into lightweight prompt objects."""
    raw_chapters = parsed_book.get("chapters") or []
    return [SimpleNamespace(**chapter) for chapter in raw_chapters]


__all__ = [
    "parse_fast_scan_result",
    "cache_roster_result",
    "build_chapter_prompt_items",
]
