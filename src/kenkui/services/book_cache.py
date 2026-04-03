"""BookCache — persistent JSON cache for parsed ebook metadata.

Stores chapter summaries (metadata only, NOT paragraph text) so that ebook
parse results survive server restarts.  Full paragraph text is always
re-read from the original ebook file when AudioBuilder needs it.
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..chapter_classifier import ChapterTags

if TYPE_CHECKING:
    from ..models import Chapter
    from ..readers import EbookMetadata


@dataclass
class ChapterSummary:
    """Lightweight summary of a chapter — no paragraph text."""

    index: int
    title: str
    word_count: int
    paragraph_count: int
    toc_index: int
    tags: ChapterTags


@dataclass
class BookEntry:
    """Cache entry for a single parsed ebook."""

    book_hash: str
    ebook_path: str       # str (not Path) for JSON serializability
    metadata: dict        # serialized EbookMetadata fields (no cover_image bytes)
    chapter_summaries: list[ChapterSummary]
    parsed_at: float      # unix timestamp


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _summary_to_dict(s: ChapterSummary) -> dict:
    return {
        "index": s.index,
        "title": s.title,
        "word_count": s.word_count,
        "paragraph_count": s.paragraph_count,
        "toc_index": s.toc_index,
        "tags": dataclasses.asdict(s.tags),
    }


def _summary_from_dict(d: dict) -> ChapterSummary:
    return ChapterSummary(
        index=d["index"],
        title=d["title"],
        word_count=d["word_count"],
        paragraph_count=d["paragraph_count"],
        toc_index=d["toc_index"],
        tags=ChapterTags(**d["tags"]),
    )


def _entry_to_dict(e: BookEntry) -> dict:
    return {
        "book_hash": e.book_hash,
        "ebook_path": e.ebook_path,
        "parsed_at": e.parsed_at,
        "metadata": e.metadata,
        "chapter_summaries": [_summary_to_dict(s) for s in e.chapter_summaries],
    }


def _entry_from_dict(d: dict) -> BookEntry:
    return BookEntry(
        book_hash=d["book_hash"],
        ebook_path=d["ebook_path"],
        parsed_at=d["parsed_at"],
        metadata=d["metadata"],
        chapter_summaries=[_summary_from_dict(s) for s in d["chapter_summaries"]],
    )


def _metadata_to_dict(metadata: "EbookMetadata") -> dict:
    """Serialize EbookMetadata, explicitly excluding cover_image bytes."""
    return {
        "title": metadata.title,
        "author": metadata.author,
        "language": metadata.language,
        "publisher": metadata.publisher,
        "description": metadata.description,
    }


def _word_count(chapter: "Chapter") -> int:
    """Estimate word count from paragraph text."""
    return sum(len(p.split()) for p in chapter.paragraphs)


# ---------------------------------------------------------------------------
# BookCache
# ---------------------------------------------------------------------------

class BookCache:
    """Persistent JSON cache for parsed ebook metadata.

    Thread-safety is not guaranteed; callers in a single-process server
    environment should be fine.  The cache file is written atomically via
    a temp-file-then-rename pattern on save.
    """

    def __init__(self, cache_path: Path | None = None) -> None:
        if cache_path is None:
            from ..config import _xdg_config_home
            cache_path = _xdg_config_home() / "kenkui" / "book_cache.json"
        self._cache_path = cache_path
        self._entries: dict[str, BookEntry] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def put(
        self,
        book_hash: str,
        ebook_path: str,
        metadata: "EbookMetadata",
        chapters: "list[Chapter]",
    ) -> BookEntry:
        """Store a parsed book in the cache and return the new BookEntry."""
        summaries = [
            ChapterSummary(
                index=ch.index,
                title=ch.title,
                word_count=_word_count(ch),
                paragraph_count=len(ch.paragraphs),
                toc_index=ch.toc_index,
                tags=ch.tags,
            )
            for ch in chapters
        ]
        entry = BookEntry(
            book_hash=book_hash,
            ebook_path=ebook_path,
            metadata=_metadata_to_dict(metadata),
            chapter_summaries=summaries,
            parsed_at=time.time(),
        )
        self._entries[book_hash] = entry
        self._save()
        return entry

    def get(self, book_hash: str) -> BookEntry | None:
        """Return the cached BookEntry for book_hash, or None if not found."""
        return self._entries.get(book_hash)

    def evict(self, book_hash: str) -> None:
        """Remove a book from the cache and persist the change."""
        self._entries.pop(book_hash, None)
        self._save()

    # ------------------------------------------------------------------
    # Internal persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Read cache from disk.  Creates an empty cache if file is missing."""
        if not self._cache_path.exists():
            self._entries = {}
            return
        try:
            raw = json.loads(self._cache_path.read_text(encoding="utf-8"))
            self._entries = {
                k: _entry_from_dict(v)
                for k, v in raw.get("entries", {}).items()
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            # Corrupted cache — start fresh rather than crashing.
            self._entries = {}

    def _save(self) -> None:
        """Write cache to disk, creating parent directories as needed."""
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"entries": {k: _entry_to_dict(v) for k, v in self._entries.items()}}
        # Atomic write: write to a temp file then replace.
        tmp = self._cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._cache_path)
