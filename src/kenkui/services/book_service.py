"""book_service — high-level functions for parsing ebooks and filtering chapters.

Public API:
  parse_book(ebook_path, cache) -> BookParseResult
  filter_chapters(book_hash_key, selection, cache) -> ChapterFilterResult
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kenkui.nlp import book_hash
from kenkui.readers import get_reader

from .book_cache import BookCache, ChapterSummary


@dataclass
class BookParseResult:
    book_hash: str
    metadata: dict            # {title, author, language, publisher, description}
    chapters: list[ChapterSummary]
    total_chapters: int
    total_word_count: int


@dataclass
class ChapterFilterResult:
    included_indices: list[int]
    chapter_count: int
    estimated_word_count: int
    chapters: list[ChapterSummary]   # only included ones


def parse_book(ebook_path: str, cache: BookCache) -> BookParseResult:
    """Parse an ebook file and return structured metadata + chapter summaries.

    Results are stored in *cache* so that repeated calls for the same
    (unmodified) file are served from cache without re-parsing.

    Raises:
        FileNotFoundError: if ebook_path does not exist on disk.
        ValueError: if the file format is not supported.
    """
    path = Path(ebook_path)
    if not path.exists():
        raise FileNotFoundError(f"Ebook not found: {ebook_path}")

    # Compute hash (sha256 of path + mtime → 32-char hex).
    book_hash_value = book_hash(path)

    # Extra guard: same mtime hash collision between different file paths is
    # astronomically unlikely, but the path check is a cheap extra sanity check.
    entry = cache.get(book_hash_value)
    if entry is not None and entry.ebook_path == str(path.resolve()):
        return BookParseResult(
            book_hash=book_hash_value,
            metadata=entry.metadata,
            chapters=entry.chapter_summaries,
            total_chapters=len(entry.chapter_summaries),
            total_word_count=sum(s.word_count for s in entry.chapter_summaries),
        )

    # Parse the ebook.
    reader = get_reader(path)
    metadata = reader.get_metadata()
    chapters = reader.get_chapters()

    # Store in cache (BookCache builds ChapterSummary objects internally).
    entry = cache.put(book_hash_value, str(path.resolve()), metadata, chapters)

    return BookParseResult(
        book_hash=book_hash_value,
        metadata=entry.metadata,
        chapters=entry.chapter_summaries,
        total_chapters=len(entry.chapter_summaries),
        total_word_count=sum(s.word_count for s in entry.chapter_summaries),
    )


def filter_chapters(
    book_hash_key: str,
    selection: "ChapterSelection",  # noqa: F821 – imported below
    cache: BookCache,
) -> ChapterFilterResult:
    """Apply a ChapterSelection to the cached chapters for *book_hash_key*.

    If `selection.preset` is `ChapterPreset.NONE`, returns an empty result by design.

    Raises:
        KeyError: if *book_hash_key* is not present in *cache*.
    """
    from kenkui.models import Chapter, ChapterPreset
    from kenkui.chapter_filter import ChapterFilter, FilterOperation

    if selection.preset == ChapterPreset.NONE:
        # "none" means empty selection by design — return early
        return ChapterFilterResult(included_indices=[], chapter_count=0, estimated_word_count=0, chapters=[])

    entry = cache.get(book_hash_key)
    if entry is None:
        raise KeyError("Book not found in cache. Call parse_book first.")

    # ChapterFilter only inspects ch.tags and ch.title — paragraphs are not
    # needed here and are intentionally omitted to avoid re-reading the ebook.
    chapters = [
        Chapter(
            index=s.index,
            title=s.title,
            paragraphs=[],
            tags=s.tags,
            toc_index=s.toc_index,
        )
        for s in entry.chapter_summaries
    ]

    # Build FilterOperation list from ChapterSelection.
    preset = selection.preset
    if preset.value in ("manual", "custom"):
        operations: list[FilterOperation] = [
            FilterOperation("index", str(idx)) for idx in selection.included
        ]
        if not operations:
            operations = [FilterOperation("preset", "content-only")]
    else:
        operations = [FilterOperation("preset", preset.value)]

    # Apply operations.
    filtered = ChapterFilter(operations).apply(chapters)

    # Returned indices are always in ascending order (ChapterFilter.apply sorts them).
    # selection.included order is NOT preserved.

    # Apply explicit exclusions.
    if selection.excluded:
        excluded_set = set(selection.excluded)
        filtered = [ch for ch in filtered if ch.index not in excluded_set]

    # Map filtered chapters back to their ChapterSummary objects.
    summary_by_index = {s.index: s for s in entry.chapter_summaries}
    included_summaries = [
        summary_by_index[ch.index] for ch in filtered if ch.index in summary_by_index
    ]

    return ChapterFilterResult(
        included_indices=[ch.index for ch in filtered],
        chapter_count=len(filtered),
        estimated_word_count=sum(s.word_count for s in included_summaries),
        chapters=included_summaries,
    )
