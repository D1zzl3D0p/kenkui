"""Assemble logical chapters without changing source order or losing annotations."""

from __future__ import annotations

import hashlib
from dataclasses import replace

from kenkui.errors import ErrorCode, SourceError
from kenkui.inspection import ChapterInspection


def slice_chapter(
    chapter: ChapterInspection, start: int, end: int, identity: str, title: str
) -> ChapterInspection:
    """Partition canonical text and clip annotations to the resulting section."""
    text = chapter.text[start:end]
    left = len(text) - len(text.lstrip())
    text = text.strip()
    start += left
    end = start + len(text)

    def clip(ranges: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
        return tuple(
            (max(a, start) - start, min(b, end) - start)
            for a, b in ranges
            if a < end and b > start
        )

    headings = clip(chapter.heading_ranges)
    return replace(
        chapter,
        id=identity,
        title=title,
        text=text,
        speech_characters=len(text),
        headings=tuple(text[a:b] for a, b in headings),
        emphasis=clip(chapter.emphasis),
        heading_ranges=headings,
        scene_ranges=clip(chapter.scene_ranges),
    )


def join_chapters(parts: list[ChapterInspection], index: int) -> ChapterInspection:
    """Join file pieces with paragraph boundaries and offset their annotations."""
    if len(parts) == 1:
        return replace(parts[0], index=index)
    text = "\n\n".join(part.text for part in parts)
    emphasis: list[tuple[int, int]] = []
    headings: list[tuple[int, int]] = []
    scenes: list[tuple[int, int]] = []
    offset = 0
    for part in parts:
        for target, ranges in (
            (emphasis, part.emphasis),
            (headings, part.heading_ranges),
            (scenes, part.scene_ranges),
        ):
            target.extend((a + offset, b + offset) for a, b in ranges)
        offset += len(part.text) + 2
    # Changed grouping changes offsets and must not reuse old chapter caches.
    identity = "\0".join(part.id for part in parts)
    digest = hashlib.sha256(identity.encode()).hexdigest()[:24]
    return ChapterInspection(
        f"ch-v2-{digest}",
        index,
        parts[0].title,
        len(text),
        text,
        tuple(heading for part in parts for heading in part.headings),
        tuple(emphasis),
        tuple(headings),
        tuple(scenes),
        parts[0].title_source,
    )


class ChapterBuilder:
    """Group TOC intervals, retaining untitled leading content as separate entries."""

    def __init__(self, max_chapters: int) -> None:
        self._max_chapters = max_chapters
        self.chapters: list[ChapterInspection] = []
        self._pending: list[ChapterInspection] = []
        self._toc_started = False

    def _flush(self) -> None:
        if self._pending:
            if len(self.chapters) >= self._max_chapters:
                raise SourceError(ErrorCode.ARCHIVE_LIMIT)
            self.chapters.append(join_chapters(self._pending, len(self.chapters)))
            self._pending = []

    def add(
        self, part: ChapterInspection, *, boundary: bool, continuation: bool
    ) -> None:
        """Append in reading order, opening a new chapter only at a boundary."""
        if boundary:
            self._flush()
            self._toc_started = True
        elif not self._toc_started and not continuation:
            self._flush()
        if not part.title:
            part = replace(
                part,
                title=f"Untitled section {len(self.chapters) + 1}",
                title_source="generated",
            )
        self._pending.append(part)

    def finish(self) -> tuple[ChapterInspection, ...]:
        """Retain trailing content in the final chapter."""
        self._flush()
        return tuple(self.chapters)
