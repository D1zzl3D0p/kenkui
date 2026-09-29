"""Resolve chapter announcements without changing source text or its offsets."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

    from kenkui.inspection import BookInspection


@dataclass(frozen=True, slots=True)
class ChapterAnnouncement:
    """One reviewable decision shared by quoting and synthesis."""

    chapter_id: str
    text: str
    kind: Literal["inserted", "existing", "omitted"]
    heading_end: int = 0

    @property
    def added_characters(self) -> int:
        """Canonical added speech; existing headings are already counted."""
        return len(self.text) if self.kind == "inserted" else 0


def _match(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def resolve_chapter_titles(
    inspection: BookInspection,
    *,
    enabled: bool = True,
    overrides: Mapping[str, str | None] | None = None,
) -> tuple[ChapterAnnouncement, ...]:
    """Resolve authored titles, explicit overrides, and matching opening headings.

    Unknown provenance is deliberately omitted unless explicitly overridden.
    This avoids speaking invented labels from legacy inspection records.
    """
    values = overrides or {}
    result: list[ChapterAnnouncement] = []
    for chapter in inspection.chapters:
        text = values.get(chapter.id, chapter.title)
        eligible = chapter.title_source in {"navigation", "heading", "document"}
        if not enabled or not text or (chapter.id not in values and not eligible):
            result.append(ChapterAnnouncement(chapter.id, "", "omitted"))
            continue
        text = " ".join(text.split())
        heading_end = next(
            (
                end
                for start, end in chapter.heading_ranges
                if start == 0 and _match(chapter.text[:end]) == _match(text)
            ),
            0,
        )
        result.append(
            ChapterAnnouncement(
                chapter.id,
                text,
                "existing" if heading_end else "inserted",
                heading_end,
            )
        )
    return tuple(result)
