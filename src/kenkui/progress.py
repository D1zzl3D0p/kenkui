"""Structured generation progress events for kenkui clients."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

ProgressStage = Literal[
    "tts_synthesis",
    "stitching",
    "normalization",
    "cover_embedding",
    "nlp_extraction",
    "nlp_attribution",
]
ProgressStatus = Literal["started", "advanced", "completed", "failed", "message"]
ProgressUnit = Literal["chars", "milliseconds", "items", "chapters", ""]


@dataclass(frozen=True)
class ChapterProgress:
    """Display-neutral progress for one active chapter worker."""

    index: int
    title: str
    completed_units: float = 0.0
    total_units: float = 0.0
    status: ProgressStatus = "advanced"


@dataclass(frozen=True)
class ProgressEvent:
    """Display-neutral facts about audiobook generation progress."""

    stage: ProgressStage
    status: ProgressStatus
    message: str = ""
    completed_units: float = 0.0
    total_units: float = 0.0
    unit: ProgressUnit = ""
    timestamp: float = field(default_factory=time.monotonic)
    book_hash: str = ""
    provider: str = ""
    model: str = ""
    active_chapters: tuple[ChapterProgress, ...] = ()
    total_chapters: int = 0
    chapter_ordinal: int = 0


__all__ = ["ChapterProgress", "ProgressEvent", "ProgressStage", "ProgressStatus", "ProgressUnit"]
