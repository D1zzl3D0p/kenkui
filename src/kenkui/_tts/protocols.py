"""Private synthesis contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class SynthesisTask:
    """Spawn-safe exact input and hard output-byte limit for one synthesis unit."""

    segment_id: str
    chapter_id: str
    text: str
    sample_rate_hz: int
    channels: int
    max_output_bytes: int


@dataclass(frozen=True, slots=True)
class SynthesizedAudio:
    """Spawn-safe signed 16-bit interleaved PCM and explicit audio metadata."""

    segment_id: str
    chapter_id: str
    pcm_s16le: bytes
    sample_rate_hz: int
    channels: int
    frame_count: int
    duration_ms: int


@runtime_checkable
class SynthesisEngine(Protocol):
    """Private provider-neutral synthesis boundary."""

    def synthesize(self, task: SynthesisTask) -> SynthesizedAudio:
        """Render the task's exact text as PCM."""
        ...
