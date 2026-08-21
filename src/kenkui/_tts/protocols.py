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
    # Which conditioning state renders this unit. A cast holds several, and a
    # worker that had to guess would emit valid audio in the wrong voice.
    voice_asset_sha256: str = ""


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


@dataclass(frozen=True, slots=True)
class SegmentAudio:
    """One segment's audio metadata once its PCM has been spilled to disk.

    Assembly needs frame counts and identities for chapter markers and probing,
    but not the samples themselves, so carrying the payload past the render
    stage is what forces a whole book into memory at once.
    """

    segment_id: str
    chapter_id: str
    sample_rate_hz: int
    channels: int
    frame_count: int
    duration_ms: int

    @property
    def byte_count(self) -> int:
        """Exact signed 16-bit interleaved payload size this metadata implies."""
        return self.frame_count * self.channels * 2


def segment_audio(audio: SynthesizedAudio) -> SegmentAudio:
    """Drop a rendered segment's payload, keeping only what assembly reads."""
    return SegmentAudio(
        audio.segment_id,
        audio.chapter_id,
        audio.sample_rate_hz,
        audio.channels,
        audio.frame_count,
        audio.duration_ms,
    )


@runtime_checkable
class SynthesisEngine(Protocol):
    """Private provider-neutral synthesis boundary."""

    def synthesize(self, task: SynthesisTask) -> SynthesizedAudio:
        """Render the task's exact text as PCM."""
        ...
