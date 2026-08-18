"""Deterministic test synthesis engine."""

from __future__ import annotations

import hashlib

from kenkui.errors import ErrorCode, RenderError

from .protocols import SynthesisTask, SynthesizedAudio

FAKE_SAMPLE_RATE_HZ = 16_000
FAKE_CHANNELS = 1
_FRAMES_PER_CHARACTER = 160


class DeterministicFakeEngine:
    """Render deterministic PCM whose length and content derive from exact text."""

    def synthesize(self, task: SynthesisTask) -> SynthesizedAudio:
        """Produce signed 16-bit mono PCM without external effects."""
        frame_count = len(task.text) * _FRAMES_PER_CHARACTER
        output_bytes = frame_count * task.channels * 2
        if (
            type(task.max_output_bytes) is not int
            or task.max_output_bytes <= 0
            or output_bytes > task.max_output_bytes
        ):
            raise RenderError(ErrorCode.INVALID_AUDIO)
        digest = hashlib.sha256(task.text.encode("utf-8")).digest()
        pcm = bytearray(output_bytes)
        for frame in range(frame_count):
            # Small centered samples make a valid, deterministic test waveform.
            sample = (digest[frame % len(digest)] - 128) * 128
            pcm[frame * 2 : frame * 2 + 2] = sample.to_bytes(2, "little", signed=True)
        return SynthesizedAudio(
            segment_id=task.segment_id,
            chapter_id=task.chapter_id,
            pcm_s16le=bytes(pcm),
            sample_rate_hz=task.sample_rate_hz,
            channels=task.channels,
            frame_count=frame_count,
            duration_ms=frame_count * 1000 // task.sample_rate_hz,
        )
