"""Uncompressed WAV assembly over the coordinator's verified PCM parts."""

from __future__ import annotations

import hashlib
import wave
from itertools import pairwise
from typing import TYPE_CHECKING

from kenkui._audio.m4b import AssemblyResult, ChapterAudio, chapter_frame_boundaries_ms

if TYPE_CHECKING:
    from kenkui._audio.m4b import AssemblyRequest

_READ_BYTES = 64 * 1024


class WavArtifactAssembler:
    """Stream PCM into a WAV with no cover, tags, or embedded chapter markers."""

    def preflight(self, *, expect_cover: bool = False) -> None:
        """WAV assembly uses the standard library and requires no external tools."""

    def assemble(self, request: AssemblyRequest) -> AssemblyResult:
        """Create a private PCM WAV and report normal coordinator audio metadata."""
        audio = request.audio
        boundaries = chapter_frame_boundaries_ms(request.plan, audio)
        with (
            request.workspace_output.open("xb") as target,
            wave.open(target, "wb") as output,
        ):
            output.setnchannels(audio[0].channels)
            output.setsampwidth(2)
            output.setframerate(audio[0].sample_rate_hz)
            output.setnframes(sum(item.frame_count for item in audio))
            for part in request.pcm_parts:
                with part.open("rb") as source:
                    while payload := source.read(_READ_BYTES):
                        output.writeframesraw(payload)
        with request.workspace_output.open("rb") as artifact:
            digest = hashlib.file_digest(artifact, "sha256").hexdigest()
        return AssemblyResult(
            request.workspace_output,
            request.workspace_output.stat().st_size,
            digest,
            audio[0].sample_rate_hz,
            audio[0].channels,
            boundaries[-1],
            tuple(
                ChapterAudio(chapter.id, end - start)
                for chapter, (start, end) in zip(
                    request.plan.output.chapters, pairwise(boundaries), strict=True
                )
            ),
        )
