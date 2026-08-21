"""Private audio assembly contracts and deterministic fake assembler."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    from kenkui._domain.planning import ExecutionPlan
    from kenkui._tts.protocols import SegmentAudio


@dataclass(frozen=True, slots=True)
class ChapterAudio:
    """Actual chapter audio metadata produced by assembly."""

    chapter_id: str
    duration_ms: int


@dataclass(frozen=True, slots=True)
class AssemblyRequest:
    """Complete ordered input to a private assembler/publisher boundary."""

    plan: ExecutionPlan
    audio: tuple[SegmentAudio, ...]
    pcm_parts: tuple[Path, ...]
    workspace_output: Path
    source_epub: Path | None = None


@dataclass(frozen=True, slots=True)
class AssemblyResult:
    """Validated candidate artifact and actual audio metadata."""

    artifact: Path
    size_bytes: int
    artifact_sha256: str
    sample_rate_hz: int
    channels: int
    duration_ms: int
    chapters: tuple[ChapterAudio, ...]


@runtime_checkable
class ArtifactAssembler(Protocol):
    """Private replaceable artifact assembler boundary."""

    def preflight(self, *, expect_cover: bool = False) -> None:
        """Validate required assembly resources before synthesis begins."""
        ...

    def assemble(self, request: AssemblyRequest) -> AssemblyResult:
        """Create a candidate artifact only at the workspace path."""
        ...


class FakeArtifactAssembler:
    """Assemble an inspectable deterministic artifact without FFmpeg."""

    def preflight(self, *, expect_cover: bool = False) -> None:
        """Accept preflight because the deterministic fake has no resources."""

    def assemble(self, request: AssemblyRequest) -> AssemblyResult:
        """Write ordered PCM identities to the requested workspace artifact."""
        audio = request.audio
        sample_rate = audio[0].sample_rate_hz
        channels = audio[0].channels
        boundaries = chapter_frame_boundaries_ms(request.plan, audio)
        chapters = tuple(
            ChapterAudio(chapter.id, end - start)
            for chapter, (start, end) in zip(
                request.plan.output.chapters, pairwise(boundaries), strict=True
            )
        )
        lines = [
            b"KENKUI-FAKE-M4B",
            request.plan.semantic_fingerprint.encode("ascii"),
            f"{sample_rate}:{channels}".encode(),
        ]
        digests = _segment_digests(audio, request.pcm_parts)
        lines.extend(
            b":".join(
                (
                    item.segment_id.encode("utf-8"),
                    item.chapter_id.encode("utf-8"),
                    str(item.duration_ms).encode(),
                    digest.encode("ascii"),
                )
            )
            for item, digest in zip(audio, digests, strict=True)
        )
        artifact_bytes = b"\n".join(lines) + b"\n"
        with request.workspace_output.open("xb") as stream:
            stream.write(artifact_bytes)
        return AssemblyResult(
            artifact=request.workspace_output,
            size_bytes=len(artifact_bytes),
            artifact_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
            sample_rate_hz=sample_rate,
            channels=channels,
            duration_ms=boundaries[-1],
            chapters=chapters,
        )


def _segment_digests(
    audio: tuple[SegmentAudio, ...], pcm_parts: tuple[Path, ...]
) -> tuple[str, ...]:
    """Hash each segment by reading its exact span from its chapter's part."""
    order: list[str] = []
    for item in audio:
        if item.chapter_id not in order:
            order.append(item.chapter_id)
    parts = dict(zip(order, pcm_parts, strict=True))
    digests: list[str] = []
    handles: dict[str, object] = {}
    try:
        for item in audio:
            stream = handles.get(item.chapter_id)
            if stream is None:
                stream = parts[item.chapter_id].open("rb")
                handles[item.chapter_id] = stream
            payload = stream.read(item.byte_count)  # type: ignore[attr-defined]
            if len(payload) != item.byte_count:
                message = "chapter part is shorter than its segment metadata"
                raise ValueError(message)
            digests.append(hashlib.sha256(payload).hexdigest())
    finally:
        for stream in handles.values():
            stream.close()  # type: ignore[attr-defined]
    return tuple(digests)


def cumulative_frame_boundaries_ms(
    audio: tuple[SegmentAudio, ...],
) -> tuple[int, ...]:
    """Convert cumulative exact frames to integral milliseconds without drift."""
    if not audio or any(
        item.sample_rate_hz != audio[0].sample_rate_hz for item in audio
    ):
        message = "audio must have one common sample rate"
        raise ValueError(message)
    rate = audio[0].sample_rate_hz
    frames = 0
    boundaries = [0]
    for item in audio:
        frames += item.frame_count
        boundaries.append(frames * 1000 // rate)
    return tuple(boundaries)


def chapter_frame_boundaries_ms(
    plan: ExecutionPlan, audio: tuple[SegmentAudio, ...]
) -> tuple[int, ...]:
    """Aggregate contiguous segment frames into exact semantic chapter boundaries."""
    if len(audio) != len(plan.segments) or not audio:
        message = "audio must match plan segments"
        raise ValueError(message)
    if any(
        item.chapter_id != segment.chapter_id
        for item, segment in zip(audio, plan.segments, strict=True)
    ):
        message = "audio chapter identity must match plan"
        raise ValueError(message)
    rate = audio[0].sample_rate_hz
    if any(item.sample_rate_hz != rate for item in audio):
        message = "audio must have one common sample rate"
        raise ValueError(message)
    by_chapter: dict[str, int] = {}
    for item in audio:
        by_chapter[item.chapter_id] = (
            by_chapter.get(item.chapter_id, 0) + item.frame_count
        )
    expected_ids = tuple(chapter.id for chapter in plan.output.chapters)
    if tuple(by_chapter) != expected_ids:
        message = "audio chapters must match selected chapter order"
        raise ValueError(message)
    frames = 0
    boundaries = [0]
    for chapter_id in expected_ids:
        frames += by_chapter[chapter_id]
        boundaries.append(frames * 1000 // rate)
    return tuple(boundaries)
