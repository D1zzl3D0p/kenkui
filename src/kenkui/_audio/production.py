"""Production FFmpeg-backed M4B assembler."""

from __future__ import annotations

import hashlib
import os
import stat
from contextlib import suppress
from typing import TYPE_CHECKING

from kenkui._audio.cover import materialize_source_cover
from kenkui._audio.ffmpeg import FFmpegTools
from kenkui._audio.m4b import (
    AssemblyResult,
    ChapterAudio,
    chapter_frame_boundaries_ms,
)
from kenkui._audio.native import run_checked
from kenkui._audio.probe import validate_artifact
from kenkui._domain.planning import CoverIntent
from kenkui.errors import EncodingError, ErrorCode

if TYPE_CHECKING:
    from pathlib import Path

    from kenkui._audio.m4b import AssemblyRequest

_ENCODE_TIMEOUT_SECONDS = 300.0
_READ_CHUNK_BYTES = 64 * 1024


class FFmpegM4BAssembler:
    """Encode ordered PCM and transactionally return only a validated candidate."""

    def __init__(self, tools: FFmpegTools | None = None) -> None:
        self._tools = tools or FFmpegTools()

    def preflight(self, *, expect_cover: bool = False) -> None:
        """Discover and capability-check FFmpeg and ffprobe."""
        self._tools.preflight(expect_cover=expect_cover)

    def assemble(self, request: AssemblyRequest) -> AssemblyResult:
        """Encode, probe, fully decode, and hash one workspace candidate."""
        candidate = request.workspace_output
        pcm = candidate.with_suffix(".pcm")
        metadata = candidate.with_suffix(".ffmetadata")
        cover = candidate.with_suffix(".cover")
        try:
            _validate_request(request)
            expect_cover = (
                request.plan.output.cover is CoverIntent.SOURCE
                and request.plan.output.source_cover_available
            )
            tools = self._tools.preflight(expect_cover=expect_cover)
            _require_absent_candidate(candidate)
            _write_pcm(pcm, request)
            _write_metadata(metadata, request)
            if expect_cover:
                if request.source_epub is None:
                    raise EncodingError(ErrorCode.COVER_FAILED)  # noqa: TRY301
                materialize_source_cover(request.source_epub, cover)
            argv = _encode_argv(
                tools.ffmpeg,
                pcm,
                metadata,
                cover if expect_cover else None,
                candidate,
                request.audio[0].sample_rate_hz,
                request.audio[0].channels,
            )
            run_checked(
                self._tools.runner,
                argv,
                timeout=_ENCODE_TIMEOUT_SECONDS,
                code=ErrorCode.ENCODING_FAILED,
            )
            _require_regular_artifact(candidate)
            validate_artifact(
                candidate,
                request.plan,
                request.audio,
                tools,
                self._tools.runner,
                expect_cover=expect_cover,
            )
            size, digest = _hash_file(candidate)
            boundaries = chapter_frame_boundaries_ms(request.plan, request.audio)
            return AssemblyResult(
                candidate,
                size,
                digest,
                request.audio[0].sample_rate_hz,
                request.audio[0].channels,
                boundaries[-1],
                tuple(
                    ChapterAudio(chapter.id, end - start)
                    for chapter, start, end in zip(
                        request.plan.output.chapters,
                        boundaries[:-1],
                        boundaries[1:],
                        strict=True,
                    )
                ),
            )
        except EncodingError:
            with suppress(OSError):
                candidate.unlink(missing_ok=True)
            raise
        except (OSError, ValueError):
            with suppress(OSError):
                candidate.unlink(missing_ok=True)
            raise EncodingError(ErrorCode.ASSEMBLY_FAILED) from None
        finally:
            for temporary in (pcm, metadata, cover):
                with suppress(OSError):
                    temporary.unlink(missing_ok=True)


def _validate_request(request: AssemblyRequest) -> None:
    audio = request.audio
    if (
        not audio
        or len(audio) != len(request.plan.segments)
        or any(
            item.sample_rate_hz != audio[0].sample_rate_hz
            or item.channels != audio[0].channels
            or item.segment_id != segment.id
            or item.chapter_id != segment.chapter_id
            or len(item.pcm_s16le) != item.frame_count * item.channels * 2
            or item.frame_count <= 0
            or item.duration_ms <= 0
            for item, segment in zip(audio, request.plan.segments, strict=True)
        )
    ):
        raise EncodingError(ErrorCode.ASSEMBLY_FAILED)


def _require_absent_candidate(candidate: Path) -> None:
    if candidate.exists() or candidate.is_symlink():
        raise EncodingError(ErrorCode.ASSEMBLY_FAILED)


def _write_pcm(path: Path, request: AssemblyRequest) -> None:
    with path.open("xb") as stream:
        for item in request.audio:
            stream.write(item.pcm_s16le)
        stream.flush()
        os.fsync(stream.fileno())


def _write_metadata(path: Path, request: AssemblyRequest) -> None:
    output = request.plan.output
    lines = [";FFMETADATA1"]
    if output.title is not None:
        lines.append(f"title={_escape(output.title)}")
    if output.author is not None:
        author = _escape(output.author)
        lines.extend((f"artist={author}", f"author={author}"))
    boundaries = chapter_frame_boundaries_ms(request.plan, request.audio)
    for chapter, start, end in zip(
        output.chapters, boundaries[:-1], boundaries[1:], strict=True
    ):
        lines.extend(
            (
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={start}",
                f"END={end}",
                f"title={_escape(chapter.title)}",
            )
        )
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _escape(value: str) -> str:
    """Escape every ffmetadata syntax character, including physical newlines."""
    return (
        value.replace("\\", "\\\\")
        .replace("=", "\\=")
        .replace(";", "\\;")
        .replace("#", "\\#")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "\\\n")
    )


def _encode_argv(  # noqa: PLR0913, PLR0917 - direct command material.
    ffmpeg: str,
    pcm: Path,
    metadata: Path,
    cover: Path | None,
    candidate: Path,
    sample_rate: int,
    channels: int,
) -> tuple[str, ...]:
    argv = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-n",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-i",
        str(pcm),
        "-f",
        "ffmetadata",
        "-i",
        str(metadata),
    ]
    if cover is not None:
        argv.extend(("-i", str(cover)))
    argv.extend(
        (
            "-map",
            "0:a:0",
            "-map_metadata",
            "1",
            "-map_chapters",
            "1",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
        )
    )
    if cover is not None:
        argv.extend(
            (
                "-map",
                "2:v:0",
                "-c:v",
                "mjpeg",
                "-disposition:v:0",
                "attached_pic",
                "-metadata:s:v:0",
                "title=Cover",
            )
        )
    argv.extend(("-movflags", "+faststart", "-f", "mp4", str(candidate)))
    return tuple(argv)


def _require_regular_artifact(path: Path) -> None:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size <= 0
    ):
        raise EncodingError(ErrorCode.INVALID_ARTIFACT)


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_READ_CHUNK_BYTES):
            size += len(chunk)
            digest.update(chunk)
    if size <= 0:
        raise EncodingError(ErrorCode.INVALID_ARTIFACT)
    return size, digest.hexdigest()
