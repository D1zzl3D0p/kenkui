"""Semantic ffprobe JSON and full-decode artifact validation."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, cast

from kenkui._audio.m4b import chapter_frame_boundaries_ms
from kenkui._audio.native import NativeCommandRunner, run_checked
from kenkui.errors import EncodingError, ErrorCode

if TYPE_CHECKING:
    from pathlib import Path

    from kenkui._audio.ffmpeg import ResolvedFFmpeg
    from kenkui._domain.planning import ExecutionPlan
    from kenkui._tts.protocols import SynthesizedAudio

_PROBE_TIMEOUT_SECONDS = 30.0
_DECODE_TIMEOUT_SECONDS = 300.0
_TIMESTAMP_TOLERANCE_SECONDS = 0.075


def validate_artifact(  # noqa: PLR0913 - explicit validation inputs.
    path: Path,
    plan: ExecutionPlan,
    audio: tuple[SynthesizedAudio, ...],
    tools: ResolvedFFmpeg,
    runner: NativeCommandRunner,
    *,
    expect_cover: bool,
) -> None:
    """Require MP4/AAC semantics, exact chapters, and a complete decode."""
    probe = run_checked(
        runner,
        (
            tools.ffprobe,
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-show_chapters",
            "-of",
            "json",
            str(path),
        ),
        timeout=_PROBE_TIMEOUT_SECONDS,
        code=ErrorCode.PROBE_FAILED,
    )
    payload: object = None
    invalid_json = False
    try:
        payload = json.loads(probe.stdout)
    except (json.JSONDecodeError, TypeError):
        invalid_json = True
    if invalid_json:
        _invalid()
    if not _valid_probe(payload, plan, audio, expect_cover=expect_cover):
        _invalid()
    run_checked(
        runner,
        (
            tools.ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0",
            "-f",
            "null",
            "-",
        ),
        timeout=_DECODE_TIMEOUT_SECONDS,
        code=ErrorCode.DECODE_FAILED,
    )


def _valid_probe(  # noqa: C901, PLR0911 - rejection predicates stay explicit.
    payload: object,
    plan: ExecutionPlan,
    audio: tuple[SynthesizedAudio, ...],
    *,
    expect_cover: bool,
) -> bool:
    if type(payload) is not dict:
        return False
    root = cast("dict[str, object]", payload)
    format_data = root.get("format")
    streams = root.get("streams")
    chapters = root.get("chapters")
    if (
        type(format_data) is not dict
        or type(streams) is not list
        or type(chapters) is not list
    ):
        return False
    container = cast("dict[str, object]", format_data)
    format_name = container.get("format_name")
    boundaries = chapter_frame_boundaries_ms(plan, audio)
    expected_duration = boundaries[-1] / 1000
    if (
        type(format_name) is not str
        or not ({"mov", "mp4", "m4a"} & set(format_name.split(",")))
        or not _close(_number(container.get("duration")), expected_duration)
    ):
        return False

    typed_streams = [item for item in streams if type(item) is dict]
    audio_streams = [
        cast("dict[str, object]", item)
        for item in typed_streams
        if item.get("codec_type") == "audio"
    ]
    video_streams = [
        cast("dict[str, object]", item)
        for item in typed_streams
        if item.get("codec_type") == "video"
    ]
    if len(audio_streams) != 1 or len(typed_streams) != len(streams):
        return False
    stream = audio_streams[0]
    if (
        stream.get("codec_name") != "aac"
        or _integer(stream.get("sample_rate")) != audio[0].sample_rate_hz
        or _integer(stream.get("channels")) != audio[0].channels
    ):
        return False
    attached = [
        stream
        for stream in video_streams
        if type(stream.get("disposition")) is dict
        and cast("dict[str, object]", stream["disposition"]).get("attached_pic") == 1
    ]
    if (expect_cover and len(attached) != 1) or (not expect_cover and video_streams):
        return False

    if len(chapters) != len(plan.output.chapters):
        return False
    for raw, expected, start_ms, end_ms in zip(
        chapters, plan.output.chapters, boundaries[:-1], boundaries[1:], strict=True
    ):
        if type(raw) is not dict:
            return False
        chapter = cast("dict[str, object]", raw)
        start = start_ms / 1000
        end = end_ms / 1000
        tags = chapter.get("tags")
        if (
            type(tags) is not dict
            or cast("dict[str, object]", tags).get("title") != expected.title
            or not _close(_number(chapter.get("start_time")), start)
            or not _close(_number(chapter.get("end_time")), end)
        ):
            return False
    return True


def _number(value: object) -> float:
    try:
        result = float(cast("str | int | float", value))
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _integer(value: object) -> int | None:
    try:
        result = int(cast("str | int", value))
    except (TypeError, ValueError):
        return None
    return result


def _close(actual: float, expected: float) -> bool:
    return (
        math.isfinite(actual) and abs(actual - expected) <= _TIMESTAMP_TOLERANCE_SECONDS
    )


def _invalid() -> None:
    raise EncodingError(ErrorCode.INVALID_ARTIFACT) from None
