"""FFmpeg/ffprobe discovery and minimum-capability preflight."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kenkui._audio.native import NativeCommandRunner, SubprocessRunner, run_checked
from kenkui.errors import EncodingError, ErrorCode

if TYPE_CHECKING:
    from collections.abc import Callable

_MINIMUM_MAJOR_VERSION = 5
_PREFLIGHT_TIMEOUT_SECONDS = 10.0
_VERSION = re.compile(r"\b(?:ffmpeg|ffprobe) version (\d+)(?:\.|\s)")


@dataclass(frozen=True, slots=True)
class ResolvedFFmpeg:
    """Absolute executable paths accepted by capability preflight."""

    ffmpeg: str
    ffprobe: str


class FFmpegTools:
    """Discover and preflight required native tools through an injectable runner."""

    def __init__(
        self,
        runner: NativeCommandRunner | None = None,
        *,
        discover: Callable[[str], str | None] = shutil.which,
    ) -> None:
        self.runner = runner or SubprocessRunner()
        self._discover = discover
        self._resolved: ResolvedFFmpeg | None = None
        self._cover_checked = False

    def preflight(self, *, expect_cover: bool = False) -> ResolvedFFmpeg:
        """Require modern tools and precisely the codecs needed by this request."""
        if self._resolved is not None and (not expect_cover or self._cover_checked):
            return self._resolved
        ffmpeg = self._discover("ffmpeg")
        ffprobe = self._discover("ffprobe")
        if ffmpeg is None:
            raise EncodingError(ErrorCode.FFMPEG_NOT_FOUND)
        if ffprobe is None:
            raise EncodingError(ErrorCode.FFPROBE_NOT_FOUND)

        ffmpeg_version = run_checked(
            self.runner,
            (ffmpeg, "-hide_banner", "-version"),
            timeout=_PREFLIGHT_TIMEOUT_SECONDS,
            code=ErrorCode.FFMPEG_UNSUPPORTED,
        )
        encoders = run_checked(
            self.runner,
            (ffmpeg, "-hide_banner", "-encoders"),
            timeout=_PREFLIGHT_TIMEOUT_SECONDS,
            code=ErrorCode.FFMPEG_UNSUPPORTED,
        )
        muxers = run_checked(
            self.runner,
            (ffmpeg, "-hide_banner", "-muxers"),
            timeout=_PREFLIGHT_TIMEOUT_SECONDS,
            code=ErrorCode.FFMPEG_UNSUPPORTED,
        )
        decoders = (
            run_checked(
                self.runner,
                (ffmpeg, "-hide_banner", "-decoders"),
                timeout=_PREFLIGHT_TIMEOUT_SECONDS,
                code=ErrorCode.FFMPEG_UNSUPPORTED,
            )
            if expect_cover
            else None
        )
        ffprobe_version = run_checked(
            self.runner,
            (ffprobe, "-hide_banner", "-version"),
            timeout=_PREFLIGHT_TIMEOUT_SECONDS,
            code=ErrorCode.FFPROBE_UNSUPPORTED,
        )
        has_aac = re.search(r"(?m)^\s*A\S*\s+aac\s", encoders.stdout)
        has_mp4 = re.search(r"(?m)^\s*E\S*\s+mp4\s", muxers.stdout)
        has_mjpeg = re.search(r"(?m)^\s*V\S*\s+mjpeg\s", encoders.stdout)
        has_image_decoders = decoders is not None and all(
            re.search(rf"(?m)^\s*V\S*\s+{codec}\s", decoders.stdout)
            for codec in ("png", "mjpeg")
        )
        if (
            not _supported_version(ffmpeg_version.stdout, "ffmpeg")
            or not has_aac
            or not has_mp4
            or (expect_cover and (not has_mjpeg or not has_image_decoders))
        ):
            raise EncodingError(ErrorCode.FFMPEG_UNSUPPORTED)
        if not _supported_version(ffprobe_version.stdout, "ffprobe"):
            raise EncodingError(ErrorCode.FFPROBE_UNSUPPORTED)
        self._resolved = ResolvedFFmpeg(ffmpeg, ffprobe)
        self._cover_checked = self._cover_checked or expect_cover
        return self._resolved


def _supported_version(output: str, executable: str) -> bool:
    match = _VERSION.search(output)
    return bool(
        match is not None
        and executable in match.group(0)
        and int(match.group(1)) >= _MINIMUM_MAJOR_VERSION
    )
