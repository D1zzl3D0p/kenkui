"""Audio assembly components extracted from ``AudioBuilder`` (QP Task 8).

Splits the m4b-assembly responsibilities that were tangled inside
``AudioBuilder`` into three focused, independently testable collaborators:

- ``AudioBatcher``   — computes per-chapter batch info and run totals.
- ``MetadataWriter`` — writes the ffmpeg concat list and FFMETADATA chapter
                       metadata, and embeds cover art into the finished file.
- ``M4BBuilder``     — runs ffmpeg to concatenate segments into the output
                       container, emitting stitching progress.

Behaviour is byte-identical to the previous inline ``AudioBuilder`` methods
(``_stitch_files`` / ``_embed_cover`` and the batch-info loop in ``run``).
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import imageio_ffmpeg

from .models import AudioResult, Chapter, _normalize_bitrate

EmitFn = Callable[..., None]


class AudioBatcher:
    """Computes chapter batch info and run totals for synthesis planning."""

    @staticmethod
    def compute(chapters: list[Chapter]) -> tuple[list[tuple[int, int, bool]], int, int]:
        from .workers import get_batch_info

        chapter_batch_info: list[tuple[int, int, bool]] = []
        for idx, ch in enumerate(chapters):
            is_first = idx == 0
            batch_count, total_chars = get_batch_info(ch, is_first_chapter=is_first)
            chapter_batch_info.append((batch_count, total_chars, is_first))

        total_batches = sum(info[0] for info in chapter_batch_info)
        total_chars = sum(info[1] for info in chapter_batch_info)
        return chapter_batch_info, total_batches, total_chars


class MetadataWriter:
    """Writes ffmpeg concat/metadata sidecar files and embeds cover art."""

    def write_concat_list(self, results: list[AudioResult], file_list: Path) -> None:
        with open(file_list, "w", encoding="utf-8") as f:
            for res in results:
                f.write(f"file '{res.file_path.resolve().as_posix()}'\n")

    def write_chapter_metadata(
        self, results: list[AudioResult], meta_file: Path, narrator_label: str = ""
    ) -> None:
        with open(meta_file, "w", encoding="utf-8") as f:
            f.write(";FFMETADATA1\n")
            if narrator_label:
                f.write(f"comment=Narrated by {narrator_label}\n")
            t = 0
            for res in results:
                start, end = int(t), int(t + res.duration_ms)
                f.write(
                    f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={start}\nEND={end}\ntitle={res.title}\n"
                )
                t += res.duration_ms

    def embed_cover(
        self,
        output_file: Path,
        get_cover: Callable[[], tuple[bytes | None, str]],
        on_message: Callable[[str], None],
    ) -> None:
        """Embed cover image (from ``get_cover``) into the M4B file."""
        try:
            from mutagen.mp4 import MP4, MP4Cover

            cover_data, mime_type = get_cover()

            if cover_data:
                image_format = (
                    MP4Cover.FORMAT_PNG if mime_type == "image/png" else MP4Cover.FORMAT_JPEG
                )
                audio = MP4(str(output_file))
                audio["covr"] = [MP4Cover(cover_data, imageformat=image_format)]
                audio.save()
                on_message("Cover embedded successfully")

        except ImportError:
            on_message("Warning: mutagen library not found. Cover not embedded.")
        except Exception as e:
            on_message(f"Warning: Could not embed cover: {e}")


class M4BBuilder:
    """Concatenates rendered chapter audio into the output container."""

    def stitch(
        self,
        results: list[AudioResult],
        output_file: Path,
        file_list: Path,
        meta_file: Path,
        *,
        bitrate: str,
        emit: EmitFn,
    ) -> None:
        total_ms = sum(r.duration_ms for r in results)

        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-v", "error",
            "-progress", "pipe:1",
            "-f", "concat",
            "-safe", "0",
            "-i", str(file_list),
            "-i", str(meta_file),
            "-map_metadata", "1",
            "-c:a", "aac" if output_file.suffix == ".m4b" else "libmp3lame",
            "-b:a",
            _normalize_bitrate(bitrate) if output_file.suffix == ".m4b" else "128k",
        ]
        if output_file.suffix == ".m4b":
            cmd.extend(["-movflags", "+faststart"])
        cmd.append(str(output_file))

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert proc.stdout is not None

        for line in proc.stdout:
            line = line.strip()
            if line.startswith("out_time_ms="):
                try:
                    out_ms = int(line.split("=", 1)[1])
                    if total_ms > 0 and out_ms > 0:
                        emit(
                            "stitching",
                            "advanced",
                            "Stitching audio files",
                            completed_units=min(out_ms, total_ms),
                            total_units=total_ms,
                            unit="milliseconds",
                        )
                except (ValueError, ZeroDivisionError):
                    pass

        proc.wait()
        if proc.returncode != 0:
            stderr_out = proc.stderr.read() if proc.stderr else ""
            raise subprocess.CalledProcessError(proc.returncode, cmd, stderr=stderr_out)


__all__ = ["AudioBatcher", "M4BBuilder", "MetadataWriter"]
