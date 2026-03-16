from __future__ import annotations

import multiprocessing
import shutil
import subprocess
import time
import warnings
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import imageio_ffmpeg

from .chapter_classifier import ChapterClassifier  # noqa: F401 – re-exported
from .models import AudioResult, Chapter, ProcessingConfig, _normalize_bitrate
from .readers import EbookReader, get_reader
from .utils import extract_epub_cover
from .workers import worker_process_chapter

# Suppress ALL warnings by default (verbose mode will re-enable them)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*characters could not be decoded.*")
warnings.filterwarnings("ignore", message=".*looks like.*")
warnings.filterwarnings("ignore", message=".*surrogate.*")


class SimpleConsole:
    """Simple console output replacement for Rich."""

    def print(self, msg: str = "", style: str = ""):
        msg = str(msg)
        # Strip Rich markup tags
        import re

        msg = re.sub(r"\[/?[a-zA-Z_ ]+\]", "", msg)
        print(msg)


def get_unique_output_path(output_file: Path) -> Path:
    """Generate a unique output path by appending a number if file exists.

    Example: If Book.m4b exists, returns Book_1.m4b, then Book_2.m4b, etc.
    """
    if not output_file.exists():
        return output_file

    import re

    stem = output_file.stem
    suffix = output_file.suffix
    parent = output_file.parent

    match = re.match(r"^(.+)_(\d+)$", stem)
    if match:
        base_stem = match.group(1)
        start_num = int(match.group(2)) + 1
    else:
        base_stem = stem
        start_num = 1

    counter = start_num
    while True:
        new_path = parent / f"{base_stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


@dataclass
class ETATracker:
    """Tracks TTS throughput for accurate ETA calculation."""

    total_chars: int
    start_time: float = field(default_factory=time.monotonic)
    processed_chars: int = 0
    chapter_rates: list[float] = field(default_factory=list)

    def update(self, chars: int) -> None:
        self.processed_chars += chars

    def on_chapter_complete(self, chars: int, elapsed: float) -> None:
        if elapsed > 0:
            self.chapter_rates.append(chars / elapsed)

    @property
    def current_rate(self) -> float:
        """Calculate chars/second with chapter-based refinement."""
        elapsed = time.monotonic() - self.start_time
        if elapsed < 1 or self.processed_chars == 0:
            return 0.0

        rate = self.processed_chars / elapsed

        if self.chapter_rates:
            avg_chapter_rate = sum(self.chapter_rates) / len(self.chapter_rates)
            rate = 0.6 * rate + 0.4 * avg_chapter_rate

        return rate

    def format_eta(self) -> str:
        rate = self.current_rate
        if rate <= 0:
            return "--:--:--"

        remaining = (self.total_chars - self.processed_chars) / rate
        hours = int(remaining // 3600)
        minutes = int((remaining % 3600) // 60)
        seconds = int(remaining % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def format_elapsed(self) -> str:
        elapsed = time.monotonic() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def format_rate(self) -> str:
        rate = self.current_rate
        if rate <= 0:
            return "0.0"
        return f"{rate:,.1f}"


class AudioBuilder:
    """Builds audiobooks from ebooks with progress tracking."""

    def __init__(
        self,
        config: ProcessingConfig,
        progress_callback: Callable[[float, str, int], None] | None = None,
    ):
        """
        Initialize AudioBuilder.

        Args:
            config: Configuration for audio building
            progress_callback: Optional callback(percent_complete, current_chapter, eta_seconds)
        """
        self.cfg = config
        self.progress_callback = progress_callback
        self.temp_dir = Path("temp_audio_build")
        self.console = SimpleConsole()
        self._reader: EbookReader | None = None
        self._total_batches = 0
        self._completed_batches = 0
        self._current_chapter = ""

    def _report_progress(self, chapter: str = "", eta: int = 0):
        """Report progress to callback if configured."""
        if self.progress_callback and self._total_batches > 0:
            percent = (self._completed_batches / self._total_batches) * 100
            self.progress_callback(percent, chapter, eta)

    def _signal_phase(self, message: str):
        """Send a named post-TTS phase status (stitching, cover, etc.) at 100%."""
        if self.progress_callback:
            self.progress_callback(100.0, message, 0)

    def _calculate_eta(self, eta_tracker: ETATracker) -> int:
        """Calculate ETA in seconds from eta_tracker."""
        try:
            eta_str = eta_tracker.format_eta()
            parts = eta_str.split(":")
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            elif len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
        except Exception:
            pass
        return 0

    def build(
        self,
        chapters: list[Chapter],
        output_file: Path,
        chapter_batch_info: dict[str, tuple[int, int]],
        total_batches: int,
        total_chars: int,
    ) -> bool:
        output_file.parent.mkdir(parents=True, exist_ok=True)

        self._total_batches = total_batches
        self._completed_batches = 0
        self._report_progress("Starting...", 0)

        with self._managed_temp_dir():
            print(f"Building audiobook: {output_file.name}")
            results = self._process_chapters(
                chapters, chapter_batch_info, total_batches, total_chars
            )
            if not results:
                print("No results generated. Aborting.")
                return False

            # ── Stitching phase ──────────────────────────────────────────
            # Signal explicitly so the UI doesn't look frozen at 100%.
            self._signal_phase("Stitching audio files…")
            print("Stitching audio files...")
            self._stitch_files(results, output_file)

            # ── Cover embedding ──────────────────────────────────────────
            self._signal_phase("Embedding cover art…")
            self._embed_cover(output_file)

            print(f"Audiobook created: {output_file}")
            return True

    def _process_chapters(
        self,
        chapters: list[Chapter],
        chapter_batch_info: dict[str, tuple[int, int]],
        total_batches: int,
        total_chars: int,
    ) -> list[AudioResult]:
        results = []
        worker_state: dict = {}
        worker_errors: list[dict] = []
        worker_logs: list[str] = []

        eta_tracker = ETATracker(total_chars)
        chapter_start_times: dict[int, float] = {}
        completed_chapters = 0
        total_chapters = len(chapters)

        manager = multiprocessing.Manager()
        queue = manager.Queue()  # type: ignore

        cfg_dict: dict = {
            "voice": self.cfg.voice,
            "pause_line_ms": self.cfg.pause_line_ms,
            "pause_chapter_ms": self.cfg.pause_chapter_ms,
            "tts_model": self.cfg.tts_model,
            "tts_provider": self.cfg.tts_provider,
            "model_name": self.cfg.model_name,
            "elevenlabs_key": self.cfg.elevenlabs_key,
            "elevenlabs_turbo": self.cfg.elevenlabs_turbo,
            "debug_html": self.cfg.debug_html,
            "verbose": self.cfg.verbose,
            # TTS quality parameters — passed through to TTSModel.load_model()
            "temp": self.cfg.temp,
            "lsd_decode_steps": self.cfg.lsd_decode_steps,
            "noise_clamp": self.cfg.noise_clamp,
        }

        try:
            with ProcessPoolExecutor(max_workers=self.cfg.workers) as pool:
                futures = {}
                for idx, ch in enumerate(chapters):
                    info = chapter_batch_info.get(ch.title, (0, 0, idx == 0))
                    is_first = bool(info[2]) if len(info) > 2 else (idx == 0)
                    fut = pool.submit(
                        worker_process_chapter,
                        ch,
                        cfg_dict,
                        self.temp_dir,
                        queue,  # type: ignore
                        is_first,
                    )
                    futures[fut] = ch

                while True:
                    while not queue.empty():
                        try:
                            msg = queue.get_nowait()
                            event, pid = msg[0], msg[1]
                            if event == "START":
                                worker_state[pid] = {
                                    "title": msg[2],
                                    "total": msg[3],
                                    "current": 0,
                                    "total_chars": msg[4] if len(msg) > 4 else 0,
                                    "is_first": msg[5] if len(msg) > 5 else False,
                                }
                                chapter_start_times[pid] = time.monotonic()
                            elif event == "UPDATE":
                                chars = msg[5] if len(msg) > 5 else 1000
                                self._completed_batches += msg[2]
                                eta_tracker.update(chars)
                                if pid in worker_state:
                                    worker_state[pid]["current"] += msg[2]
                                    self._current_chapter = worker_state[pid].get(
                                        "title", ""
                                    )
                                eta_seconds = self._calculate_eta(eta_tracker)
                                self._report_progress(
                                    self._current_chapter, eta_seconds
                                )
                            elif event == "DONE":
                                if pid in worker_state:
                                    if pid in chapter_start_times:
                                        elapsed = (
                                            time.monotonic() - chapter_start_times[pid]
                                        )
                                        chars = worker_state[pid].get("total_chars", 0)
                                        eta_tracker.on_chapter_complete(chars, elapsed)
                                        del chapter_start_times[pid]
                                    del worker_state[pid]
                                    completed_chapters += 1
                            elif event == "ERROR":
                                worker_errors.append(
                                    {
                                        "pid": pid,
                                        "chapter": msg[2],
                                        "message": msg[3],
                                        "traceback": msg[4],
                                    }
                                )
                            elif event == "LOG":
                                worker_logs.append(f"[{pid}] {msg[2]}")
                                if len(worker_logs) > 20:
                                    worker_logs.pop(0)
                        except Exception:
                            break

                    if (
                        all(f.done() for f in futures)
                        and not worker_state
                        and queue.empty()
                    ):
                        break

                for future in as_completed(futures):
                    res = future.result()
                    if res:
                        results.append(res)

        except KeyboardInterrupt:
            print("Interrupted by user. Shutting down workers...")
            if "pool" in locals():
                pool.shutdown(wait=False, cancel_futures=True)
            return []
        finally:
            if worker_errors:
                print("Worker errors encountered:")
                for err in worker_errors:
                    print(f"- PID {err['pid']} {err['chapter']}: {err['message']}")
                    if self.cfg.debug_html:
                        print(err["traceback"])

        # Suppress unused variable warning — total_chapters used in loop above
        _ = total_chapters

        return sorted(results, key=lambda x: x.chapter_index)

    def _stitch_files(self, results: list[AudioResult], output_file: Path):
        file_list = self.temp_dir / "files.txt"
        meta_file = self.temp_dir / "metadata.txt"

        with open(file_list, "w", encoding="utf-8") as f:
            for res in results:
                f.write(f"file '{res.file_path.resolve().as_posix()}'\n")

        with open(meta_file, "w", encoding="utf-8") as f:
            f.write(";FFMETADATA1\n")
            t = 0
            for res in results:
                start, end = int(t), int(t + res.duration_ms)
                f.write(
                    f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={start}\nEND={end}\ntitle={res.title}\n"
                )
                t += res.duration_ms

        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-v",
            "error",
            "-stats",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(file_list),
            "-i",
            str(meta_file),
            "-map_metadata",
            "1",
            "-c:a",
            "aac" if output_file.suffix == ".m4b" else "libmp3lame",
            "-b:a",
            _normalize_bitrate(self.cfg.m4b_bitrate)
            if output_file.suffix == ".m4b"
            else "128k",
        ]
        if output_file.suffix == ".m4b":
            cmd.extend(["-movflags", "+faststart"])
        cmd.append(str(output_file))
        subprocess.run(cmd, check=True)

    def _embed_cover(self, output_file: Path) -> None:
        """Embed cover image from ebook into the M4B file."""
        try:
            from mutagen.mp4 import MP4, MP4Cover

            if self._reader is not None:
                cover_data, mime_type = self._reader.get_cover()
            else:
                cover_data, mime_type = extract_epub_cover(self.cfg.ebook_path)

            if cover_data:
                image_format = (
                    MP4Cover.FORMAT_PNG
                    if mime_type == "image/png"
                    else MP4Cover.FORMAT_JPEG
                )
                audio = MP4(str(output_file))
                audio["covr"] = [MP4Cover(cover_data, imageformat=image_format)]
                audio.save()
                self.console.print("Cover embedded successfully")

        except ImportError:
            self.console.print(
                "Warning: mutagen library not found. Cover not embedded."
            )
        except Exception as e:
            self.console.print(f"Warning: Could not embed cover: {e}")

    def run(self) -> bool:
        """Main entry point for audiobook creation."""
        self._reader = get_reader(self.cfg.ebook_path, self.cfg.verbose)

        all_chapters = self._reader.get_chapters()

        if not all_chapters:
            self.console.print(f"No chapters found in {self._reader.format_name}")
            return False

        from .chapter_filter import ChapterFilter

        filter_chain = ChapterFilter(self.cfg.chapter_filters)
        chapters = filter_chain.apply(all_chapters)

        self.console.print(
            f"Extracted {len(all_chapters)} chapters, {len(chapters)} after filtering"
        )

        if not chapters:
            self.console.print("No chapters match the specified filters")
            return False

        from .workers import get_batch_info

        chapter_batch_info = {}
        for idx, ch in enumerate(chapters):
            is_first = idx == 0
            batch_count, total_chars = get_batch_info(ch, is_first_chapter=is_first)
            chapter_batch_info[ch.title] = (batch_count, total_chars, is_first)

        total_batches = sum(info[0] for info in chapter_batch_info.values())
        total_chars = sum(info[1] for info in chapter_batch_info.values())

        if self.cfg.output_path and self.cfg.output_path.suffix:
            output_file = self.cfg.output_path
        else:
            metadata = self._reader.get_metadata()
            book_title = metadata.title
            output_dir = (
                self.cfg.output_path
                if self.cfg.output_path
                else self.cfg.ebook_path.parent
            )
            output_file = output_dir / f"{book_title}.m4b"

        output_file = get_unique_output_path(output_file)

        return self.build(
            chapters, output_file, chapter_batch_info, total_batches, total_chars
        )

    @contextmanager
    def _managed_temp_dir(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(parents=True)
        try:
            yield
        finally:
            if not self.cfg.keep_temp and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)


__all__ = ["ETATracker", "AudioBuilder"]
