from __future__ import annotations

import json
import logging
import multiprocessing
import re
import shutil
import subprocess
import time
import warnings
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

import imageio_ffmpeg

from .analytics import StageRecord, append_record, now_utc
from .chapter_classifier import ChapterClassifier  # noqa: F401 – re-exported
from .models import (
    AudioResult,
    Chapter,
    ProcessingConfig,
    _migrate_speaker_voices_keys,
    _normalize_bitrate,
)
from .nlp.models import _SPEAKER_SENTINELS
from .nlp.models import slugify as _slugify
from .progress import ChapterProgress, ProgressEvent, ProgressStage, ProgressStatus, ProgressUnit
from .readers import EbookReader, get_reader
from .utils import extract_epub_cover
from .voice_loader import load_voice
from .workers import worker_process_chapter

# ---------------------------------------------------------------------------
# Pre-flight speaker/voice validation
# ---------------------------------------------------------------------------


def _load_character_genders(roster_cache_path: Path | None) -> dict[str, str]:
    """Load {character_id -> gender_pronoun} from the roster cache, if available."""
    if not roster_cache_path:
        return {}
    try:
        from .models import FastScanResult
        data = json.loads(Path(roster_cache_path).read_text(encoding="utf-8"))
        result = FastScanResult.from_dict(data.get("roster_data") or data)
        return {
            _slugify(c.character_id): c.gender_pronoun
            for c in result.characters
            if c.gender_pronoun
        }
    except Exception:
        return {}


def _load_roster_slugs(roster_cache_path: Path | None) -> set[str]:
    """Load valid character slugs from a roster cache, if available."""
    if not roster_cache_path:
        return set()
    try:
        from .models import FastScanResult
        data = json.loads(Path(roster_cache_path).read_text(encoding="utf-8"))
        result = FastScanResult.from_dict(data.get("roster_data") or data)
        slugs = {c.slug for c in result.roster.characters}
        slugs.update(_slugify(c.character_id) for c in result.characters)
        return {s for s in slugs if s}
    except Exception:
        return set()


def _auto_assign_unmapped_speakers(
    chapters: list[Chapter],
    speaker_voices: dict[str, str],
    narrator_voice: str,
    log: Callable[[str], None],
    config_path: str | None = None,
    character_genders: dict[str, str] | None = None,
    roster_slugs: set[str] | None = None,
) -> dict[str, str]:
    """Assign voices to speakers missing from speaker_voices.

    Assignment is gender-aware, prominence-aware, and chapter-conflict-avoiding:
    - Male speakers are drawn from the male voice pool; female from female; neutral/unknown
      from whichever gender pool has been used less (balanced fill).
    - Speakers are sorted by segment count descending so high-prominence characters
      get first pick of fresh voices; their voices are never shared.
    - When fresh voices run out, low-prominence speakers share a voice with any
      non-co-occurring, same-gender-pool character (different chapters only).
    - Only built-in and compiled voices are used — uncompiled (.wav) voices are
      excluded because we do not hold rights to them.

    ``character_genders`` maps character_id → gender_pronoun string (e.g. "she/her").
    When absent, all speakers are treated as gender-neutral.

    Returns an updated speaker_voices dict.
    """
    from .services.voice_service import gender_from_pronoun, list_voices

    # Collect stats for all speakers in segments.
    speaker_prominence: dict[str, int] = {}
    speaker_chapters: dict[str, set[int]] = {}
    valid_slugs = roster_slugs or set()
    for ch_idx, ch in enumerate(chapters):
        for seg in ch.segments or []:
            if seg.is_scene_break or seg.speaker in _SPEAKER_SENTINELS:
                continue
            if valid_slugs and seg.speaker not in valid_slugs:
                continue
            speaker_prominence[seg.speaker] = speaker_prominence.get(seg.speaker, 0) + 1
            speaker_chapters.setdefault(seg.speaker, set()).add(ch_idx)

    unmapped = [s for s in speaker_prominence if s not in speaker_voices]
    if not unmapped:
        return speaker_voices

    # Sort most-prominent first so they get first pick of fresh exclusive voices.
    unmapped.sort(key=lambda s: speaker_prominence[s], reverse=True)

    # Pool: catalog voices explicitly enabled for assignment.
    all_voices = list_voices(config_path=config_path)
    licensed = [v for v in all_voices if v.pool_enabled and v.status == "available"]

    used_voices = set(speaker_voices.values()) | {narrator_voice}

    all_male = [v.voice_id for v in licensed if v.gender.lower() == "male" and v.voice_id != narrator_voice]
    all_female = [v.voice_id for v in licensed if v.gender.lower() == "female" and v.voice_id != narrator_voice]

    if not all_male and not all_female:
        # Absolute fallback: use all licensed voices gender-agnostically.
        fallback = [v.voice_id for v in licensed if v.voice_id != narrator_voice] or [narrator_voice]
        all_male = all_female = list(fallback)

    male_used_count = female_used_count = 0

    # Seed chapter-usage tracking from existing assignments.
    voice_chapters: dict[str, set[int]] = {}
    for spk, v in speaker_voices.items():
        for ch_idx in speaker_chapters.get(spk, set()):
            voice_chapters.setdefault(v, set()).add(ch_idx)

    exclusive_voices: set[str] = set()
    updated = dict(speaker_voices)
    genders = character_genders or {}

    def _pick_shared(pool: list[str], my_chapters: set[int]) -> str | None:
        """Find first non-exclusive pool voice that doesn't co-occur in my_chapters."""
        for v in pool:
            if v in exclusive_voices:
                continue
            if not (voice_chapters.get(v, set()) & my_chapters):
                return v
        return None

    for speaker in unmapped:
        my_chapters = speaker_chapters.get(speaker, set())
        raw_pronoun = genders.get(speaker, "")
        gender = gender_from_pronoun(raw_pronoun)

        if gender == "male":
            pool = all_male
        elif gender == "female":
            pool = all_female
        else:
            # Neutral: pick the gender pool with fewer assigned speakers so far.
            if male_used_count <= female_used_count:
                pool = all_male
                gender = "male"
            else:
                pool = all_female
                gender = "female"

        if not pool:
            pool = [v.voice_id for v in licensed if v.voice_id != narrator_voice] or [narrator_voice]

        # Try a fresh voice from the gender pool first (exclusive assignment).
        fresh = [v for v in pool if v not in used_voices and v not in exclusive_voices]
        if fresh:
            voice = fresh[0]
            exclusive_voices.add(voice)
        else:
            # Try to share with a non-co-occurring character in the same pool.
            voice = _pick_shared(pool, my_chapters)
            if voice is None:
                # All pool voices conflict: pick the least-overlapping one.
                non_excl = [v for v in pool if v not in exclusive_voices]
                voice = min(
                    non_excl or pool,
                    key=lambda v: len(voice_chapters.get(v, set()) & my_chapters),
                )

        if gender == "male":
            male_used_count += 1
        else:
            female_used_count += 1

        updated[speaker] = voice
        voice_chapters.setdefault(voice, set()).update(my_chapters)
        log(
            f"INFO: auto-assigned voice '{voice}' ({gender}) "
            f"to valid roster speaker '{speaker}'"
        )

    return updated


def _warn_unresolvable_speakers(
    chapters: list[Chapter],
    speaker_voices: dict[str, str],
    log: Callable[[str], None],
) -> None:
    """Emit warnings for speakers that have no voice mapping or a missing safetensors path."""
    seen: set[str] = set()
    for ch in chapters:
        for seg in ch.segments or []:
            if seg.is_scene_break or seg.speaker in seen or seg.speaker in _SPEAKER_SENTINELS:
                continue
            seen.add(seg.speaker)
            voice_name = speaker_voices.get(seg.speaker)
            if voice_name is None:
                log(
                    f"WARNING: speaker '{seg.speaker}' has no voice mapping"
                    " — will use narrator fallback"
                )
            else:
                path = load_voice(voice_name)
                if str(path).endswith(".safetensors") and not Path(path).exists():
                    log(
                        f"WARNING: voice '{voice_name}' for '{seg.speaker}'"
                        f" → '{path}' not found on disk"
                    )


# ---------------------------------------------------------------------------
# Multi-voice cache error
# ---------------------------------------------------------------------------


class AnnotatedChaptersCacheMissError(Exception):
    """Raised when annotated_chapters_path is set but the file does not exist.

    The WorkerServer catches this and marks the job with a CACHE_MISS sentinel
    so the UI can offer the user a recovery choice (re-analyse or fall back to
    single-voice).
    """


# ---------------------------------------------------------------------------
# Annotated chapter loading helper
# ---------------------------------------------------------------------------


def _load_annotated_chapters(
    cache_path: Path,
    included_indices: list[int],
    roster_cache_path: Path | None = None,
    log: Callable[[str], None] | None = None,
) -> list[Chapter]:
    """Load annotated chapters from an NLP cache JSON file.

    Args:
        cache_path:       Path to the NLP cache JSON written by
                          ``nlp.cache_result()``.
        included_indices: Chapter indices that should be included (from the
                          job's ChapterSelection.included list).  An empty
                          list means include all chapters in the cache.

    Returns:
        List of Chapter objects with ``.segments`` populated, filtered to
        ``included_indices`` and sorted by chapter index.

    Raises:
        AnnotatedChaptersCacheMissError: If ``cache_path`` does not exist.
    """
    if not cache_path.exists():
        raise AnnotatedChaptersCacheMissError(
            f"NLP cache file not found: {cache_path}\n"
            "CACHE_MISS: The annotated chapters cache file is missing. "
            "Please re-analyse the book or fall back to single-voice mode."
        )

    data = json.loads(cache_path.read_text(encoding="utf-8"))
    chapters = [Chapter.from_dict(ch) for ch in data.get("chapters", [])]

    roster_slugs = _load_roster_slugs(roster_cache_path)
    warn = log or logger.warning

    for chapter in chapters:
        if chapter.segments:
            for seg in chapter.segments:
                if not seg.is_scene_break and seg.speaker and seg.speaker not in _SPEAKER_SENTINELS:
                    original = seg.speaker
                    seg.speaker = _slugify(seg.speaker)
                    if roster_slugs and seg.speaker not in roster_slugs:
                        warn(
                            f"Annotated cache speaker {original!r} is not in roster; "
                            "remapping to 'Unknown'"
                        )
                        seg.speaker = "Unknown"

    if included_indices:
        idx_set = set(included_indices)
        chapters = [ch for ch in chapters if ch.index in idx_set]

    return sorted(chapters, key=lambda c: c.index)


# Suppress ALL warnings by default (verbose mode will re-enable them)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*characters could not be decoded.*")
warnings.filterwarnings("ignore", message=".*looks like.*")
warnings.filterwarnings("ignore", message=".*surrogate.*")


class LogSink:
    """Small logging-backed compatibility shim for old Rich-style call sites."""

    def emit(self, msg: str = "", style: str = ""):
        _ = style
        msg = str(msg)
        msg = re.sub(r"\[/?[a-zA-Z_ ]+\]", "", msg)
        logger.info("%s", msg)


def _sanitize_for_filename(s: str) -> str:
    """Remove characters that are invalid in filenames across platforms."""
    return re.sub(r'[<>:"/\\|?*]', "", s).strip()


def _make_output_filename(book_title: str, voice: str, is_multi: bool) -> str:
    """Build an output filename that includes the voice label.

    Examples:
        Single-voice: "Pride and Prejudice [alba].m4b"
        Multi-voice:  "Pride and Prejudice [multi-voice].m4b"
        Custom voice: "Pride and Prejudice [my_voice].m4b"
    """
    safe_title = _sanitize_for_filename(book_title)
    if is_multi:
        label = "multi-voice"
    else:
        # Strip hf:// URLs or absolute file paths down to just the stem
        label = Path(voice).stem if ("/" in voice or "\\" in voice) else voice
    return f"{safe_title} [{label}].m4b"


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


class AudioBuilder:
    """Builds audiobooks from ebooks with progress tracking."""

    def __init__(
        self,
        config: ProcessingConfig,
        progress_callback: Callable[[ProgressEvent], None] | None = None,
    ):
        """
        Initialize AudioBuilder.

        Args:
            config: Configuration for audio building
            progress_callback: Optional callback receiving ProgressEvent facts.
        """
        self.cfg = config
        self.progress_callback = progress_callback
        self.temp_dir = Path("temp_audio_build")
        self.console = LogSink()
        self._reader: EbookReader | None = None
        self._total_batches = 0
        self._completed_batches = 0
        self._completed_tts_units = 0
        self._current_chapter = ""
        self._book_hash = ""
        self.pause_check: Callable[[], bool] | None = None
        self.was_paused: bool = False

    def _emit_progress(
        self,
        stage: ProgressStage,
        status: ProgressStatus,
        message: str = "",
        *,
        completed_units: float = 0.0,
        total_units: float = 0.0,
        unit: ProgressUnit = "",
        active_chapters: tuple[ChapterProgress, ...] = (),
    ) -> None:
        """Report structured generation progress facts to the callback."""
        if self.progress_callback:
            self.progress_callback(
                ProgressEvent(
                    stage=stage,
                    status=status,
                    message=message,
                    completed_units=completed_units,
                    total_units=total_units,
                    unit=unit,
                    book_hash=self._book_hash,
                    provider=self.cfg.tts_provider or "kokoro",
                    model=self.cfg.tts_model or "",
                    active_chapters=active_chapters,
                )
            )

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
        self._completed_tts_units = 0
        self._emit_progress(
            "tts_synthesis",
            "started",
            "Starting synthesis",
            completed_units=0,
            total_units=total_chars,
            unit="chars",
        )

        self.cfg.speaker_voices = _migrate_speaker_voices_keys(self.cfg.speaker_voices)
        is_multi = bool(self.cfg.speaker_voices)
        narrator_label = "multi-voice" if is_multi else (
            Path(self.cfg.voice).stem if ("/" in self.cfg.voice or "\\" in self.cfg.voice)
            else self.cfg.voice
        )

        _book_char_count = sum(
            len(p) for ch in chapters for p in ch.paragraphs
        )
        _narration_mode = (
            "multi_voice" if self.cfg.speaker_voices else
            "chapter_voice" if self.cfg.chapter_voices else
            "single"
        )
        _chapter_count = len(chapters)
        from .nlp import book_hash as _nlp_book_hash
        try:
            _book_hash = _nlp_book_hash(self.cfg.ebook_path)
        except Exception:
            _book_hash = ""
        self._book_hash = _book_hash
        try:
            _book_title = self._reader.get_metadata().title or "" if self._reader is not None else ""
        except Exception:
            _book_title = ""

        with self._managed_temp_dir():
            logger.info("Building audiobook: %s", output_file.name)

            # Pre-flight: auto-assign voices to new speakers, then warn about remaining issues
            if any(ch.segments for ch in chapters):
                _roster_cache_path = getattr(self.cfg, "roster_cache_path", None)
                _char_genders = _load_character_genders(_roster_cache_path)
                _roster_slugs = _load_roster_slugs(_roster_cache_path)
                self.cfg.speaker_voices = _auto_assign_unmapped_speakers(
                    chapters, self.cfg.speaker_voices, self.cfg.voice, logger.info,
                    character_genders=_char_genders,
                    roster_slugs=_roster_slugs,
                )
                _warn_unresolvable_speakers(chapters, self.cfg.speaker_voices, logger.warning)

            _tts_start = now_utc()
            t0 = time.monotonic()
            results = self._process_chapters(
                chapters, chapter_batch_info, total_batches, total_chars
            )
            _tts_dur = time.monotonic() - t0
            logger.info("Phase 'processing' completed in %.1fs", _tts_dur)
            append_record(StageRecord(
                stage="tts_synthesis",
                started_at=_tts_start,
                duration_seconds=_tts_dur,
                success=bool(results),
                book_hash=_book_hash,
                book_title=_book_title,
                book_char_count=_book_char_count,
                chapter_count=_chapter_count,
                provider=self.cfg.tts_provider or "kokoro",
                model=self.cfg.tts_model or "",
                narration_mode=_narration_mode,
                chars_per_second=_book_char_count / _tts_dur if _tts_dur > 0 else 0.0,
            ))

            if not results:
                logger.error("No results generated. Aborting.")
                self._emit_progress("tts_synthesis", "failed", "No audio generated")
                return False
            self._emit_progress(
                "tts_synthesis",
                "completed",
                "Synthesis complete",
                completed_units=total_chars,
                total_units=total_chars,
                unit="chars",
            )

            # ── Stitching phase ──────────────────────────────────────────
            # Signal explicitly so the UI doesn't look frozen at 100%.
            stitch_total_ms = sum(r.duration_ms for r in results)
            self._emit_progress(
                "stitching",
                "started",
                "Stitching audio files",
                total_units=stitch_total_ms,
                unit="milliseconds",
            )
            logger.info("Stitching audio files...")
            _stitch_start = now_utc()
            t0 = time.monotonic()
            self._stitch_files(results, output_file, narrator_label=narrator_label)
            _stitch_dur = time.monotonic() - t0
            logger.info("Phase 'stitching' completed in %.1fs", _stitch_dur)
            append_record(StageRecord(
                stage="stitching",
                started_at=_stitch_start,
                duration_seconds=_stitch_dur,
                success=True,
                book_hash=_book_hash,
                book_title=_book_title,
                book_char_count=_book_char_count,
                chapter_count=_chapter_count,
            ))
            self._emit_progress(
                "stitching",
                "completed",
                "Stitching complete",
                completed_units=stitch_total_ms,
                total_units=stitch_total_ms,
                unit="milliseconds",
            )

            # ── Loudness normalization (optional) ────────────────────────
            if self.cfg.post_processing.enabled and self.cfg.post_processing.normalize:
                from .post_processing import normalize_output

                self._emit_progress("normalization", "started", "Normalizing loudness")
                _norm_start = now_utc()
                t0 = time.monotonic()
                normalize_output(output_file, self.cfg.post_processing)
                _norm_dur = time.monotonic() - t0
                logger.info("Phase 'normalization' completed in %.1fs", _norm_dur)
                append_record(StageRecord(
                    stage="normalization",
                    started_at=_norm_start,
                    duration_seconds=_norm_dur,
                    success=True,
                    book_hash=_book_hash,
                    book_title=_book_title,
                    book_char_count=_book_char_count,
                    chapter_count=_chapter_count,
                ))
                self._emit_progress("normalization", "completed", "Normalization complete")

            # ── Cover embedding ──────────────────────────────────────────
            self._emit_progress("cover_embedding", "started", "Embedding cover art")
            _cover_start = now_utc()
            t0 = time.monotonic()
            self._embed_cover(output_file)
            _cover_dur = time.monotonic() - t0
            logger.info("Phase 'cover_embedding' completed in %.1fs", _cover_dur)
            append_record(StageRecord(
                stage="cover_embedding",
                started_at=_cover_start,
                duration_seconds=_cover_dur,
                success=True,
                book_hash=_book_hash,
                book_title=_book_title,
                book_char_count=_book_char_count,
                chapter_count=_chapter_count,
            ))
            self._emit_progress("cover_embedding", "completed", "Cover embedding complete")

            logger.info("Audiobook created: %s", output_file)
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

        completed_chapters = 0
        total_chapters = len(chapters)

        manager = multiprocessing.Manager()
        queue = manager.Queue()  # type: ignore

        def _active_chapter_progress() -> tuple[ChapterProgress, ...]:
            return tuple(
                ChapterProgress(
                    index=int(state.get("index", 0)),
                    title=str(state.get("title", "")),
                    completed_units=float(state.get("current", 0)),
                    total_units=float(state.get("total", 0)),
                    status=str(state.get("status", "advanced")),  # type: ignore[arg-type]
                )
                for state in sorted(worker_state.values(), key=lambda item: int(item.get("index", 0)))
            )

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
            "eos_threshold": self.cfg.eos_threshold,
            "frames_after_eos": self.cfg.frames_after_eos,
            # Multi-voice: character id → voice name mapping
            "speaker_voices": _migrate_speaker_voices_keys(self.cfg.speaker_voices),
            # Chapter-voice mode: str(chapter_index) → voice name
            "chapter_voices": self.cfg.chapter_voices,
            # Audio post-processing effects chain
            "post_processing": self.cfg.post_processing.to_dict(),
            "apostrophe_mode": self.cfg.apostrophe_mode.value,
        }

        pool: ProcessPoolExecutor | None = None
        try:
            # Do NOT use `with ProcessPoolExecutor(...) as pool:` — Python's context
            # manager calls shutdown(wait=True) before re-raising KeyboardInterrupt,
            # blocking until every worker finishes (which can take minutes per chapter).
            # Managing the pool manually lets us terminate running processes immediately.
            pool = ProcessPoolExecutor(max_workers=self.cfg.workers)
            futures = {}
            for idx, ch in enumerate(chapters):
                if self.pause_check is not None and self.pause_check():
                    self.was_paused = True
                    break
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
                                "index": msg[6] if len(msg) > 6 else 0,
                                "status": "started",
                            }
                            self._current_chapter = worker_state[pid].get("title", "")
                            self._emit_progress(
                                "tts_synthesis",
                                "message",
                                self._current_chapter,
                                completed_units=self._completed_tts_units,
                                total_units=total_chars,
                                unit="chars",
                                active_chapters=_active_chapter_progress(),
                            )
                        elif event == "UPDATE":
                            chars = msg[5] if len(msg) > 5 else 0
                            self._completed_batches += msg[2]
                            self._completed_tts_units = min(
                                total_chars,
                                self._completed_tts_units + max(0, chars),
                            )
                            if pid in worker_state:
                                worker_state[pid]["current"] += msg[2]
                                worker_state[pid]["status"] = "advanced"
                                self._current_chapter = worker_state[pid].get("title", "")
                            self._emit_progress(
                                "tts_synthesis",
                                "advanced",
                                self._current_chapter,
                                completed_units=self._completed_tts_units,
                                total_units=total_chars,
                                unit="chars",
                                active_chapters=_active_chapter_progress(),
                            )
                        elif event == "DONE":
                            if pid in worker_state:
                                worker_state[pid]["status"] = "completed"
                                worker_state[pid]["current"] = worker_state[pid].get("total", 0)
                                self._emit_progress(
                                    "tts_synthesis",
                                    "advanced",
                                    worker_state[pid].get("title", ""),
                                    completed_units=self._completed_tts_units,
                                    total_units=total_chars,
                                    unit="chars",
                                    active_chapters=_active_chapter_progress(),
                                )
                                del worker_state[pid]
                                completed_chapters += 1
                        elif event == "ERROR":
                            if pid in worker_state:
                                worker_state[pid]["status"] = "failed"
                            self._emit_progress(
                                "tts_synthesis",
                                "failed",
                                msg[3],
                                completed_units=self._completed_tts_units,
                                total_units=total_chars,
                                unit="chars",
                                active_chapters=_active_chapter_progress(),
                            )
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

                if all(f.done() for f in futures) and not worker_state and queue.empty():
                    break

            for future in as_completed(futures):
                res = future.result()
                if res:
                    results.append(res)

        except KeyboardInterrupt:
            logger.info("Interrupted by caller. Shutting down workers...")
            if pool is not None:
                # Terminate running worker processes immediately (SIGTERM).
                for proc in pool._processes.values():
                    proc.terminate()
                pool.shutdown(wait=False, cancel_futures=True)
            return []
        finally:
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)
            if worker_errors:
                logger.error("Worker errors encountered:")
                for err in worker_errors:
                    logger.error("- PID %s %s: %s", err["pid"], err["chapter"], err["message"])
                    tb = err.get("traceback", "")
                    if tb:
                        logger.error("%s", tb)

        # Suppress unused variable warning — total_chapters used in loop above
        _ = total_chapters

        return sorted(results, key=lambda x: x.chapter_index)

    def _stitch_files(
        self, results: list[AudioResult], output_file: Path, narrator_label: str = ""
    ):
        file_list = self.temp_dir / "files.txt"
        meta_file = self.temp_dir / "metadata.txt"

        with open(file_list, "w", encoding="utf-8") as f:
            for res in results:
                f.write(f"file '{res.file_path.resolve().as_posix()}'\n")

        total_ms = sum(r.duration_ms for r in results)

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
            _normalize_bitrate(self.cfg.m4b_bitrate) if output_file.suffix == ".m4b" else "128k",
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
                        self._emit_progress(
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
                    MP4Cover.FORMAT_PNG if mime_type == "image/png" else MP4Cover.FORMAT_JPEG
                )
                audio = MP4(str(output_file))
                audio["covr"] = [MP4Cover(cover_data, imageformat=image_format)]
                audio.save()
                self.console.emit("Cover embedded successfully")

        except ImportError:
            self.console.emit("Warning: mutagen library not found. Cover not embedded.")
        except Exception as e:
            self.console.emit(f"Warning: Could not embed cover: {e}")

    def run(self) -> bool:
        """Main entry point for audiobook creation."""
        self._reader = get_reader(self.cfg.ebook_path, self.cfg.verbose)
        self._configure_reader(self._reader)

        # ── Chapter loading ───────────────────────────────────────────────
        # Multi-voice jobs reference an NLP cache file.  Load annotated
        # chapters from cache when available; raise AnnotatedChaptersCacheMissError
        # if the cache file has gone missing so the server can surface a
        # recovery dialog in the UI.
        if self.cfg.annotated_chapters_path is not None:
            # This call raises AnnotatedChaptersCacheMissError if file missing.
            included = getattr(self.cfg, "_included_indices", [])
            chapters = _load_annotated_chapters(
                self.cfg.annotated_chapters_path,
                included,
                getattr(self.cfg, "roster_cache_path", None),
            )
            self.console.emit(f"Loaded {len(chapters)} annotated chapters from NLP cache")
        else:
            all_chapters = self._reader.get_chapters()

            if not all_chapters:
                self.console.emit(f"No chapters found in {self._reader.format_name}")
                return False

            from .chapter_filter import ChapterFilter

            filter_chain = ChapterFilter(self.cfg.chapter_filters)
            chapters = filter_chain.apply(all_chapters)

            self.console.emit(
                f"Extracted {len(all_chapters)} chapters, {len(chapters)} after filtering"
            )

        if not chapters:
            self.console.emit("No chapters match the specified filters")
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
                self.cfg.output_path if self.cfg.output_path else self.cfg.ebook_path.parent
            )
            is_multi = bool(self.cfg.speaker_voices)
            output_file = output_dir / _make_output_filename(book_title, self.cfg.voice, is_multi)

        output_file = get_unique_output_path(output_file)

        if hasattr(self._reader, "get_transcript_sections"):
            self._write_pdf_transcripts(output_file, chapters)

        return self.build(chapters, output_file, chapter_batch_info, total_batches, total_chars)

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

    def _configure_reader(self, reader) -> None:
        pdf_options = {
            "drop_code_blocks": bool(getattr(self.cfg, "pdf_drop_code_blocks", False)),
            "drop_notes": bool(getattr(self.cfg, "pdf_drop_notes", False)),
            "drop_asides": bool(getattr(self.cfg, "pdf_drop_asides", False)),
        }
        if any(pdf_options.values()):
            reader.configure_pdf_extraction(pdf_options)

    def _write_pdf_transcripts(self, output_file: Path, filtered_chapters: list[Chapter]) -> None:
        if self._reader is None or not hasattr(self._reader, "get_transcript_sections"):
            return
        sections = self._reader.get_transcript_sections()
        transcript_dir = output_file.parent
        transcript_dir.mkdir(parents=True, exist_ok=True)

        raw_path = transcript_dir / f"{output_file.stem}.transcript.raw.txt"
        filtered_path = transcript_dir / f"{output_file.stem}.transcript.filtered.txt"
        raw_path.write_text(self._format_pdf_raw_transcript(sections), encoding="utf-8")
        filtered_path.write_text(
            self._format_pdf_filtered_transcript(filtered_chapters),
            encoding="utf-8",
        )
        self.console.emit(f"text transcript copied to {transcript_dir}")

    @staticmethod
    def _format_pdf_raw_transcript(sections) -> str:
        lines: list[str] = ["# PDF Raw Transcript", ""]
        for section in sections:
            lines.append(f"## {section.title}")
            lines.append(f"Pages: {section.start_page + 1}-{section.end_page + 1}")
            lines.append("")
            paragraphs = list(getattr(section, "raw_paragraphs", []) or [])
            if not paragraphs:
                lines.append("[no extractable text]")
                lines.append("")
                continue
            for paragraph in paragraphs:
                lines.extend(paragraph.splitlines() or [paragraph])
                lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _format_pdf_filtered_transcript(chapters: list[Chapter]) -> str:
        lines: list[str] = ["# PDF Filtered Transcript", ""]
        for chapter in chapters:
            lines.append(f"## {chapter.title}")
            lines.append("")
            if chapter.segments is None:
                paragraphs = chapter.paragraphs
                for paragraph in paragraphs:
                    lines.extend(paragraph.splitlines() or [paragraph])
                    lines.append("")
                continue
            for segment in chapter.segments:
                if segment.is_scene_break:
                    lines.append("*** SCENE BREAK ***")
                elif segment.speaker and segment.speaker != "NARRATOR":
                    lines.append(f"[{segment.speaker}] {segment.text}")
                else:
                    lines.append(segment.text)
                lines.append("")
        return "\n".join(lines).rstrip() + "\n"


__all__ = ["AudioBuilder"]
