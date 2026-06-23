"""TTS worker functions executed in subprocess workers via ProcessPoolExecutor.

Design notes:
- Each worker process loads the TTS model once and caches it for the lifetime
  of the process.  With ProcessPoolExecutor the same process handles multiple
  chapters sequentially, so subsequent chapters pay no model-load cost.
- Voice states are cached inside TTSModel via its internal LRU cache, so
  switching between speakers within a multi-voice chapter is cheap after the
  first call.
- ``frames_after_eos=0`` is passed to ``generate_audio`` to suppress the
  trailing noise artifacts that the model sometimes appends after end-of-speech.
- The ``Chapter.segments`` field (populated by the NLP pipeline) activates multi-voice
  mode.  When ``None`` the existing single-voice paragraph path is used
  unchanged.
"""

from __future__ import annotations

import contextlib
import io
import logging
import math
import multiprocessing
import os
import re
import traceback
from pathlib import Path

import scipy.io.wavfile
from pydub import AudioSegment

from .models import AudioResult, Chapter, Segment, _migrate_speaker_voices_keys
from .nlp.models import _SPEAKER_SENTINELS
from .nlp.models import slugify as _slugify
from .text_rules import is_scene_break, split_at_scene_breaks
from .utils import ApostropheMode, batch_text, ensure_terminal_punct, normalize_for_tts
from .voice_loader import load_voice_conditioning_source as load_voice

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Batch-size constants (single source of truth)
# ---------------------------------------------------------------------------

FIRST_CHAPTER_BATCH_SIZE = 250  # Smaller → more frequent ETA updates
DEFAULT_BATCH_SIZE = 800  # Larger → fewer TTS calls, better throughput
UNBOUNDED_TTS_MAX_TOKENS = 100_000


_DISABLED_MALLOC_VALUES = {"", "0", "false", "no", "off", "disable", "disabled"}


def _sanitize_disabled_malloc_debug_env() -> None:
    """Drop inherited macOS malloc debug toggles only when explicitly disabled."""
    for key in ("MallocStackLogging", "MallocStackLoggingNoCompact"):
        value = os.environ.get(key)
        if value is not None and value.strip().lower() in _DISABLED_MALLOC_VALUES:
            os.environ.pop(key, None)


# ---------------------------------------------------------------------------
# Per-process model cache
# ---------------------------------------------------------------------------

# Keyed by (temp, lsd_decode_steps, noise_clamp) so different quality settings
# each get their own cached model instance.
_model_cache: dict[tuple, object] = {}
_logged_tts_max_tokens_pids: set[int] = set()


def _effective_tts_max_tokens(config_dict: dict) -> int:
    try:
        configured = int(config_dict.get("tts_max_tokens_per_chunk", 0) or 0)
    except (TypeError, ValueError):
        configured = 0
    return configured if configured > 0 else UNBOUNDED_TTS_MAX_TOKENS


def _uses_unbounded_tts_chunks(config_dict: dict) -> bool:
    return _effective_tts_max_tokens(config_dict) == UNBOUNDED_TTS_MAX_TOKENS


def _log_tts_max_tokens_once(pid: int, max_tokens: int, log_message) -> None:
    if pid in _logged_tts_max_tokens_pids:
        return
    _logged_tts_max_tokens_pids.add(pid)
    log_message(f"[Worker {pid}] Pocket max_tokens per generation: {max_tokens}")


def _estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text.split()) * 1.3))


def _text_preview(text: str, limit: int = 120) -> str:
    return " ".join(text.split())[:limit]


def _get_or_load_model(
    temp: float,
    lsd_decode_steps: int,
    noise_clamp: float | None,
    eos_threshold: float = -4.0,
):
    """Return a cached TTSModel, loading it on first call for this config."""
    try:
        from pocket_tts import TTSModel
    except ImportError as exc:
        from kenkui.errors import KenkuiDependencyError

        raise KenkuiDependencyError(
            "pocket-tts is required for local TTS execution. Install the local TTS "
            "extra or select tts_execution_mode='modal'."
        ) from exc

    key = (temp, lsd_decode_steps, noise_clamp, eos_threshold)
    if key not in _model_cache:
        logger.debug(
            "Loading TTSModel: temp=%s lsd_decode_steps=%s noise_clamp=%s eos_threshold=%s",
            temp,
            lsd_decode_steps,
            noise_clamp,
            eos_threshold,
        )
        _model_cache[key] = TTSModel.load_model(
            language="english_2026-04",
            temp=temp,
            lsd_decode_steps=lsd_decode_steps,
            noise_clamp=noise_clamp,
            eos_threshold=eos_threshold,
        )
        logger.debug("TTSModel loaded and cached for key %s", key)
    return _model_cache[key]


def _is_scene_break(text: str) -> bool:
    """Return True if *text* is a scene-break marker (or pure whitespace)."""
    return is_scene_break(text)


def _split_at_scene_breaks(paragraphs: list[str]) -> list[list[str]]:
    """Split a paragraph list into sub-groups at scene-break markers.

    Scene-break paragraphs are dropped; remaining paragraphs are grouped
    between break points.  Always returns at least one group.
    """
    return split_at_scene_breaks(paragraphs)


def _autogain_segment(seg: AudioSegment, target_db: float) -> AudioSegment:
    """RMS-normalize *seg* to *target_db* dBFS in-place.

    Uses numpy (transitive dependency) and pydub's ``apply_gain()``.
    Returns *seg* unchanged if it is silence or numpy is not available.
    """
    try:
        import numpy as np
    except ImportError:
        return seg

    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
    max_val = float(2 ** (seg.sample_width * 8 - 1))
    if max_val == 0:
        return seg
    samples /= max_val
    rms = float(np.sqrt(np.mean(samples**2)))
    if rms < 1e-8:
        return seg  # silence — skip
    target_linear = 10 ** (target_db / 20.0)
    gain_db = float(np.clip(20 * np.log10(target_linear / rms), -24.0, 24.0))
    return seg.apply_gain(gain_db)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def _pause_for_segment(pause_line_ms: int, text: str) -> AudioSegment:
    """Return silence whose duration scales logarithmically with paragraph-break count.

    With ``pause_line_ms=1000`` (the base value for a single paragraph boundary):
    - 0 breaks (short dialogue span): 500 ms
    - 1 break:                        1000 ms
    - 2 breaks:                       ~1485 ms  (≈1.5 s)
    - 3 breaks:                       ~1769 ms  (≈1.8 s)
    - N breaks:  pause_line_ms × (1 + 0.7 × ln(N))
    """
    n = text.count("\n\n")
    if n <= 0:
        ms = pause_line_ms // 2
    else:
        ms = int(pause_line_ms * (1 + 0.7 * math.log(n)))
    return AudioSegment.silent(duration=ms)


def get_batch_info(chapter: Chapter, is_first_chapter: bool = False) -> tuple[int, int]:
    """Pre-calculate batch count and total characters for progress estimation.

    For multi-voice chapters (with NLP segments) the batch count is the number
    of segments — each segment produces exactly one TTS call.

    Returns:
        ``(batch_count, total_characters)``
    """
    if chapter.segments:
        batches = [
            part
            for s in chapter.segments
            if not s.is_scene_break
            for part in _split_paragraph_text(s.text)
        ]
        total_chars = sum(len(b) for b in batches)
        return len(batches), total_chars
    batches = [p for p in chapter.paragraphs if p.strip()]
    total_chars = sum(len(b) for b in batches)
    return len(batches), total_chars


def _split_paragraph_text(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]


# ---------------------------------------------------------------------------
# Chapter-title audio helper
# ---------------------------------------------------------------------------


def _render_chapter_title_audio(
    title: str,
    model,
    voice_state,
    log_message,
    pid: int,
    total_batches: int,
    pause_before_ms: int,
    pause_after_ms: int,
    intra_segment_ms: int,
    apostrophe_mode: str,
    max_tokens: int,
) -> AudioSegment:
    """Render a chapter title as audio with configurable silence.

    Titles with colon-delimited segments (e.g. "Chapter 1: Darrow: Castaway")
    are split on ": " and each segment is rendered separately with
    *intra_segment_ms* silence between them, so the listener hears a brief
    pause between the chapter number, POV name, and subtitle.
    """
    audio = AudioSegment.silent(duration=pause_before_ms)
    segments = title.split(": ")
    for i, seg in enumerate(segments):
        seg_audio = _render_text(
            model,
            voice_state,
            seg,
            log_message,
            pid,
            0,
            total_batches,
            frames_after_eos=0,
            apostrophe_mode=apostrophe_mode,
            max_tokens=max_tokens,
            chapter_title=title,
            speaker="chapter_title",
        )
        if seg_audio is not None:
            audio += seg_audio
        if i < len(segments) - 1:
            audio += AudioSegment.silent(duration=intra_segment_ms)
    audio += AudioSegment.silent(duration=pause_after_ms)
    return audio


# ---------------------------------------------------------------------------
# Top-level worker entry point
# ---------------------------------------------------------------------------


def worker_process_chapter(
    chapter: Chapter,
    config_dict: dict,
    temp_dir: Path,
    queue: multiprocessing.Queue,
    is_first_chapter: bool = False,
) -> AudioResult | None:
    """Process a single chapter, retrying up to 2 times on failure.

    Executed inside a subprocess worker via ``ProcessPoolExecutor``.
    """
    _sanitize_disabled_malloc_debug_env()

    # Convert SIGTERM to KeyboardInterrupt so the existing cleanup path runs.
    import signal as _signal

    def _sigterm(signum, frame):
        raise KeyboardInterrupt("SIGTERM")

    _signal.signal(_signal.SIGTERM, _sigterm)

    pid = os.getpid()
    max_retries = 2

    def log_message(msg: str):
        if config_dict.get("verbose", False):
            queue.put(("LOG", pid, msg))

    # Suppress all Python-level stdout/stderr in non-verbose mode so that
    # library chatter doesn't bleed through.
    verbose = config_dict.get("verbose", False)

    # Configure logging before any redirect so file handlers write to the actual
    # log file rather than the /dev/null descriptor captured at StreamHandler
    # construction time.  When neither verbose nor KENKUI_LOG_FILE is set we
    # deliberately set up logging INSIDE the redirect so the StreamHandler binds
    # to /dev/null — this suppresses worker DEBUG noise in silent mode.
    _use_file_log = bool(os.environ.get("KENKUI_LOG_FILE"))
    if verbose or _use_file_log:
        try:
            from .log import setup_logging
            setup_logging("workers")
        except Exception:
            pass

    last_error: Exception | None = None
    result: AudioResult | None = None

    with open(os.devnull, "w") as _devnull:
        if verbose:
            _stdout_suppress = contextlib.nullcontext()
            _stderr_suppress = contextlib.nullcontext()
        else:
            _stdout_suppress = contextlib.redirect_stdout(_devnull)
            _stderr_suppress = contextlib.redirect_stderr(_devnull)
        with _stdout_suppress, _stderr_suppress:
            if not verbose and not _use_file_log:
                try:
                    from .log import setup_logging
                    setup_logging("workers")
                except Exception:
                    pass
            for retry_attempt in range(max_retries + 1):
                try:
                    result = _process_chapter_inner(
                        chapter,
                        config_dict,
                        temp_dir,
                        queue,
                        is_first_chapter,
                        pid,
                        log_message,
                    )
                    if result is not None:
                        return result
                    if retry_attempt < max_retries:
                        log_message(
                            f"[Worker {pid}] Chapter returned None, "
                            f"retrying ({retry_attempt + 1}/{max_retries})…"
                        )
                except Exception as exc:
                    last_error = exc
                    if retry_attempt < max_retries:
                        log_message(
                            f"[Worker {pid}] Exception: {exc}, "
                            f"retrying ({retry_attempt + 1}/{max_retries})…"
                        )

    error_msg = str(last_error) if last_error else "Unknown error after all retries"
    log_message(f"[Worker {pid}] ✗ Failed after {max_retries + 1} attempts: {error_msg}")
    queue.put(("ERROR", pid, chapter.title, error_msg, f"Failed after {max_retries + 1} attempts"))
    queue.put(("DONE", pid))
    return None


# ---------------------------------------------------------------------------
# Inner chapter processor
# ---------------------------------------------------------------------------


def _process_chapter_inner(
    chapter: Chapter,
    config_dict: dict,
    temp_dir: Path,
    queue: multiprocessing.Queue,
    is_first_chapter: bool,
    pid: int,
    log_message,
) -> AudioResult | None:
    """Process one chapter — raises on failure (caller handles retry)."""
    try:
        log_message(f"[Worker {pid}] Chapter: {chapter.title}")

        model = _get_or_load_model(
            temp=config_dict.get("temp", 0.7),
            lsd_decode_steps=config_dict.get("lsd_decode_steps", 1),
            noise_clamp=config_dict.get("noise_clamp"),
            eos_threshold=config_dict.get("eos_threshold", -4.0),
        )

        tts_max_tokens = _effective_tts_max_tokens(config_dict)
        _log_tts_max_tokens_once(pid, tts_max_tokens, log_message)
        apostrophe_mode = ApostropheMode(config_dict.get("apostrophe_mode", "expand_contractions"))

        # ── Multi-voice path (NLP segments present and non-empty) ────────
        if chapter.segments:
            return _render_multi_voice(
                chapter, model, config_dict, temp_dir, queue, pid, log_message,
                apostrophe_mode=apostrophe_mode,
            )

        # ── Per-chapter voice override (chapter-voice mode) ───────────────
        chapter_voice = config_dict.get("chapter_voices", {}).get(str(chapter.index))
        if chapter_voice:
            config_dict = {**config_dict, "voice": chapter_voice}

        # ── Single-voice path ─────────────────────────────────────────────
        voice_name = config_dict.get("voice") or "alba"
        voice_path = load_voice(voice_name)
        log_message(f"[Worker {pid}] Voice: {voice_name} -> {voice_path}")

        voice_state = model.get_state_for_audio_prompt(voice_path)
        log_message(f"[Worker {pid}] Voice state ready")

        batch_size = FIRST_CHAPTER_BATCH_SIZE if is_first_chapter else DEFAULT_BATCH_SIZE
        sub_groups = _split_at_scene_breaks(chapter.paragraphs)
        # Pre-calculate totals across all sub-groups for progress reporting
        if _uses_unbounded_tts_chunks(config_dict):
            all_batches = [[p for p in g if p.strip()] for g in sub_groups]
        else:
            all_batches = [batch_text(g, max_chars=batch_size) for g in sub_groups]
        total_batches = sum(len(b) for b in all_batches)
        total_chars = sum(len(b) for batches in all_batches for b in batches)
        log_message(
            f"[Worker {pid}] {len(chapter.paragraphs)} paragraphs → "
            f"{len(sub_groups)} scene group(s), {total_batches} batches ({total_chars} chars)"
        )

        queue.put(("START", pid, chapter.title, total_batches, total_chars, is_first_chapter, chapter.index))

        pause_line_ms = config_dict.get("pause_line_ms", 400)
        pause_scene_break_ms = config_dict.get("pause_scene_break_ms", 4000)
        speak_chapter_titles = config_dict.get("speak_chapter_titles", True)
        pause_before_title_ms = config_dict.get("pause_before_chapter_title_ms", 2000)
        pause_after_title_ms = config_dict.get("pause_after_chapter_title_ms", 3000)
        pp = config_dict.get("post_processing", {})
        autogain_enabled = bool(pp.get("autogain", True)) if pp.get("enabled", True) else False
        autogain_target_db = float(pp.get("autogain_target_lufs", -23.0))
        full_audio = AudioSegment.empty()

        if speak_chapter_titles and chapter.title:
            full_audio += _render_chapter_title_audio(
                chapter.title,
                model,
                voice_state,
                log_message,
                pid,
                total_batches,
                pause_before_ms=pause_before_title_ms,
                pause_after_ms=pause_after_title_ms,
                intra_segment_ms=config_dict.get("pause_chapter_title_segment_ms", 600),
                apostrophe_mode=apostrophe_mode,
                max_tokens=tts_max_tokens,
            )

        fae = config_dict.get("frames_after_eos")
        global_batch_idx = 0
        for group_idx, (para_group, batches) in enumerate(zip(sub_groups, all_batches)):
            if group_idx > 0:
                full_audio += AudioSegment.silent(duration=pause_scene_break_ms)
            for batch in batches:
                batch_fae = fae if fae is not None else max(3, len(batch) // 150)
                audio_seg = _render_text(
                    model,
                    voice_state,
                    batch,
                    log_message,
                    pid,
                    global_batch_idx,
                    total_batches,
                    frames_after_eos=batch_fae,
                    apostrophe_mode=apostrophe_mode,
                    max_tokens=tts_max_tokens,
                    chapter_title=chapter.title,
                    speaker=voice_name,
                )
                if audio_seg is not None:
                    if autogain_enabled:
                        audio_seg = _autogain_segment(audio_seg, autogain_target_db)
                    full_audio += audio_seg + _pause_for_segment(pause_line_ms, batch)
                global_batch_idx += 1
                queue.put(("UPDATE", pid, 1, global_batch_idx, total_batches, len(batch)))

        return _finalise_chapter(
            chapter, full_audio, config_dict, temp_dir, queue, pid, log_message
        )

    except KeyboardInterrupt:
        log_message(f"[Worker {pid}] Interrupted")
        queue.put(("ERROR", pid, chapter.title, "KeyboardInterrupt", "Worker interrupted"))
        queue.put(("DONE", pid))
        return None
    except Exception as exc:
        error_text = traceback.format_exc()
        log_message(f"[Worker {pid}] ✗ {exc}\n{error_text[:400]}")
        queue.put(("ERROR", pid, chapter.title, str(exc), error_text))
        queue.put(("DONE", pid))
        return None


# ---------------------------------------------------------------------------
# Multi-voice rendering
# ---------------------------------------------------------------------------


def _render_multi_voice(
    chapter: Chapter,
    model,
    config_dict: dict,
    temp_dir: Path,
    queue: multiprocessing.Queue,
    pid: int,
    log_message,
    apostrophe_mode: ApostropheMode | None = None,
) -> AudioResult | None:
    """Render a chapter that has NLP-assigned speaker segments.

    Groups segments by speaker so voice_state is loaded once per speaker,
    then reassembles audio in the original segment order.
    """
    assert chapter.segments is not None
    segments: list[Segment] = chapter.segments
    for seg in segments:
        if not seg.is_scene_break and seg.speaker and seg.speaker not in _SPEAKER_SENTINELS:
            seg.speaker = _slugify(seg.speaker)

    # Collect unique speakers (exclude SCENE_BREAK — no voice state needed)
    unique_speakers: list[str] = list(
        dict.fromkeys(s.speaker for s in segments if not s.is_scene_break)
    )
    log_message(
        f"[Worker {pid}] Multi-voice: {len(segments)} segments, "
        f"{len(unique_speakers)} speakers: {unique_speakers}"
    )

    # Load voice state for each unique speaker.
    # Per-character voice mappings are injected via config_dict["speaker_voices"].
    speaker_voices: dict[str, str] = _migrate_speaker_voices_keys(
        config_dict.get("speaker_voices", {})
    )
    speaker_states: dict[str, object] = {}
    for speaker in unique_speakers:
        voice_name: str = str(speaker_voices.get(speaker) or config_dict.get("voice") or "alba")
        voice_path = load_voice(voice_name)
        try:
            speaker_states[speaker] = model.get_state_for_audio_prompt(voice_path)
            log_message(f"[Worker {pid}]   {speaker} → {voice_path}")
        except Exception as primary_exc:
            log_message(
                f"[Worker {pid}] WARNING: failed to load voice '{voice_name}' for '{speaker}'"
                f" ({primary_exc!r}) — trying fallbacks"
            )
            fallback_voices: list[str] = []
            seen: set[str] = {voice_name}
            for v in [config_dict.get("voice") or "alba", "alba"]:
                if v not in seen:
                    fallback_voices.append(v)
                    seen.add(v)
            loaded = False
            for fallback_name in fallback_voices:
                fallback_path = load_voice(fallback_name)
                try:
                    speaker_states[speaker] = model.get_state_for_audio_prompt(fallback_path)
                    log_message(f"[Worker {pid}]   {speaker} → {fallback_path} (fallback)")
                    loaded = True
                    break
                except Exception:
                    continue
            if not loaded:
                raise RuntimeError(
                    f"All voice fallbacks failed for speaker '{speaker}' "
                    f"(tried: {[voice_name] + fallback_voices})"
                ) from primary_exc

    use_unbounded_chunks = _uses_unbounded_tts_chunks(config_dict)
    total_segments = sum(
        1 if seg.is_scene_break else len(_split_paragraph_text(seg.text) if use_unbounded_chunks else [seg.text])
        for seg in segments
    )
    queue.put(
        (
            "START",
            pid,
            chapter.title,
            total_segments,
            sum(len(s.text) for s in segments),
            False,
            chapter.index,
        )
    )

    pause_line_ms = config_dict.get("pause_line_ms", 400)
    pause_scene_break_ms = config_dict.get("pause_scene_break_ms", 4000)
    speak_chapter_titles = config_dict.get("speak_chapter_titles", True)
    pause_before_title_ms = config_dict.get("pause_before_chapter_title_ms", 2000)
    pause_after_title_ms = config_dict.get("pause_after_chapter_title_ms", 3000)
    pp = config_dict.get("post_processing", {})
    autogain_enabled = bool(pp.get("autogain", True)) if pp.get("enabled", True) else False
    autogain_target_db = float(pp.get("autogain_target_lufs", -23.0))
    fae_cfg = config_dict.get("frames_after_eos")
    tts_max_tokens = _effective_tts_max_tokens(config_dict)
    _log_tts_max_tokens_once(pid, tts_max_tokens, log_message)

    # Build initial audio with chapter title if enabled
    initial_audio = AudioSegment.empty()
    if speak_chapter_titles and chapter.title and speaker_states:
        narrator_state = speaker_states.get("NARRATOR") or next(iter(speaker_states.values()))
        initial_audio += _render_chapter_title_audio(
            chapter.title,
            model,
            narrator_state,
            log_message,
            pid,
            total_segments,
            pause_before_ms=pause_before_title_ms,
            pause_after_ms=pause_after_title_ms,
            intra_segment_ms=config_dict.get("pause_chapter_title_segment_ms", 600),
            apostrophe_mode=apostrophe_mode,
            max_tokens=tts_max_tokens,
        )

    # Render each segment with its speaker's voice state
    rendered: dict[int, tuple[AudioSegment, str]] = {}
    completed_units = 0
    for seg_idx, seg in enumerate(segments):
        if seg.is_scene_break:
            rendered[seg.index] = (AudioSegment.silent(duration=pause_scene_break_ms), "")
            completed_units += 1
            queue.put(("UPDATE", pid, 1, completed_units, total_segments, 0))
            continue
        voice_state = speaker_states[seg.speaker]
        parts = _split_paragraph_text(seg.text) if use_unbounded_chunks else [seg.text]
        combined = AudioSegment.empty()
        for part_idx, part in enumerate(parts):
            seg_fae = fae_cfg if fae_cfg is not None else max(3, len(part) // 150)
            audio_seg = _render_text(
                model,
                voice_state,
                part,
                log_message,
                pid,
                seg_idx,
                total_segments,
                frames_after_eos=seg_fae,
                apostrophe_mode=apostrophe_mode,
                max_tokens=tts_max_tokens,
                chapter_title=chapter.title,
                speaker=seg.speaker,
                segment_index=seg_idx,
            )
            if audio_seg is not None and autogain_enabled:
                audio_seg = _autogain_segment(audio_seg, autogain_target_db)
            if audio_seg is not None:
                combined += audio_seg
                if part_idx < len(parts) - 1:
                    combined += _pause_for_segment(pause_line_ms, part)
            completed_units += 1
            queue.put(("UPDATE", pid, 1, completed_units, total_segments, len(part)))
        rendered[seg.index] = (combined, seg.text)

    # Reassemble in original index order
    full_audio = initial_audio
    for idx in sorted(rendered):
        audio_seg, seg_text = rendered[idx]
        full_audio += audio_seg + _pause_for_segment(pause_line_ms, seg_text)

    return _finalise_chapter(chapter, full_audio, config_dict, temp_dir, queue, pid, log_message)


# ---------------------------------------------------------------------------
# Low-level rendering helpers
# ---------------------------------------------------------------------------


def _tensor_to_audio(tensor, sample_rate: int) -> AudioSegment | None:
    """Convert a TTS output tensor to a pydub AudioSegment."""
    if tensor is None or tensor.numel() == 0:
        return None

    # Normalise dimensionality to 1D
    if tensor.dim() > 1:
        if tensor.dim() == 2:
            tensor = tensor.squeeze()
        else:
            try:
                tensor = tensor.view(-1)
            except Exception:
                tensor = tensor.squeeze().flatten()

    if tensor.dim() != 1 or tensor.numel() == 0:
        return None

    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, sample_rate, tensor.numpy())
    buf.seek(0)
    return AudioSegment.from_wav(buf)


def _render_text(
    model,
    voice_state: dict,
    text: str,
    log_message,
    pid: int,
    batch_idx: int,
    total_batches: int,
    frames_after_eos: int = 0,
    apostrophe_mode: ApostropheMode | None = None,
    max_tokens: int = UNBOUNDED_TTS_MAX_TOKENS,
    chapter_title: str = "",
    speaker: str = "",
    segment_index: int | None = None,
) -> AudioSegment | None:
    """Generate audio for one text batch, retrying once on failure.

    ``frames_after_eos`` controls how many frames are appended after the
    end-of-speech cutoff.  0 (default) suppresses trailing noise artifacts;
    higher values add a brief silence tail.
    """
    for attempt in range(2):
        try:
            # Strip any italic STX/ETX markers that may have survived into this
            # segment (e.g. in single-voice mode that bypasses the NLP pipeline).
            text = text.replace("\x02", "").replace("\x03", "")
            # Collapse paragraph breaks to a single space so the TTS model
            # doesn't treat \n\n as a hard restart mid-segment.  Multi-voice
            # narrator segments are joined with \n\n; single-voice batches
            # never contain \n\n because batch_text uses " ".join().
            text = " ".join(text.split("\n\n"))
            # Expand n't contractions so TTS pronounces them correctly.
            effective_mode = apostrophe_mode if apostrophe_mode is not None else ApostropheMode.EXPAND_CONTRACTIONS
            text = normalize_for_tts(text, mode=effective_mode)
            text = ensure_terminal_punct(text)
            log_message(f"  Batch {batch_idx + 1}/{total_batches}: {text[:80]}…")
            tensor = model.generate_audio(
                voice_state,
                text,
                frames_after_eos=frames_after_eos,
                max_tokens=max_tokens,
            )
            seg = _tensor_to_audio(tensor, model.sample_rate)
            if seg is not None:
                return seg
            token_estimate = _estimate_tokens(text)
            if token_estimate > 50:
                log_message(
                    "  ✗ Empty tensor on attempt "
                    f"{attempt + 1}: chapter={chapter_title!r} speaker={speaker or 'unknown'!r} "
                    f"segment={segment_index if segment_index is not None else batch_idx} "
                    f"tokens~{token_estimate} preview={_text_preview(text)!r}"
                )
            else:
                log_message(f"  ✗ Empty tensor on attempt {attempt + 1}")
        except Exception as exc:
            log_message(f"  ✗ Attempt {attempt + 1} failed: {exc}")
            if attempt == 1:
                log_message(f"  Traceback: {traceback.format_exc()[:300]}")
    return None


def _finalise_chapter(
    chapter: Chapter,
    full_audio: AudioSegment,
    config_dict: dict,
    temp_dir: Path,
    queue: multiprocessing.Queue,
    pid: int,
    log_message,
) -> AudioResult | None:
    """Append chapter silence, write WAV, and return AudioResult."""
    if len(full_audio) < 1000:
        log_message(f"[Worker {pid}] ✗ Audio too short ({len(full_audio)}ms), skipping")
        queue.put(("DONE", pid))
        return None

    full_audio += AudioSegment.silent(duration=config_dict.get("pause_chapter_ms", 2000))
    filename = temp_dir / f"ch_{chapter.index:04d}.wav"
    full_audio.export(str(filename), format="wav")

    # Audio post-processing effects chain (EQ, compression, noise reduction)
    pp_data = config_dict.get("post_processing", {})
    if pp_data and pp_data.get("enabled"):
        from .models import PostProcessingConfig
        from .post_processing import apply_chapter_effects

        apply_chapter_effects(filename, PostProcessingConfig.from_dict(pp_data))

    log_message(f"[Worker {pid}] ✓ {chapter.title}: {len(full_audio)}ms saved")
    queue.put(("DONE", pid))
    return AudioResult(chapter.index, chapter.title, filename, len(full_audio))


__all__ = [
    "get_batch_info",
    "worker_process_chapter",
    "_pause_for_segment",
    "_is_scene_break",
    "_split_at_scene_breaks",
    "_autogain_segment",
    "UNBOUNDED_TTS_MAX_TOKENS",
]
