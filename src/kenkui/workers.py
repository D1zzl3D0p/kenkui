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

from .models import AudioResult, Chapter, Segment
from .utils import batch_text
from .voice_loader import load_voice

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Batch-size constants (single source of truth)
# ---------------------------------------------------------------------------

FIRST_CHAPTER_BATCH_SIZE = 250  # Smaller → more frequent ETA updates
DEFAULT_BATCH_SIZE = 800  # Larger → fewer TTS calls, better throughput

# ---------------------------------------------------------------------------
# Per-process model cache
# ---------------------------------------------------------------------------

# Keyed by (temp, lsd_decode_steps, noise_clamp) so different quality settings
# each get their own cached model instance.
_model_cache: dict[tuple, object] = {}


def _get_or_load_model(
    temp: float,
    lsd_decode_steps: int,
    noise_clamp: float | None,
    eos_threshold: float = -4.0,
):
    """Return a cached TTSModel, loading it on first call for this config."""
    from pocket_tts import TTSModel

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
            temp=temp,
            lsd_decode_steps=lsd_decode_steps,
            noise_clamp=noise_clamp,
            eos_threshold=eos_threshold,
        )
        logger.debug("TTSModel loaded and cached for key %s", key)
    return _model_cache[key]


# ---------------------------------------------------------------------------
# Scene-break helpers
# ---------------------------------------------------------------------------

_SCENE_BREAK_RE = re.compile(
    r"^\s*(\*\s*){2,}\s*$"
    r"|^\s*[-\u2014]{2,}\s*$"
    r"|^\s*#\s*$",
)


def _is_scene_break(text: str) -> bool:
    """Return True if *text* is a scene-break marker (or pure whitespace)."""
    stripped = text.strip()
    return not stripped or bool(_SCENE_BREAK_RE.match(stripped))


def _split_at_scene_breaks(paragraphs: list[str]) -> list[list[str]]:
    """Split a paragraph list into sub-groups at scene-break markers.

    Scene-break paragraphs are dropped; remaining paragraphs are grouped
    between break points.  Always returns at least one group.
    """
    groups: list[list[str]] = []
    current: list[str] = []
    for para in paragraphs:
        if _is_scene_break(para):
            if current:
                groups.append(current)
                current = []
        else:
            current.append(para)
    if current:
        groups.append(current)
    return groups or [[]]


def _autogain_segment(seg: "AudioSegment", target_db: float) -> "AudioSegment":
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
    rms = float(np.sqrt(np.mean(samples ** 2)))
    if rms < 1e-8:
        return seg  # silence — skip
    target_linear = 10 ** (target_db / 20.0)
    gain_db = float(np.clip(20 * np.log10(target_linear / rms), -24.0, 24.0))
    return seg.apply_gain(gain_db)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def _pause_for_segment(pause_line_ms: int, text: str) -> "AudioSegment":
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
    if chapter.segments is not None:
        renderable = [s for s in chapter.segments if not s.is_scene_break]
        total_chars = sum(len(s.text) for s in renderable)
        return len(renderable), total_chars
    batch_size = FIRST_CHAPTER_BATCH_SIZE if is_first_chapter else DEFAULT_BATCH_SIZE
    batches = batch_text(chapter.paragraphs, max_chars=batch_size)
    total_chars = sum(len(b) for b in batches)
    return len(batches), total_chars


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
    # Configure logging for this worker process on first chapter call.
    # setup_logging() is idempotent — subsequent calls for the same
    # process_name are no-ops, so this pays no cost after the first chapter.
    try:
        from .log import setup_logging
        setup_logging("workers")
    except Exception:
        pass

    pid = os.getpid()
    max_retries = 2

    def log_message(msg: str):
        if config_dict.get("verbose", False):
            queue.put(("LOG", pid, msg))

    # Suppress all Python-level stdout/stderr in non-verbose mode so that
    # library chatter doesn't bleed through.
    verbose = config_dict.get("verbose", False)
    last_error: Exception | None = None
    result: AudioResult | None = None

    with open(os.devnull, "w") as _devnull:
        _suppress = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(_devnull)
        with _suppress:
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

        # ── Multi-voice path (NLP segments present) ──────────────────────
        if chapter.segments is not None:
            return _render_multi_voice(
                chapter, model, config_dict, temp_dir, queue, pid, log_message
            )

        # ── Per-chapter voice override (chapter-voice mode) ───────────────
        chapter_voice = config_dict.get("chapter_voices", {}).get(str(chapter.index))
        if chapter_voice:
            config_dict = {**config_dict, "voice": chapter_voice}

        # ── Single-voice path ─────────────────────────────────────────────
        voice_path = load_voice(config_dict.get("voice") or "alba")
        log_message(f"[Worker {pid}] Voice: {voice_path}")

        voice_state = model.get_state_for_audio_prompt(voice_path)
        log_message(f"[Worker {pid}] Voice state ready")

        batch_size = FIRST_CHAPTER_BATCH_SIZE if is_first_chapter else DEFAULT_BATCH_SIZE
        sub_groups = _split_at_scene_breaks(chapter.paragraphs)
        # Pre-calculate totals across all sub-groups for progress reporting
        all_batches = [batch_text(g, max_chars=batch_size) for g in sub_groups]
        total_batches = sum(len(b) for b in all_batches)
        total_chars = sum(len(b) for batches in all_batches for b in batches)
        log_message(
            f"[Worker {pid}] {len(chapter.paragraphs)} paragraphs → "
            f"{len(sub_groups)} scene group(s), {total_batches} batches ({total_chars} chars)"
        )

        queue.put(("START", pid, chapter.title, total_batches, total_chars, is_first_chapter))

        pause_line_ms = config_dict.get("pause_line_ms", 400)
        pause_scene_break_ms = config_dict.get("pause_scene_break_ms", 4000)
        pp = config_dict.get("post_processing", {})
        autogain_enabled = bool(pp.get("autogain", True)) if pp.get("enabled", True) else False
        autogain_target_db = float(pp.get("autogain_target_lufs", -23.0))
        full_audio = AudioSegment.empty()

        fae = config_dict.get("frames_after_eos")
        global_batch_idx = 0
        for group_idx, (para_group, batches) in enumerate(zip(sub_groups, all_batches)):
            if group_idx > 0:
                full_audio += AudioSegment.silent(duration=pause_scene_break_ms)
            for batch in batches:
                batch_fae = fae if fae is not None else max(3, len(batch) // 150)
                audio_seg = _render_text(
                    model, voice_state, batch, log_message, pid, global_batch_idx, total_batches,
                    frames_after_eos=batch_fae,
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
) -> AudioResult | None:
    """Render a chapter that has NLP-assigned speaker segments.

    Groups segments by speaker so voice_state is loaded once per speaker,
    then reassembles audio in the original segment order.
    """
    assert chapter.segments is not None
    segments: list[Segment] = chapter.segments

    # Collect unique speakers (exclude SCENE_BREAK — no voice state needed)
    unique_speakers: list[str] = list(dict.fromkeys(
        s.speaker for s in segments if not s.is_scene_break
    ))
    log_message(
        f"[Worker {pid}] Multi-voice: {len(segments)} segments, "
        f"{len(unique_speakers)} speakers: {unique_speakers}"
    )

    # Load voice state for each unique speaker.
    # Per-character voice mappings are injected via config_dict["speaker_voices"].
    speaker_voices: dict[str, str] = config_dict.get("speaker_voices", {})
    speaker_states: dict[str, object] = {}
    for speaker in unique_speakers:
        voice_name: str = str(speaker_voices.get(speaker) or config_dict.get("voice") or "alba")
        voice_path = load_voice(voice_name)
        speaker_states[speaker] = model.get_state_for_audio_prompt(voice_path)
        log_message(f"[Worker {pid}]   {speaker} → {voice_path}")

    total_segments = len(segments)
    queue.put(
        (
            "START",
            pid,
            chapter.title,
            total_segments,
            sum(len(s.text) for s in segments),
            False,
        )
    )

    pause_line_ms = config_dict.get("pause_line_ms", 400)
    pause_scene_break_ms = config_dict.get("pause_scene_break_ms", 4000)
    pp = config_dict.get("post_processing", {})
    autogain_enabled = bool(pp.get("autogain", True)) if pp.get("enabled", True) else False
    autogain_target_db = float(pp.get("autogain_target_lufs", -23.0))
    fae_cfg = config_dict.get("frames_after_eos")

    # Render each segment with its speaker's voice state
    rendered: dict[int, tuple[AudioSegment, str]] = {}
    for seg_idx, seg in enumerate(segments):
        if seg.is_scene_break:
            rendered[seg.index] = (AudioSegment.silent(duration=pause_scene_break_ms), "")
            queue.put(("UPDATE", pid, 1, seg_idx + 1, total_segments, 0))
            continue
        voice_state = speaker_states[seg.speaker]
        seg_fae = fae_cfg if fae_cfg is not None else max(3, len(seg.text) // 150)
        audio_seg = _render_text(
            model, voice_state, seg.text, log_message, pid, seg_idx, total_segments,
            frames_after_eos=seg_fae,
        )
        if audio_seg is not None and autogain_enabled:
            audio_seg = _autogain_segment(audio_seg, autogain_target_db)
        rendered[seg.index] = (
            audio_seg if audio_seg is not None else AudioSegment.empty(),
            seg.text,
        )
        queue.put(("UPDATE", pid, 1, seg_idx + 1, total_segments, len(seg.text)))

    # Reassemble in original index order
    full_audio = AudioSegment.empty()
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
) -> AudioSegment | None:
    """Generate audio for one text batch, retrying once on failure.

    ``frames_after_eos`` controls how many frames are appended after the
    end-of-speech cutoff.  0 (default) suppresses trailing noise artifacts;
    higher values add a brief silence tail.
    """
    for attempt in range(2):
        try:
            log_message(f"  Batch {batch_idx + 1}/{total_batches}: {text[:80]}…")
            tensor = model.generate_audio(
                voice_state,
                text,
                frames_after_eos=frames_after_eos,
            )
            seg = _tensor_to_audio(tensor, model.sample_rate)
            if seg is not None:
                return seg
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
]
