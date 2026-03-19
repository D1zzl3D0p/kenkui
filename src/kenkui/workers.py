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
- The ``Chapter.segments`` field (populated by BookNLP) activates multi-voice
  mode.  When ``None`` the existing single-voice paragraph path is used
  unchanged.
"""

from __future__ import annotations

import contextlib
import io
import logging
import multiprocessing
import os
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


def _get_or_load_model(temp: float, lsd_decode_steps: int, noise_clamp: float | None):
    """Return a cached TTSModel, loading it on first call for this config."""
    from pocket_tts import TTSModel

    key = (temp, lsd_decode_steps, noise_clamp)
    if key not in _model_cache:
        logger.debug(
            "Loading TTSModel: temp=%s lsd_decode_steps=%s noise_clamp=%s",
            temp,
            lsd_decode_steps,
            noise_clamp,
        )
        _model_cache[key] = TTSModel.load_model(
            temp=temp,
            lsd_decode_steps=lsd_decode_steps,
            noise_clamp=noise_clamp,
        )
        logger.debug("TTSModel loaded and cached for key %s", key)
    return _model_cache[key]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_batch_info(chapter: Chapter, is_first_chapter: bool = False) -> tuple[int, int]:
    """Pre-calculate batch count and total characters for progress estimation.

    Returns:
        ``(batch_count, total_characters)``
    """
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
        )

        # ── Multi-voice path (BookNLP segments present) ───────────────────
        if chapter.segments is not None:
            return _render_multi_voice(
                chapter, model, config_dict, temp_dir, queue, pid, log_message
            )

        # ── Single-voice path ─────────────────────────────────────────────
        voice_path = load_voice(config_dict.get("voice") or "alba")
        log_message(f"[Worker {pid}] Voice: {voice_path}")

        voice_state = model.get_state_for_audio_prompt(voice_path)
        log_message(f"[Worker {pid}] Voice state ready")

        batch_size = FIRST_CHAPTER_BATCH_SIZE if is_first_chapter else DEFAULT_BATCH_SIZE
        batches = batch_text(chapter.paragraphs, max_chars=batch_size)
        total_batches = len(batches)
        total_chars = sum(len(b) for b in batches)
        log_message(
            f"[Worker {pid}] {len(chapter.paragraphs)} paragraphs → "
            f"{total_batches} batches ({total_chars} chars)"
        )

        queue.put(("START", pid, chapter.title, total_batches, total_chars, is_first_chapter))

        silence = AudioSegment.silent(duration=config_dict.get("pause_line_ms", 400))
        full_audio = AudioSegment.empty()

        for batch_idx, batch in enumerate(batches):
            audio_seg = _render_text(
                model, voice_state, batch, log_message, pid, batch_idx, total_batches
            )
            if audio_seg is not None:
                full_audio += audio_seg + silence
            queue.put(("UPDATE", pid, 1, batch_idx + 1, total_batches, len(batch)))

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
    """Render a chapter that has BookNLP-assigned speaker segments.

    Groups segments by speaker so voice_state is loaded once per speaker,
    then reassembles audio in the original segment order.
    """
    assert chapter.segments is not None
    segments: list[Segment] = chapter.segments

    # Collect unique speakers
    unique_speakers: list[str] = list(dict.fromkeys(s.speaker for s in segments))
    log_message(
        f"[Worker {pid}] Multi-voice: {len(segments)} segments, "
        f"{len(unique_speakers)} speakers: {unique_speakers}"
    )

    # Load voice state for each unique speaker.
    # For now all speakers default to the configured voice — BookNLP will
    # inject per-character voice mappings via config_dict["speaker_voices"].
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

    silence = AudioSegment.silent(duration=config_dict.get("pause_line_ms", 400))

    # Render each segment with its speaker's voice state
    rendered: dict[int, AudioSegment] = {}
    for seg_idx, seg in enumerate(segments):
        voice_state = speaker_states[seg.speaker]
        audio_seg = _render_text(
            model, voice_state, seg.text, log_message, pid, seg_idx, total_segments
        )
        rendered[seg.index] = audio_seg if audio_seg is not None else AudioSegment.empty()
        queue.put(("UPDATE", pid, 1, seg_idx + 1, total_segments, len(seg.text)))

    # Reassemble in original index order
    full_audio = AudioSegment.empty()
    for idx in sorted(rendered):
        full_audio += rendered[idx] + silence

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
) -> AudioSegment | None:
    """Generate audio for one text batch, retrying once on failure.

    Passes ``frames_after_eos=0`` to suppress trailing noise artifacts.
    """
    for attempt in range(2):
        try:
            log_message(f"  Batch {batch_idx + 1}/{total_batches}: {text[:80]}…")
            tensor = model.generate_audio(
                voice_state,
                text,
                frames_after_eos=0,
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
    log_message(f"[Worker {pid}] ✓ {chapter.title}: {len(full_audio)}ms saved")
    queue.put(("DONE", pid))
    return AudioResult(chapter.index, chapter.title, filename, len(full_audio))


__all__ = ["get_batch_info", "worker_process_chapter"]
