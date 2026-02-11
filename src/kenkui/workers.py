import importlib
import importlib.resources
import io
import os
import multiprocessing
import sys
import traceback
from pathlib import Path
from urllib.parse import urlparse

from pydub import AudioSegment
import scipy.io.wavfile

from .helpers import Chapter, AudioResult, get_bundled_voices
from .utils import batch_text


def _load_voice(voice: str, verbose: bool = False) -> str:
    """Load a voice, handling built-in names, local paths, and hf:// URLs.

    Returns the voice identifier/path suitable for the TTS model.
    """
    if verbose:
        print(f"[Voice] Loading voice: {voice}")

    # Check if it's an hf:// URL
    if voice.startswith("hf://"):
        try:
            from huggingface_hub import hf_hub_download

            # Parse hf:// URL format: hf://user/repo/filename
            # Convert to: user/repo, filename
            parsed = urlparse(voice)
            path_parts = parsed.path.strip("/").split("/")

            if len(path_parts) < 2:
                raise ValueError(f"Invalid hf:// URL format: {voice}")

            # First two parts are user/repo, rest is the file path
            repo_id = f"{path_parts[0]}/{path_parts[1]}"
            filename = "/".join(path_parts[2:]) if len(path_parts) > 2 else "voice.wav"

            if verbose:
                print(f"[Voice] Downloading from HuggingFace: {repo_id}/{filename}")

            # Download to a temporary directory (will be cleaned up after process ends)
            local_path = hf_hub_download(repo_id=repo_id, filename=filename)

            if verbose:
                print(f"[Voice] Downloaded to: {local_path}")

            return local_path
        except Exception as e:
            print(f"[Voice] Error loading hf:// voice: {e}")
            # Fall back to default voice
            return "alba"

    # Check if it's a local file path
    voice_path = Path(voice)
    if voice_path.exists():
        if verbose:
            print(f"[Voice] Using local file: {voice_path}")
        return str(voice_path)

    # Check if it's a bundled voice (without .wav extension)
    voice_filename = f"{voice}.wav"
    bundled_voices = get_bundled_voices()
    if voice_filename in bundled_voices:
        try:
            pkg_name = "kenkui"
            # Get the path to the voice file in the bundled voices directory
            voice_file = importlib.resources.files(pkg_name) / "voices" / voice_filename
            if verbose:
                print(f"[Voice] Using bundled voice: {voice_file}")
            return str(voice_file)
        except Exception:
            # Fall through to built-in voice if we can't resolve the bundled voice
            pass

    # Otherwise, assume it's a built-in voice name
    if verbose:
        print(f"[Voice] Using built-in voice: {voice}")
    return voice


def get_batch_info(chapter: Chapter, is_first_chapter: bool = False) -> tuple[int, int]:
    """Pre-calculate batch count and total characters for a chapter.

    Uses adaptive batch sizing: smaller batches for first chapter (better ETA),
    larger batches for remaining chapters (better performance).

    Returns:
        tuple of (batch_count, total_characters)
    """
    batch_size = 250 if is_first_chapter else 800
    batches = batch_text(chapter.paragraphs, max_chars=batch_size)
    total_chars = sum(len(batch) for batch in batches)
    return len(batches), total_chars


def worker_process_chapter(
    chapter: Chapter,
    config_dict: dict,
    temp_dir: Path,
    queue: multiprocessing.Queue,
    is_first_chapter: bool = False,
) -> AudioResult | None:
    pid = os.getpid()
    # Only redirect output if not in verbose mode
    if not config_dict.get("verbose", False):
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    def log_message(msg: str):
        """Send log message via queue if verbose mode is enabled"""
        if config_dict.get("verbose", False):
            queue.put(("LOG", pid, msg))

    try:
        from pocket_tts import TTSModel

        log_message(f"[Worker {pid}] Starting chapter: {chapter.title}")
        log_message(
            f"[Worker {pid}] Voice config: {config_dict.get('voice', 'NOT SET')}"
        )

        log_message(f"[Worker {pid}] Loading TTS model...")
        model = TTSModel.load_model()
        log_message(f"[Worker {pid}] Model loaded successfully")

        # Load voice with support for built-in names, local paths, and hf:// URLs
        log_message(f"[Worker {pid}] Loading voice...")
        voice = _load_voice(
            config_dict["voice"], verbose=config_dict.get("verbose", False)
        )
        log_message(f"[Worker {pid}] Voice loaded: {voice}")

        log_message(f"[Worker {pid}] Getting voice state...")
        voice_state = model.get_state_for_audio_prompt(voice)
        log_message(f"[Worker {pid}] Voice state initialized")

        # Adaptive batching: smaller batches for first chapter (better ETA), larger for rest (performance)
        batch_size = 250 if is_first_chapter else 800
        batches = batch_text(chapter.paragraphs, max_chars=batch_size)
        total_batches = len(batches)
        total_chars = sum(len(batch) for batch in batches)
        log_message(
            f"[Worker {pid}] Batched {len(chapter.paragraphs)} paragraphs into "
            f"{total_batches} batches ({total_chars} chars) using {batch_size}-char batches"
        )

        # Report start with batch count and total characters for progress tracking
        queue.put(
            ("START", pid, chapter.title, total_batches, total_chars, is_first_chapter)
        )

        silence = AudioSegment.silent(duration=config_dict["pause_line_ms"])
        full_audio = AudioSegment.empty()

        for batch_idx, batch in enumerate(batches):
            for attempt in range(2):  # Try twice per batch
                try:
                    log_message(
                        f"Processing batch {batch_idx + 1}/{total_batches}: {batch[:100]}..."
                    )

                    audio_tensor = model.generate_audio(
                        voice_state,
                        batch,
                    )

                    # Log initial tensor shape
                    if audio_tensor is not None:
                        log_message(
                            f"[Tensor] Initial shape: {audio_tensor.shape}, dims: {audio_tensor.dim()}"
                        )

                    if audio_tensor is not None and audio_tensor.numel() > 0:
                        # Handle multi-dimensional tensors from custom voices
                        if audio_tensor.dim() > 1:
                            log_message(
                                f"[Tensor] Reshaping tensor from {audio_tensor.shape}..."
                            )

                            if audio_tensor.dim() == 2:
                                # Squeeze 2D tensor [1, N] or [N, 1] to 1D
                                audio_tensor = audio_tensor.squeeze()
                                log_message(
                                    f"[Tensor] Squeezed 2D to: {audio_tensor.shape}"
                                )
                            elif audio_tensor.dim() == 3:
                                # Handle 3D tensor - try to reshape or flatten
                                try:
                                    # Attempt to flatten to 1D
                                    audio_tensor = audio_tensor.view(-1)
                                    log_message(
                                        f"[Tensor] Flattened 3D to: {audio_tensor.shape}"
                                    )
                                except Exception as reshape_err:
                                    log_message(
                                        f"[Tensor] Failed to reshape 3D tensor: {reshape_err}"
                                    )
                                    # Try squeeze then flatten as fallback
                                    audio_tensor = audio_tensor.squeeze().flatten()
                                    log_message(
                                        f"[Tensor] Fallback reshape to: {audio_tensor.shape}"
                                    )

                            # Final validation that we have a 1D tensor
                            if audio_tensor.dim() != 1:
                                log_message(
                                    f"[Tensor] Cannot convert to 1D, final shape: {audio_tensor.shape}"
                                )
                                raise ValueError(
                                    f"Cannot process tensor with shape {audio_tensor.shape}"
                                )

                        # Validate final tensor shape before writing
                        if audio_tensor.dim() == 1 and audio_tensor.numel() > 0:
                            wav_buffer = io.BytesIO()
                            scipy.io.wavfile.write(
                                wav_buffer, model.sample_rate, audio_tensor.numpy()
                            )
                            wav_buffer.seek(0)
                            full_audio += AudioSegment.from_wav(wav_buffer) + silence

                            log_message(
                                f"✓ Generated audio (final shape: {audio_tensor.shape})"
                            )

                            break
                        else:
                            log_message(
                                f"✗ Invalid final tensor shape: {audio_tensor.shape}, numel: {audio_tensor.numel()}"
                            )
                            break
                    else:
                        log_message("✗ Empty or None tensor")
                        break
                except Exception as e:
                    log_message(f"✗ Attempt {attempt + 1} failed: {e}")
                    if attempt == 1:
                        log_message(
                            f"[Tensor] Traceback: {traceback.format_exc()[:500]}"
                        )
                    continue

            # Report progress per batch with batch index info and character count
            queue.put(("UPDATE", pid, 1, batch_idx + 1, total_batches, len(batch)))

        if len(full_audio) < 1000:
            log_message(f"✗ Audio too short ({len(full_audio)}ms), skipping chapter")
            queue.put(("DONE", pid))
            return None

        full_audio += AudioSegment.silent(duration=config_dict["pause_chapter_ms"])
        filename = temp_dir / f"ch_{chapter.index:04d}.wav"
        full_audio.export(str(filename), format="wav")

        log_message(f"✓ Chapter complete: {len(full_audio)}ms audio saved")

        queue.put(("DONE", pid))
        return AudioResult(chapter.index, chapter.title, filename, len(full_audio))

    except KeyboardInterrupt:
        log_message("⚠ Interrupted by user")
        queue.put(
            ("ERROR", pid, chapter.title, "KeyboardInterrupt", "Worker interrupted")
        )
        queue.put(("DONE", pid))
        return None
    except Exception as e:
        error_msg = str(e)
        log_message(f"✗ ERROR: {error_msg}")
        log_message(f"✗ Error type: {type(e).__name__}")
        log_message(f"✗ Voice value: {config_dict.get('voice', 'NOT SET')}")
        log_message(f"✗ Config keys: {list(config_dict.keys())}")
        error_text = traceback.format_exc()
        log_message(f"✗ Traceback: {error_text[:500]}")
        queue.put(("ERROR", pid, chapter.title, error_msg, error_text))
        queue.put(("DONE", pid))
        return None


__all__ = ["get_batch_info", "worker_process_chapter"]
