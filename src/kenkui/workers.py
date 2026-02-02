import io
import os
import multiprocessing
import sys
import traceback
from pathlib import Path
from typing import Optional

from pydub import AudioSegment
import scipy.io.wavfile

from .helpers import Chapter, AudioResult


def worker_process_chapter(
    chapter: Chapter, config_dict: dict, temp_dir: Path, queue: multiprocessing.Queue
) -> Optional[AudioResult]:
    pid = os.getpid()
    # Always redirect output - logs will be sent via queue in verbose mode
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

    def log_message(msg: str):
        """Send log message via queue if verbose mode is enabled"""
        if config_dict.get("verbose", False):
            queue.put(("LOG", pid, msg))

    try:
        from pocket_tts import TTSModel

        log_message(f"Starting chapter: {chapter.title}")
        queue.put(("START", pid, chapter.title, len(chapter.paragraphs)))

        model = TTSModel.load_model()
        voice_state = model.get_state_for_audio_prompt(config_dict["voice"])

        silence = AudioSegment.silent(duration=config_dict["pause_line_ms"])
        full_audio = AudioSegment.empty()

        for paragraph in chapter.paragraphs:
            success = False
            for attempt in range(2):  # Try twice per paragraph
                try:
                    log_message(f"Processing: {paragraph[:100]}...")

                    audio_tensor = model.generate_audio(voice_state, paragraph)
                    if audio_tensor is not None and audio_tensor.numel() > 0:
                        # Validate tensor shape before processing
                        if audio_tensor.dim() > 0 and all(
                            dim > 0 for dim in audio_tensor.shape
                        ):
                            wav_buffer = io.BytesIO()
                            scipy.io.wavfile.write(
                                wav_buffer, model.sample_rate, audio_tensor.numpy()
                            )
                            wav_buffer.seek(0)
                            full_audio += AudioSegment.from_wav(wav_buffer) + silence

                            log_message(
                                f"✓ Generated audio (shape: {audio_tensor.shape})"
                            )

                            success = True
                            break
                        else:
                            # Skip invalid tensor shapes
                            log_message(f"✗ Invalid tensor shape: {audio_tensor.shape}")
                            break
                except Exception as e:
                    log_message(f"✗ Attempt {attempt + 1} failed: {e}")
                    if attempt == 1:  # Only skip on second failure
                        pass
                    continue

            queue.put(("UPDATE", pid, 1))

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
        log_message(f"✗ ERROR: {e}")
        error_text = traceback.format_exc()
        queue.put(("ERROR", pid, chapter.title, str(e), error_text))
        queue.put(("DONE", pid))
        return None
    except Exception as e:
        if config_dict.get("verbose", False):
            print(f"[Worker {pid}] ✗ ERROR: {e}")
        error_text = traceback.format_exc()
        queue.put(("ERROR", pid, chapter.title, str(e), error_text))
        queue.put(("DONE", pid))
        return None
    except Exception as e:
        error_text = traceback.format_exc()
        queue.put(("ERROR", pid, chapter.title, str(e), error_text))
        queue.put(("DONE", pid))
        return None
