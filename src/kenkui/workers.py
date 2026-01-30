import io
import os
import multiprocessing
import sys
from pathlib import Path
from typing import Optional

from pydub import AudioSegment
import scipy.io.wavfile

from .helpers import Chapter, AudioResult


def worker_process_chapter(
    chapter: Chapter, config_dict: dict, temp_dir: Path, queue: multiprocessing.Queue
) -> Optional[AudioResult]:
    pid = os.getpid()
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

    try:
        from pocket_tts import TTSModel

        queue.put(("START", pid, chapter.title, len(chapter.paragraphs)))

        model = TTSModel.load_model()
        voice_state = model.get_state_for_audio_prompt(config_dict["voice"])

        silence = AudioSegment.silent(duration=config_dict["pause_line_ms"])
        full_audio = AudioSegment.empty()

        for paragraph in chapter.paragraphs:
            audio_tensor = model.generate_audio(voice_state, paragraph)
            if audio_tensor is not None:
                wav_buffer = io.BytesIO()
                scipy.io.wavfile.write(
                    wav_buffer, model.sample_rate, audio_tensor.numpy()
                )
                wav_buffer.seek(0)
                full_audio += AudioSegment.from_wav(wav_buffer) + silence

            queue.put(("UPDATE", pid, 1))

        if len(full_audio) < 1000:
            queue.put(("DONE", pid))
            return None

        full_audio += AudioSegment.silent(duration=config_dict["pause_chapter_ms"])
        filename = temp_dir / f"ch_{chapter.index:04d}.wav"
        full_audio.export(str(filename), format="wav")

        queue.put(("DONE", pid))
        return AudioResult(chapter.index, chapter.title, filename, len(full_audio))

    except Exception:
        queue.put(("DONE", pid))
        return None
