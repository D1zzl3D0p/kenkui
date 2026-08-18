#!/usr/bin/env python3
"""Reproducible Pocket-TTS 2.1.0 runtime probe (macOS arm64 exercised)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing
import platform
import queue
import resource
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import scipy.io.wavfile
import torch
from pocket_tts import TTSModel

TEXT = "This is a Pocket TTS compatibility probe."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PeakRSS:
    """Sample this process and descendants; values are process RSS sums."""

    def __init__(self) -> None:
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        process = psutil.Process()
        while not self._stop.is_set():
            total = 0
            for item in [process, *process.children(recursive=True)]:
                try:
                    total += item.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self.peak = max(self.peak, total)
            self._stop.wait(0.01)

    def __enter__(self) -> PeakRSS:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()


def tensor_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, dict):
        return {str(key): tensor_tree(item) for key, item in value.items()}
    return type(value).__name__


def timed(call: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    value = call()
    return value, time.perf_counter() - started


def capture_error(call: Any) -> dict[str, str]:
    try:
        call()
    except Exception as error:  # Intentional compatibility observation.
        return {"type": type(error).__name__, "message": str(error)}
    return {"type": "NO_ERROR", "message": ""}


def run_inference(output_wav: Path, voice: str = "alba") -> dict[str, Any]:
    with PeakRSS() as rss:
        model, load_seconds = timed(TTSModel.load_model)
        state, voice_seconds = timed(lambda: model.get_state_for_audio_prompt(voice))
        audio, generation_seconds = timed(lambda: model.generate_audio(state, TEXT))
        second_audio, second_generation_seconds = timed(
            lambda: model.generate_audio(state, TEXT)
        )

        stream = model.generate_audio_stream(state, TEXT)
        first_chunk, first_chunk_seconds = timed(lambda: next(stream))
        stream.close()
        # Closing the generator does not signal Pocket-TTS's daemon generation/decoder
        # threads. Wait briefly and report the observed thread count.
        time.sleep(0.25)
        threads_after_stream_close = threading.active_count()

        empty_text = capture_error(lambda: model.generate_audio(state, ""))
        missing_audio = capture_error(
            lambda: model.get_state_for_audio_prompt(
                Path("definitely-missing-voice.wav")
            )
        )
        bad_language = capture_error(
            lambda: TTSModel.load_model(language="not-a-language")
        )
        conflicting_config = capture_error(
            lambda: TTSModel.load_model(language="english", config="anything.yaml")
        )

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    pcm = audio.detach().cpu().numpy()
    scipy.io.wavfile.write(output_wav, model.sample_rate, pcm)
    return {
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "pocket_tts": importlib.metadata.version("pocket-tts"),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "multiprocessing_start_method": multiprocessing.get_start_method(),
        },
        "model": {
            "device": str(model.device),
            "sample_rate": model.sample_rate,
            "has_voice_cloning": model.has_voice_cloning,
            "origin": str(model.origin),
            "temperature": model.temp,
            "lsd_decode_steps": model.lsd_decode_steps,
            "noise_clamp": model.noise_clamp,
            "eos_threshold": model.eos_threshold,
        },
        "voice": {
            "input": voice,
            "note": (
                "Upstream predefined embedding used transiently; "
                "not a Kenkui-owned fixture."
            ),
            "state": tensor_tree(state),
        },
        "pcm": {
            "shape": list(audio.shape),
            "dtype": str(audio.dtype),
            "device": str(audio.device),
            "samples": audio.numel(),
            "duration_seconds": audio.numel() / model.sample_rate,
            "minimum": float(audio.min()),
            "maximum": float(audio.max()),
            "wav_sha256": sha256(output_wav),
            "repeat_equal": bool(torch.equal(audio, second_audio)),
            "first_stream_chunk_shape": list(first_chunk.shape),
            "first_stream_chunk_dtype": str(first_chunk.dtype),
        },
        "timing_seconds": {
            "load_model": load_seconds,
            "load_voice_state": voice_seconds,
            "generate_first": generation_seconds,
            "generate_second": second_generation_seconds,
            "first_stream_chunk": first_chunk_seconds,
        },
        "memory": {
            "sampled_peak_rss_bytes": rss.peak,
            "ru_maxrss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "cancellation_observation": {
            "boundary": "consumer can close only between yielded ~80 ms chunks",
            "threads_after_generator_close_and_250ms": threads_after_stream_close,
            "api_cancel_token": False,
        },
        "errors": {
            "empty_text": empty_text,
            "missing_local_audio": missing_audio,
            "bad_language": bad_language,
            "language_and_config": conflicting_config,
        },
    }


def child_entry(result_queue: multiprocessing.Queue[Any], output_wav: str) -> None:
    started = time.perf_counter()
    try:
        result_queue.put(
            {
                "ok": True,
                "wall_seconds": time.perf_counter() - started,
                "result": run_inference(Path(output_wav)),
            }
        )
    except BaseException as error:
        result_queue.put(
            {"ok": False, "type": type(error).__name__, "message": str(error)}
        )


def run_spawn(output_wav: Path) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=child_entry, args=(result_queue, str(output_wav)))
    started = time.perf_counter()
    process.start()
    try:
        result = result_queue.get(timeout=300)
    except queue.Empty:
        process.terminate()
        result = {
            "ok": False,
            "type": "TimeoutError",
            "message": "spawn probe exceeded 300 seconds",
        }
    process.join(30)
    return {
        "start_method": "spawn",
        "pid": process.pid,
        "exitcode": process.exitcode,
        "parent_wall_seconds": time.perf_counter() - started,
        "child": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("inference", "spawn"), required=True)
    parser.add_argument("--output-wav", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    result = (
        run_inference(args.output_wav)
        if args.mode == "inference"
        else run_spawn(args.output_wav)
    )
    result["command_wall_seconds"] = time.perf_counter() - started
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
