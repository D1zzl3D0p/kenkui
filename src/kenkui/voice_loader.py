"""Voice loading utilities shared across TTS workers and future multi-voice modules.

Supports three voice source types:
- Built-in pocket-tts voice names (e.g. ``"alba"``)
- Bundled .wav files shipped with the package (e.g. ``"RafeBeckley"``)
- Local file paths (e.g. ``"/home/user/my_voice.wav"``)
- HuggingFace URLs  (e.g. ``"hf://user/repo/voice.wav"``)
"""

from __future__ import annotations

import importlib.resources
import logging
from pathlib import Path
from urllib.parse import urlparse

from .helpers import get_bundled_voices

logger = logging.getLogger(__name__)


def load_voice(voice: str) -> str:
    """Resolve a voice identifier to a path or built-in name for TTSModel.

    Args:
        voice: One of:
            - A pocket-tts built-in name (``"alba"``, ``"cosette"``, …)
            - A bundled voice name (with or without ``.wav``)
            - An absolute or relative file path that exists on disk
            - An ``hf://user/repo/filename`` URL

    Returns:
        A string suitable for passing to ``TTSModel.get_state_for_audio_prompt``:
        either a local filesystem path or a built-in name unchanged.

    On any error the function falls back to ``"alba"`` and logs a warning.
    """
    logger.debug("Loading voice: %s", voice)

    # ── HuggingFace URL ────────────────────────────────────────────────────
    if voice.startswith("hf://"):
        try:
            from huggingface_hub import hf_hub_download

            parsed = urlparse(voice)
            # urlparse treats the first component as netloc (host):
            # hf://user/repo/path/file.wav → netloc=user, path=/repo/path/file.wav
            user = parsed.netloc
            path_parts = parsed.path.strip("/").split("/")
            if not user or len(path_parts) < 1:
                raise ValueError(f"Invalid hf:// URL: {voice}")
            repo_id = f"{user}/{path_parts[0]}"
            filename = "/".join(path_parts[1:]) if len(path_parts) > 1 else "voice.wav"
            logger.debug("Downloading from HuggingFace: %s / %s", repo_id, filename)
            local_path = hf_hub_download(repo_id=repo_id, filename=filename)
            logger.debug("Downloaded to: %s", local_path)
            return local_path
        except Exception as exc:
            logger.warning("Failed to load hf:// voice %r: %s — falling back to 'alba'", voice, exc)
            return "alba"

    # ── Local file path ────────────────────────────────────────────────────
    voice_path = Path(voice)
    if voice_path.exists():
        logger.debug("Using local file: %s", voice_path)
        return str(voice_path)

    # ── Bundled voice ──────────────────────────────────────────────────────
    # Accept both "RafeBeckley" and "RafeBeckley.wav"
    voice_filename = voice if voice.endswith(".wav") else f"{voice}.wav"
    if voice_filename in get_bundled_voices():
        try:
            voice_file = importlib.resources.files("kenkui") / "voices" / voice_filename
            resolved = str(voice_file)
            logger.debug("Using bundled voice: %s", resolved)
            return resolved
        except Exception as exc:
            logger.debug("Could not resolve bundled voice %r: %s", voice_filename, exc)
            # Fall through to built-in name assumption

    # ── Built-in pocket-tts name ───────────────────────────────────────────
    logger.debug("Using built-in voice name: %s", voice)
    return voice


__all__ = ["load_voice"]
