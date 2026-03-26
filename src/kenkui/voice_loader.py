"""Voice loading utilities shared across TTS workers and future multi-voice modules.

Supports four voice source types (resolved in priority order):
1. HuggingFace URLs  (e.g. ``"hf://user/repo/voice.wav"``)
2. Local file paths  (e.g. ``"/home/user/my_voice.wav"``)
3. Registry voices   — compiled ``.safetensors``, uncompiled ``.wav``, or built-in names
4. Raw string fallback → passed unchanged to pocket-tts (handles built-in names)
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def load_voice(voice: str) -> str:
    """Resolve a voice identifier to a path or built-in name for TTSModel.

    Args:
        voice: One of:
            - A pocket-tts built-in name (``"alba"``, ``"cosette"``, …)
            - A compiled voice name (e.g. ``"Alasdair"``)
            - An uncompiled voice name (e.g. ``"RafeBeckley"``)
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

    # ── Registry lookup (compiled / uncompiled / builtin) ─────────────────
    from .voice_registry import get_registry

    meta = get_registry().resolve(voice)
    if meta is not None:
        if meta.file_path is not None:
            logger.debug("Using registry voice: %s (%s)", voice, meta.source)
            return str(meta.file_path)
        # Built-in voices have no file_path — pass name to pocket-tts
        logger.debug("Using built-in voice name: %s", meta.name)
        return meta.name

    # ── Raw fallback (unknown voice — let pocket-tts decide) ───────────────
    logger.debug("Voice %r not in registry — passing raw string to pocket-tts", voice)
    return voice


__all__ = ["load_voice"]
