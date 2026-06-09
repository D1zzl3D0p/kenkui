"""Voice loading utilities shared across TTS workers."""

from __future__ import annotations

from pathlib import Path


def load_voice(voice_id: str) -> str:
    """Resolve a canonical ``voice_id`` to a TTS prompt path or built-in ID."""
    from .voice_registry import BUILTIN_VOICE_NAMES, get_catalog

    entry = get_catalog().resolve(voice_id)
    if entry is None:
        if voice_id not in BUILTIN_VOICE_NAMES:
            from .voice_download import voices_are_present

            if not voices_are_present():
                raise KeyError(
                    f"Unknown voice_id: {voice_id} "
                    "(compiled voice pack is missing or stale; download the current pack)"
                )
        raise KeyError(f"Unknown voice_id: {voice_id}")
    if entry.asset_kind == "pocket_tts_builtin":
        return entry.voice_id
    if entry.status != "available":
        if entry.origin == "kenkui_compiled":
            raise RuntimeError(
                f"Compiled voice {voice_id!r} is stale or unavailable; "
                "rebuild or download the current voice pack before using it"
            )
        raise FileNotFoundError(f"Voice {voice_id!r} does not have an available compiled asset")
    if entry.path is None:
        raise FileNotFoundError(f"Voice {voice_id!r} does not have a local compiled asset")
    path = Path(entry.path)
    if not path.exists():
        raise FileNotFoundError(f"Voice {voice_id!r} asset is missing: {path}")
    return str(path)


def load_voice_conditioning_source(voice_id: str) -> str:
    """Return a conditioning source compatible with ``get_state_for_audio_prompt``.

    Built-in Pocket TTS voices are resolved to their original audio prompt URLs
    because the installed Pocket TTS release can reject some precomputed voice
    embedding files. Compiled voices continue to use their local safetensors
    assets through :func:`load_voice`.
    """
    from .voice_registry import get_catalog

    entry = get_catalog().resolve(voice_id)
    if entry is None:
        raise KeyError(f"Unknown voice_id: {voice_id}")

    if entry.asset_kind == "pocket_tts_builtin":
        try:
            from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES
        except Exception:
            return load_voice(voice_id)

        source = _ORIGINS_OF_PREDEFINED_VOICES.get(entry.voice_id)
        if source:
            return source

    return load_voice(voice_id)


__all__ = ["load_voice", "load_voice_conditioning_source"]
