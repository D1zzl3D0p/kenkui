"""Voice loading utilities shared across TTS workers."""

from __future__ import annotations

from pathlib import Path


def load_voice(voice_id: str) -> str:
    """Resolve a canonical ``voice_id`` to a TTS prompt path or built-in ID."""
    from .voice_registry import get_catalog

    entry = get_catalog().resolve(voice_id)
    if entry is None:
        raise KeyError(f"Unknown voice_id: {voice_id}")
    if entry.asset_kind == "pocket_tts_builtin":
        return entry.voice_id
    if entry.path is None:
        raise FileNotFoundError(f"Voice {voice_id!r} does not have a local compiled asset")
    path = Path(entry.path)
    if not path.exists():
        raise FileNotFoundError(f"Voice {voice_id!r} asset is missing: {path}")
    return str(path)


__all__ = ["load_voice"]
