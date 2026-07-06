"""Formatting helpers for voice catalog metadata.

These accessors read the user-facing fields of a kenkui voice object while
tolerating legacy attribute names, so UI layers can format voices without
duplicating the fallback rules.  They accept any object exposing the relevant
attributes (current ``VoiceResponse`` objects or older/legacy shapes).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "voice_id",
    "voice_source",
    "voice_source_group",
    "voice_excluded",
    "voice_label",
]

_SOURCE_GROUPS = {
    "kenkui_compiled": "compiled",
    "compiled": "compiled",
    "pocket_tts_builtin": "builtin",
    "builtin": "builtin",
    "custom_compiled": "custom",
    "custom": "custom",
    "uncompiled": "custom",
}


def voice_id(voice: Any) -> str:
    """Return the canonical voice id from current or legacy voice objects."""
    return getattr(voice, "voice_id", None) or getattr(voice, "name", "")


def voice_source(voice: Any) -> str:
    """Return the upstream source/origin string from current or legacy voice objects."""
    return getattr(voice, "origin", None) or getattr(voice, "source", "")


def voice_source_group(voice: Any) -> str:
    """Return the display grouping (compiled/builtin/custom) for a voice source."""
    source = voice_source(voice)
    return _SOURCE_GROUPS.get(source, source)


def voice_excluded(voice: Any) -> bool:
    """Return whether a voice is excluded from automatic assignment."""
    pool_enabled = getattr(voice, "pool_enabled", None)
    if pool_enabled is not None:
        return not bool(pool_enabled)
    return bool(getattr(voice, "excluded", False))


def voice_label(voice: Any) -> str:
    """Return the best user-facing label for a voice."""
    return (
        getattr(voice, "display_label", None)
        or getattr(voice, "description", None)
        or getattr(voice, "display_name", None)
        or voice_id(voice)
    )
