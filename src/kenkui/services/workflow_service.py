"""workflow_service — pure state-transition helpers for setup flows."""

from __future__ import annotations

from typing import Any


def reset_chapter_selection(state: dict[str, Any], app_config) -> dict[str, Any]:
    """Reset chapter selection to app defaults."""
    return {
        **state,
        "chapter_selection": {
            "preset": getattr(app_config, "default_chapter_preset", "content-only"),
            "included": [],
            "excluded": [],
        },
    }


def reset_voice_mode(state: dict[str, Any], app_config) -> dict[str, Any]:
    """Reset voice-related state to single-narrator defaults."""
    return {
        **state,
        "voice": getattr(app_config, "default_voice", "alba"),
        "narration_mode": "single",
        "speaker_voices": {},
        "chapter_voices": {},
        "roster_cache_path": None,
        "series_slug": None,
        "_series_manifest": None,
    }


def set_single_voice_mode(state: dict[str, Any]) -> dict[str, Any]:
    """Switch to single-voice mode and clear mode-specific assignments."""
    return {
        **state,
        "narration_mode": "single",
        "speaker_voices": {},
        "chapter_voices": {},
    }


def apply_multi_voice_setup(
    state: dict[str, Any],
    *,
    speaker_voices: dict[str, str],
    roster_cache_path: str | None,
    manifest=None,
) -> dict[str, Any]:
    """Apply the result of a completed multi-voice setup flow to wizard state."""
    return {
        **state,
        "narration_mode": "multi",
        "speaker_voices": speaker_voices,
        "chapter_voices": {},
        "roster_cache_path": roster_cache_path,
        "_series_manifest": manifest,
        "series_slug": getattr(manifest, "slug", None) if manifest else None,
    }


def apply_chapter_voice_setup(
    state: dict[str, Any],
    chapter_voices: dict[str, str],
) -> dict[str, Any]:
    """Apply the result of chapter-voice assignment to wizard state."""
    return {
        **state,
        "narration_mode": "single",
        "chapter_voices": chapter_voices,
        "speaker_voices": {},
    }


def reset_quality_overrides(state: dict[str, Any]) -> dict[str, Any]:
    """Clear per-job audio quality overrides."""
    return {**state, "quality_overrides": {}}


def reset_post_processing_overrides(state: dict[str, Any]) -> dict[str, Any]:
    """Clear per-job post-processing overrides."""
    return {**state, "pp_overrides": {}}


def describe_voice_mode(state: dict[str, Any]) -> str:
    """Return a short human-readable label for the current voice mode."""
    current_mode = state.get("narration_mode", "single")
    has_chapter_voices = bool(state.get("chapter_voices"))
    if has_chapter_voices:
        return "chapter-voice"
    if current_mode == "multi":
        return "multi-voice NLP"
    return "single narrator"


__all__ = [
    "reset_chapter_selection",
    "reset_voice_mode",
    "set_single_voice_mode",
    "apply_multi_voice_setup",
    "apply_chapter_voice_setup",
    "reset_quality_overrides",
    "reset_post_processing_overrides",
    "describe_voice_mode",
]
