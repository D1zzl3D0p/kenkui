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
    nlp_provider: str | None = None,
    nlp_model: str | None = None,
) -> dict[str, Any]:
    """Apply the result of a completed multi-voice setup flow to wizard state."""
    update: dict[str, Any] = {
        **state,
        "narration_mode": "multi",
        "speaker_voices": speaker_voices,
        "chapter_voices": {},
        "roster_cache_path": roster_cache_path,
        "_series_manifest": manifest,
        "series_slug": getattr(manifest, "slug", None) if manifest else None,
    }
    if nlp_provider is not None:
        update["job_nlp_provider"] = nlp_provider
    if nlp_model is not None:
        update["job_nlp_model"] = nlp_model
    return update


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


def reset_output_location(state: dict[str, Any], app_config) -> dict[str, Any]:
    """Reset output directory to app config default."""
    from pathlib import Path
    default = str(getattr(app_config, "default_output_dir", None) or Path(state["_book_path"]).parent)
    return {**state, "output_dir": default}


def describe_voice_mode(state: dict[str, Any]) -> str:
    """Return a short human-readable label for the current voice mode."""
    current_mode = state.get("narration_mode", "single")
    has_chapter_voices = bool(state.get("chapter_voices"))
    if has_chapter_voices:
        return "chapter-voice"
    if current_mode == "multi":
        return "multi-voice NLP"
    return "single narrator"


def set_characters_config(
    state: dict[str, Any],
    method: "str | None",
    provider: "str | None",
    model: "str | None",
    execution_mode: "str | None",
) -> dict[str, Any]:
    """Apply character-discovery step settings to wizard state."""
    update = dict(state)
    if method is not None:
        update["job_character_discovery_method"] = method
    if provider is not None:
        update["job_nlp_provider"] = provider
    if model is not None:
        update["job_nlp_model"] = model
    if execution_mode is not None:
        update["job_nlp_execution_mode"] = execution_mode
    return update


def set_attribution_config(
    state: dict[str, Any],
    provider: "str | None",
    model: "str | None",
    execution_mode: "str | None",
) -> dict[str, Any]:
    """Apply attribution step settings to wizard state."""
    update = dict(state)
    if provider is not None:
        update["job_attribution_provider"] = provider
    if model is not None:
        update["job_attribution_model"] = model
    if execution_mode is not None:
        update["job_attribution_execution_mode"] = execution_mode
    return update


def set_tts_execution_mode(state: dict[str, Any], mode: str) -> dict[str, Any]:
    """Set the per-job TTS execution mode."""
    return {**state, "tts_execution_mode": mode}


def reset_characters_config(state: dict[str, Any], app_config) -> dict[str, Any]:
    """Reset character-discovery config to app defaults."""
    return {
        **state,
        "job_character_discovery_method": getattr(app_config, "nlp_discovery_method", "auto") or "auto",
        "job_nlp_provider": getattr(app_config, "nlp_provider", "ollama"),
        "job_nlp_model": getattr(app_config, "nlp_model", "llama3.2"),
        "job_nlp_execution_mode": None,
    }


def reset_attribution_config(state: dict[str, Any], app_config) -> dict[str, Any]:
    """Reset attribution config to app defaults."""
    return {
        **state,
        "job_attribution_provider": None,
        "job_attribution_model": None,
        "job_attribution_execution_mode": None,
    }


__all__ = [
    "reset_chapter_selection",
    "reset_voice_mode",
    "set_single_voice_mode",
    "apply_multi_voice_setup",
    "apply_chapter_voice_setup",
    "reset_quality_overrides",
    "reset_post_processing_overrides",
    "reset_output_location",
    "describe_voice_mode",
    "set_characters_config",
    "set_attribution_config",
    "set_tts_execution_mode",
    "reset_characters_config",
    "reset_attribution_config",
]
