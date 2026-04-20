"""confirmation_service — shared helpers for confirmation-screen state."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def quality_overrides_from_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Extract per-job quality overrides from a saved profile."""
    return dict(profile.get("quality_overrides") or {})


def init_confirmation_state(book_path: Path, app_config, profile: dict[str, Any]) -> dict[str, Any]:
    """Build initial confirmation-screen state from a saved profile and app defaults."""
    return {
        "_book_path": book_path,
        "_app_config": app_config,
        "voice": profile.get("voice") or app_config.default_voice,
        "narration_mode": profile.get("narration_mode", "single"),
        "job_nlp_provider": profile.get("job_nlp_provider") or getattr(app_config, "nlp_provider", "ollama"),
        "job_nlp_model": profile.get("job_nlp_model") or getattr(app_config, "nlp_model", "llama3.2"),
        "chapter_selection": {
            "preset": profile.get("chapter_preset", app_config.default_chapter_preset),
            "included": [],
            "excluded": [],
        },
        "output_dir": profile.get("output_dir") or str(
            getattr(app_config, "default_output_dir", None) or book_path.parent
        ),
        "quality_overrides": quality_overrides_from_profile(profile),
        "pp_overrides": dict(profile.get("pp_overrides") or {}),
        "speaker_voices": {},
        "chapter_voices": {},
        "roster_cache_path": None,
        "series_slug": None,
        "_series_manifest": None,
    }


def confirmation_state_to_profile(state: dict[str, Any]) -> dict[str, Any]:
    """Extract the persisted profile subset from confirmation-screen state."""
    chapter_selection = state.get("chapter_selection", {})
    profile: dict[str, Any] = {
        "voice": state.get("voice", ""),
        "narration_mode": state.get("narration_mode", "single"),
        "chapter_preset": chapter_selection.get("preset", "content-only"),
        "output_dir": state.get("output_dir", ""),
        "quality_overrides": state.get("quality_overrides") or {},
        "pp_overrides": state.get("pp_overrides") or {},
    }
    if state.get("job_nlp_provider"):
        profile["job_nlp_provider"] = state["job_nlp_provider"]
    if state.get("job_nlp_model"):
        profile["job_nlp_model"] = state["job_nlp_model"]
    return profile


def summarize_confirmation_state(state: dict[str, Any], app_config) -> dict[str, str]:
    """Return display-ready summary strings for the confirmation screen."""
    book_name = Path(state["_book_path"]).name

    chapter_selection = state.get("chapter_selection", {})
    preset = chapter_selection.get(
        "preset",
        getattr(app_config, "default_chapter_preset", "content-only"),
    )
    included = chapter_selection.get("included", [])
    chapter_count = f"{len(included)} selected" if included else "not yet selected"
    chapter_tag = (
        "[DEFAULT]"
        if preset == getattr(app_config, "default_chapter_preset", "content-only")
        else "[CUSTOM]"
    )

    voice = state.get("voice", getattr(app_config, "default_voice", "alba"))
    mode = state.get("narration_mode", "single")
    default_voice = getattr(app_config, "default_voice", "alba")
    has_chapter_voices = bool(state.get("chapter_voices"))
    if has_chapter_voices:
        mode_display = "chapter-voice"
    elif mode == "multi":
        mode_display = "multi-voice NLP"
    else:
        mode_display = "single"
    voice_tag = (
        "[DEFAULT]"
        if (voice == default_voice and mode == "single" and not has_chapter_voices)
        else "[CUSTOM]"
    )

    quality_tag = "[DEFAULT]" if not state.get("quality_overrides") else "[CUSTOM]"
    output = state.get("output_dir", "")

    return {
        "book_name": book_name,
        "chapter_line": f"{preset} ({chapter_count})  {chapter_tag}",
        "voice_line": f"{voice} / {mode_display}  {voice_tag}",
        "quality_line": quality_tag,
        "output_line": str(output),
    }


__all__ = [
    "quality_overrides_from_profile",
    "init_confirmation_state",
    "confirmation_state_to_profile",
    "summarize_confirmation_state",
]
