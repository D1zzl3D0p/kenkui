"""job_service — shared helpers for assembling queue job payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_headless_job_kwargs(args, app_config) -> dict[str, Any]:
    """Build add_job kwargs for a non-interactive submission path."""
    from ..models import ChapterPreset, ChapterSelection

    try:
        preset_enum = ChapterPreset(app_config.default_chapter_preset)
    except ValueError:
        preset_enum = ChapterPreset.CONTENT_ONLY

    chapter_selection = ChapterSelection(preset=preset_enum).to_dict()
    output_dir = (
        str(Path(args.output).expanduser().resolve())
        if getattr(args, "output", None)
        else (
            str(app_config.default_output_dir)
            if app_config.default_output_dir
            else str(args.book.parent)
        )
    )

    from ..utils import ApostropheMode

    kwargs = {
        "ebook_path": str(args.book),
        "voice": app_config.default_voice,
        "chapter_selection": chapter_selection,
        "output_path": output_dir,
    }
    if getattr(args, "apostrophe_mode", None):
        kwargs["job_apostrophe_mode"] = ApostropheMode(args.apostrophe_mode)
    return kwargs


def build_job_kwargs_from_state(state: dict[str, Any]) -> dict[str, Any]:
    """Convert wizard/client state into the kwargs shape accepted by add_job()."""
    book_path = Path(state["_book_path"])
    chapter_selection = state.get("chapter_selection", {})
    narration_mode = state.get("narration_mode", "single")
    voice = state.get("voice", "alba")
    speaker_voices = state.get("speaker_voices", {})
    chapter_voices = state.get("chapter_voices", {})
    quality_overrides = state.get("quality_overrides", {})
    output_dir = state.get("output_dir", str(book_path.parent))
    roster_cache_path = state.get("roster_cache_path")
    series_slug = state.get("series_slug")
    job_nlp_provider = state.get("job_nlp_provider")
    job_nlp_model = state.get("job_nlp_model")

    kwargs: dict[str, Any] = dict(
        ebook_path=str(book_path),
        voice=voice,
        chapter_selection=chapter_selection,
        output_path=output_dir,
        narration_mode=narration_mode,
        speaker_voices=speaker_voices or None,
        annotated_chapters_path=None,
        roster_cache_path=roster_cache_path,
        chapter_voices=chapter_voices or None,
        series_slug=series_slug,
        **quality_overrides,
    )
    if job_nlp_provider is not None:
        kwargs["job_nlp_provider"] = job_nlp_provider
    if job_nlp_model is not None:
        kwargs["job_nlp_model"] = job_nlp_model
    return kwargs


__all__ = ["build_headless_job_kwargs", "build_job_kwargs_from_state"]
