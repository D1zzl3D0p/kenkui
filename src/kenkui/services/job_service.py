"""job_service — shared helpers for assembling queue job payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..chapter_filter import ChapterFilter, FilterOperation
from ..models import (
    ChapterPreset,
    ChapterSelection,
    JobConfig,
    PostProcessingConfig,
    ProcessingConfig,
)
from ..models.common import _normalize_bitrate
from ..utils import ApostropheMode


def _resolve(job_val, app_val):
    """Return job_val if explicitly set, else fall back to app_val."""
    return job_val if job_val is not None else app_val


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

    kwargs = {
        "ebook_path": str(args.book),
        "voice": app_config.default_voice,
        "chapter_selection": chapter_selection,
        "output_path": output_dir,
    }
    apostrophe_mode = getattr(args, "apostrophe_mode", None)
    if apostrophe_mode is not None:
        kwargs["job_apostrophe_mode"] = ApostropheMode(apostrophe_mode)
    return kwargs


def chapter_filters_from_selection(selection: ChapterSelection) -> tuple[list[FilterOperation], list[int]]:
    """Build rendering filters and included indices from a chapter selection."""
    if selection.preset in (ChapterPreset.MANUAL, ChapterPreset.CUSTOM) and selection.included:
        return [FilterOperation("index", str(idx)) for idx in selection.included], list(selection.included)
    if selection.preset.value in ChapterFilter.PRESETS:
        return [FilterOperation("preset", selection.preset.value)], list(selection.included)
    return [FilterOperation("preset", "content-only")], list(selection.included)


def post_processing_from_app_config(app_config, enabled_override: bool | None = None) -> PostProcessingConfig:
    """Return a PostProcessingConfig copy with an optional per-job enabled override."""
    app_pp = getattr(app_config, "post_processing", None)
    if app_pp is not None:
        if hasattr(app_pp, "to_dict"):
            pp_dict = app_pp.to_dict()
        else:
            pp_dict = app_pp.model_dump(mode="json", exclude_none=True)
        pp_config = PostProcessingConfig.from_dict(pp_dict)
    else:
        pp_config = PostProcessingConfig()

    if enabled_override is None:
        return pp_config
    return PostProcessingConfig.from_dict({**pp_config.to_dict(), "enabled": enabled_override})


def build_processing_config(job: JobConfig, app_config) -> ProcessingConfig:
    """Convert a queued/API JobConfig plus AppConfig into ProcessingConfig."""
    chapter_filters, included_indices = chapter_filters_from_selection(job.chapter_selection)
    output_path = job.output_path or job.ebook_path.parent
    return ProcessingConfig(
        voice=job.voice or getattr(app_config, "default_voice", "alba"),
        ebook_path=job.ebook_path,
        output_path=output_path,
        pause_line_ms=_resolve(job.job_pause_line_ms, app_config.pause_line_ms),
        pause_chapter_ms=_resolve(job.job_pause_chapter_ms, app_config.pause_chapter_ms),
        speak_chapter_titles=_resolve(
            job.job_speak_chapter_titles,
            app_config.speak_chapter_titles,
        ),
        pause_before_chapter_title_ms=_resolve(
            job.job_pause_before_chapter_title_ms,
            app_config.pause_before_chapter_title_ms,
        ),
        pause_after_chapter_title_ms=_resolve(
            job.job_pause_after_chapter_title_ms,
            app_config.pause_after_chapter_title_ms,
        ),
        workers=app_config.workers,
        m4b_bitrate=_normalize_bitrate(_resolve(job.job_m4b_bitrate, app_config.m4b_bitrate)),
        keep_temp=app_config.keep_temp,
        debug_html=app_config.verbose,
        chapter_filters=chapter_filters,
        verbose=app_config.verbose,
        temp=_resolve(job.job_temp, app_config.temp),
        lsd_decode_steps=_resolve(job.job_lsd_decode_steps, app_config.lsd_decode_steps),
        noise_clamp=_resolve(job.job_noise_clamp, app_config.noise_clamp),
        eos_threshold=_resolve(job.job_eos_threshold, app_config.eos_threshold),
        frames_after_eos=_resolve(job.job_frames_after_eos, app_config.frames_after_eos),
        tts_max_tokens_per_chunk=app_config.tts_max_tokens_per_chunk,
        pdf_drop_code_blocks=app_config.pdf_drop_code_blocks,
        pdf_drop_notes=app_config.pdf_drop_notes,
        pdf_drop_asides=app_config.pdf_drop_asides,
        pdf_drop_margin_notes=app_config.pdf_drop_margin_notes,
        pdf_header_zone_ratio=app_config.pdf_header_zone_ratio,
        pdf_footer_zone_ratio=app_config.pdf_footer_zone_ratio,
        pdf_force_ocr=app_config.pdf_force_ocr,
        speaker_voices=job.speaker_voices,
        annotated_chapters_path=job.annotated_chapters_path,
        roster_cache_path=job.roster_cache_path,
        chapter_voices=job.chapter_voices,
        post_processing=post_processing_from_app_config(
            app_config,
            enabled_override=job.job_post_processing_enabled,
        ),
        _included_indices=included_indices,
        apostrophe_mode=_resolve(job.job_apostrophe_mode, app_config.apostrophe_mode),
    )


def build_job_kwargs_from_state(state: dict[str, Any]) -> dict[str, Any]:
    """Convert wizard/client state into the kwargs shape accepted by add_job()."""
    book_path = Path(state["_book_path"])
    chapter_selection = state.get("chapter_selection", {})
    narration_mode = state.get("narration_mode", "single")
    voice = state.get("voice", "alba")
    speaker_voices = state.get("speaker_voices", {})
    chapter_voices = state.get("chapter_voices", {})
    quality_overrides = state.get("quality_overrides", {})
    pp_enabled_override = state.get("pp_enabled_override")  # None | bool
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
    if pp_enabled_override is not None:
        kwargs["job_post_processing_enabled"] = pp_enabled_override
    job_nlp_execution_mode = state.get("job_nlp_execution_mode")
    if job_nlp_execution_mode is not None:
        from kenkui.models import NlpExecutionMode
        kwargs["job_nlp_execution_mode"] = NlpExecutionMode(job_nlp_execution_mode)
    job_attribution_execution_mode = state.get("job_attribution_execution_mode")
    if job_attribution_execution_mode is not None:
        from kenkui.models import AttributionExecutionMode
        kwargs["job_attribution_execution_mode"] = AttributionExecutionMode(job_attribution_execution_mode)
    job_character_discovery_method = state.get("job_character_discovery_method")
    if job_character_discovery_method is not None:
        kwargs["job_character_discovery_method"] = job_character_discovery_method
    job_attribution_provider = state.get("job_attribution_provider")
    if job_attribution_provider is not None:
        kwargs["job_attribution_provider"] = job_attribution_provider
    job_attribution_model = state.get("job_attribution_model")
    if job_attribution_model is not None:
        kwargs["job_attribution_model"] = job_attribution_model
    tts_execution_mode = state.get("tts_execution_mode")
    if tts_execution_mode is not None:
        from kenkui.models import TTSExecutionMode
        kwargs["tts_execution_mode"] = TTSExecutionMode(tts_execution_mode)
    voice_assignment_mode = state.get("voice_assignment_mode")
    if voice_assignment_mode is not None:
        kwargs["voice_assignment_mode"] = voice_assignment_mode
    simple_male_voice = state.get("simple_male_voice")
    if simple_male_voice is not None:
        kwargs["simple_male_voice"] = simple_male_voice
    simple_female_voice = state.get("simple_female_voice")
    if simple_female_voice is not None:
        kwargs["simple_female_voice"] = simple_female_voice
    return kwargs


__all__ = [
    "build_headless_job_kwargs",
    "build_job_kwargs_from_state",
    "build_processing_config",
    "chapter_filters_from_selection",
    "post_processing_from_app_config",
]
