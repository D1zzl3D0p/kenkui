"""Kenkui — Convert Ebooks to Audiobooks with custom voice samples.

This package provides tools for converting ebooks to audiobooks using
text-to-speech synthesis with support for custom voice samples.

Example:
    >>> from kenkui import run_job, parse_book, list_voices
    >>> config = load_config()
    >>> run_job(config)
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .analytics import StageRecord
from .chapter_classifier import ChapterClassifier, ChapterTags
from .chapter_filter import ChapterFilter, FilterOperation, FilterPreset
from .config import (
    CACHE_DIR,
    CONFIG_DIR,
    ProviderCredentials,
    load_provider_credentials,
    save_provider_credentials,
)
from .huggingface_auth import is_custom_voice
from .models import (
    AppConfig,
    AttributionTool,
    AudioResult,
    Chapter,
    ChapterPreset,
    ChapterSelection,
    CharacterInfo,
    ExtractionTool,
    FastScanResult,
    NarrationMode,
    NLPResult,
    PostProcessingConfig,
    ProcessingConfig,
    Segment,
)
from .nlp_config import NLPConfig
from .parsing import AudioBuilder
from .progress import ChapterProgress, ProgressEvent
from .readers import get_reader
from .readers.epub import EpubReader
from .series import (
    SeriesCharacter,
    SeriesManifest,
    build_manifest_from_predecessor,
    list_roster_candidates,
    load_series_roster,
    match_characters,
    save_series_roster,
    slugify,
)
from .series import (
    list_series as list_local_series,
)
from .series import (
    load_series as load_local_series,
)
from .series import (
    save_series as save_local_series,
)
from .services.auth_service import HFAuthStatus, HFLoginResult
from .services.series_service import (
    ListSeriesResult,
    RosterCandidateEntry,
    RosterCandidateListResult,
    SeriesCharacterEntry,
    SeriesEntry,
    SeriesMatchResult,
    build_series_from_candidate,
    delete_series,
    get_roster,
    match_series_characters,
    update_roster,
)
from .services.series_service import (
    list_roster_candidates as list_series_roster_candidates,
)
from .utils import ApostropheMode
from .voice_loader import load_voice
from .voice_registry import get_bundled_voices
from .workers import worker_process_chapter

try:
    __version__ = importlib.metadata.version("kenkui")
except importlib.metadata.PackageNotFoundError:
    __version__ = "1.0.0"

__author__ = "Sumner MacArthur"
__license__ = "GPL-3.0"


# ---------------------------------------------------------------------------
# Config API
# ---------------------------------------------------------------------------


def load_config(name: str | None = None) -> AppConfig:
    """Load an AppConfig from the named config file (or the default config).

    Args:
        name: Config file name (without .toml) or path. None uses the default.

    Returns:
        Populated AppConfig. Creates and persists the default config on first run.
    """
    from .config import load_app_config
    return load_app_config(name)


def save_config(config: AppConfig, name: str | None = None) -> Path:
    """Save an AppConfig to a named config file.

    Args:
        config: The AppConfig to persist.
        name: Destination name (without .toml) or full path. None saves to the default.

    Returns:
        The resolved Path that was written.
    """
    from .config import DEFAULT_CONFIG_PATH, resolve_config_path, save_app_config
    dest = resolve_config_path(name) if name is not None else DEFAULT_CONFIG_PATH
    return save_app_config(config, dest)


def list_configs() -> list[str]:
    """Return the names of all saved kenkui config files (without .toml extension)."""
    from .config import _kenkui_config_dir
    config_dir = _kenkui_config_dir()
    return sorted(p.stem for p in config_dir.glob("*.toml"))


# ---------------------------------------------------------------------------
# Book / chapter API
# ---------------------------------------------------------------------------

_book_cache = None


def _get_book_cache():
    """Return a lazily initialised BookCache using the XDG cache dir."""
    global _book_cache
    if _book_cache is None:
        from .services.book_cache import BookCache
        _book_cache = BookCache()
    return _book_cache


def parse_book(ebook_path: str | Path) -> BookParseResult:  # noqa: F821
    """Parse an ebook file and return structured metadata + chapter summaries.

    Results are cached on disk so repeated calls for the same file are fast.

    Args:
        ebook_path: Path to the ebook file.

    Returns:
        BookParseResult with metadata and chapter summaries.

    Raises:
        FileNotFoundError: if ebook_path does not exist.
        ValueError: if the file format is not supported.
    """
    from .services.book_service import parse_book as _parse_book
    return _parse_book(str(ebook_path), _get_book_cache())


def filter_chapters(book_hash: str, selection) -> ChapterFilterResult:  # noqa: F821
    """Apply a ChapterSelection to previously parsed chapters.

    Args:
        book_hash: The hash returned by parse_book().
        selection:  A ChapterSelection (from kenkui.models) specifying which chapters to keep.

    Returns:
        ChapterFilterResult with the filtered chapter list.

    Raises:
        KeyError: if book_hash has not been parsed in this session.
    """
    from .services.book_service import filter_chapters as _filter_chapters
    return _filter_chapters(book_hash, selection, _get_book_cache())


def load_chapters(
    ebook_path: str | Path,
    chapter_filters: list[FilterOperation] | None = None,
) -> list[Chapter]:
    """Load full Chapter objects from an ebook, optionally applying filters."""
    reader = get_reader(Path(ebook_path))
    chapters = reader.get_chapters()
    if chapter_filters:
        chapters = ChapterFilter(chapter_filters).apply(chapters)
    return chapters


def full_analysis(
    ebook_path: str | Path,
    nlp_model: str | None = None,
    nlp_provider: str | None = None,
    config_path: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    extraction_progress_callback: Callable[[int, str], None] | None = None,
    attribution_progress_callback: Callable[[int, str], None] | None = None,
    extraction_progress_event_callback: Callable[[ProgressEvent], None] | None = None,
    attribution_progress_event_callback: Callable[[ProgressEvent], None] | None = None,
    series_slug: str | None = None,
    book_slug: str | None = None,
    discovery_method: str | None = None,
    attribution_provider: str | None = None,
    attribution_model: str | None = None,
    use_cache: bool = True,
) -> NLPResult:  # noqa: F821
    """Run the full NLP speaker-attribution pipeline (Stages 1-4).

    Args:
        ebook_path:                    Path to the ebook file.
        nlp_model:                     Override NLP model name.
        nlp_provider:                  Override NLP provider name.
        config_path:                   Optional config file name or path.
        progress_callback:             Optional (percent, message) callback (for backward compat).
                                       If provided and the extraction/attribution callbacks are
                                       not set, this is forwarded to both new callback params.
        extraction_progress_callback:  Optional (percent, message) callback for extraction phase.
        attribution_progress_callback: Optional (percent, message) callback for attribution phase.
        extraction_progress_event_callback:  Optional structured callback for extraction phase.
        attribution_progress_event_callback: Optional structured callback for attribution phase.
        series_slug:                   Series slug for cross-book roster merging.
        book_slug:                     Slug for this book (series first-appearance tracking).
        discovery_method:              Override discovery method.
        attribution_provider:          Override attribution provider.
        attribution_model:             Override attribution model.
        use_cache:                     Return cached result if available (default True).

    Returns:
        NLPResult with characters and annotated chapters.
    """
    from .services.nlp_service import full_analysis as _full_analysis
    return _full_analysis(
        str(ebook_path),
        nlp_model=nlp_model,
        nlp_provider=nlp_provider,
        config_path=config_path,
        extraction_progress_callback=extraction_progress_callback or progress_callback,
        attribution_progress_callback=attribution_progress_callback or progress_callback,
        extraction_progress_event_callback=extraction_progress_event_callback,
        attribution_progress_event_callback=attribution_progress_event_callback,
        series_slug=series_slug,
        book_slug=book_slug,
        discovery_method=discovery_method,
        attribution_provider=attribution_provider,
        attribution_model=attribution_model,
        use_cache=use_cache,
    )


def fast_scan(
    ebook_path: str | Path,
    nlp_model: str | None = None,
    config_path: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    progress_event_callback: Callable[[ProgressEvent], None] | None = None,
    series_slug: str | None = None,
    book_slug: str | None = None,
    nlp_provider: str | None = None,
    discovery_method: str | None = None,
    use_cache: bool = True,
) -> FastScanResult:  # noqa: F821
    """Run Stage 1-2 NLP (entity detection + character clustering).

    Args:
        ebook_path:        Path to the ebook file.
        nlp_model:         Override NLP model name.
        config_path:       Optional config file name or path.
        progress_callback: Optional (percent, message) callback.
        progress_event_callback: Optional structured progress callback.
        series_slug:       Existing series slug to load and merge roster into.
        book_slug:         Slug for this book (for series first-appearance tracking).
        nlp_provider:      Override NLP provider ("ollama", "spacy", "booknlp").
        discovery_method:  Override discovery method ("auto", "spacy", "booknlp", "ollama").
        use_cache:         Return cached result if available (default True).

    Returns:
        FastScanResult with characters sorted by mention_count descending.
    """
    from .services.nlp_service import fast_scan as _fast_scan
    return _fast_scan(
        str(ebook_path),
        nlp_model=nlp_model,
        config_path=config_path,
        progress_callback=progress_callback,
        progress_event_callback=progress_event_callback,
        series_slug=series_slug,
        book_slug=book_slug,
        nlp_provider=nlp_provider,
        discovery_method=discovery_method,
        use_cache=use_cache,
    )


def attribute_only(
    *,
    roster: Any,
    chapters: list[Chapter],
    ebook_path: str | Path,
    nlp_model: str | None = None,
    nlp_provider: str | None = None,
    config_path: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    progress_event_callback: Callable[[ProgressEvent], None] | None = None,
    attribution_provider: str | None = None,
    attribution_model: str | None = None,
) -> NLPResult:
    """Run speaker attribution against an already discovered roster."""
    from .services.nlp_service import attribute_only as _attribute_only
    return _attribute_only(
        roster=roster,
        chapters=chapters,
        ebook_path=str(ebook_path),
        nlp_model=nlp_model,
        nlp_provider=nlp_provider,
        config_path=config_path,
        progress_callback=progress_callback,
        progress_event_callback=progress_event_callback,
        attribution_provider=attribution_provider,
        attribution_model=attribution_model,
    )


# ---------------------------------------------------------------------------
# Voice API
# ---------------------------------------------------------------------------


def list_voices(
    gender: str | None = None,
    accent: str | None = None,
    dataset: str | None = None,
    source: str | None = None,
    config_path: str | None = None,
) -> list:
    """Return all voices matching the given filters.

    Returns:
        List of VoiceInfo dataclasses with name, gender, accent, source, excluded flag.
    """
    from .services.voice_service import list_voices as _list_voices
    return _list_voices(
        gender=gender,
        accent=accent,
        dataset=dataset,
        source=source,
        config_path=config_path,
    )


def get_voice(name: str, config_path: str | None = None):
    """Look up a single voice by name.

    Returns:
        VoiceInfo if found, None otherwise.
    """
    from .services.voice_service import get_voice as _get_voice
    return _get_voice(name, config_path=config_path)


def suggest_cast(
    *,
    roster: list,
    excluded_voices: list[str],
    default_voice: str,
    chapters: list | None = None,
    config_path: str | None = None,
) -> SuggestCastResult:  # noqa: F821
    """Assign voices to characters using round-robin pool with conflict resolution.

    Args:
        roster:          List of CharacterInfo objects to assign voices to.
        excluded_voices: Voice names to exclude from auto-assignment.
        default_voice:   Narrator voice (excluded from character pool).
        chapters:        Optional chapter list for conflict resolution.
        config_path:     Optional config file name or path.

    Returns:
        SuggestCastResult with speaker_voices mapping and any warnings.
    """
    from .services.voice_service import suggest_cast as _suggest_cast
    return _suggest_cast(
        roster=roster,
        excluded_voices=excluded_voices,
        default_voice=default_voice,
        chapters=chapters,
        config_path=config_path,
    )


def recommend_narrator(
    roster: list,
    excluded: list[str] | None = None,
    default_voice: str = "",
) -> str:
    """Recommend a narrator voice based on dominant character gender in roster.

    Args:
        roster:        List of CharacterInfo objects.
        excluded:      Voice names to exclude from consideration.
        default_voice: Fallback voice name.

    Returns:
        Recommended narrator voice name.
    """
    from .services.voice_service import top_gender_matched_voice
    return top_gender_matched_voice(
        roster,
        excluded=excluded or [],
        default_voice=default_voice,
    )


def assign_simple_cast(*, roster: list, narrator_voice: str, male_voice: str, female_voice: str) -> dict[str, str]:
    """Assign one male and one female character voice plus narrator."""
    from .services.voice_service import assign_simple_cast as _assign_simple_cast
    return _assign_simple_cast(
        roster=roster,
        narrator_voice=narrator_voice,
        male_voice=male_voice,
        female_voice=female_voice,
    )


def merge_speaker_voices(
    base: dict[str, str],
    inherited: dict[str, str] | None,
) -> dict[str, str]:
    """Merge inherited speaker voice pins over generated assignments."""
    from .services.voice_service import merge_speaker_voices as _merge_speaker_voices
    return _merge_speaker_voices(base, inherited)


def format_character_review_label(ch, voice: str, pinned: set[str], series_name: str | None = None) -> str:
    """Return display text for a character voice-review entry."""
    from .services.voice_service import format_character_review_label as _format
    return _format(ch, voice, pinned=pinned, series_name=series_name)


def build_voice_users(speaker_voices: dict[str, str], characters: list) -> dict[str, list[str]]:
    """Return voice name to character display-name mapping."""
    from .services.voice_service import build_voice_users as _build_voice_users
    return _build_voice_users(speaker_voices, characters)


def annotate_voice_choices(
    voice_choices: list[dict],
    voice_users: dict[str, list[str]],
    exclude_char_name: str | None = None,
) -> list[dict]:
    """Annotate voice picker labels with current character users."""
    from .services.voice_service import annotate_voice_choices as _annotate
    return _annotate(voice_choices, voice_users, exclude_char_name=exclude_char_name)


def format_unresolved_conflict_warnings(
    unresolved_conflicts: list[tuple[str, str]] | None,
    pinned: set[str],
) -> list[str]:
    """Return display warnings for unresolved voice conflicts."""
    from .services.voice_service import format_unresolved_conflict_warnings as _format
    return _format(unresolved_conflicts, pinned)


def build_character_review_choices(
    characters: list,
    speaker_voices: dict[str, str],
    narrator_voice: str,
    pinned: set[str],
    series_name: str | None = None,
) -> list[dict]:
    """Build picker choices for character voice review."""
    from .services.voice_service import build_character_review_choices as _build
    return _build(
        characters,
        speaker_voices,
        narrator_voice,
        pinned=pinned,
        series_name=series_name,
    )


def exclude_voice(name: str, config_path: str | None = None) -> ExcludeResult:  # noqa: F821
    """Add a voice to the excluded-from-auto-assignment list.

    Returns:
        ExcludeResult with updated excluded list and optional gender-pool warning.
    """
    from .services.voice_service import exclude_voice as _exclude_voice
    return _exclude_voice(name, config_path=config_path)


def include_voice(name: str, config_path: str | None = None) -> IncludeResult:  # noqa: F821
    """Remove a voice from the excluded list (restores it to auto-assignment).

    Returns:
        IncludeResult with updated excluded list.
    """
    from .services.voice_service import include_voice as _include_voice
    return _include_voice(name, config_path=config_path)


def audition_voice(
    voice_name: str,
    text: str | None = None,
    config_path: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> AudioPreviewResult:  # noqa: F821
    """Synthesize a short audio preview for a voice.

    Saves output to ~/.cache/kenkui/previews/{voice_name}.wav.

    Returns:
        AudioPreviewResult with audio_path and duration_ms.
    """
    from .services.voice_service import audition_voice as _audition_voice
    return _audition_voice(
        voice_name,
        text=text,
        config_path=config_path,
        progress_callback=progress_callback,
    )


def download_voice(
    force: bool = False,
    progress_callback: Callable[[int, str], None] | None = None,
) -> DownloadResult:  # noqa: F821
    """Download compiled voices from HuggingFace.

    Returns:
        DownloadResult with success flag and path.
    """
    from .services.download_service import download_compiled
    return download_compiled(force=force, progress_callback=progress_callback)


def compiled_voices_available() -> bool:
    """Return True when downloaded compiled voices are present locally."""
    from .voice_download import voices_are_present
    return voices_are_present()


def fetch_voice(
    repo_id: str | None = None,
    patterns: list[str] | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> DownloadResult:  # noqa: F821
    """Fetch uncompiled voice sources from HuggingFace.

    Returns:
        DownloadResult with success flag and path.
    """
    from .services.download_service import fetch_uncompiled
    return fetch_uncompiled(repo_id=repo_id, patterns=patterns, progress_callback=progress_callback)


# ---------------------------------------------------------------------------
# Series API
# ---------------------------------------------------------------------------


def list_series() -> ListSeriesResult:  # noqa: F821
    """Return all saved series manifests.

    Returns:
        ListSeriesResult with series list and total count.
    """
    from .services.series_service import list_series as _list_series
    return _list_series()


def get_series(slug: str) -> SeriesEntry:  # noqa: F821
    """Load a series manifest by slug.

    Returns:
        SeriesEntry.

    Raises:
        KeyError: if the slug is not found.
    """
    from .services.series_service import load_series
    return load_series(slug)


def create_series(name: str) -> SeriesEntry:  # noqa: F821
    """Create and persist an empty series manifest.

    Args:
        name: Human-readable series name. A slug is derived automatically.

    Returns:
        The newly created SeriesEntry.
    """
    from .services.series_service import create_empty_series
    return create_empty_series(name)


def update_series(entry) -> None:
    """Persist changes to an existing SeriesEntry.

    Args:
        entry: A SeriesEntry (from kenkui.services.series_service).
    """
    from .services.series_service import save_series
    save_series(entry)


# ---------------------------------------------------------------------------
# Auth API
# ---------------------------------------------------------------------------


def authenticate_huggingface(token: str) -> HFLoginResult:  # noqa: F821
    """Authenticate with HuggingFace using a user-supplied token.

    Args:
        token: HuggingFace access token.

    Returns:
        HFLoginResult with authenticated flag, username, and optional error.
    """
    from .services.auth_service import login
    return login(token)


def get_huggingface_status(model_id: str = "kyutai/pocket-tts") -> HFAuthStatus:
    """Return structured HuggingFace auth status without prompting."""
    from .services.auth_service import get_hf_status
    return get_hf_status(model_id)


# ---------------------------------------------------------------------------
# NLP cache API
# ---------------------------------------------------------------------------


def book_hash(book_path: str | Path) -> str:
    """Return the cache hash used for an ebook path."""
    from .nlp import book_hash as _book_hash
    return _book_hash(Path(book_path))


def list_cached_rosters(book_path: str | Path):
    """Return cached roster metadata entries for an ebook path."""
    from .nlp import list_cached_rosters as _list_cached_rosters
    return _list_cached_rosters(Path(book_path))


def get_cached_nlp_result(book_path: str | Path, provider: str | None = None) -> NLPResult | None:
    """Return a cached attributed NLP result, if available."""
    from .nlp import get_cached_result
    return get_cached_result(Path(book_path), provider=provider)


def cache_nlp_result(result: NLPResult, book_path: str | Path, provider: str | None = None) -> Path:
    """Persist an attributed NLP result and return the cache path."""
    from .nlp import cache_result
    return cache_result(result, Path(book_path), provider=provider)


def cache_roster(
    result,
    book_path: str | Path,
    method: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    description: str | None = None,
) -> Path:
    """Persist a fast-scan roster result and return the cache path."""
    from .nlp import cache_roster as _cache_roster
    return _cache_roster(
        result,
        Path(book_path),
        method=method,
        provider=provider,
        model=model,
        description=description,
    )


def nlp_attribution_cache_path(book_path: str | Path, provider: str | None = None) -> Path:
    """Return the expected attributed NLP cache path for an ebook path."""
    from .nlp import attribution_cache_path
    return attribution_cache_path(Path(book_path), provider=provider)


# ---------------------------------------------------------------------------
# Top-level job runner
# ---------------------------------------------------------------------------


def run_job(
    config: ProcessingConfig,
    progress_callback: Callable[[ProgressEvent], None] | None = None,
) -> bool:
    """Convert an ebook to an audiobook using the given configuration.

    This is the main library entry point. It handles the full pipeline:
    parse → chapter filtering → TTS → stitching → M4B output.

    For multi-voice jobs, set config.annotated_chapters_path to the NLP
    cache file produced by fast_scan / full_analysis first.

    Args:
        config:            Full ProcessingConfig for the conversion.
        progress_callback: Optional callback receiving ProgressEvent facts.

    Returns:
        True on success, False on failure.
    """
    builder = AudioBuilder(config, progress_callback=progress_callback)
    return builder.run()


def load_stage_records(path: Path | None = None) -> list[StageRecord]:
    """Load historical pipeline stage analytics records."""
    from .analytics import load_records
    return load_records(path)


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    # Data models
    "AppConfig",
    "ApostropheMode",
    "ProcessingConfig",
    "PostProcessingConfig",
    "ProgressEvent",
    "ChapterProgress",
    "StageRecord",
    "ChapterPreset",
    "ChapterSelection",
    "Chapter",
    "Segment",
    "AudioResult",
    # Multi-voice / NLP
    "NLPResult",
    "CharacterInfo",
    "NarrationMode",
    # NLP pipeline config and tool enums
    "NLPConfig",
    "ExtractionTool",
    "AttributionTool",
    # Ebook reading
    "EpubReader",
    "AudioBuilder",
    # Chapter handling
    "ChapterTags",
    "ChapterClassifier",
    "ChapterFilter",
    "FilterPreset",
    "FilterOperation",
    # Low-level voice helpers
    "get_bundled_voices",
    "load_voice",
    # Workers
    "worker_process_chapter",
    # HuggingFace auth
    "is_custom_voice",
    # Config API
    "load_config",
    "save_config",
    "list_configs",
    # Book API
    "parse_book",
    "filter_chapters",
    "load_chapters",
    "fast_scan",
    "full_analysis",
    "attribute_only",
    # Voice API
    "list_voices",
    "get_voice",
    "suggest_cast",
    "recommend_narrator",
    "assign_simple_cast",
    "merge_speaker_voices",
    "format_character_review_label",
    "build_voice_users",
    "annotate_voice_choices",
    "format_unresolved_conflict_warnings",
    "build_character_review_choices",
    "exclude_voice",
    "include_voice",
    "audition_voice",
    "download_voice",
    "compiled_voices_available",
    "fetch_voice",
    # Series API
    "list_series",
    "get_series",
    "create_series",
    "update_series",
    "delete_series",
    "build_series_from_candidate",
    "list_series_roster_candidates",
    "match_series_characters",
    "get_roster",
    "update_roster",
    "slugify",
    "list_local_series",
    "load_local_series",
    "save_local_series",
    "SeriesCharacter",
    "SeriesManifest",
    "SeriesCharacterEntry",
    "SeriesEntry",
    "ListSeriesResult",
    "RosterCandidateEntry",
    "RosterCandidateListResult",
    "SeriesMatchResult",
    "load_series_roster",
    "save_series_roster",
    "match_characters",
    "list_roster_candidates",
    "build_manifest_from_predecessor",
    # Auth API
    "authenticate_huggingface",
    "get_huggingface_status",
    "HFAuthStatus",
    "HFLoginResult",
    "ProviderCredentials",
    "CONFIG_DIR",
    "CACHE_DIR",
    "load_provider_credentials",
    "save_provider_credentials",
    # NLP cache API
    "book_hash",
    "list_cached_rosters",
    "get_cached_nlp_result",
    "cache_nlp_result",
    "cache_roster",
    "nlp_attribution_cache_path",
    # Job runner
    "run_job",
    "load_stage_records",
    # Package metadata
    "__version__",
    "__author__",
    "__license__",
]
