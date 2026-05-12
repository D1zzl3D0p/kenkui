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
import os
from collections.abc import Callable
from pathlib import Path

from .chapter_classifier import ChapterClassifier, ChapterTags
from .chapter_filter import ChapterFilter, FilterOperation, FilterPreset
from .voice_registry import get_bundled_voices
from .huggingface_auth import (
    check_voice_access,
    ensure_huggingface_access,
    is_custom_voice,
)
from .models import (
    AppConfig,
    AudioResult,
    Chapter,
    CharacterInfo,
    NarrationMode,
    NLPResult,
    ProcessingConfig,
    Segment,
)
from .parsing import AudioBuilder
from .readers.epub import EpubReader
from .voice_loader import load_voice
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
    from .config import save_app_config, DEFAULT_CONFIG_PATH, resolve_config_path
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


def parse_book(ebook_path: str | Path) -> "BookParseResult":  # noqa: F821
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


def filter_chapters(book_hash: str, selection) -> "ChapterFilterResult":  # noqa: F821
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


def fast_scan(
    ebook_path: str | Path,
    nlp_model: str | None = None,
    config_path: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    series_slug: str | None = None,
    book_slug: str | None = None,
    nlp_provider: str | None = None,
    discovery_method: str | None = None,
) -> "FastScanResult":  # noqa: F821
    """Run Stage 1-2 NLP (entity detection + character clustering).

    Args:
        ebook_path:        Path to the ebook file.
        nlp_model:         Override NLP model name.
        config_path:       Optional config file name or path.
        progress_callback: Optional (percent, message) callback.
        series_slug:       Existing series slug to load and merge roster into.
        book_slug:         Slug for this book (for series first-appearance tracking).
        nlp_provider:      Override NLP provider ("ollama", "spacy", "booknlp").
        discovery_method:  Override discovery method ("auto", "spacy", "booknlp", "ollama").

    Returns:
        FastScanResult with characters sorted by mention_count descending.
    """
    from .services.nlp_service import fast_scan as _fast_scan
    return _fast_scan(
        str(ebook_path),
        nlp_model=nlp_model,
        config_path=config_path,
        progress_callback=progress_callback,
        series_slug=series_slug,
        book_slug=book_slug,
        nlp_provider=nlp_provider,
        discovery_method=discovery_method,
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
) -> "SuggestCastResult":  # noqa: F821
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


def exclude_voice(name: str, config_path: str | None = None) -> "ExcludeResult":  # noqa: F821
    """Add a voice to the excluded-from-auto-assignment list.

    Returns:
        ExcludeResult with updated excluded list and optional gender-pool warning.
    """
    from .services.voice_service import exclude_voice as _exclude_voice
    return _exclude_voice(name, config_path=config_path)


def include_voice(name: str, config_path: str | None = None) -> "IncludeResult":  # noqa: F821
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
) -> "AudioPreviewResult":  # noqa: F821
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
) -> "DownloadResult":  # noqa: F821
    """Download compiled voices from HuggingFace.

    Returns:
        DownloadResult with success flag and path.
    """
    from .services.download_service import download_compiled
    return download_compiled(force=force, progress_callback=progress_callback)


def fetch_voice(
    repo_id: str | None = None,
    patterns: list[str] | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> "DownloadResult":  # noqa: F821
    """Fetch uncompiled voice sources from HuggingFace.

    Returns:
        DownloadResult with success flag and path.
    """
    from .services.download_service import fetch_uncompiled
    return fetch_uncompiled(repo_id=repo_id, patterns=patterns, progress_callback=progress_callback)


# ---------------------------------------------------------------------------
# Series API
# ---------------------------------------------------------------------------


def list_series() -> "ListSeriesResult":  # noqa: F821
    """Return all saved series manifests.

    Returns:
        ListSeriesResult with series list and total count.
    """
    from .services.series_service import list_series as _list_series
    return _list_series()


def get_series(slug: str) -> "SeriesEntry":  # noqa: F821
    """Load a series manifest by slug.

    Returns:
        SeriesEntry.

    Raises:
        KeyError: if the slug is not found.
    """
    from .services.series_service import load_series
    return load_series(slug)


def create_series(name: str) -> "SeriesEntry":  # noqa: F821
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


def authenticate_huggingface(token: str) -> "HFLoginResult":  # noqa: F821
    """Authenticate with HuggingFace using a user-supplied token.

    Args:
        token: HuggingFace access token.

    Returns:
        HFLoginResult with authenticated flag, username, and optional error.
    """
    from .services.auth_service import login
    return login(token)


# ---------------------------------------------------------------------------
# Top-level job runner
# ---------------------------------------------------------------------------


def run_job(
    config: ProcessingConfig,
    progress_callback: Callable[[float, str, int], None] | None = None,
) -> bool:
    """Convert an ebook to an audiobook using the given configuration.

    This is the main library entry point. It handles the full pipeline:
    parse → chapter filtering → TTS → stitching → M4B output.

    For multi-voice jobs, set config.annotated_chapters_path to the NLP
    cache file produced by fast_scan / full_analysis first.

    Args:
        config:            Full ProcessingConfig for the conversion.
        progress_callback: Optional (percent, chapter_title, eta_seconds) callback.

    Returns:
        True on success, False on failure.
    """
    builder = AudioBuilder(config, progress_callback=progress_callback)
    return builder.run()


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    # Data models
    "AppConfig",
    "ProcessingConfig",
    "Chapter",
    "Segment",
    "AudioResult",
    # Multi-voice / NLP
    "NLPResult",
    "CharacterInfo",
    "NarrationMode",
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
    # HuggingFace auth (low-level)
    "ensure_huggingface_access",
    "is_custom_voice",
    "check_voice_access",
    # Config API
    "load_config",
    "save_config",
    "list_configs",
    # Book API
    "parse_book",
    "filter_chapters",
    "fast_scan",
    # Voice API
    "list_voices",
    "get_voice",
    "suggest_cast",
    "recommend_narrator",
    "exclude_voice",
    "include_voice",
    "audition_voice",
    "download_voice",
    "fetch_voice",
    # Series API
    "list_series",
    "get_series",
    "create_series",
    "update_series",
    # Auth API
    "authenticate_huggingface",
    # Job runner
    "run_job",
    # Package metadata
    "__version__",
    "__author__",
    "__license__",
]
