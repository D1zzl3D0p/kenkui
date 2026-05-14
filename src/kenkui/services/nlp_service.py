"""nlp_service — provider-aware NLP service layer.

Public API:
  fast_scan(ebook_path, nlp_model, config_path, progress_callback) -> FastScanResult
  full_analysis(ebook_path, nlp_model, config_path, progress_callback) -> NLPResult
  attribute_only(roster, chapters, ebook_path, nlp_model, nlp_provider, ...) -> NLPResult

The uniform service-layer progress callback is ``Callable[[int, str], None]``
(percent: int, message: str).  The underlying provider callbacks use
``Callable[[str], None]`` (message only).  This module adapts between the two.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import replace as _replace
from pathlib import Path

from kenkui.config import load_app_config
from kenkui.models import (
    CharacterInfo,
    CharacterRecord as AppCharacterRecord,
    FastScanResult,
    NLPResult,
)
from kenkui.nlp import (
    _attribution_to_segments,
    book_hash,
    cache_result,
    cache_roster,
    get_cached_result,
    get_cached_roster,
)
from kenkui.nlp_config import NLPConfig
from kenkui.nlp.pipeline import NLPPipeline
from kenkui.readers import get_reader

# Adapter constants for progress-callback translation (provider string-only → int+str).
_FAST_SCAN_START_PCT = 10
_FAST_SCAN_BUMP = 15
_FAST_SCAN_CAP = 90
_FULL_ROSTER_START_PCT = 5
_FULL_ROSTER_BUMP = 8
_FULL_ROSTER_CAP = 45
_FULL_ATTRIB_START_PCT = 50
_FULL_ATTRIB_CAP = 95


def _make_adapter(
    progress_callback: Callable[[int, str], None] | None,
    start_pct: int,
    bump: int,
    cap: int,
) -> Callable[[str], None]:
    """Return a string-only callback that maps progress to the int+str service callback."""
    _pct = [start_pct]

    def _adapt(msg: str) -> None:
        _pct[0] = min(cap, _pct[0] + bump)
        if progress_callback:
            progress_callback(_pct[0], msg)

    return _adapt


def fast_scan(
    ebook_path: str,
    nlp_model: str | None = None,
    config_path: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    series_slug: str | None = None,
    book_slug: str | None = None,
    nlp_provider: str | None = None,
    discovery_method: str | None = None,
    use_cache: bool = True,
) -> FastScanResult:
    """Run Stage 1-2 NLP (quote extraction + entity clustering + mention counting).

    Args:
        ebook_path:        Path to the source ebook file.
        nlp_model:         Override model name.  Falls back to AppConfig.nlp_model.
        config_path:       Optional path/name for the kenkui config file.
        progress_callback: Optional ``(percent: int, message: str) -> None``.
        series_slug:       If provided, load the series CharacterRoster before the
                           scan and merge new characters back into it afterward.
        book_slug:         Slug for the current book (used for first_appearance
                           tracking when *series_slug* is set).
        nlp_provider:      Override provider name.  Falls back to AppConfig.nlp_provider.

    Returns:
        ``FastScanResult`` with characters sorted by mention_count descending.

    Raises:
        FileNotFoundError: if *ebook_path* does not exist on disk.
    """
    if not Path(ebook_path).exists():
        raise FileNotFoundError(f"Ebook not found: {ebook_path}")

    if progress_callback:
        progress_callback(0, "Parsing ebook")

    cfg = load_app_config(config_path)
    if nlp_model is not None:
        cfg = cfg.model_copy(update={"nlp_model": nlp_model})
    if nlp_provider is not None:
        cfg = cfg.model_copy(update={"nlp_provider": nlp_provider})
    if discovery_method is not None:
        cfg = cfg.model_copy(update={"nlp_discovery_method": discovery_method})

    _discovery_method = getattr(cfg, "nlp_discovery_method", "auto") or "auto"

    # Return cached result before the expensive parse + NLP pass.
    if use_cache:
        cached = get_cached_roster(Path(ebook_path), method=_discovery_method if _discovery_method != "auto" else None, provider=nlp_provider)
        if cached is not None:
            if progress_callback:
                progress_callback(100, "Scan complete (cached)")
            return cached

    reader = get_reader(Path(ebook_path))
    chapters = reader.get_chapters()

    if progress_callback:
        progress_callback(_FAST_SCAN_START_PCT, "Starting NLP scan")

    # Fetch existing series roster before extraction so providers can inject it.
    series_roster = None
    if series_slug:
        from kenkui.services.series_service import get_roster as _get_roster
        series_roster = _get_roster(series_slug)

    nlp_config = NLPConfig.from_app_config(cfg)
    pipeline = NLPPipeline(nlp_config)
    roster = pipeline.extract(
        book_path=Path(ebook_path),
        chapters=chapters,
        series_roster=series_roster,
        progress_callback=progress_callback,
        use_cache=False,  # nlp_service handles its own roster cache above
    )

    # Update series roster with newly discovered characters.
    if series_slug and book_slug:
        from kenkui.services.series_service import update_roster as _update_roster
        _update_roster(series_slug, roster, book_slug)

    characters: list[CharacterInfo] = [
        AppCharacterRecord.from_nlp(rec).to_character_info()
        for rec in roster.characters
    ]
    characters.sort(key=lambda c: c.mention_count or c.quote_count, reverse=True)

    result = FastScanResult(
        roster=roster,
        characters=characters,
        book_hash=book_hash(Path(ebook_path)),
    )
    cache_roster(
        result,
        Path(ebook_path),
        method=_discovery_method,
        provider=cfg.nlp_provider,
        model=cfg.nlp_model,
    )

    if progress_callback:
        progress_callback(100, "Scan complete")

    return result


def full_analysis(
    ebook_path: str,
    nlp_model: str | None = None,
    config_path: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    series_slug: str | None = None,
    book_slug: str | None = None,
    discovery_method: str | None = None,
    attribution_provider: str | None = None,
    attribution_model: str | None = None,
    use_cache: bool = True,
) -> NLPResult:
    """Run the full NLP speaker-attribution pipeline.

    Args:
        ebook_path:        Path to the source ebook file.
        nlp_model:         Override model name.  Falls back to AppConfig.nlp_model.
        config_path:       Optional path/name for the kenkui config file.
        progress_callback: Optional ``(percent: int, message: str) -> None``.
        series_slug:       If provided, load the series CharacterRoster before the
                           analysis and merge new characters back into it afterward.
        book_slug:         Slug for the current book (used for first_appearance
                           tracking when *series_slug* is set).

    Returns:
        ``NLPResult`` with both ``mention_count`` and ``quote_count`` populated.

    Raises:
        FileNotFoundError: if *ebook_path* does not exist on disk.
    """
    if not Path(ebook_path).exists():
        raise FileNotFoundError(f"Ebook not found: {ebook_path}")

    if progress_callback:
        progress_callback(0, "Parsing ebook")

    cfg = load_app_config(config_path)
    if nlp_model is not None:
        cfg = cfg.model_copy(update={"nlp_model": nlp_model})
    if discovery_method is not None:
        cfg = cfg.model_copy(update={"nlp_discovery_method": discovery_method})
    if attribution_provider is not None:
        cfg = cfg.model_copy(update={"nlp_attribution_provider": attribution_provider})
    if attribution_model is not None:
        cfg = cfg.model_copy(update={"nlp_attribution_model": attribution_model})

    _attr_provider_name = getattr(cfg, "nlp_attribution_provider", "") or cfg.nlp_provider

    # Return cached result before the expensive parse + NLP pass.
    if use_cache:
        cached = get_cached_result(Path(ebook_path), provider=_attr_provider_name)
        if cached is not None:
            if progress_callback:
                progress_callback(100, "Analysis complete (cached)")
            return cached

    reader = get_reader(Path(ebook_path))
    chapters = reader.get_chapters()

    if progress_callback:
        progress_callback(_FULL_ROSTER_START_PCT, "Starting NLP analysis")

    # Fetch existing series roster before extraction so providers can inject it.
    series_roster = None
    if series_slug:
        from kenkui.services.series_service import get_roster as _get_roster
        series_roster = _get_roster(series_slug)

    nlp_config = NLPConfig.from_app_config(cfg)
    pipeline = NLPPipeline(nlp_config)

    # Phase 1: Build character roster (5–45 %)
    roster = pipeline.extract(
        book_path=Path(ebook_path),
        chapters=chapters,
        series_roster=series_roster,
        progress_callback=progress_callback,
        use_cache=False,  # nlp_service handles its own cache
    )

    # Update series roster with newly discovered characters.
    if series_slug and book_slug:
        from kenkui.services.series_service import update_roster as _update_roster
        _update_roster(series_slug, roster, book_slug)

    if progress_callback:
        progress_callback(_FULL_ATTRIB_START_PCT, "Attributing dialogue")

    # Phase 2: Attribute each chapter (50–95 %)
    attrib_bump = max(1, (_FULL_ATTRIB_CAP - _FULL_ATTRIB_START_PCT) // max(1, len(chapters)))
    attrib_adapt = _make_adapter(
        progress_callback, _FULL_ATTRIB_START_PCT, attrib_bump, _FULL_ATTRIB_CAP
    )

    attribution_counts: dict[str, int] = defaultdict(int)
    attributed_chapters = []

    for chapter in chapters:
        attr_result = pipeline._attribution.attribute_chapter(chapter, roster, progress_callback=attrib_adapt)
        segments = _attribution_to_segments(chapter, attr_result, roster)
        attributed_chapters.append(_replace(chapter, segments=segments))
        for item in attr_result.attributions:
            if item.speaker not in ("NARRATOR", "Unknown"):
                attribution_counts[item.speaker] += 1

    # Build CharacterInfo list with both mention_count and quote_count.
    characters: list[CharacterInfo] = []
    for rec in roster.characters:
        ci = AppCharacterRecord.from_nlp(rec).to_character_info()
        ci.quote_count = attribution_counts.get(rec.slug, 0)
        characters.append(ci)
    characters.sort(key=lambda c: c.prominence, reverse=True)

    result = NLPResult(
        characters=characters,
        chapters=attributed_chapters,
        book_hash=book_hash(Path(ebook_path)),
    )
    cache_result(result, Path(ebook_path), provider=_attr_provider_name)

    if progress_callback:
        progress_callback(100, "Analysis complete")

    return result


def attribute_only(
    roster: "CharacterRoster",
    chapters: list,
    ebook_path: str,
    nlp_model: str | None = None,
    nlp_provider: str | None = None,
    config_path: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    attribution_provider: str | None = None,
    attribution_model: str | None = None,
) -> "NLPResult":
    """Run Stage 3-4 speaker attribution against a pre-built roster.

    Used by the job worker when the roster already exists (from a prior scan).
    Uses the configured provider (cloud or ollama) rather than the hardcoded
    legacy ``run_attribution`` path.

    Args:
        roster:            Pre-built ``CharacterRoster`` from a prior scan.
        chapters:          List of ``Chapter`` objects (paragraphs populated).
        ebook_path:        Path to the source ebook (used as cache key).
        nlp_model:         Override model name.
        nlp_provider:      Override provider name.
        config_path:       Optional kenkui config path.
        progress_callback: Optional ``(percent: int, message: str) -> None``.

    Returns:
        ``NLPResult`` with attributed chapters and quote counts.
    """
    cfg = load_app_config(config_path)
    if nlp_model is not None:
        cfg = cfg.model_copy(update={"nlp_model": nlp_model})
    if nlp_provider is not None:
        cfg = cfg.model_copy(update={"nlp_provider": nlp_provider})
    if attribution_provider is not None:
        cfg = cfg.model_copy(update={"nlp_attribution_provider": attribution_provider})
    if attribution_model is not None:
        cfg = cfg.model_copy(update={"nlp_attribution_model": attribution_model})

    _effective_provider = getattr(cfg, "nlp_attribution_provider", "") or cfg.nlp_provider

    if progress_callback:
        progress_callback(5, "Starting attribution")

    nlp_config = NLPConfig.from_app_config(cfg)
    pipeline = NLPPipeline(nlp_config)
    result = pipeline.attribute(
        book_path=Path(ebook_path),
        chapters=chapters,
        roster=roster,
        progress_callback=progress_callback,
        use_cache=False,
    )

    # Override cache with the effective provider name used by nlp_service
    cache_result(result, Path(ebook_path), provider=_effective_provider)

    if progress_callback:
        progress_callback(100, "Attribution complete")

    return result
