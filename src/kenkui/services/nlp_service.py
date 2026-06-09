"""nlp_service — provider-aware NLP service layer.

Public API:
  fast_scan(ebook_path, nlp_model, config_path, progress_callback) -> FastScanResult
  full_analysis(ebook_path, nlp_model, config_path, extraction_progress_callback, attribution_progress_callback) -> NLPResult
  attribute_only(roster, chapters, ebook_path, nlp_model, nlp_provider, ...) -> NLPResult

The legacy service-layer progress callback is ``Callable[[int, str], None]``
(percent: int, message: str). Structured ``ProgressEvent`` callbacks are also
supported for clients that need display-neutral work units.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from dataclasses import replace as _replace
from pathlib import Path
from typing import TYPE_CHECKING

from kenkui.analytics import StageRecord, append_record, now_utc
from kenkui.config import load_app_config
from kenkui.models import (
    CharacterInfo,
    FastScanResult,
    NLPResult,
)
from kenkui.models import (
    CharacterRecord as AppCharacterRecord,
)
from kenkui.nlp import (
    _attribution_to_segments,
    book_hash,
    cache_result,
    cache_roster,
    get_cached_result,
    get_cached_roster,
)
from kenkui.nlp.pipeline import NLPPipeline
from kenkui.nlp_config import NLPConfig
from kenkui.progress import ProgressEvent
from kenkui.readers import get_reader

if TYPE_CHECKING:
    from kenkui.nlp.models import CharacterRoster


@dataclass
class ProgressTracker:
    total: int
    callback: Callable[[int, str], None] | None
    _done: int = field(default=0, init=False)

    def advance(self, msg: str) -> None:
        self._done = min(self._done + 1, self.total)
        if self.callback:
            pct = int(self._done / self.total * 100) if self.total else 100
            self.callback(pct, msg)


_ATTRIBUTION_JOBS_RE = re.compile(r"Attribution jobs \[(\d+)/(\d+)\]")


def _completed_attribution_units(message: str, fallback: float) -> float:
    match = _ATTRIBUTION_JOBS_RE.search(message)
    if match is None:
        return fallback
    return float(match.group(1))


def _chapter_label(chapter) -> str:
    return chapter.title or f"Chapter {chapter.index}"


def _emit_nlp_progress(
    callback: Callable[[ProgressEvent], None] | None,
    *,
    stage: str,
    status: str,
    message: str,
    completed_units: float,
    total_units: float,
    provider: str = "",
    model: str = "",
    book_hash_value: str = "",
    active_chapters: tuple = (),
) -> None:
    if callback is None:
        return
    callback(
        ProgressEvent(
            stage=stage,
            status=status,
            message=message,
            completed_units=completed_units,
            total_units=total_units,
            unit="chapters",
            provider=provider,
            model=model,
            book_hash=book_hash_value,
            active_chapters=active_chapters,
        )
    )


def _extraction_step_count(method: str) -> int:
    if method == "spacy":
        return 1   # one synthetic step at completion
    if method == "booknlp":
        return 4   # extract + deduplicate + resolve_epithets + normalize
    return 4       # worst-case upper bound for auto/llm paths


def fast_scan(
    ebook_path: str,
    nlp_model: str | None = None,
    config_path: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    progress_event_callback: Callable[[ProgressEvent], None] | None = None,
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
        progress_event_callback: Optional structured ``ProgressEvent`` callback.
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

    cfg = load_app_config(config_path)
    if nlp_model is not None:
        cfg = cfg.model_copy(update={"nlp_model": nlp_model})
    if nlp_provider is not None:
        cfg = cfg.model_copy(update={"nlp_provider": nlp_provider})
    if discovery_method is not None:
        cfg = cfg.model_copy(update={"nlp_discovery_method": discovery_method})

    _discovery_method = getattr(cfg, "nlp_discovery_method", "auto") or "auto"
    provider_name = cfg.nlp_provider or ""
    model_name = cfg.nlp_model or ""
    ebook = Path(ebook_path)
    ebook_hash = book_hash(ebook)

    # Return cached result before the expensive parse + NLP pass.
    if use_cache:
        cached = get_cached_roster(
            ebook,
            method=_discovery_method if _discovery_method != "auto" else None,
            provider=provider_name,
            model=model_name,
        )
        if cached is not None:
            if progress_callback:
                progress_callback(100, "Scan complete (cached)")
            if progress_event_callback:
                try:
                    cached_chapters = get_reader(ebook).get_chapters()
                    cached_total = len(cached_chapters)
                except Exception:
                    cached_total = 1
                _emit_nlp_progress(
                    progress_event_callback,
                    stage="nlp_extraction",
                    status="completed",
                    message="Character discovery complete (cached)",
                    completed_units=cached_total,
                    total_units=cached_total,
                    provider=provider_name,
                    model=model_name,
                    book_hash_value=ebook_hash,
                )
            try:
                _cached_title = get_reader(ebook).get_metadata().title or ""
            except Exception:
                _cached_title = ""
            append_record(StageRecord(
                stage="nlp_extraction",
                started_at=now_utc(),
                duration_seconds=0.0,
                success=True,
                book_hash=ebook_hash,
                book_title=_cached_title,
                provider=provider_name,
                model=model_name,
                cache_hit=True,
            ))
            return cached

    reader = get_reader(ebook)
    chapters = reader.get_chapters()
    _book_char_count = sum(len(p) for ch in chapters for p in ch.paragraphs)
    try:
        _book_title = reader.get_metadata().title or ""
    except Exception:
        _book_title = ""

    _method = _discovery_method
    total = _extraction_step_count(_method)
    tracker = ProgressTracker(total, progress_callback)
    if progress_callback:
        progress_callback(0, "Starting extraction")
    chapter_total = len(chapters)
    _emit_nlp_progress(
        progress_event_callback,
        stage="nlp_extraction",
        status="started",
        message="Starting character discovery",
        completed_units=0,
        total_units=chapter_total,
        provider=provider_name,
        model=model_name,
        book_hash_value=ebook_hash,
    )

    # Fetch existing series roster before extraction so providers can inject it.
    series_roster = None
    if series_slug:
        from kenkui.services.series_service import get_roster as _get_roster
        series_roster = _get_roster(series_slug)

    nlp_config = NLPConfig.from_app_config(cfg)
    pipeline = NLPPipeline(nlp_config)
    for scanned_chapters, chapter in enumerate(chapters, start=1):
        completed = min(scanned_chapters, max(chapter_total - 1, 0))
        _emit_nlp_progress(
            progress_event_callback,
            stage="nlp_extraction",
            status="advanced",
            message=f"Scanning [{scanned_chapters}/{chapter_total}] {_chapter_label(chapter)}",
            completed_units=completed,
            total_units=chapter_total,
            provider=provider_name,
            model=model_name,
            book_hash_value=ebook_hash,
        )

    def _extraction_message(_pct: int, msg: str) -> None:
        _emit_nlp_progress(
            progress_event_callback,
            stage="nlp_extraction",
            status="advanced",
            message=msg,
            completed_units=max(chapter_total - 1, 0),
            total_units=chapter_total,
            provider=provider_name,
            model=model_name,
            book_hash_value=ebook_hash,
        )

    _extract_start = now_utc()
    _t0 = time.monotonic()
    roster = pipeline.extract(
        book_path=Path(ebook_path),
        chapters=chapters,
        series_roster=series_roster,
        progress_callback=_extraction_message,
        step_callback=tracker.advance,
        use_cache=False,  # nlp_service handles its own roster cache above
    )
    _extract_dur = time.monotonic() - _t0

    append_record(StageRecord(
        stage="nlp_extraction",
        started_at=_extract_start,
        duration_seconds=_extract_dur,
        success=True,
        book_hash=ebook_hash,
        book_title=_book_title,
        book_char_count=_book_char_count,
        chapter_count=len(chapters),
        provider=provider_name,
        model=model_name,
        cache_hit=False,
    ))

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
        book_hash=ebook_hash,
    )
    cache_roster(
        result,
        Path(ebook_path),
        method=_discovery_method,
        provider=cfg.nlp_provider,
        model=cfg.nlp_model,
    )

    if progress_callback:
        progress_callback(100, "Extraction complete")
    _emit_nlp_progress(
        progress_event_callback,
        stage="nlp_extraction",
        status="completed",
        message="Character discovery complete",
        completed_units=chapter_total,
        total_units=chapter_total,
        provider=provider_name,
        model=model_name,
        book_hash_value=ebook_hash,
    )

    return result


def full_analysis(
    ebook_path: str,
    nlp_model: str | None = None,
    nlp_provider: str | None = None,
    config_path: str | None = None,
    extraction_progress_callback: Callable[[int, str], None] | None = None,
    attribution_progress_callback: Callable[[int, str], None] | None = None,
    extraction_progress_event_callback: Callable[[ProgressEvent], None] | None = None,
    attribution_progress_event_callback: Callable[[ProgressEvent], None] | None = None,
    series_slug: str | None = None,
    book_slug: str | None = None,
    discovery_method: str | None = None,
    attribution_provider: str | None = None,
    attribution_model: str | None = None,
    openrouter_attribution_concurrency: int | None = None,
    attribution_max_quotes_per_call: int | None = None,
    attribution_review_confidence: bool | None = None,
    review_model: str | None = None,
    use_cache: bool = True,
) -> NLPResult:
    """Run the full NLP speaker-attribution pipeline.

    Args:
        ebook_path:                    Path to the source ebook file.
        nlp_model:                     Override model name.  Falls back to AppConfig.nlp_model.
        nlp_provider:                  Override provider name. Falls back to AppConfig.nlp_provider.
        config_path:                   Optional path/name for the kenkui config file.
        extraction_progress_callback:  Optional ``(percent: int, message: str) -> None``
                                       called during the extraction (roster-building) phase.
        attribution_progress_callback: Optional ``(percent: int, message: str) -> None``
                                       called during the attribution (dialogue-tagging) phase.
        extraction_progress_event_callback: Optional structured extraction progress callback.
        attribution_progress_event_callback: Optional structured attribution progress callback.
        series_slug:                   If provided, load the series CharacterRoster before the
                                       analysis and merge new characters back into it afterward.
        book_slug:                     Slug for the current book (used for first_appearance
                                       tracking when *series_slug* is set).

    Returns:
        ``NLPResult`` with both ``mention_count`` and ``quote_count`` populated.

    Raises:
        FileNotFoundError: if *ebook_path* does not exist on disk.
    """
    if not Path(ebook_path).exists():
        raise FileNotFoundError(f"Ebook not found: {ebook_path}")

    cfg = load_app_config(config_path)
    if nlp_model is not None:
        cfg = cfg.model_copy(update={"nlp_model": nlp_model})
    if nlp_provider is not None:
        cfg = cfg.model_copy(update={"nlp_provider": nlp_provider})
    if discovery_method is not None:
        cfg = cfg.model_copy(update={"nlp_discovery_method": discovery_method})
    if attribution_provider is not None:
        cfg = cfg.model_copy(update={"nlp_attribution_provider": attribution_provider})
    if attribution_model is not None:
        cfg = cfg.model_copy(update={"nlp_attribution_model": attribution_model})
    if openrouter_attribution_concurrency is not None:
        cfg = cfg.model_copy(update={
            "nlp_openrouter_attribution_concurrency": openrouter_attribution_concurrency
        })
    if attribution_max_quotes_per_call is not None:
        cfg = cfg.model_copy(update={
            "nlp_attribution_max_quotes_per_call": attribution_max_quotes_per_call
        })
    if attribution_review_confidence is not None:
        cfg = cfg.model_copy(update={
            "nlp_attribution_review_confidence": attribution_review_confidence
        })
    if review_model is not None:
        cfg = cfg.model_copy(update={"nlp_review_model": review_model})

    _attr_provider_name = getattr(cfg, "nlp_attribution_provider", "") or cfg.nlp_provider
    extraction_provider_name = cfg.nlp_provider or ""
    extraction_model_name = cfg.nlp_model or ""
    attribution_model_name = getattr(cfg, "nlp_attribution_model", None) or cfg.nlp_model or ""
    ebook = Path(ebook_path)
    ebook_hash = book_hash(ebook)

    # Return cached result before the expensive parse + NLP pass.
    if use_cache:
        cached = get_cached_result(
            ebook,
            provider=_attr_provider_name,
            model=attribution_model_name,
        )
        if cached is not None:
            if attribution_progress_callback:
                attribution_progress_callback(100, "Analysis complete (cached)")
            if attribution_progress_event_callback:
                cached_total = len(getattr(cached, "chapters", []) or []) or 1
                _emit_nlp_progress(
                    attribution_progress_event_callback,
                    stage="nlp_attribution",
                    status="completed",
                    message="Attribution complete (cached)",
                    completed_units=cached_total,
                    total_units=cached_total,
                    provider=_attr_provider_name or "",
                    model=attribution_model_name,
                    book_hash_value=ebook_hash,
                )
            try:
                _cached_title = get_reader(ebook).get_metadata().title or ""
            except Exception:
                _cached_title = ""
            append_record(StageRecord(
                stage="nlp_attribution",
                started_at=now_utc(),
                duration_seconds=0.0,
                success=True,
                book_hash=ebook_hash,
                book_title=_cached_title,
                provider=_attr_provider_name or "",
                model=attribution_model_name,
                cache_hit=True,
            ))
            return cached

    reader = get_reader(ebook)
    chapters = reader.get_chapters()
    _book_char_count = sum(len(p) for ch in chapters for p in ch.paragraphs)
    try:
        _book_title = reader.get_metadata().title or ""
    except Exception:
        _book_title = ""

    # Fetch existing series roster before extraction so providers can inject it.
    series_roster = None
    if series_slug:
        from kenkui.services.series_service import get_roster as _get_roster
        series_roster = _get_roster(series_slug)

    nlp_config = NLPConfig.from_app_config(cfg)
    pipeline = NLPPipeline(nlp_config)

    # Phase 1: Build character roster
    _method = getattr(cfg, "nlp_discovery_method", "auto") or "auto"
    extract_total = _extraction_step_count(_method)
    extract_tracker = ProgressTracker(extract_total, extraction_progress_callback)
    if extraction_progress_callback:
        extraction_progress_callback(0, "Starting extraction")
    chapter_total = len(chapters)
    _emit_nlp_progress(
        extraction_progress_event_callback,
        stage="nlp_extraction",
        status="started",
        message="Starting character discovery",
        completed_units=0,
        total_units=chapter_total,
        provider=extraction_provider_name,
        model=extraction_model_name,
        book_hash_value=ebook_hash,
    )

    roster = None
    if use_cache:
        cached_roster = get_cached_roster(
            ebook,
            method=_method if _method != "auto" else None,
            provider=cfg.nlp_provider,
            model=extraction_model_name,
        )
        if cached_roster is not None:
            roster = cached_roster.roster
            if extraction_progress_callback:
                extraction_progress_callback(100, "Extraction complete (cached)")
            _emit_nlp_progress(
                extraction_progress_event_callback,
                stage="nlp_extraction",
                status="completed",
                message="Character discovery complete (cached)",
                completed_units=chapter_total,
                total_units=chapter_total,
                provider=extraction_provider_name,
                model=extraction_model_name,
                book_hash_value=ebook_hash,
            )
            append_record(StageRecord(
                stage="nlp_extraction",
                started_at=now_utc(),
                duration_seconds=0.0,
                success=True,
                book_hash=ebook_hash,
                book_title=_book_title,
                book_char_count=_book_char_count,
                chapter_count=len(chapters),
                provider=extraction_provider_name,
                model=extraction_model_name,
                cache_hit=True,
            ))

    if roster is None:
        for scanned_chapters, chapter in enumerate(chapters, start=1):
            completed = min(scanned_chapters, max(chapter_total - 1, 0))
            _emit_nlp_progress(
                extraction_progress_event_callback,
                stage="nlp_extraction",
                status="advanced",
                message=f"Scanning [{scanned_chapters}/{chapter_total}] {_chapter_label(chapter)}",
                completed_units=completed,
                total_units=chapter_total,
                provider=extraction_provider_name,
                model=extraction_model_name,
                book_hash_value=ebook_hash,
            )

        def _extraction_message(_pct: int, msg: str) -> None:
            _emit_nlp_progress(
                extraction_progress_event_callback,
                stage="nlp_extraction",
                status="advanced",
                message=msg,
                completed_units=max(chapter_total - 1, 0),
                total_units=chapter_total,
                provider=extraction_provider_name,
                model=extraction_model_name,
                book_hash_value=ebook_hash,
            )

        _extract_start = now_utc()
        _t0 = time.monotonic()
        roster = pipeline.extract(
            book_path=ebook,
            chapters=chapters,
            series_roster=series_roster,
            progress_callback=_extraction_message,
            step_callback=extract_tracker.advance,
            use_cache=False,  # nlp_service handles its own cache
        )
        _extract_dur = time.monotonic() - _t0
        if extraction_progress_callback:
            extraction_progress_callback(100, "Extraction complete")
        _emit_nlp_progress(
            extraction_progress_event_callback,
            stage="nlp_extraction",
            status="completed",
            message="Character discovery complete",
            completed_units=chapter_total,
            total_units=chapter_total,
            provider=extraction_provider_name,
            model=extraction_model_name,
            book_hash_value=ebook_hash,
        )
        append_record(StageRecord(
            stage="nlp_extraction",
            started_at=_extract_start,
            duration_seconds=_extract_dur,
            success=True,
            book_hash=ebook_hash,
            book_title=_book_title,
            book_char_count=_book_char_count,
            chapter_count=len(chapters),
            provider=extraction_provider_name,
            model=extraction_model_name,
            cache_hit=False,
        ))

    # Update series roster with newly discovered characters.
    if series_slug and book_slug:
        from kenkui.services.series_service import update_roster as _update_roster
        _update_roster(series_slug, roster, book_slug)

    # Phase 2: Attribute each chapter
    attrib_tracker = ProgressTracker(total=len(chapters), callback=attribution_progress_callback)
    if attribution_progress_callback:
        attribution_progress_callback(0, "Starting attribution")
    _emit_nlp_progress(
        attribution_progress_event_callback,
        stage="nlp_attribution",
        status="started",
        message="Starting attribution",
        completed_units=0,
        total_units=chapter_total,
        provider=_attr_provider_name or "",
        model=attribution_model_name,
        book_hash_value=ebook_hash,
    )

    attribution_counts: dict[str, int] = defaultdict(int)
    attributed_chapters = []

    _attrib_start = now_utc()
    _t1 = time.monotonic()

    if (_attr_provider_name or "").lower() == "openrouter":
        last_completed = 0.0

        def _attribution_message(pct: int, msg: str) -> None:
            nonlocal last_completed
            if attribution_progress_callback:
                attribution_progress_callback(pct, msg)
            last_completed = _completed_attribution_units(msg, last_completed)
            _emit_nlp_progress(
                attribution_progress_event_callback,
                stage="nlp_attribution",
                status="advanced",
                message=msg,
                completed_units=last_completed,
                total_units=chapter_total,
                provider=_attr_provider_name or "",
                model=attribution_model_name,
                book_hash_value=ebook_hash,
            )

        result = pipeline.attribute(
            book_path=ebook,
            chapters=chapters,
            roster=roster,
            progress_callback=_attribution_message,
            use_cache=False,
        )
    else:
        for done, chapter in enumerate(chapters, start=1):
            attr_result = pipeline._attribution.attribute_chapter(chapter, roster, progress_callback=None)
            segments = _attribution_to_segments(chapter, attr_result, roster)
            attributed_chapters.append(_replace(chapter, segments=segments))
            for item in attr_result.attributions:
                if item.speaker not in ("NARRATOR", "Unknown"):
                    attribution_counts[item.speaker] += 1
            attrib_tracker.advance(chapter.title or f"Chapter {chapter.index}")
            _emit_nlp_progress(
                attribution_progress_event_callback,
                stage="nlp_attribution",
                status="advanced",
                message=f"Attributing [{done}/{chapter_total}] {_chapter_label(chapter)}",
                completed_units=done,
                total_units=chapter_total,
                provider=_attr_provider_name or "",
                model=attribution_model_name,
                book_hash_value=ebook_hash,
            )

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
            book_hash=ebook_hash,
        )
    _attrib_dur = time.monotonic() - _t1
    cache_result(
        result,
        Path(ebook_path),
        provider=_attr_provider_name,
        model=attribution_model_name,
    )

    append_record(StageRecord(
        stage="nlp_attribution",
        started_at=_attrib_start,
        duration_seconds=_attrib_dur,
        success=True,
        book_hash=ebook_hash,
        book_title=_book_title,
        book_char_count=_book_char_count,
        chapter_count=len(chapters),
        provider=_attr_provider_name or "",
        model=attribution_model_name,
        cache_hit=False,
    ))

    if attribution_progress_callback:
        attribution_progress_callback(100, "Attribution complete")
    _emit_nlp_progress(
        attribution_progress_event_callback,
        stage="nlp_attribution",
        status="completed",
        message="Attribution complete",
        completed_units=chapter_total,
        total_units=chapter_total,
        provider=_attr_provider_name or "",
        model=attribution_model_name,
        book_hash_value=ebook_hash,
    )

    return result


def attribute_only(
    roster: CharacterRoster,
    chapters: list,
    ebook_path: str,
    nlp_model: str | None = None,
    nlp_provider: str | None = None,
    config_path: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    progress_event_callback: Callable[[ProgressEvent], None] | None = None,
    attribution_provider: str | None = None,
    attribution_model: str | None = None,
    openrouter_attribution_concurrency: int | None = None,
    attribution_max_quotes_per_call: int | None = None,
    attribution_review_confidence: bool | None = None,
    review_model: str | None = None,
) -> NLPResult:
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
        progress_event_callback: Optional structured ``ProgressEvent`` callback.

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
    if openrouter_attribution_concurrency is not None:
        cfg = cfg.model_copy(update={
            "nlp_openrouter_attribution_concurrency": openrouter_attribution_concurrency
        })
    if attribution_max_quotes_per_call is not None:
        cfg = cfg.model_copy(update={
            "nlp_attribution_max_quotes_per_call": attribution_max_quotes_per_call
        })
    if attribution_review_confidence is not None:
        cfg = cfg.model_copy(update={
            "nlp_attribution_review_confidence": attribution_review_confidence
        })
    if review_model is not None:
        cfg = cfg.model_copy(update={"nlp_review_model": review_model})

    _effective_provider = getattr(cfg, "nlp_attribution_provider", "") or cfg.nlp_provider
    _effective_model = getattr(cfg, "nlp_attribution_model", None) or cfg.nlp_model or ""
    ebook = Path(ebook_path)
    ebook_hash = book_hash(ebook)

    nlp_config = NLPConfig.from_app_config(cfg)
    pipeline = NLPPipeline(nlp_config)

    tracker = ProgressTracker(len(chapters), progress_callback)
    if progress_callback:
        progress_callback(0, "Starting attribution")
    chapter_total = len(chapters)
    _emit_nlp_progress(
        progress_event_callback,
        stage="nlp_attribution",
        status="started",
        message="Starting attribution",
        completed_units=0,
        total_units=chapter_total,
        provider=_effective_provider or "",
        model=_effective_model,
        book_hash_value=ebook_hash,
    )

    attribution_counts: dict[str, int] = defaultdict(int)
    attributed_chapters = []

    if (_effective_provider or "").lower() == "openrouter":
        last_completed = 0.0

        def _attribution_message(pct: int, msg: str) -> None:
            nonlocal last_completed
            if progress_callback:
                progress_callback(pct, msg)
            last_completed = _completed_attribution_units(msg, last_completed)
            _emit_nlp_progress(
                progress_event_callback,
                stage="nlp_attribution",
                status="advanced",
                message=msg,
                completed_units=last_completed,
                total_units=chapter_total,
                provider=_effective_provider or "",
                model=_effective_model,
                book_hash_value=ebook_hash,
            )

        result = pipeline.attribute(
            book_path=ebook,
            chapters=chapters,
            roster=roster,
            progress_callback=_attribution_message,
            use_cache=False,
        )
    else:
        for done, chapter in enumerate(chapters, start=1):
            attr_result = pipeline._attribution.attribute_chapter(chapter, roster, progress_callback=None)
            segments = _attribution_to_segments(chapter, attr_result, roster)
            attributed_chapters.append(_replace(chapter, segments=segments))
            for item in attr_result.attributions:
                if item.speaker not in ("NARRATOR", "Unknown"):
                    attribution_counts[item.speaker] += 1
            tracker.advance(chapter.title or f"Chapter {chapter.index}")
            _emit_nlp_progress(
                progress_event_callback,
                stage="nlp_attribution",
                status="advanced",
                message=f"Attributing [{done}/{chapter_total}] {_chapter_label(chapter)}",
                completed_units=done,
                total_units=chapter_total,
                provider=_effective_provider or "",
                model=_effective_model,
                book_hash_value=ebook_hash,
            )

        # Build CharacterInfo list with quote counts from the just-run attribution.
        characters: list[CharacterInfo] = []
        for rec in roster.characters:
            ci = AppCharacterRecord.from_nlp(rec).to_character_info()
            ci.quote_count = attribution_counts.get(rec.slug, 0)
            characters.append(ci)
        characters.sort(key=lambda c: c.prominence, reverse=True)

        result = NLPResult(
            characters=characters,
            chapters=attributed_chapters,
            book_hash=ebook_hash,
        )

    cache_result(
        result,
        Path(ebook_path),
        provider=_effective_provider,
        model=_effective_model,
    )

    if progress_callback:
        progress_callback(100, "Attribution complete")
    _emit_nlp_progress(
        progress_event_callback,
        stage="nlp_attribution",
        status="completed",
        message="Attribution complete",
        completed_units=chapter_total,
        total_units=chapter_total,
        provider=_effective_provider or "",
        model=_effective_model,
        book_hash_value=ebook_hash,
    )

    return result
