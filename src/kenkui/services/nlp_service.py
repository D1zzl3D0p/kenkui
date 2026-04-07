"""nlp_service — thin wrapper around kenkui.nlp for the service layer.

Public API:
  fast_scan(ebook_path, nlp_model, config_path, progress_callback) -> FastScanResult
  full_analysis(ebook_path, nlp_model, config_path, progress_callback) -> NLPResult

The uniform service-layer progress callback is ``Callable[[int, str], None]``
(percent: int, message: str).  The underlying ``kenkui.nlp`` functions use
``Callable[[str], None]`` (message only).  This module adapts between the two.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from kenkui.config import load_app_config
from kenkui.models import FastScanResult, NLPResult
from kenkui.nlp import run_analysis, run_fast_scan
from kenkui.readers import get_reader

# Adapter constants for progress-callback translation (NLP string-only → int+str).
_FAST_SCAN_START_PCT = 10
_FAST_SCAN_BUMP = 15
_FAST_SCAN_CAP = 90
_FULL_ANALYSIS_START_PCT = 5
_FULL_ANALYSIS_BUMP = 8
_FULL_ANALYSIS_CAP = 95


def fast_scan(
    ebook_path: str,
    nlp_model: str | None = None,
    config_path: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> FastScanResult:
    """Run Stage 1-2 NLP (quote extraction + entity clustering + mention counting).

    Args:
        ebook_path:        Path to the source ebook file.
        nlp_model:         Ollama model name.  Falls back to AppConfig.nlp_model.
        config_path:       Optional path/name for the kenkui config file.
        progress_callback: Optional ``(percent: int, message: str) -> None``.

    Returns:
        ``FastScanResult`` with characters sorted by mention_count descending.

    Raises:
        FileNotFoundError: if *ebook_path* does not exist on disk.
    """
    if not Path(ebook_path).exists():
        raise FileNotFoundError(f"Ebook not found: {ebook_path}")

    if progress_callback:
        progress_callback(0, "Parsing ebook")

    if nlp_model is None:
        nlp_model = load_app_config(config_path).nlp_model

    reader = get_reader(Path(ebook_path))
    chapters = reader.get_chapters()

    if progress_callback:
        progress_callback(_FAST_SCAN_START_PCT, "Starting NLP scan")

    _pct = [_FAST_SCAN_START_PCT]

    def _adapt(msg: str) -> None:
        _pct[0] = min(_FAST_SCAN_CAP, _pct[0] + _FAST_SCAN_BUMP)
        if progress_callback:
            progress_callback(_pct[0], msg)

    result = run_fast_scan(chapters, Path(ebook_path), nlp_model, progress_callback=_adapt)

    if progress_callback:
        progress_callback(100, "Scan complete")

    return result


def full_analysis(
    ebook_path: str,
    nlp_model: str | None = None,
    config_path: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> NLPResult:
    """Run the full NLP speaker-attribution pipeline.

    Args:
        ebook_path:        Path to the source ebook file.
        nlp_model:         Ollama model name.  Falls back to AppConfig.nlp_model.
        config_path:       Optional path/name for the kenkui config file.
        progress_callback: Optional ``(percent: int, message: str) -> None``.

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
    if nlp_model is None:
        nlp_model = cfg.nlp_model

    reader = get_reader(Path(ebook_path))
    chapters = reader.get_chapters()

    if progress_callback:
        progress_callback(_FULL_ANALYSIS_START_PCT, "Starting NLP analysis")

    _pct = [_FULL_ANALYSIS_START_PCT]

    def _adapt(msg: str) -> None:
        _pct[0] = min(_FULL_ANALYSIS_CAP, _pct[0] + _FULL_ANALYSIS_BUMP)
        if progress_callback:
            progress_callback(_pct[0], msg)

    result = run_analysis(
        chapters,
        Path(ebook_path),
        nlp_model,
        progress_callback=_adapt,
        confidence_threshold=cfg.nlp_confidence_threshold,
        review_model=cfg.nlp_review_model,
    )

    if progress_callback:
        progress_callback(100, "Analysis complete")

    return result
