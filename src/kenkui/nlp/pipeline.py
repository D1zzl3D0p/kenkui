"""NLPPipeline — high-level orchestrator for the kenkui NLP pipeline.

Wraps ExtractionProvider + AttributionProvider with:
- Cache read/write (via kenkui.nlp._cache)
- Retry logic (via kenkui.nlp._retry)
- SIGTERM handling during long-running provider calls
- Thread-based async job API (NLPJob)
- Pre-flight tool validation (validate_tools)
"""

from __future__ import annotations

import dataclasses
import signal
import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from kenkui.nlp._cache import get_cache, put_cache
from kenkui.nlp._retry import with_retry
from kenkui.nlp import _attribution_to_segments, book_hash

if TYPE_CHECKING:
    from kenkui.models import NLPResult, Chapter
    from kenkui.nlp.models import CharacterRoster
    from kenkui.nlp_config import NLPConfig


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------


class NLPJobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of a pre-flight tool availability check."""

    tool: str    # e.g. "ollama", "booknlp", "litellm"
    step: str    # "extraction" or "attribution"
    ok: bool
    message: str  # Human-readable status


# ---------------------------------------------------------------------------
# NLPJob
# ---------------------------------------------------------------------------


@dataclass
class NLPJob:
    """A running or completed NLP pipeline job.

    Wraps a daemon thread and exposes poll()/wait()/cancel() for callers
    that want non-blocking progress tracking.
    """

    job_id: str
    status: NLPJobStatus
    progress: int   # 0-100
    message: str
    result: "CharacterRoster | NLPResult | None"  # populated when DONE
    error: Exception | None                        # populated when FAILED

    # Internal — excluded from poll() snapshots
    _thread: threading.Thread | None = field(default=None, repr=False)
    _cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def poll(self) -> "NLPJob":
        """Return a frozen snapshot of current state (non-blocking)."""
        return dataclasses.replace(self, _thread=None)

    def wait(self, timeout: float | None = None) -> "NLPJob":
        """Block until done/failed or timeout. Returns snapshot."""
        if self._thread:
            self._thread.join(timeout=timeout)
        return self.poll()

    def cancel(self) -> None:
        """Request cancellation. Sets the cancel event; the running thread checks it."""
        self._cancel_event.set()


# ---------------------------------------------------------------------------
# NLPPipeline
# ---------------------------------------------------------------------------


class NLPPipeline:
    """Orchestrates extraction and attribution with caching, retry, and job support."""

    def __init__(self, config: "NLPConfig") -> None:
        self._config = config
        from kenkui.nlp.providers._factory import (
            get_extraction_provider,
            get_attribution_provider,
        )
        self._extraction = get_extraction_provider(config)
        self._attribution = get_attribution_provider(config)

    # ------------------------------------------------------------------
    # validate_tools
    # ------------------------------------------------------------------

    def validate_tools(self) -> list[ValidationResult]:
        """Pre-flight check for required NLP tool packages.

        Never raises — all errors become ValidationResult(ok=False, ...).
        Returns one ValidationResult per step (extraction, attribution).
        """
        from kenkui.models import ExtractionTool, AttributionTool

        results: list[ValidationResult] = []

        # --- Extraction tool ---
        extraction_tool = self._config.extraction_tool.value
        try:
            match self._config.extraction_tool:
                case ExtractionTool.OLLAMA:
                    import ollama as _ollama  # noqa: F401
                case ExtractionTool.BOOKNLP:
                    from booknlp.booknlp import BookNLP as _bnlp  # noqa: F401
                case ExtractionTool.LITELLM:
                    import litellm as _ll  # noqa: F401
            results.append(ValidationResult(
                tool=extraction_tool, step="extraction", ok=True, message="Available"
            ))
        except ImportError as exc:
            results.append(ValidationResult(
                tool=extraction_tool, step="extraction", ok=False, message=str(exc)
            ))

        # --- Attribution tool ---
        attribution_tool = self._config.attribution_tool.value
        try:
            match self._config.attribution_tool:
                case AttributionTool.OLLAMA:
                    import ollama as _ollama  # noqa: F401
                case AttributionTool.BOOKNLP:
                    from booknlp.booknlp import BookNLP as _bnlp  # noqa: F401
                case AttributionTool.LITELLM:
                    import litellm as _ll  # noqa: F401
            results.append(ValidationResult(
                tool=attribution_tool, step="attribution", ok=True, message="Available"
            ))
        except ImportError as exc:
            results.append(ValidationResult(
                tool=attribution_tool, step="attribution", ok=False, message=str(exc)
            ))

        return results

    # ------------------------------------------------------------------
    # extract
    # ------------------------------------------------------------------

    def extract(
        self,
        book_path: Path,
        chapters: list["Chapter"],
        series_roster: "CharacterRoster | None" = None,
        progress_callback: Callable[[int, str], None] | None = None,
        step_callback: Callable[[str], None] | None = None,
        use_cache: bool = True,
    ) -> "CharacterRoster":
        """Run Stage 1-2: character extraction + coreference resolution.

        Checks cache first (if use_cache), runs the extraction provider with
        retry + SIGTERM handling, writes result to cache.

        Returns:
            CharacterRoster — the extracted character roster.
        """
        from kenkui.nlp.models import CharacterRoster

        tool = self._config.extraction_tool.value
        model = self._config.extraction_model

        # Cache read
        if use_cache:
            cached = get_cache(book_path, step="extraction", tool=tool, model=model)
            if cached is not None:
                try:
                    return CharacterRoster.model_validate(cached)
                except Exception:
                    pass  # Corrupt/stale cache — fall through to fresh run

        # Build progress adapter: (int, str) → str
        _pct = [0]

        def _adapt(msg: str) -> None:
            _pct[0] = min(90, _pct[0] + 5)
            if progress_callback:
                progress_callback(_pct[0], msg)

        # SIGTERM handling
        _orig_sigterm = signal.getsignal(signal.SIGTERM)

        def _sigterm_handler(signum: int, frame: object) -> None:
            raise KeyboardInterrupt("SIGTERM received")

        signal.signal(signal.SIGTERM, _sigterm_handler)
        try:
            roster: CharacterRoster = with_retry(
                self._extraction.build_roster,
                max_attempts=self._config.retry_max_attempts,
                backoff_base=self._config.retry_backoff_base,
            )(
                chapters,
                series_roster=series_roster,
                progress_callback=None if step_callback is not None else _adapt,
                step_callback=step_callback,
                book_path=book_path,
            )
        finally:
            signal.signal(signal.SIGTERM, _orig_sigterm)

        # Cache write
        try:
            put_cache(
                roster.model_dump(),
                book_path,
                step="extraction",
                tool=tool,
                model=model,
            )
        except Exception:
            pass  # Cache write failure is non-fatal

        return roster

    # ------------------------------------------------------------------
    # attribute
    # ------------------------------------------------------------------

    def attribute(
        self,
        book_path: Path,
        chapters: list["Chapter"],
        roster: "CharacterRoster",
        progress_callback: Callable[[int, str], None] | None = None,
        use_cache: bool = True,
    ) -> "NLPResult":
        """Run Stage 3-4: LLM speaker attribution using a pre-built roster.

        Checks cache first (if use_cache), runs attribution per chapter
        with retry + SIGTERM handling, writes result to cache.

        Returns:
            NLPResult — chapters with segments and character quote counts.
        """
        from dataclasses import replace as _replace

        from kenkui.models import CharacterInfo, CharacterRecord as AppCharacterRecord, NLPResult

        tool = self._config.attribution_tool.value
        model = self._config.attribution_model

        # Cache read
        if use_cache:
            cached = get_cache(book_path, step="attribution", tool=tool, model=model)
            if cached is not None:
                try:
                    return NLPResult.from_dict(cached)
                except Exception:
                    pass  # Corrupt/stale cache — fall through to fresh run

        # Build progress adapter
        total_chapters = max(1, len(chapters))
        _bump = max(1, 80 // total_chapters)
        _pct = [10]

        def _adapt(msg: str) -> None:
            _pct[0] = min(90, _pct[0] + _bump)
            if progress_callback:
                progress_callback(_pct[0], msg)

        # SIGTERM handling
        _orig_sigterm = signal.getsignal(signal.SIGTERM)

        def _sigterm_handler(signum: int, frame: object) -> None:
            raise KeyboardInterrupt("SIGTERM received")

        signal.signal(signal.SIGTERM, _sigterm_handler)
        try:
            attribution_counts: dict[str, int] = defaultdict(int)
            attributed_chapters = []

            for chapter in chapters:
                attr_result = with_retry(
                    self._attribution.attribute_chapter,
                    max_attempts=self._config.retry_max_attempts,
                    backoff_base=self._config.retry_backoff_base,
                )(chapter, roster, progress_callback=_adapt)

                segments = _attribution_to_segments(chapter, attr_result, roster)
                attributed_chapters.append(_replace(chapter, segments=segments))

                for item in attr_result.attributions:
                    if item.speaker not in ("NARRATOR", "Unknown"):
                        attribution_counts[item.speaker] += 1
        finally:
            signal.signal(signal.SIGTERM, _orig_sigterm)

        # Build CharacterInfo with quote_count
        characters: list[CharacterInfo] = []
        for rec in roster.characters:
            ci = AppCharacterRecord.from_nlp(rec).to_character_info()
            ci.quote_count = attribution_counts.get(rec.slug, 0)
            characters.append(ci)
        characters.sort(key=lambda c: c.prominence, reverse=True)

        result = NLPResult(
            characters=characters,
            chapters=attributed_chapters,
            book_hash=book_hash(book_path),
        )

        # Cache write
        try:
            put_cache(
                result.to_dict(),
                book_path,
                step="attribution",
                tool=tool,
                model=model,
            )
        except Exception:
            pass  # Cache write failure is non-fatal

        return result

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(
        self,
        book_path: Path,
        chapters: list["Chapter"],
        series_roster: "CharacterRoster | None" = None,
        progress_callback: Callable[[int, str], None] | None = None,
        use_cache: bool = True,
    ) -> "NLPResult":
        """Run the full pipeline: extraction then attribution.

        Returns:
            NLPResult with attributed chapters and character quote counts.
        """
        roster = self.extract(book_path, chapters, series_roster, progress_callback, use_cache)
        return self.attribute(book_path, chapters, roster, progress_callback, use_cache)

    # ------------------------------------------------------------------
    # extract_job
    # ------------------------------------------------------------------

    def extract_job(
        self,
        book_path: Path,
        chapters: list["Chapter"],
        series_roster: "CharacterRoster | None" = None,
        use_cache: bool = True,
    ) -> NLPJob:
        """Start extraction in a background thread and return an NLPJob immediately."""
        import uuid

        job = NLPJob(
            job_id=str(uuid.uuid4()),
            status=NLPJobStatus.PENDING,
            progress=0,
            message="Queued",
            result=None,
            error=None,
        )

        def _progress(pct: int, msg: str) -> None:
            job.progress = pct
            job.message = msg

        def _run() -> None:
            job.status = NLPJobStatus.RUNNING
            try:
                result = self.extract(book_path, chapters, series_roster, _progress, use_cache)
                job.result = result
                job.status = NLPJobStatus.DONE
                job.progress = 100
                job.message = "Done"
            except Exception as exc:
                job.error = exc
                job.status = NLPJobStatus.FAILED
                job.message = str(exc)

        t = threading.Thread(target=_run, daemon=True)
        job._thread = t
        t.start()
        return job

    # ------------------------------------------------------------------
    # attribute_job
    # ------------------------------------------------------------------

    def attribute_job(
        self,
        book_path: Path,
        chapters: list["Chapter"],
        roster: "CharacterRoster",
        use_cache: bool = True,
    ) -> NLPJob:
        """Start attribution in a background thread and return an NLPJob immediately."""
        import uuid

        job = NLPJob(
            job_id=str(uuid.uuid4()),
            status=NLPJobStatus.PENDING,
            progress=0,
            message="Queued",
            result=None,
            error=None,
        )

        def _progress(pct: int, msg: str) -> None:
            job.progress = pct
            job.message = msg

        def _run() -> None:
            job.status = NLPJobStatus.RUNNING
            try:
                result = self.attribute(book_path, chapters, roster, _progress, use_cache)
                job.result = result
                job.status = NLPJobStatus.DONE
                job.progress = 100
                job.message = "Done"
            except Exception as exc:
                job.error = exc
                job.status = NLPJobStatus.FAILED
                job.message = str(exc)

        t = threading.Thread(target=_run, daemon=True)
        job._thread = t
        t.start()
        return job


__all__ = [
    "NLPJobStatus",
    "ValidationResult",
    "NLPJob",
    "NLPPipeline",
]
