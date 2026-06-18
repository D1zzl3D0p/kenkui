"""NLPPipeline — high-level orchestrator for the kenkui NLP pipeline.

Wraps ExtractionProvider + AttributionProvider with:
- Cache read/write (via kenkui.nlp._cache)
- Retry logic (via kenkui.nlp._retry)
- SIGTERM handling during long-running provider calls
- Thread-based async job API (NLPJob)
- Pre-flight tool validation (validate_tools)
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import logging
import re
import signal
import threading
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from kenkui.nlp import _attribution_to_segments, book_hash
from kenkui.nlp._cache import (
    clear_checkpoints,
    get_cache,
    get_chapter_checkpoint,
    put_cache,
    put_chapter_checkpoint,
)
from kenkui.nlp._filters import _PRONOUNS
from kenkui.nlp._retry import with_retry
from kenkui.nlp.models import _SPEAKER_SENTINELS
from kenkui.nlp.models import slugify as _slugify

if TYPE_CHECKING:
    from kenkui.models import Chapter, NLPResult
    from kenkui.nlp.models import CharacterRoster
    from kenkui.nlp_config import NLPConfig

_logger = logging.getLogger(__name__)


def _run_coroutine_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


def _chapter_label(chapter: Chapter) -> str:
    return chapter.title or f"Chapter {chapter.index}"


def _format_attribution_jobs(
    active: dict[int, tuple[Chapter, str]],
    *,
    completed: int,
    total: int,
) -> str:
    if not active:
        return f"Attribution jobs [{completed}/{total}]"

    width = len(str(total))
    lines = [f"Attribution jobs [{completed}/{total}]"]
    for position, (chapter, status) in sorted(active.items()):
        lines.append(
            f"  [{position:>{width}}/{total}] running  {_chapter_label(chapter)} - {status}"
        )
    return "\n".join(lines)


def _store_attribution_checkpoint(
    *,
    book_path: Path,
    chapter: Chapter,
    attr_result: object,
    tool: str,
    model: str,
) -> None:
    try:
        put_chapter_checkpoint(
            {"attribution_result": attr_result.model_dump()},
            book_path,
            chapter,
            step="attribution",
            tool=tool,
            model=model,
        )
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "Could not write attribution checkpoint for chapter %s: %s",
            getattr(chapter, "index", "?"),
            exc,
        )


async def _with_retry_async(
    fn: Callable[..., object],
    *args: object,
    max_attempts: int,
    backoff_base: float,
    **kwargs: object,
) -> object:
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit, ValueError, TypeError, AttributeError):
            raise
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == max_attempts:
                break
            wait = backoff_base ** (attempt - 1)
            _logger.warning(
                "Attempt %d/%d failed (%s: %s); retrying in %.1fs",
                attempt,
                max_attempts,
                type(exc).__name__,
                exc,
                wait,
            )
            _logger.debug(
                "Full retryable NLP provider error on attempt %d/%d: %s: %s",
                attempt,
                max_attempts,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            await asyncio.sleep(wait)
    raise last_exc  # type: ignore[misc]


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
    result: CharacterRoster | NLPResult | None  # populated when DONE
    error: Exception | None                        # populated when FAILED

    # Internal — excluded from poll() snapshots
    _thread: threading.Thread | None = field(default=None, repr=False)
    _cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def poll(self) -> NLPJob:
        """Return a frozen snapshot of current state (non-blocking)."""
        return dataclasses.replace(self, _thread=None)

    def wait(self, timeout: float | None = None) -> NLPJob:
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

    def __init__(self, config: NLPConfig) -> None:
        self._config = config
        from kenkui.nlp.providers._factory import (
            get_attribution_provider,
            get_extraction_provider,
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
        from kenkui.models import AttributionTool, ExtractionTool

        results: list[ValidationResult] = []

        # --- Extraction tool ---
        extraction_tool = self._config.extraction_tool.value
        try:
            match self._config.extraction_tool:
                case ExtractionTool.OLLAMA:
                    import ollama as _ollama  # noqa: F401
                case ExtractionTool.BOOKNLP:
                    from booknlp.booknlp import BookNLP as _bnlp  # noqa: F401
                case (
                    ExtractionTool.LITELLM
                    | ExtractionTool.OPENROUTER
                    | ExtractionTool.ANTHROPIC
                    | ExtractionTool.OPENAI
                    | ExtractionTool.GOOGLE
                ):
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
                case (
                    AttributionTool.LITELLM
                    | AttributionTool.OPENROUTER
                    | AttributionTool.ANTHROPIC
                    | AttributionTool.OPENAI
                    | AttributionTool.GOOGLE
                ):
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
        chapters: list[Chapter],
        series_roster: CharacterRoster | None = None,
        progress_callback: Callable[[int, str], None] | None = None,
        step_callback: Callable[[str], None] | None = None,
        use_cache: bool = True,
    ) -> CharacterRoster:
        """Run Stage 1-2: character extraction + coreference resolution.

        Checks cache first (if use_cache), runs the extraction provider with
        retry + SIGTERM handling, writes result to cache.

        Returns:
            CharacterRoster — the extracted character roster.
        """
        from kenkui.nlp.models import CharacterRoster

        tool = self._config.extraction_tool.value
        model = self._config.extraction_model

        if not use_cache:
            clear_checkpoints(book_path, step="extraction", tool=tool, model=model)

        # Cache read
        if use_cache:
            cached = get_cache(book_path, step="extraction", tool=tool, model=model)
            if cached is not None:
                try:
                    return CharacterRoster.model_validate(cached)
                except Exception:
                    pass  # Corrupt/stale cache — fall through to fresh run

        # Build progress adapter: str → (int, str).
        # Parses "Block N/M" messages for accurate block-level percentage;
        # falls back to fixed +5% increments for named milestones.
        _BLOCK_RE = re.compile(r"Block (\d+)/(\d+)")
        _pct = [0]

        def _adapt(msg: str) -> None:
            m = _BLOCK_RE.search(msg)
            if m:
                current, total = int(m.group(1)), int(m.group(2))
                _pct[0] = int(current / total * 90) if total > 0 else _pct[0]
            else:
                _pct[0] = min(90, _pct[0] + 5)
            if progress_callback:
                progress_callback(_pct[0], msg)

        # SIGTERM handlers are process-global and Python only allows installing
        # them from the main interpreter thread. HTTP analysis tasks run in the
        # task-service worker pool, so preserve graceful CLI cancellation when
        # possible without breaking server-side threaded execution.
        should_install_sigterm_handler = threading.current_thread() is threading.main_thread()
        _orig_sigterm = signal.getsignal(signal.SIGTERM) if should_install_sigterm_handler else None

        def _sigterm_handler(signum: int, frame: object) -> None:
            raise KeyboardInterrupt("SIGTERM received")

        if should_install_sigterm_handler:
            signal.signal(signal.SIGTERM, _sigterm_handler)
        try:
            roster: CharacterRoster = with_retry(
                self._extraction.build_roster,
                max_attempts=self._config.retry_max_attempts,
                backoff_base=self._config.retry_backoff_base,
            )(
                chapters,
                series_roster=series_roster,
                progress_callback=_adapt,
                step_callback=step_callback,
                book_path=book_path,
            )
        finally:
            if should_install_sigterm_handler:
                signal.signal(signal.SIGTERM, _orig_sigterm)

        # Strip pronoun-slug characters produced by hallucinating LLMs
        roster = CharacterRoster(
            characters=[ch for ch in roster.characters if ch.slug not in _PRONOUNS]
        )

        # Cache write
        try:
            put_cache(
                roster.model_dump(),
                book_path,
                step="extraction",
                tool=tool,
                model=model,
            )
            cached = get_cache(book_path, step="extraction", tool=tool, model=model)
            if cached is None:
                raise OSError("extraction cache verification failed")
            CharacterRoster.model_validate(cached)
        except Exception:
            pass  # Cache write failure is non-fatal

        return roster

    # ------------------------------------------------------------------
    # attribute
    # ------------------------------------------------------------------

    def attribute(
        self,
        book_path: Path,
        chapters: list[Chapter],
        roster: CharacterRoster,
        progress_callback: Callable[[int, str], None] | None = None,
        use_cache: bool = True,
    ) -> NLPResult:
        """Run Stage 3-4: LLM speaker attribution using a pre-built roster.

        Checks cache first (if use_cache), runs attribution per chapter
        with retry + SIGTERM handling, writes result to cache.

        Returns:
            NLPResult — chapters with segments and character quote counts.
        """
        from dataclasses import replace as _replace

        from kenkui.models import CharacterInfo, NLPResult
        from kenkui.models import CharacterRecord as AppCharacterRecord
        from kenkui.nlp.models import AttributionResult

        tool = self._config.attribution_tool.value
        model = self._config.attribution_model

        if not use_cache:
            clear_checkpoints(book_path, step="attribution", tool=tool, model=model)

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
            async_attr = getattr(self._attribution, "attribute_chapter_async", None)
            if tool == "openrouter" and inspect.iscoroutinefunction(async_attr):
                chapter_results = _run_coroutine_sync(
                    self._attribute_chapters_async(
                        book_path,
                        chapters,
                        roster,
                        progress_callback,
                        use_cache=use_cache,
                        tool=tool,
                        model=model,
                    )
                )
            else:
                chapter_results = []
                for chapter in chapters:
                    cached_payload = (
                        get_chapter_checkpoint(
                            book_path,
                            chapter,
                            step="attribution",
                            tool=tool,
                            model=model,
                        )
                        if use_cache
                        else None
                    )
                    if cached_payload is not None:
                        chapter_results.append(
                            (
                                chapter,
                                AttributionResult.model_validate(
                                    cached_payload["attribution_result"]
                                ),
                            )
                        )
                        _adapt(chapter.title or f"Chapter {chapter.index}")
                        continue

                    attr_result = with_retry(
                        self._attribution.attribute_chapter,
                        max_attempts=self._config.retry_max_attempts,
                        backoff_base=self._config.retry_backoff_base,
                    )(chapter, roster, progress_callback=_adapt)
                    _store_attribution_checkpoint(
                        book_path=book_path,
                        chapter=chapter,
                        attr_result=attr_result,
                        tool=tool,
                        model=model,
                    )
                    chapter_results.append((chapter, attr_result))
        finally:
            signal.signal(signal.SIGTERM, _orig_sigterm)

        attribution_counts: dict[str, int] = defaultdict(int)
        attributed_chapters = []
        for chapter, attr_result in chapter_results:
            segments = _attribution_to_segments(chapter, attr_result, roster)
            attributed_chapters.append(_replace(chapter, segments=segments))

            for item in attr_result.attributions:
                if item.speaker and item.speaker not in _SPEAKER_SENTINELS:
                    attribution_counts[_slugify(item.speaker)] += 1

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
            cached = get_cache(book_path, step="attribution", tool=tool, model=model)
            if cached is None:
                raise OSError("attribution cache verification failed")
            NLPResult.from_dict(cached)
            clear_checkpoints(book_path, step="attribution", tool=tool, model=model)
        except Exception:
            pass  # Cache write failure is non-fatal

        return result

    async def _attribute_chapters_async(
        self,
        book_path: Path,
        chapters: list[Chapter],
        roster: CharacterRoster,
        progress_callback: Callable[[int, str], None] | None,
        *,
        use_cache: bool,
        tool: str,
        model: str,
    ) -> list[tuple[Chapter, object]]:
        from kenkui.nlp.models import AttributionResult

        async_attr = getattr(self._attribution, "attribute_chapter_async", None)
        if not inspect.iscoroutinefunction(async_attr):
            raise TypeError("Attribution provider does not expose async chapter attribution")

        concurrency = max(1, getattr(self._config, "openrouter_attribution_concurrency", 4))
        semaphore = asyncio.Semaphore(concurrency)
        total = max(1, len(chapters))
        active: dict[int, tuple[Chapter, str]] = {}
        completed = 0

        def _emit_snapshot() -> None:
            if progress_callback is None:
                return
            pct = min(90, 10 + int(completed / total * 80))
            progress_callback(
                pct,
                _format_attribution_jobs(active, completed=completed, total=total),
            )

        async def _attribute_one(position: int, chapter: Chapter) -> tuple[Chapter, object]:
            nonlocal completed
            cached_payload = (
                get_chapter_checkpoint(
                    book_path,
                    chapter,
                    step="attribution",
                    tool=tool,
                    model=model,
                )
                if use_cache
                else None
            )
            if cached_payload is not None:
                completed += 1
                _emit_snapshot()
                return (
                    chapter,
                    AttributionResult.model_validate(cached_payload["attribution_result"]),
                )

            async with semaphore:
                active[position] = (chapter, "queued")
                _emit_snapshot()

                def _chapter_progress(msg: str) -> None:
                    active[position] = (chapter, msg)
                    _emit_snapshot()

                result = await _with_retry_async(
                    async_attr,
                    chapter,
                    roster,
                    progress_callback=_chapter_progress,
                    max_attempts=self._config.retry_max_attempts,
                    backoff_base=self._config.retry_backoff_base,
                )
                completed += 1
                active.pop(position, None)
                _store_attribution_checkpoint(
                    book_path=book_path,
                    chapter=chapter,
                    attr_result=result,
                    tool=tool,
                    model=model,
                )
                _emit_snapshot()
                return chapter, result

        tasks = [
            asyncio.create_task(_attribute_one(position, chapter))
            for position, chapter in enumerate(chapters, start=1)
        ]
        results: list[tuple[Chapter, object]] = []
        try:
            for task in asyncio.as_completed(tasks):
                results.append(await task)
        except Exception:
            for task in tasks:
                task.cancel()
            raise
        return sorted(results, key=lambda item: getattr(item[0], "index", 0))

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(
        self,
        book_path: Path,
        chapters: list[Chapter],
        series_roster: CharacterRoster | None = None,
        progress_callback: Callable[[int, str], None] | None = None,
        use_cache: bool = True,
    ) -> NLPResult:
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
        chapters: list[Chapter],
        series_roster: CharacterRoster | None = None,
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
        chapters: list[Chapter],
        roster: CharacterRoster,
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
