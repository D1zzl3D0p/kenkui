"""Background job orchestration extracted from ``KenkuiService``.

Lock ownership
--------------
``JobExecutor`` deliberately does **not** own a private lock. The original
monolith mutated the job-orchestration flags (``_running`` /
``_pause_requested`` / ``_cancel_requested_job_id`` / ``_current_id``) *and* the
queue-item state atomically under a single ``RLock``. Introducing a second,
executor-private lock would create a lock-ordering hazard (two locks acquired in
different orders across ``cancel_job`` / ``_process_job``) and would break those
atomic transitions.

To preserve the exact synchronization guarantees, ``JobExecutor`` shares the
:class:`~kenkui.services.queue_manager.QueueManager` lock — it is the single
reentrant lock guarding the whole queue/job domain. ``JobExecutor`` owns the
*thread lifecycle* (``self._thread`` and ``start_processing`` / ``stop``) and the
orchestration flags; all queue reads/writes are delegated to the QueueManager,
whose methods reacquire the same (reentrant) lock.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from kenkui.models import CostStatus, JobStatus, QueueItem
from kenkui.progress import ProgressEvent
from kenkui.services.execution_service import actionable_tts_error_message
from kenkui.services.job_service import build_processing_config
from kenkui.services.queue_manager import QueueManager, _queue_item_summary

logger = logging.getLogger(__name__)


def _progress_percent(event: ProgressEvent) -> float:
    if event.total_units:
        return max(0.0, min(100.0, (event.completed_units / event.total_units) * 100.0))
    if event.status == "completed":
        return 100.0
    return 0.0


def progress_update_from_args(*args: Any) -> tuple[float, str, int]:
    if len(args) == 1 and isinstance(args[0], ProgressEvent):
        event = args[0]
        title = event.message
        if event.active_chapters:
            title = event.active_chapters[0].title or title
        return _progress_percent(event), title, 0

    if len(args) == 3:
        progress, chapter, eta = args
        return float(progress or 0.0), str(chapter or ""), int(eta or 0)

    if len(args) == 2:
        progress, message = args
        return float(progress or 0.0), str(message or ""), 0

    if len(args) == 1:
        return 0.0, str(args[0] or ""), 0

    raise TypeError(f"Unsupported progress callback payload: {args!r}")


class JobExecutor:
    """Drives the pending queue through the TTS execution providers."""

    def __init__(
        self,
        *,
        queue: QueueManager,
        resolve_provider: Callable[[QueueItem], Any],
    ) -> None:
        self._queue = queue
        self._lock = queue.lock
        self._resolve_provider = resolve_provider
        self._thread: threading.Thread | None = None
        self._running = False
        self._pause_requested = False
        self._cancel_requested_job_id: str | None = None
        self._current_id: str | None = None

    # -- orchestration state --------------------------------------------
    @property
    def running(self) -> bool:
        return self._running

    @running.setter
    def running(self, value: bool) -> None:
        self._running = value

    @property
    def current_id(self) -> str | None:
        return self._current_id

    # -- lifecycle ------------------------------------------------------
    def start_job(self, job_id: str) -> bool:
        item = self._queue.get(job_id)
        if item is None or item.status != JobStatus.PENDING or self._running:
            return False
        with self._lock:
            item.status = JobStatus.PROCESSING
            item.started_at = time.time()
            self._current_id = item.id
            self._queue.save()
        logger.info("Started queued job %s", _queue_item_summary(item))
        self.start_processing()
        return True

    def start_processing(self) -> bool:
        if self._running:
            return False
        self._running = True
        self._thread = threading.Thread(target=self.process_loop, daemon=True)
        self._thread.start()
        logger.info(
            "Processing loop started current_job=%s pending=%d",
            self._current_id or "",
            len(self._queue.pending_items),
        )
        return True

    def stop_processing(self) -> None:
        current = self._queue.processing_item()
        if current is not None:
            with self._lock:
                self._cancel_requested_job_id = current.id
                self._pause_requested = False
            try:
                self._resolve_provider(current).cancel(current)
            except Exception as exc:
                logger.warning("Could not cancel provider job %s: %s", current.id, exc)
            else:
                logger.info("Requested cancel for current job job_id=%s", current.id)
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Processing loop stopped current_job=%s", current.id if current else "")

    def pause_job(self, job_id: str) -> bool:
        with self._lock:
            item = self._queue.get(job_id)
            if item is None or item.status != JobStatus.PROCESSING:
                return False
            self._pause_requested = True
        logger.info("Pause requested job_id=%s", job_id)
        return True

    def resume_job(self, job_id: str) -> bool:
        with self._lock:
            item = self._queue.get(job_id)
            if item is None or item.status != JobStatus.PAUSED:
                return False
            item.status = JobStatus.PENDING
            self._pause_requested = False
            self._cancel_requested_job_id = None
            self._queue.save()
        logger.info("Resumed job job_id=%s", job_id)
        self.start_processing()
        return True

    def retry_job(self, job_id: str) -> bool:
        with self._lock:
            item = self._queue.get(job_id)
            if item is None or item.status != JobStatus.FAILED:
                return False
            item.status = JobStatus.PENDING
            item.progress = 0.0
            item.current_chapter = ""
            item.eta_seconds = 0
            item.error_message = ""
            item.output_path = ""
            item.started_at = 0.0
            item.completed_at = 0.0
            item.execution_provider = item.job.tts_execution_mode.value
            item.remote_job_id = ""
            item.estimated_cost_usd = None
            item.actual_cost_usd = None
            item.cost_status = CostStatus.NONE
            item.artifact_uri = ""
            item.artifact_source = ""
            item.provider_status = "retrying"
            self._pause_requested = False
            if self._cancel_requested_job_id == item.id:
                self._cancel_requested_job_id = None
            self._queue.save()
        logger.info("Retrying failed job job_id=%s", job_id)
        self.start_processing()
        return True

    def cancel_job(self, job_id: str) -> bool:
        provider = None
        item: QueueItem | None = None
        with self._lock:
            item = self._queue.get(job_id)
            if item is None:
                return False
            if item.status == JobStatus.CANCELLED:
                return True
            if item.status == JobStatus.PROCESSING:
                self._cancel_requested_job_id = item.id
                self._pause_requested = False
                item.provider_status = "cancelling"
                self._queue.save()
                try:
                    provider = self._resolve_provider(item)
                except Exception as exc:
                    logger.warning("Could not load provider for cancel job %s: %s", item.id, exc)
                    provider = None
            elif item.status == JobStatus.PAUSED:
                item.status = JobStatus.CANCELLED
                item.current_chapter = ""
                item.error_message = ""
                item.provider_status = "cancelled"
                item.completed_at = time.time()
                self._queue.save()
                logger.info("Cancelled paused job job_id=%s", job_id)
                return True
            elif item.status == JobStatus.PENDING:
                item.status = JobStatus.CANCELLED
                item.current_chapter = ""
                item.error_message = ""
                item.provider_status = "cancelled"
                item.completed_at = time.time()
                self._queue.save()
                logger.info("Cancelled queued job job_id=%s", job_id)
                return True
            else:
                return False
        if provider is not None and item is not None:
            try:
                provider.cancel(item)
            except Exception as exc:
                logger.warning("Could not cancel provider job %s: %s", item.id, exc)
            else:
                logger.info("Requested cancel for current job job_id=%s", item.id)
        return True

    # -- processing -----------------------------------------------------
    def process_loop(self) -> None:
        try:
            logger.info("Processing loop running")
            while self._running:
                item = self._queue.next_pending()
                if item is None:
                    break
                with self._lock:
                    item.status = JobStatus.PROCESSING
                    item.started_at = time.time()
                    self._current_id = item.id
                    self._cancel_requested_job_id = None
                    self._queue.save()
                logger.info("Dequeued job %s", _queue_item_summary(item))
                self.process_job(item)
                if item.status == JobStatus.PAUSED:
                    break
        finally:
            self._running = False
            self._current_id = None
            logger.info("Processing loop idle")

    def process_job(self, item: QueueItem) -> None:
        started_at = time.time()
        app_config = self._queue.app_config
        try:
            cfg = build_processing_config(item.job, app_config)
            provider = self._resolve_provider(item)
            logger.info(
                "Processing job job_id=%s provider=%s output_path=%s voice=%s",
                item.id,
                item.execution_provider,
                cfg.output_path,
                cfg.voice,
            )
            self._queue.update_job_metadata(
                item.id,
                execution_provider=item.job.tts_execution_mode.value,
                provider_status="starting",
            )
            outcome = provider.execute(
                item=item,
                cfg=cfg,
                app_config=app_config,
                progress_callback=self._progress_callback_for_job(item.id),
                metadata_callback=lambda **fields: self._queue.update_job_metadata(item.id, **fields),
                pause_check=lambda: self._pause_requested,
                cancel_check=lambda: self._cancel_requested_job_id == item.id,
            )
            if outcome.cancelled or self._cancel_requested_job_id == item.id:
                with self._lock:
                    item.status = JobStatus.CANCELLED
                    item.current_chapter = ""
                    item.error_message = ""
                    item.provider_status = outcome.provider_status or "cancelled"
                    item.completed_at = time.time()
                    self._pause_requested = False
                    self._cancel_requested_job_id = None
                    self._queue.save()
                logger.info("Job cancelled job_id=%s duration_s=%.1f", item.id, time.time() - started_at)
                return
            if outcome.paused:
                with self._lock:
                    item.status = JobStatus.PAUSED
                    self._pause_requested = False
                    self._cancel_requested_job_id = None
                    self._queue.save()
                logger.info("Job paused job_id=%s duration_s=%.1f", item.id, time.time() - started_at)
                return
            self._queue.update_job_metadata(
                item.id,
                remote_job_id=outcome.remote_job_id,
                estimated_cost_usd=outcome.estimated_cost_usd,
                actual_cost_usd=outcome.actual_cost_usd,
                cost_status="final"
                if outcome.actual_cost_usd is not None
                else ("estimated" if outcome.estimated_cost_usd is not None else "none"),
                artifact_uri=outcome.artifact_uri,
                artifact_source=outcome.artifact_source,
                provider_status=outcome.provider_status or ("completed" if outcome.success else "failed"),
            )
            if outcome.success:
                output_path = outcome.output_path or str(cfg.output_path / f"{item.job.name}.m4b")
                self._queue.complete(item.id, output_path)
                logger.info(
                    "Job completed job_id=%s output_path=%s duration_s=%.1f",
                    item.id,
                    output_path,
                    time.time() - started_at,
                )
            else:
                failure_message = actionable_tts_error_message(
                    outcome.error_message or "Conversion failed"
                )
                self._queue.fail(item.id, failure_message)
                logger.warning(
                    "Job failed job_id=%s provider_status=%s duration_s=%.1f error=%s",
                    item.id,
                    outcome.provider_status or "",
                    time.time() - started_at,
                    failure_message,
                )
        except Exception as exc:
            logger.exception("Job %s failed: %s", item.id, exc)
            self._queue.fail(item.id, actionable_tts_error_message(exc))

    def _progress_callback_for_job(self, job_id: str) -> Callable[..., None]:
        def progress_callback(*args: Any) -> None:
            progress, current_chapter, eta_seconds = progress_update_from_args(*args)
            self._queue.update_progress(job_id, progress, current_chapter, eta_seconds)

        return progress_callback
