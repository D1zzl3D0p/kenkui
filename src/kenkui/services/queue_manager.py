"""Queue persistence + item-state store extracted from ``KenkuiService``.

Lock ownership
--------------
``QueueManager`` owns the single reentrant lock (``self.lock``) that guards the
in-memory queue (``self._items``), the persisted ``AppConfig`` snapshot, and
every mutation that is followed by a ``save()``. It is the *synchronization
authority* for the queue/job domain.

Historically the queue items and the job-orchestration flags
(``_running`` / ``_pause_requested`` / ``_cancel_requested_job_id``) were mutated
atomically under one ``RLock``. To preserve those atomic transitions without
introducing a lock-ordering hazard, :class:`JobExecutor` shares *this* lock
rather than owning a private one — see its module docstring. The lock is
therefore exposed as a public attribute on purpose.
"""

from __future__ import annotations

import logging
import threading
import time
import tomllib
from pathlib import Path
from typing import Any

import tomli_w

from kenkui.models import AppConfig, CostStatus, JobConfig, JobStatus, QueueItem

logger = logging.getLogger(__name__)


def _strip_none(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(v) for v in obj if v is not None]
    return obj


def _job_summary(job: JobConfig, *, job_id: str | None = None, status: str | None = None) -> str:
    parts = [
        f"job_id={job_id}" if job_id else None,
        f"status={status}" if status else None,
        f"ebook_path={job.ebook_path}",
        f"output_path={job.output_path}" if job.output_path else None,
        f"voice={job.voice or ''}",
        f"tts_mode={job.tts_execution_mode.value}",
        f"narration_mode={job.narration_mode.value}",
    ]
    return " ".join(part for part in parts if part)


def _queue_item_summary(item: QueueItem) -> str:
    return _job_summary(item.job, job_id=item.id, status=item.status.value)


class QueueManager:
    """Owns the persisted job queue and its guarding lock."""

    def __init__(self, *, queue_file: Path, app_config: AppConfig) -> None:
        self.queue_file = queue_file
        self.legacy_queue_file = queue_file.with_suffix(".yaml")
        self.lock = threading.RLock()
        self._items: list[QueueItem] = []
        self._app_config = app_config
        self.load()

    # -- persistence -----------------------------------------------------
    def load(self) -> None:
        if not self.queue_file.exists() and self.legacy_queue_file.exists():
            self._migrate_yaml_to_toml()
        if self.queue_file.exists():
            try:
                data = tomllib.loads(self.queue_file.read_text(encoding="utf-8"))
                self._items = [QueueItem.from_dict(d) for d in data.get("items", [])]
                if "app_config" in data:
                    self._app_config = AppConfig.from_dict(data.get("app_config", {}))
                logger.info(
                    "Loaded queue file path=%s items=%d",
                    self.queue_file,
                    len(self._items),
                )
            except Exception as exc:
                logger.warning("Could not load queue file %s: %s", self.queue_file, exc)
        self.reset_stale_processing()

    def _migrate_yaml_to_toml(self) -> None:
        try:
            import yaml

            data = yaml.safe_load(self.legacy_queue_file.read_text())
            if data:
                self.queue_file.parent.mkdir(parents=True, exist_ok=True)
                self.queue_file.write_bytes(tomli_w.dumps(_strip_none(data)).encode("utf-8"))
            self.legacy_queue_file.unlink(missing_ok=True)
            logger.info(
                "Migrated legacy queue file from=%s to=%s",
                self.legacy_queue_file,
                self.queue_file,
            )
        except Exception as exc:
            logger.warning("Could not migrate legacy queue yaml: %s", exc)

    def save(self) -> None:
        raw = {
            "items": [item.to_dict() for item in self._items],
            "app_config": self._app_config.to_dict(),
        }
        self.queue_file.parent.mkdir(parents=True, exist_ok=True)
        self.queue_file.write_bytes(tomli_w.dumps(_strip_none(raw)).encode("utf-8"))

    # -- app config ------------------------------------------------------
    @property
    def app_config(self) -> AppConfig:
        return self._app_config

    @app_config.setter
    def app_config(self, config: AppConfig) -> None:
        with self.lock:
            self._app_config = config
            self.save()

    # -- read views ------------------------------------------------------
    @property
    def all_items(self) -> list[QueueItem]:
        with self.lock:
            return list(self._items)

    @property
    def pending_items(self) -> list[QueueItem]:
        with self.lock:
            return [i for i in self._items if i.status == JobStatus.PENDING]

    @property
    def completed_items(self) -> list[QueueItem]:
        with self.lock:
            return [i for i in self._items if i.status == JobStatus.COMPLETED]

    @property
    def failed_items(self) -> list[QueueItem]:
        with self.lock:
            return [i for i in self._items if i.status == JobStatus.FAILED]

    @property
    def current_item(self) -> QueueItem | None:
        with self.lock:
            current = next((i for i in self._items if i.status == JobStatus.PROCESSING), None)
            if current is not None:
                return current
            return next((i for i in self._items if i.status == JobStatus.PAUSED), None)

    def processing_item(self) -> QueueItem | None:
        with self.lock:
            return next((i for i in self._items if i.status == JobStatus.PROCESSING), None)

    def next_pending(self) -> QueueItem | None:
        with self.lock:
            return next((i for i in self._items if i.status == JobStatus.PENDING), None)

    def get(self, job_id: str) -> QueueItem | None:
        with self.lock:
            return next((i for i in self._items if i.id == job_id), None)

    # -- mutations -------------------------------------------------------
    def add(self, item: QueueItem) -> QueueItem:
        with self.lock:
            self._items.append(item)
            self.save()
        logger.info("Queued %s", _queue_item_summary(item))
        return item

    def remove(self, job_id: str) -> bool:
        with self.lock:
            for i, item in enumerate(self._items):
                if item.id == job_id:
                    if item.status not in {
                        JobStatus.COMPLETED,
                        JobStatus.FAILED,
                        JobStatus.CANCELLED,
                    }:
                        logger.info(
                            "Refused queue removal job_id=%s status=%s",
                            item.id,
                            item.status.value,
                        )
                        return False
                    self._items.pop(i)
                    self.save()
                    logger.info(
                        "Removed queued job job_id=%s status=%s", item.id, item.status.value
                    )
                    return True
        return False

    def clear_all(self) -> int:
        with self.lock:
            removed = len(self._items)
            self._items = []
            self.save()
        logger.info("Cleared queue removed=%d", removed)
        return removed

    def reset_stale_processing(self) -> None:
        changed = False
        reset_ids: list[str] = []
        with self.lock:
            for item in self._items:
                if item.status == JobStatus.PROCESSING:
                    item.status = JobStatus.PENDING
                    item.progress = 0.0
                    item.current_chapter = ""
                    item.error_message = ""
                    changed = True
                    reset_ids.append(item.id)
            if changed:
                self.save()
        if reset_ids:
            logger.warning("Reset stale processing jobs job_ids=%s", ",".join(reset_ids))

    def update_progress(
        self, job_id: str, progress: float, current_chapter: str, eta_seconds: int
    ) -> None:
        with self.lock:
            item = self.get(job_id)
            if item is not None:
                item.progress = progress
                item.current_chapter = current_chapter
                item.eta_seconds = eta_seconds
                self.save()
        logger.debug(
            "Job progress job_id=%s progress=%.1f chapter=%s eta_seconds=%s",
            job_id,
            progress,
            current_chapter,
            eta_seconds,
        )

    def update_job_metadata(self, job_id: str, **fields: Any) -> None:
        with self.lock:
            item = self.get(job_id)
            if item is None:
                return
            for key, value in fields.items():
                if value is None:
                    continue
                if key == "cost_status" and not isinstance(value, CostStatus):
                    value = CostStatus(str(value))
                setattr(item, key, value)
            self.save()

    def complete(self, job_id: str, output_path: str = "") -> None:
        with self.lock:
            item = self.get(job_id)
            if item is not None:
                item.status = JobStatus.COMPLETED
                item.progress = 100.0
                item.current_chapter = ""
                item.output_path = output_path
                item.completed_at = time.time()
                self.save()
        logger.info("Marked job completed job_id=%s output_path=%s", job_id, output_path)

    def fail(self, job_id: str, error: str) -> None:
        with self.lock:
            item = self.get(job_id)
            if item is not None:
                item.status = JobStatus.FAILED
                item.error_message = error
                self.save()
        logger.warning("Marked job failed job_id=%s error=%s", job_id, error)
