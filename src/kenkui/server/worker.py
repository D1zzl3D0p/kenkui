"""WorkerServer - Queue and processing logic for the kenkui server."""

from __future__ import annotations

import logging
import threading
import tomllib
import uuid
from collections.abc import Callable

import tomli_w

from ..chapter_filter import FilterOperation
from ..config import CONFIG_DIR
from ..models import AppConfig, JobConfig, JobStatus, QueueItem
from ..parsing import AnnotatedChaptersCacheMissError, AudioBuilder

logger = logging.getLogger(__name__)

QUEUE_FILE = CONFIG_DIR / "queue.toml"
_LEGACY_QUEUE_FILE = CONFIG_DIR / "queue.yaml"


def _strip_none(obj: object) -> object:
    """Recursively remove None values — TOML has no null type."""
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(v) for v in obj if v is not None]
    return obj


class WorkerServer:
    """Server managing job queue and audio processing.

    This class combines queue management and processing logic,
    providing a thread-safe interface for the HTTP API.
    """

    def __init__(self):
        self._items: list[QueueItem] = []
        self._current_id: str | None = None
        self._app_config = AppConfig()
        self._lock = threading.RLock()
        self._processing_thread: threading.Thread | None = None
        self._running = False
        self._progress_callback: Callable[[float, str, int], None] | None = None
        self._load()

    def _load(self):
        # Auto-migrate from legacy queue.yaml if queue.toml does not exist yet.
        if not QUEUE_FILE.exists() and _LEGACY_QUEUE_FILE.exists():
            self._migrate_yaml_to_toml()

        if QUEUE_FILE.exists():
            try:
                data = tomllib.loads(QUEUE_FILE.read_text(encoding="utf-8"))
                if data:
                    self._items = [QueueItem.from_dict(d) for d in data.get("items", [])]
                    self._app_config = AppConfig.from_dict(data.get("app_config", {}))
            except Exception:
                pass
        self._reset_stale_processing()

    def _migrate_yaml_to_toml(self) -> None:
        """Convert queue.yaml → queue.toml and remove the old file."""
        try:
            import yaml  # pyyaml may still be present as a transitive dep

            data = yaml.safe_load(_LEGACY_QUEUE_FILE.read_text())
            if data:
                QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
                QUEUE_FILE.write_bytes(tomli_w.dumps(_strip_none(data)).encode("utf-8"))
            _LEGACY_QUEUE_FILE.unlink(missing_ok=True)
            logger.info("Migrated queue.yaml → queue.toml")
        except Exception as exc:
            logger.warning("Could not migrate queue.yaml: %s — starting fresh", exc)

    def _save(self):
        raw = {
            "items": [item.to_dict() for item in self._items],
            "app_config": self._app_config.to_dict(),
        }
        data: dict = _strip_none(raw)  # type: ignore[assignment]
        QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
        QUEUE_FILE.write_bytes(tomli_w.dumps(data).encode("utf-8"))

    @property
    def app_config(self) -> AppConfig:
        return self._app_config

    @app_config.setter
    def app_config(self, config: AppConfig):
        with self._lock:
            self._app_config = config
            self._save()

    @property
    def current_item(self) -> QueueItem | None:
        with self._lock:
            return next((i for i in self._items if i.status == JobStatus.PROCESSING), None)

    @property
    def pending_items(self) -> list[QueueItem]:
        with self._lock:
            return [i for i in self._items if i.status == JobStatus.PENDING]

    @property
    def completed_items(self) -> list[QueueItem]:
        with self._lock:
            return [i for i in self._items if i.status == JobStatus.COMPLETED]

    @property
    def failed_items(self) -> list[QueueItem]:
        with self._lock:
            return [i for i in self._items if i.status == JobStatus.FAILED]

    @property
    def all_items(self) -> list[QueueItem]:
        with self._lock:
            return list(self._items)

    @property
    def is_running(self) -> bool:
        return self._running

    def add_job(self, job: JobConfig) -> QueueItem:
        with self._lock:
            item = QueueItem(
                id=str(uuid.uuid4())[:8],
                job=job,
                status=JobStatus.PENDING,
            )
            self._items.append(item)
            self._save()
            return item

    def get_job(self, job_id: str) -> QueueItem | None:
        with self._lock:
            return next((i for i in self._items if i.id == job_id), None)

    def remove_job(self, job_id: str) -> bool:
        with self._lock:
            for i, item in enumerate(self._items):
                if item.id == job_id:
                    if item.status == JobStatus.PROCESSING:
                        return False
                    self._items.pop(i)
                    self._save()
                    return True
            return False

    def clear_all_jobs(self):
        with self._lock:
            self._items = []
            self._save()

    def _reset_stale_processing(self):
        """Reset any PROCESSING jobs to PENDING (e.g., from previous session)."""
        with self._lock:
            for item in self._items:
                if item.status == JobStatus.PROCESSING:
                    item.status = JobStatus.PENDING
                    item.progress = 0.0
                    item.current_chapter = ""
                    item.error_message = ""
            if any(i.status == JobStatus.PENDING for i in self._items):
                self._save()

    def get_next_pending(self) -> QueueItem | None:
        with self._lock:
            return next((i for i in self._items if i.status == JobStatus.PENDING), None)

    def start_next_job(self) -> QueueItem | None:
        with self._lock:
            item = self.get_next_pending()
            if item:
                item.status = JobStatus.PROCESSING
                self._current_id = item.id
                self._save()
            return item

    def update_progress(self, job_id: str, progress: float, current_chapter: str, eta_seconds: int):
        with self._lock:
            for item in self._items:
                if item.id == job_id:
                    item.progress = progress
                    item.current_chapter = current_chapter
                    item.eta_seconds = eta_seconds
                    break
            self._save()

    def complete_job(self, job_id: str, output_path: str = ""):
        with self._lock:
            for item in self._items:
                if item.id == job_id:
                    item.status = JobStatus.COMPLETED
                    item.progress = 100.0
                    item.current_chapter = ""
                    item.output_path = output_path
                    break
            self._save()

    def fail_job(self, job_id: str, error: str):
        with self._lock:
            for item in self._items:
                if item.id == job_id:
                    item.status = JobStatus.FAILED
                    item.error_message = error
                    break
            self._save()

    def cancel_current_job(self) -> bool:
        with self._lock:
            if self.current_item:
                self.current_item.status = JobStatus.CANCELLED
                self._save()
                return True
            return False

    def start_processing(self, progress_callback: Callable[[float, str, int], None] | None = None):
        """Start processing the next job in the queue."""
        if self._running:
            return False

        # Create a wrapper callback that also updates server state
        def progress_wrapper(percent: float, chapter: str, eta: int):
            # Call the external callback if provided
            if progress_callback:
                progress_callback(percent, chapter, eta)
            # Always update server state for API queries
            if self._current_id:
                self.update_progress(self._current_id, percent, chapter, eta)

        self._progress_callback = progress_wrapper
        self._running = True

        self._processing_thread = threading.Thread(target=self._process_loop)
        self._processing_thread.start()
        return True

    def stop_processing(self):
        """Stop the current processing job."""
        self._running = False
        if self._processing_thread:
            self._processing_thread.join(timeout=5)

    def _process_loop(self):
        """Main processing loop - runs in background thread."""
        while self._running:
            item = self.start_next_job()
            if not item:
                break

            self._process_job(item)

            if not self._running:
                break

    def _process_job(self, item: QueueItem):
        """Process a single job."""
        job = item.job

        try:
            cfg = self._build_config(job)
            builder = AudioBuilder(cfg, progress_callback=self._progress_callback)

            result = builder.run()

            if result:
                output_path = str(cfg.output_path / f"{job.name}.m4b")
                self.complete_job(item.id, output_path)
            else:
                self.fail_job(item.id, "Conversion failed")

        except AnnotatedChaptersCacheMissError as e:
            # Surface the CACHE_MISS sentinel so QueueScreen can present the
            # recovery dialog (re-analyse or fall back to single voice).
            self.fail_job(item.id, f"CACHE_MISS: {e}")
        except Exception as e:
            self.fail_job(item.id, str(e))

    def _build_config(self, job: JobConfig):
        """Build ProcessingConfig from JobConfig and AppConfig."""
        from ..models import ProcessingConfig

        preset = job.chapter_selection.preset
        if preset.value in ("manual", "custom"):
            # Use explicit index list from the UI checkbox selection
            operations = [
                FilterOperation("index", str(idx)) for idx in job.chapter_selection.included
            ]
            if not operations:
                operations = [FilterOperation("preset", "content-only")]
        else:
            operations = [FilterOperation("preset", preset.value)]

        output_path = job.output_path or job.ebook_path.parent

        cfg = ProcessingConfig(
            voice=job.voice,
            ebook_path=job.ebook_path,
            output_path=output_path,
            pause_line_ms=self._app_config.pause_line_ms,
            pause_chapter_ms=self._app_config.pause_chapter_ms,
            workers=self._app_config.workers,
            m4b_bitrate=self._app_config.m4b_bitrate,
            keep_temp=self._app_config.keep_temp,
            debug_html=self._app_config.verbose,
            chapter_filters=operations,
            verbose=self._app_config.verbose,
            temp=self._app_config.temp,
            lsd_decode_steps=self._app_config.lsd_decode_steps,
            noise_clamp=self._app_config.noise_clamp,
            # Multi-voice fields
            speaker_voices=job.speaker_voices,
            annotated_chapters_path=job.annotated_chapters_path,
            _included_indices=job.chapter_selection.included,
        )
        return cfg


_server: WorkerServer | None = None


def get_server() -> WorkerServer:
    """Get the global WorkerServer instance."""
    global _server
    if _server is None:
        _server = WorkerServer()
    return _server


def reset_server():
    """Reset the global WorkerServer instance."""
    global _server
    if _server:
        _server.stop_processing()
    _server = None
