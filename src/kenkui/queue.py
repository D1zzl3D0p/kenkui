import logging
import uuid

import yaml

from .config import CONFIG_DIR

logger = logging.getLogger(__name__)
from .models import AppConfig, JobConfig, JobStatus, QueueItem

QUEUE_FILE = CONFIG_DIR / "queue.yaml"


class QueueManager:
    def __init__(self):
        self.items: list[QueueItem] = []
        self._current_id: str | None = None
        self._app_config = AppConfig()
        self._load()

    def _load(self):
        if QUEUE_FILE.exists():
            try:
                data = yaml.safe_load(QUEUE_FILE.read_text())
                if data:
                    self.items = [QueueItem.from_dict(d) for d in data.get("items", [])]
                    self._app_config = AppConfig.from_dict(data.get("app_config", {}))
            except Exception:
                pass
        self.reset_stale_processing()

    def _save(self):
        data = {
            "items": [item.to_dict() for item in self.items],
            "app_config": self._app_config.to_dict(),
        }
        QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
        QUEUE_FILE.write_text(yaml.dump(data, default_flow_style=False))

    @property
    def app_config(self) -> AppConfig:
        return self._app_config

    @app_config.setter
    def app_config(self, config: AppConfig):
        self._app_config = config
        self._save()

    @property
    def current_item(self) -> QueueItem | None:
        return next((i for i in self.items if i.status == JobStatus.PROCESSING), None)

    @property
    def pending_items(self) -> list[QueueItem]:
        result = [i for i in self.items if i.status == JobStatus.PENDING]
        logger.debug(f"[DEBUG Queue] pending_items: {len(result)} items")
        for item in result:
            logger.debug(f"  - {item.id}: {item.job.name} ({item.status})")
        return result

    @property
    def completed_items(self) -> list[QueueItem]:
        return [i for i in self.items if i.status == JobStatus.COMPLETED]

    @property
    def failed_items(self) -> list[QueueItem]:
        return [i for i in self.items if i.status == JobStatus.FAILED]

    def add(self, job: JobConfig) -> QueueItem:
        item = QueueItem(
            id=str(uuid.uuid4())[:8],
            job=job,
            status=JobStatus.PENDING,
        )
        self.items.append(item)
        self._save()
        logger.debug(f"[DEBUG Queue] Added job {item.id} ({job.name}) to queue")
        return item

    def reset_stale_processing(self):
        """Reset any PROCESSING jobs to PENDING (e.g., from previous session)."""
        for item in self.items:
            if item.status == JobStatus.PROCESSING:
                item.status = JobStatus.PENDING
                item.progress = 0.0
                item.current_chapter = ""
                item.error_message = ""
        if any(i.status == JobStatus.PENDING for i in self.items):
            self._save()

    def remove(self, item_id: str) -> bool:
        for i, item in enumerate(self.items):
            if item.id == item_id:
                if item.status == JobStatus.PROCESSING:
                    return False
                self.items.pop(i)
                self._save()
                return True
        return False

    def remove_or_stop(self, item_id: str) -> bool:
        """Remove an item, stopping it first if it's processing."""
        for i, item in enumerate(self.items):
            if item.id == item_id:
                if item.status == JobStatus.PROCESSING:
                    from .processor import get_processor

                    get_processor().stop()
                    item.status = JobStatus.CANCELLED
                self.items.pop(i)
                self._save()
                return True
        return False

    def clear_all(self):
        self.items = []
        self._save()

    def get_next_pending(self) -> QueueItem | None:
        result = next((i for i in self.items if i.status == JobStatus.PENDING), None)
        logger.debug("get_next_pending returned: %s", result.id if result else None)
        return result

    def start_next(self) -> QueueItem | None:
        logger.debug("[DEBUG Queue] start_next() called")
        item = self.get_next_pending()
        logger.debug(f"[DEBUG Queue] Got item: {item}")
        if item:
            logger.debug(f"[DEBUG Queue] Setting item {item.id} status to PROCESSING")
            item.status = JobStatus.PROCESSING
            self._current_id = item.id
            self._save()
            logger.debug("[DEBUG Queue] Saved queue with item now PROCESSING")
        return item

    def update_progress(
        self, item_id: str, progress: float, current_chapter: str, eta_seconds: int
    ):
        for item in self.items:
            if item.id == item_id:
                item.progress = progress
                item.current_chapter = current_chapter
                item.eta_seconds = eta_seconds
                break
        self._save()

    def complete(self, item_id: str):
        for item in self.items:
            if item.id == item_id:
                item.status = JobStatus.COMPLETED
                item.progress = 100.0
                item.current_chapter = ""
                break
        self._save()

    def fail(self, item_id: str, error: str):
        for item in self.items:
            if item.id == item_id:
                item.status = JobStatus.FAILED
                item.error_message = error
                break
        self._save()

    def cancel_current(self) -> bool:
        if self.current_item:
            self.current_item.status = JobStatus.CANCELLED
            self._save()
            return True
        return False


_queue_manager: QueueManager | None = None


def get_queue_manager() -> QueueManager:
    global _queue_manager
    if _queue_manager is None:
        _queue_manager = QueueManager()
    return _queue_manager
