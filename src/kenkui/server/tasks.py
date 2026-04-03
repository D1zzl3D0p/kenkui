"""Task registry and runner for async service operations."""
from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskType(str, Enum):
    FAST_SCAN = "fast_scan"
    AUDITION = "audition"
    VOICE_DOWNLOAD = "voice_download"
    VOICE_FETCH = "voice_fetch"


@dataclass
class Task:
    task_id: str
    type: TaskType
    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0
    message: str = "Queued"
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)


class TaskRegistry:
    """Thread-safe in-memory task store with TTL-based eviction."""

    DEFAULT_TTL = 3600  # 1 hour

    def __init__(self, ttl: float = DEFAULT_TTL) -> None:
        self._tasks: dict[str, Task] = {}
        self._lock = Lock()
        self._ttl = ttl

    def create(self, task_type: TaskType) -> Task:
        task = Task(task_id=str(uuid.uuid4()), type=task_type)
        with self._lock:
            self._tasks[task.task_id] = task
        return task

    def get(self, task_id: str) -> Task | None:
        with self._lock:
            return self._tasks.get(task_id)

    def update(self, task_id: str, *, progress: int, message: str) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is not None:
                task.status = TaskStatus.RUNNING
                task.progress = progress
                task.message = message

    def complete(self, task_id: str, result: Any) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is not None:
                task.status = TaskStatus.COMPLETED
                task.progress = 100
                task.message = "Done"
                task.result = result

    def fail(self, task_id: str, error: str) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is not None:
                task.status = TaskStatus.FAILED
                task.message = "Failed"
                task.error = error

    def evict_stale(self) -> None:
        """Remove tasks older than TTL."""
        cutoff = time.time() - self._ttl
        with self._lock:
            stale = [tid for tid, t in self._tasks.items() if t.created_at < cutoff]
            for tid in stale:
                del self._tasks[tid]


class TaskRunner:
    """Submits service calls to a thread pool; injects progress_callback → TaskRegistry."""

    def __init__(self, registry: TaskRegistry, max_workers: int = 4) -> None:
        self._registry = registry
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, task_type: TaskType, fn: Callable, /, **kwargs: Any) -> Task:
        """Submit fn(**kwargs) as a background task.

        Injects `progress_callback` into kwargs. The callback writes to TaskRegistry.
        Returns the Task immediately (status=PENDING).
        """
        task = self._registry.create(task_type)

        def _progress(percent: int, message: str) -> None:
            self._registry.update(task.task_id, progress=percent, message=message)

        kwargs["progress_callback"] = _progress

        def _run() -> None:
            try:
                result = fn(**kwargs)
                self._registry.complete(task.task_id, result)
            except Exception as exc:
                self._registry.fail(task.task_id, str(exc))

        self._executor.submit(_run)
        return task

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)


__all__ = [
    "TaskStatus",
    "TaskType",
    "Task",
    "TaskRegistry",
    "TaskRunner",
]
