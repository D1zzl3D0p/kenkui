"""Task registry + runner coordination extracted from ``KenkuiService``.

Lock ownership
--------------
``TaskCoordinator`` owns no lock of its own: all synchronization lives inside
the wrapped :class:`~kenkui.services.task_service.TaskRegistry` and
:class:`~kenkui.services.task_service.TaskRunner` (thread pool). It owns the
lifecycle of the runner's worker threads and is responsible for shutting them
down. It shares no mutable state with the queue/job machinery, so it can be
reasoned about in complete isolation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kenkui.services.task_service import Task, TaskRegistry, TaskRunner, TaskType


class TaskCoordinator:
    """Owns the async task registry and its worker-thread pool."""

    def __init__(self, *, max_workers: int = 4) -> None:
        self.registry = TaskRegistry()
        self.runner = TaskRunner(self.registry, max_workers=max_workers)

    def submit(self, task_type: TaskType, fn: Callable, /, **kwargs: Any) -> Task:
        return self.runner.submit(task_type, fn, **kwargs)

    def get(self, task_id: str) -> Task | None:
        return self.registry.get(task_id)

    def shutdown(self, *, wait: bool = False) -> None:
        self.runner.shutdown(wait=wait)
