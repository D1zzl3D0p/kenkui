from __future__ import annotations

import threading

from kenkui.services.task_coordinator import TaskCoordinator
from kenkui.services.task_service import TaskStatus, TaskType


def test_submit_runs_task_and_registry_get_returns_it():
    coordinator = TaskCoordinator(max_workers=2)
    done = threading.Event()

    def work(*, value: int, progress_callback) -> dict[str, int]:
        progress_callback(100, "done")
        done.set()
        return {"value": value}

    try:
        task = coordinator.submit(TaskType.FAST_SCAN, work, value=7)
        assert coordinator.get(task.task_id) is task
        assert done.wait(timeout=2)
        # Runner threads set terminal state after the callable returns.
        for _ in range(200):
            fetched = coordinator.get(task.task_id)
            if fetched is not None and fetched.status == TaskStatus.COMPLETED:
                break
            threading.Event().wait(0.01)
        assert coordinator.get(task.task_id).status == TaskStatus.COMPLETED
        assert coordinator.get(task.task_id).result == {"value": 7}
    finally:
        coordinator.shutdown(wait=True)


def test_get_unknown_task_returns_none():
    coordinator = TaskCoordinator(max_workers=1)
    try:
        assert coordinator.get("does-not-exist") is None
    finally:
        coordinator.shutdown(wait=True)


def test_shutdown_is_idempotent():
    coordinator = TaskCoordinator(max_workers=1)
    coordinator.shutdown(wait=True)
    # A second shutdown must not raise.
    coordinator.shutdown(wait=False)
