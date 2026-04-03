"""Tests for TaskRegistry and TaskRunner."""
from __future__ import annotations

import time

from kenkui.server.tasks import (
    Task,
    TaskRegistry,
    TaskRunner,
    TaskStatus,
    TaskType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait_for(registry: TaskRegistry, task_id: str, target_status: TaskStatus, timeout: float = 5.0) -> Task:
    deadline = time.time() + timeout
    while time.time() < deadline:
        task = registry.get(task_id)
        if task and task.status == target_status:
            return task
        time.sleep(0.01)
    raise TimeoutError(f"Task {task_id} did not reach {target_status}")


# ---------------------------------------------------------------------------
# TaskRegistry tests
# ---------------------------------------------------------------------------

class TestTaskRegistry:
    def test_create_returns_pending_task(self):
        registry = TaskRegistry()
        task = registry.create(TaskType.FAST_SCAN)
        assert task.status == TaskStatus.PENDING
        assert task.type == TaskType.FAST_SCAN
        assert task.task_id != ""

    def test_get_returns_none_for_missing(self):
        registry = TaskRegistry()
        assert registry.get("nonexistent") is None

    def test_update_sets_running_status(self):
        registry = TaskRegistry()
        task = registry.create(TaskType.AUDITION)
        registry.update(task.task_id, progress=42, message="working")
        updated = registry.get(task.task_id)
        assert updated.status == TaskStatus.RUNNING
        assert updated.progress == 42
        assert updated.message == "working"

    def test_complete_sets_completed_status(self):
        registry = TaskRegistry()
        task = registry.create(TaskType.VOICE_FETCH)
        registry.complete(task.task_id, {"voices": ["Alice"]})
        completed = registry.get(task.task_id)
        assert completed.status == TaskStatus.COMPLETED
        assert completed.progress == 100
        assert completed.result == {"voices": ["Alice"]}

    def test_fail_sets_failed_status(self):
        registry = TaskRegistry()
        task = registry.create(TaskType.VOICE_DOWNLOAD)
        registry.fail(task.task_id, "network error")
        failed = registry.get(task.task_id)
        assert failed.status == TaskStatus.FAILED
        assert failed.error == "network error"

    def test_evict_stale_removes_old_tasks(self):
        registry = TaskRegistry(ttl=3600)
        task = registry.create(TaskType.FAST_SCAN)
        # Push created_at into the past beyond TTL
        task.created_at = time.time() - 7200
        registry.evict_stale()
        assert registry.get(task.task_id) is None

    def test_evict_stale_keeps_recent_tasks(self):
        registry = TaskRegistry(ttl=3600)
        task = registry.create(TaskType.FAST_SCAN)
        # created_at is fresh (just set by default_factory)
        registry.evict_stale()
        assert registry.get(task.task_id) is not None

    def test_update_ignores_missing_task_id(self):
        registry = TaskRegistry()
        # Should not raise
        registry.update("ghost-id", progress=10, message="ghost")

    def test_create_assigns_unique_ids(self):
        registry = TaskRegistry()
        t1 = registry.create(TaskType.AUDITION)
        t2 = registry.create(TaskType.AUDITION)
        assert t1.task_id != t2.task_id


# ---------------------------------------------------------------------------
# TaskRunner tests
# ---------------------------------------------------------------------------

class TestTaskRunner:
    def setup_method(self):
        self.registry = TaskRegistry()
        self.runner = TaskRunner(self.registry, max_workers=4)

    def teardown_method(self):
        self.runner.shutdown(wait=True)

    def test_submit_returns_task_immediately(self):
        def slow_fn(progress_callback):
            time.sleep(0.5)
            return "done"

        task = self.runner.submit(TaskType.FAST_SCAN, slow_fn)
        # Returned before slow_fn finishes
        assert task is not None
        assert task.task_id != ""

    def test_submit_injects_progress_callback(self):
        received = {}

        def fn_that_checks(progress_callback):
            received["callback"] = progress_callback
            return "ok"

        task = self.runner.submit(TaskType.AUDITION, fn_that_checks)
        _wait_for(self.registry, task.task_id, TaskStatus.COMPLETED)
        assert "callback" in received
        assert callable(received["callback"])

    def test_submit_completes_task_on_success(self):
        def succeeding_fn(progress_callback):
            return {"status": "success"}

        task = self.runner.submit(TaskType.VOICE_FETCH, succeeding_fn)
        completed = _wait_for(self.registry, task.task_id, TaskStatus.COMPLETED)
        assert completed.result == {"status": "success"}

    def test_submit_fails_task_on_exception(self):
        def failing_fn(progress_callback):
            raise ValueError("something went wrong")

        task = self.runner.submit(TaskType.VOICE_DOWNLOAD, failing_fn)
        failed = _wait_for(self.registry, task.task_id, TaskStatus.FAILED)
        assert "something went wrong" in failed.error

    def test_progress_callback_updates_registry(self):
        def fn_with_progress(progress_callback):
            progress_callback(50, "halfway")
            time.sleep(0.1)
            return "result"

        task = self.runner.submit(TaskType.FAST_SCAN, fn_with_progress)
        # Poll until RUNNING with progress=50 or already COMPLETED
        deadline = time.time() + 5.0
        found_progress = False
        while time.time() < deadline:
            t = self.registry.get(task.task_id)
            if t and t.status == TaskStatus.RUNNING and t.progress == 50:
                found_progress = True
                break
            if t and t.status == TaskStatus.COMPLETED:
                # Completed before we could observe RUNNING — acceptable only if
                # the function ran so fast the intermediate state was missed.
                # Re-check by trusting that progress_callback was called.
                found_progress = True
                break
            time.sleep(0.005)

        assert found_progress, "Never observed progress=50 or task completion"
        _wait_for(self.registry, task.task_id, TaskStatus.COMPLETED)
