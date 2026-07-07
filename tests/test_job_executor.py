from __future__ import annotations

from pathlib import Path

from kenkui.models import AppConfig, JobConfig, JobStatus, QueueItem
from kenkui.services.execution_service import ExecutionOutcome
from kenkui.services.job_executor import JobExecutor, progress_update_from_args
from kenkui.services.queue_manager import QueueManager


def _queue(tmp_path) -> QueueManager:
    return QueueManager(queue_file=tmp_path / "queue.toml", app_config=AppConfig())


def _add(qm: QueueManager, job_id: str, status: JobStatus = JobStatus.PENDING) -> QueueItem:
    return qm.add(
        QueueItem(
            id=job_id,
            job=JobConfig(ebook_path=Path("book.epub")),
            status=status,
            execution_provider="local",
        )
    )


class _Provider:
    def __init__(self, outcome: ExecutionOutcome) -> None:
        self._outcome = outcome
        self.cancel_calls: list[str] = []

    def execute(self, *, item, cfg, app_config, progress_callback, metadata_callback, pause_check, cancel_check):
        del item, cfg, app_config, progress_callback, metadata_callback, pause_check, cancel_check
        return self._outcome

    def cancel(self, item):
        self.cancel_calls.append(item.id)


def test_process_job_success_marks_completed(tmp_path):
    qm = _queue(tmp_path)
    item = _add(qm, "aaa")
    provider = _Provider(ExecutionOutcome(success=True, output_path=str(tmp_path / "a.m4b")))
    executor = JobExecutor(queue=qm, resolve_provider=lambda i: provider)

    executor.process_job(item)

    assert item.status == JobStatus.COMPLETED
    assert item.progress == 100.0


def test_process_job_failure_marks_failed(tmp_path):
    qm = _queue(tmp_path)
    item = _add(qm, "bad")
    provider = _Provider(ExecutionOutcome(success=False, error_message="boom"))
    executor = JobExecutor(queue=qm, resolve_provider=lambda i: provider)

    executor.process_job(item)

    assert item.status == JobStatus.FAILED
    assert item.error_message


def test_pause_and_resume_transitions(tmp_path):
    qm = _queue(tmp_path)
    item = _add(qm, "p", JobStatus.PROCESSING)
    executor = JobExecutor(queue=qm, resolve_provider=lambda i: _Provider(ExecutionOutcome(success=True)))

    assert executor.pause_job("p") is True
    assert executor._pause_requested is True

    item.status = JobStatus.PAUSED
    # start_processing would spawn a thread; stub it out for a unit test.
    executor.start_processing = lambda: True  # type: ignore[assignment]
    assert executor.resume_job("p") is True
    assert item.status == JobStatus.PENDING
    assert executor._pause_requested is False


def test_cancel_pending_job_is_terminal_without_provider(tmp_path):
    qm = _queue(tmp_path)
    _add(qm, "c", JobStatus.PENDING)
    executor = JobExecutor(queue=qm, resolve_provider=lambda i: _Provider(ExecutionOutcome(success=True)))

    assert executor.cancel_job("c") is True
    assert qm.get("c").status == JobStatus.CANCELLED


def test_cancel_processing_job_sets_flag_and_calls_provider(tmp_path):
    qm = _queue(tmp_path)
    _add(qm, "run", JobStatus.PROCESSING)
    provider = _Provider(ExecutionOutcome(success=True))
    executor = JobExecutor(queue=qm, resolve_provider=lambda i: provider)

    assert executor.cancel_job("run") is True
    assert executor._cancel_requested_job_id == "run"
    assert provider.cancel_calls == ["run"]


def test_progress_update_from_args_shapes():
    assert progress_update_from_args(40, "Chapter 4", 90) == (40.0, "Chapter 4", 90)
    assert progress_update_from_args("Preparing") == (0.0, "Preparing", 0)
