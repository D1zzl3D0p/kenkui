"""Unit tests for the WorkerServer class."""

import tempfile
from pathlib import Path

import pytest

from kenkui.models import AppConfig, JobConfig, JobStatus
from kenkui.server.worker import WorkerServer, get_server, reset_server

TEST_EBOOK_PATH = Path("/tmp/test_ebook.epub")


@pytest.fixture
def temp_queue_file():
    """Create a temporary queue file for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from kenkui import config

        original_dir = config.CONFIG_DIR
        config.CONFIG_DIR = Path(tmpdir)

        from kenkui.server import worker

        original_file = worker.QUEUE_FILE
        worker.QUEUE_FILE = Path(tmpdir) / "queue.yaml"

        yield Path(tmpdir) / "queue.yaml"

        config.CONFIG_DIR = original_dir
        worker.QUEUE_FILE = original_file


@pytest.fixture
def server(temp_queue_file):
    """Create a fresh WorkerServer for each test."""
    reset_server()
    s = WorkerServer()
    s._items = []
    yield s
    s.stop_processing()
    reset_server()


class TestWorkerServerInit:
    def test_server_initialization(self, server):
        assert server is not None
        assert server.is_running is False
        assert len(server.all_items) == 0


class TestJobManagement:
    def test_add_job(self, server):
        job = JobConfig(ebook_path=TEST_EBOOK_PATH)
        item = server.add_job(job)

        assert item is not None
        assert item.id is not None
        assert item.status == JobStatus.PENDING
        assert len(server.all_items) == 1

    def test_get_job(self, server):
        job = JobConfig(ebook_path=TEST_EBOOK_PATH)
        item = server.add_job(job)

        retrieved = server.get_job(item.id)
        assert retrieved is not None
        assert retrieved.id == item.id

    def test_get_job_not_found(self, server):
        job = server.get_job("nonexistent")
        assert job is None

    def test_remove_job(self, server):
        job = JobConfig(ebook_path=TEST_EBOOK_PATH)
        item = server.add_job(job)

        result = server.remove_job(item.id)
        assert result is True
        assert len(server.all_items) == 0

    def test_remove_processing_job_fails(self, server):
        job = JobConfig(ebook_path=TEST_EBOOK_PATH)
        item = server.add_job(job)
        item.status = JobStatus.PROCESSING

        result = server.remove_job(item.id)
        assert result is False

    def test_clear_all_jobs(self, server):
        server.add_job(JobConfig(ebook_path=TEST_EBOOK_PATH))
        server.add_job(JobConfig(ebook_path=Path("/tmp/test2.epub")))

        server.clear_all_jobs()

        assert len(server.all_items) == 0


class TestQueueProperties:
    def test_pending_items(self, server):
        job1 = JobConfig(ebook_path=TEST_EBOOK_PATH)
        job2 = JobConfig(ebook_path=Path("/tmp/test2.epub"))

        server.add_job(job1)
        server.add_job(job2)

        pending = server.pending_items
        assert len(pending) == 2

    def test_current_item_none_when_idle(self, server):
        assert server.current_item is None


class TestProcessingControl:
    def test_start_processing_no_pending(self, server):
        result = server.start_processing()
        assert result is True

    def test_start_next_job(self, server):
        job = JobConfig(ebook_path=TEST_EBOOK_PATH)
        server.add_job(job)

        item = server.start_next_job()

        assert item is not None
        assert item.status == JobStatus.PROCESSING

    def test_start_next_job_no_pending(self, server):
        item = server.start_next_job()
        assert item is None


class TestProgressUpdates:
    def test_update_progress(self, server):
        job = JobConfig(ebook_path=TEST_EBOOK_PATH)
        item = server.add_job(job)

        server.update_progress(item.id, 50.0, "Chapter 1", 300)

        updated = server.get_job(item.id)
        assert updated.progress == 50.0
        assert updated.current_chapter == "Chapter 1"
        assert updated.eta_seconds == 300

    def test_complete_job(self, server):
        job = JobConfig(ebook_path=TEST_EBOOK_PATH)
        item = server.add_job(job)
        server.start_next_job()

        server.complete_job(item.id)

        completed = server.get_job(item.id)
        assert completed.status == JobStatus.COMPLETED
        assert completed.progress == 100.0

    def test_fail_job(self, server):
        job = JobConfig(ebook_path=TEST_EBOOK_PATH)
        item = server.add_job(job)
        server.start_next_job()

        server.fail_job(item.id, "Test error")

        failed = server.get_job(item.id)
        assert failed.status == JobStatus.FAILED
        assert failed.error_message == "Test error"


class TestConfig:
    def test_get_set_app_config(self, server):
        config = AppConfig(workers=4, verbose=True)
        server.app_config = config

        assert server.app_config.workers == 4
        assert server.app_config.verbose is True


class TestSingleton:
    def test_get_server_returns_singleton(self):
        reset_server()
        s1 = get_server()
        s2 = get_server()

        assert s1 is s2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
