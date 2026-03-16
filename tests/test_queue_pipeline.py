"""Tests for kenkui queue and processing pipeline."""

import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Test constants
TEST_EBOOK_PATH = Path("/tmp/test_ebook.epub")


class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_job_status_values(self):
        from kenkui.models import JobStatus

        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.PROCESSING.value == "processing"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"


class TestJobConfig:
    """Tests for JobConfig model."""

    def test_job_config_creation(self):
        from kenkui.models import JobConfig

        job = JobConfig(ebook_path=TEST_EBOOK_PATH)
        assert job.ebook_path == TEST_EBOOK_PATH
        assert job.voice == "alba"  # default
        assert job.name == TEST_EBOOK_PATH.stem  # auto-set from path

    def test_job_config_with_custom_voice(self):
        from kenkui.models import JobConfig

        job = JobConfig(ebook_path=TEST_EBOOK_PATH, voice="custom_voice")
        assert job.voice == "custom_voice"

    def test_job_config_to_dict(self):
        from kenkui.models import JobConfig

        job = JobConfig(ebook_path=TEST_EBOOK_PATH, voice="test_voice")
        data = job.to_dict()
        assert data["ebook_path"] == str(TEST_EBOOK_PATH)
        assert data["voice"] == "test_voice"

    def test_job_config_from_dict(self):
        from kenkui.models import JobConfig

        data = {
            "ebook_path": str(TEST_EBOOK_PATH),
            "voice": "test_voice",
            "chapter_selection": {},
            "output_path": None,
            "name": "test",
        }
        job = JobConfig.from_dict(data)
        assert job.ebook_path == TEST_EBOOK_PATH
        assert job.voice == "test_voice"


class TestQueueItem:
    """Tests for QueueItem model."""

    def test_queue_item_creation(self):
        from kenkui.models import JobConfig, JobStatus, QueueItem

        job = JobConfig(ebook_path=TEST_EBOOK_PATH)
        item = QueueItem(id="test123", job=job, status=JobStatus.PENDING)
        assert item.id == "test123"
        assert item.job == job
        assert item.status == JobStatus.PENDING
        assert item.progress == 0.0

    def test_queue_item_to_dict(self):
        from kenkui.models import JobConfig, JobStatus, QueueItem

        job = JobConfig(ebook_path=TEST_EBOOK_PATH)
        item = QueueItem(id="test123", job=job, status=JobStatus.PENDING)
        data = item.to_dict()
        assert data["id"] == "test123"
        assert data["status"] == "pending"

    def test_queue_item_from_dict(self):
        from kenkui.models import JobStatus, QueueItem

        data = {
            "id": "test123",
            "job": {
                "ebook_path": str(TEST_EBOOK_PATH),
                "voice": "alba",
                "chapter_selection": {},
                "output_path": None,
                "name": "test",
            },
            "status": "pending",
            "progress": 0.0,
            "current_chapter": "",
            "eta_seconds": 0,
            "error_message": "",
        }
        item = QueueItem.from_dict(data)
        assert item.id == "test123"
        assert item.status == JobStatus.PENDING
        assert item.progress == 0.0


class TestQueueManager:
    """Tests for QueueManager class."""

    @pytest.fixture
    def temp_queue_file(self):
        """Create a temporary queue file for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Patch CONFIG_DIR to use temp directory
            from kenkui import config

            original_dir = config.CONFIG_DIR
            config.CONFIG_DIR = Path(tmpdir)
            yield Path(tmpdir) / "queue.yaml"
            config.CONFIG_DIR = original_dir

    @pytest.fixture
    def queue_manager(self, temp_queue_file):
        """Create a fresh QueueManager for each test."""
        # Patch the queue file path
        import kenkui.queue
        from kenkui.queue import QueueManager

        original_file = kenkui.queue.QUEUE_FILE
        kenkui.queue.QUEUE_FILE = temp_queue_file

        # Create new instance
        mgr = QueueManager()
        mgr.items = []  # Clear any loaded items

        yield mgr

        # Cleanup
        kenkui.queue.QUEUE_FILE = original_file
        kenkui.queue._queue_manager = None

    def test_queue_manager_initialization(self, queue_manager):
        """Test QueueManager can be initialized."""
        assert queue_manager is not None
        assert isinstance(queue_manager.items, list)

    def test_add_job(self, queue_manager):
        """Test adding a job to the queue."""
        from kenkui.models import JobConfig

        job = JobConfig(ebook_path=TEST_EBOOK_PATH)
        item = queue_manager.add(job)

        assert len(queue_manager.items) == 1
        assert item.id is not None
        assert item.status.value == "pending"

    def test_pending_items_property(self, queue_manager):
        """Test getting pending items."""
        from kenkui.models import JobConfig

        job1 = JobConfig(ebook_path=Path("/tmp/book1.epub"))
        job2 = JobConfig(ebook_path=Path("/tmp/book2.epub"))

        queue_manager.add(job1)
        queue_manager.add(job2)

        pending = queue_manager.pending_items
        assert len(pending) == 2

    def test_start_next_returns_pending_item(self, queue_manager):
        """Test start_next returns a pending item."""
        from kenkui.models import JobConfig, JobStatus

        job = JobConfig(ebook_path=TEST_EBOOK_PATH)
        queue_manager.add(job)

        item = queue_manager.start_next()

        assert item is not None
        assert item.status == JobStatus.PROCESSING

    def test_start_next_returns_none_when_no_pending(self, queue_manager):
        """Test start_next returns None when no pending items."""
        item = queue_manager.start_next()
        assert item is None

    def test_current_item_property(self, queue_manager):
        """Test current_item returns PROCESSING item."""
        from kenkui.models import JobConfig, JobStatus

        job = JobConfig(ebook_path=TEST_EBOOK_PATH)
        queue_manager.add(job)

        # Initially no current item
        assert queue_manager.current_item is None

        # After starting, should return the processing item
        queue_manager.start_next()
        current = queue_manager.current_item
        assert current is not None
        assert current.status == JobStatus.PROCESSING

    def test_remove_pending_item(self, queue_manager):
        """Test removing a pending item."""
        from kenkui.models import JobConfig

        job = JobConfig(ebook_path=TEST_EBOOK_PATH)
        item = queue_manager.add(job)

        result = queue_manager.remove(item.id)

        assert result is True
        assert len(queue_manager.items) == 0

    def test_remove_processing_item_fails(self, queue_manager):
        """Test removing a processing item returns False."""
        from kenkui.models import JobConfig

        job = JobConfig(ebook_path=TEST_EBOOK_PATH)
        item = queue_manager.add(job)
        queue_manager.start_next()  # Mark as processing

        result = queue_manager.remove(item.id)

        assert result is False
        assert len(queue_manager.items) == 1

    def test_clear_all(self, queue_manager):
        """Test clearing all items."""
        from kenkui.models import JobConfig

        queue_manager.add(JobConfig(ebook_path=Path("/tmp/book1.epub")))
        queue_manager.add(JobConfig(ebook_path=Path("/tmp/book2.epub")))

        queue_manager.clear_all()

        assert len(queue_manager.items) == 0

    def test_reset_stale_processing(self, queue_manager):
        """Test resetting stale PROCESSING jobs to PENDING."""
        from kenkui.models import JobConfig, JobStatus

        job = JobConfig(ebook_path=TEST_EBOOK_PATH)
        item = queue_manager.add(job)
        item.status = JobStatus.PROCESSING  # Simulate stale state
        item.progress = 50.0

        queue_manager.reset_stale_processing()

        assert item.status == JobStatus.PENDING
        assert item.progress == 0.0


class TestProcessor:
    """Tests for Processor class."""

    @pytest.fixture
    def processor(self):
        """Create a fresh Processor for each test."""
        from kenkui.processor import Processor

        p = Processor()
        yield p
        # Cleanup
        p.stop()

    @pytest.fixture
    def mock_queue_item(self):
        """Create a mock QueueItem for testing."""
        from kenkui.models import JobConfig, JobStatus, QueueItem

        job = JobConfig(ebook_path=TEST_EBOOK_PATH)
        item = QueueItem(id="test123", job=job, status=JobStatus.PENDING)
        return item

    @pytest.fixture
    def mock_app_config(self):
        """Create a mock AppConfig for testing."""
        from kenkui.models import AppConfig

        return AppConfig()

    def test_processor_initialization(self, processor):
        """Test Processor can be initialized."""
        assert processor is not None
        assert processor._running is False
        assert processor.is_running is False

    def test_start_sets_running_flag(self, processor, mock_queue_item, mock_app_config):
        """Test that start() sets the running flag."""
        # Mock the _process method to avoid actual processing
        with patch.object(processor, "_process"):
            result = processor.start(mock_queue_item, mock_app_config)

            assert result is True
            assert processor._running is True

    def test_start_returns_false_when_already_running(
        self, processor, mock_queue_item, mock_app_config
    ):
        """Test that start() returns False when already running."""
        with patch.object(processor, "_process"):
            processor.start(mock_queue_item, mock_app_config)
            result = processor.start(mock_queue_item, mock_app_config)

            assert result is False

    def test_stop_resets_running_flag(self, processor, mock_queue_item, mock_app_config):
        """Test that stop() resets the running flag."""
        with patch.object(processor, "_process"):
            processor.start(mock_queue_item, mock_app_config)
            processor.stop()

            assert processor._running is False

    def test_callback_registration(self, processor):
        """Test event callback registration."""
        callback = Mock()
        processor.on("started", callback)

        processor._emit("started", "test_id")

        callback.assert_called_once_with("test_id")

    def test_progress_callback_called(self, processor):
        """Test that _progress_callback is invoked directly and that _emit
        fires registered event callbacks (these are separate mechanisms)."""

        # Test 1: _progress_callback is called directly
        called = threading.Event()
        values: dict = {}

        def progress_callback(progress, chapter, eta):
            values["progress"] = progress
            values["chapter"] = chapter
            values["eta"] = eta
            called.set()

        processor._progress_callback = progress_callback
        # Call it directly (as the Processor._process method would)
        processor._progress_callback(50.0, "Chapter 1", 300)

        assert called.is_set()
        assert values["progress"] == 50.0
        assert values["chapter"] == "Chapter 1"

        # Test 2: _emit fires registered event callbacks
        event_called = threading.Event()
        processor.on("started", lambda job_id: event_called.set())
        processor._emit("started", "test_id")
        assert event_called.is_set()


class TestProcessorIntegration:
    """Integration tests for Processor with mocks."""

    def test_processor_thread_actually_starts(self):
        """Test that the processing thread actually starts."""
        from kenkui.models import AppConfig, JobConfig, JobStatus, QueueItem
        from kenkui.processor import Processor

        processor = Processor()
        job = JobConfig(ebook_path=TEST_EBOOK_PATH)
        item = QueueItem(id="test123", job=job, status=JobStatus.PENDING)
        app_config = AppConfig()

        thread_started = {"started": False}

        def mock_process(item, app_config):
            thread_started["started"] = True
            # Don't actually run processing, just emit started and exit
            processor._emit("started", item.id)
            time.sleep(0.1)  # Small delay to allow thread to be detected
            processor._running = False

        processor._process = mock_process

        result = processor.start(item, app_config)

        # Wait a bit for thread to start
        time.sleep(0.2)

        assert result is True
        assert thread_started["started"] is True
        assert processor.is_running is False  # After mock_process exits

        processor.stop()

    def test_processor_stores_current_item(self):
        """Test that processor stores the current item being processed."""
        from kenkui.models import AppConfig, JobConfig, JobStatus, QueueItem
        from kenkui.processor import Processor

        processor = Processor()
        job = JobConfig(ebook_path=TEST_EBOOK_PATH)
        item = QueueItem(id="test123", job=job, status=JobStatus.PENDING)
        app_config = AppConfig()

        with patch.object(processor, "_process"):
            processor.start(item, app_config)

            assert processor._current_item is not None
            assert processor._current_item.id == "test123"


# Debug test to understand actual behavior
class TestDebugInfo:
    """Debug tests to understand actual behavior."""

    def test_queue_singleton_behavior(self):
        """Test that get_queue_manager returns a singleton."""
        from kenkui.queue import get_queue_manager

        q1 = get_queue_manager()
        q2 = get_queue_manager()
        assert q1 is q2

    def test_processor_singleton_behavior(self):
        """Test that get_processor returns a singleton."""
        from kenkui.processor import get_processor, reset_processor

        reset_processor()  # Start fresh
        p1 = get_processor()
        p2 = get_processor()
        assert p1 is p2

    def test_queue_persistence_file_created(self):
        """Test that queue saves to file."""
        import kenkui.queue
        from kenkui.queue import QueueManager

        with tempfile.TemporaryDirectory() as tmpdir:
            original = kenkui.queue.QUEUE_FILE
            kenkui.queue.QUEUE_FILE = Path(tmpdir) / "queue.yaml"

            try:
                q = QueueManager()
                from kenkui.models import JobConfig

                q.add(JobConfig(ebook_path=TEST_EBOOK_PATH))

                assert kenkui.queue.QUEUE_FILE.exists()
            finally:
                kenkui.queue.QUEUE_FILE = original
                kenkui.queue._queue_manager = None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
