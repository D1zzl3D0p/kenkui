"""Tests for kenkui queue and processing pipeline."""

from pathlib import Path

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
