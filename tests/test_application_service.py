from __future__ import annotations

from pathlib import Path

import pytest

from kenkui.models import AppConfig, ChapterPreset, ChapterSelection, JobConfig, JobStatus
from kenkui.models.api import JobCreateRequest
from kenkui.services.application_service import KenkuiService, job_create_request_to_config
from kenkui.services.job_service import build_processing_config


def test_job_create_request_to_config_preserves_api_fields(tmp_path):
    req = JobCreateRequest(
        ebook_path=str(tmp_path / "book.epub"),
        voice="alba",
        chapter_selection=ChapterSelection(
            preset=ChapterPreset.MANUAL,
            included=[1, 3],
        ),
        output_path=str(tmp_path / "out"),
        narration_mode="multi",
        speaker_voices={"NARRATOR": "alba", "alice": "clara"},
        job_pause_line_ms=1200,
        job_apostrophe_mode="always_remove",
    )

    job = job_create_request_to_config(req)

    assert job.ebook_path == tmp_path / "book.epub"
    assert job.output_path == tmp_path / "out"
    assert job.chapter_selection.included == [1, 3]
    assert job.narration_mode.value == "multi"
    assert job.speaker_voices["alice"] == "clara"
    assert job.job_pause_line_ms == 1200
    assert job.job_apostrophe_mode.value == "always_remove"


def test_build_processing_config_is_shared_job_boundary(tmp_path):
    job = JobConfig(
        ebook_path=tmp_path / "book.epub",
        voice="clara",
        chapter_selection=ChapterSelection(
            preset=ChapterPreset.MANUAL,
            included=[2],
        ),
        output_path=tmp_path / "audio",
        speaker_voices={"NARRATOR": "clara"},
        job_m4b_bitrate="64",
        job_post_processing_enabled=False,
    )
    app_config = AppConfig(default_voice="alba", workers=3, pause_line_ms=900)

    cfg = build_processing_config(job, app_config)

    assert cfg.voice == "clara"
    assert cfg.output_path == tmp_path / "audio"
    assert cfg.workers == 3
    assert cfg.pause_line_ms == 900
    assert cfg.m4b_bitrate == "64k"
    assert cfg.chapter_filters[0].type == "index"
    assert cfg.chapter_filters[0].value == "2"
    assert cfg._included_indices == [2]
    assert cfg.post_processing.enabled is False


def test_application_service_queue_contract_persists(tmp_path):
    queue_file = tmp_path / "queue.toml"
    service = KenkuiService(queue_file=queue_file, app_config=AppConfig(default_voice="alba"))

    item = service.add_job(JobConfig(ebook_path=Path("book.epub"), voice="clara"))
    queue = service.queue()

    assert queue.pending_count == 1
    assert queue.items[0].id == item.id
    assert queue.items[0].status == "pending"
    assert queue.items[0].job["voice"] == "clara"
    assert queue_file.exists()

    restored = KenkuiService(queue_file=queue_file)
    assert restored.queue().items[0].id == item.id
    assert restored.get_job_item(item.id).status == JobStatus.PENDING


def test_application_service_resets_stale_processing_jobs(tmp_path):
    queue_file = tmp_path / "queue.toml"
    service = KenkuiService(queue_file=queue_file)
    item = service.add_job(JobConfig(ebook_path=Path("book.epub")))
    item.status = JobStatus.PROCESSING
    service._save()

    restored = KenkuiService(queue_file=queue_file)

    assert restored.get_job_item(item.id).status == JobStatus.PENDING


def test_http_adapter_uses_service_contract(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kenkui.server import api

    service = KenkuiService(queue_file=tmp_path / "http-queue.toml")
    monkeypatch.setattr(api, "get_service", lambda: service)

    client = TestClient(api.create_app())
    response = client.get("/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["api_version"] == "v1"
    assert "local-queue" in body["capabilities"]
