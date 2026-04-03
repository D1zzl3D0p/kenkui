"""Tests for new API routes added in Task 9."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from kenkui.server.api import app
from kenkui.server.tasks import Task, TaskStatus, TaskType


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Minimal fake objects
# ---------------------------------------------------------------------------


def _make_fake_task(task_id: str = "fake-task-id") -> Task:
    task = Task(task_id=task_id, type=TaskType.FAST_SCAN)
    return task


def _make_mock_server(task: Task | None = None):
    server = MagicMock()
    server.get_job.return_value = None
    server.book_cache = MagicMock()
    server.task_registry = MagicMock()
    server.task_registry.get.return_value = None
    if task is not None:
        server.task_runner = MagicMock()
        server.task_runner.submit.return_value = task
    return server


# ---------------------------------------------------------------------------
# Books
# ---------------------------------------------------------------------------


def test_parse_book_not_found(client):
    """POST /books/parse with a nonexistent path should return 404."""
    with patch("kenkui.server.api.get_server") as mock_get_server:
        mock_server = _make_mock_server()
        mock_get_server.return_value = mock_server

        with patch("kenkui.services.book_service.parse_book") as mock_parse:
            mock_parse.side_effect = FileNotFoundError("File not found: /no/such/file.epub")
            resp = client.post("/books/parse", json={"ebook_path": "/no/such/file.epub"})

    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_scan_book_returns_202(client):
    """POST /books/scan should return 202 with a task_id when task_runner.submit works."""
    fake_task = _make_fake_task("scan-task-001")

    with patch("kenkui.server.api.get_server") as mock_get_server:
        mock_server = _make_mock_server(task=fake_task)
        mock_get_server.return_value = mock_server

        resp = client.post("/books/scan", json={"ebook_path": "/tmp/book.epub"})

    assert resp.status_code == 202
    data = resp.json()
    assert data["task_id"] == "scan-task-001"
    assert data["status"] == TaskStatus.PENDING.value


# ---------------------------------------------------------------------------
# Voices
# ---------------------------------------------------------------------------


def test_list_voices_returns_list(client):
    """GET /voices should return 200 with voices list and total."""
    @dataclass
    class FakeVoice:
        name: str = "alba"
        source: str = "compiled"
        gender: str | None = "Female"
        accent: str | None = "British"
        dataset: str | None = None
        speaker_id: str | None = None
        description: str = "A lovely voice"
        display_label: str = "Alba (British Female)"
        excluded: bool = False

    with patch("kenkui.services.voice_service.list_voices", return_value=[FakeVoice()]):
        resp = client.get("/voices")

    assert resp.status_code == 200
    data = resp.json()
    assert "voices" in data
    assert "total" in data
    assert data["total"] == 1
    assert data["voices"][0]["name"] == "alba"


def test_get_voice_not_found(client):
    """GET /voices/{name} with an unknown voice should return 404."""
    with patch("kenkui.services.voice_service.get_voice", return_value=None):
        resp = client.get("/voices/nonexistent_voice_xyz")

    assert resp.status_code == 404
    assert "nonexistent_voice_xyz" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def test_get_task_not_found(client):
    """GET /tasks/{task_id} with an unknown id should return 404."""
    with patch("kenkui.server.api.get_server") as mock_get_server:
        mock_server = _make_mock_server()
        mock_server.task_registry.get.return_value = None
        mock_get_server.return_value = mock_server

        resp = client.get("/tasks/nonexistent-id")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_get_hf_auth_returns_status(client):
    """GET /auth/huggingface should return 200 with an authenticated field."""
    @dataclass
    class FakeHFStatus:
        authenticated: bool = False
        username: str | None = None
        has_pocket_tts_access: bool = False

    with patch("kenkui.services.auth_service.get_hf_status", return_value=FakeHFStatus()):
        resp = client.get("/auth/huggingface")

    assert resp.status_code == 200
    data = resp.json()
    assert "authenticated" in data


def test_login_hf_success(client):
    """POST /auth/huggingface with valid token returns authenticated status."""
    mock_login_result = MagicMock()
    mock_login_result.authenticated = True
    mock_login_result.error = None
    mock_status = MagicMock()
    mock_status.authenticated = True
    mock_status.username = "testuser"
    mock_status.has_pocket_tts_access = True
    with patch("kenkui.services.auth_service.login", return_value=mock_login_result), \
         patch("kenkui.services.auth_service.get_hf_status", return_value=mock_status):
        response = client.post("/auth/huggingface", json={"token": "hf_testtoken"})
    assert response.status_code == 200
    data = response.json()
    assert data["authenticated"] is True
    assert data["error"] is None


# ---------------------------------------------------------------------------
# Audition
# ---------------------------------------------------------------------------


def test_audition_returns_202(client):
    """POST /voices/audition should return 202 with a task_id."""
    fake_task = _make_fake_task("audition-task-001")
    fake_task.type = TaskType.AUDITION

    with patch("kenkui.server.api.get_server") as mock_get_server:
        mock_server = _make_mock_server(task=fake_task)
        mock_get_server.return_value = mock_server

        resp = client.post(
            "/voices/audition",
            json={"voice_name": "alba", "text": "Hello world"},
        )

    assert resp.status_code == 202
    data = resp.json()
    assert data["task_id"] == "audition-task-001"


# ---------------------------------------------------------------------------
# Queue Cast
# ---------------------------------------------------------------------------


def test_get_cast_job_not_found(client):
    """GET /queue/{job_id}/cast with unknown job_id should return 404."""
    with patch("kenkui.server.api.get_server") as mock_get_server:
        mock_server = _make_mock_server()
        mock_server.get_job.return_value = None
        mock_get_server.return_value = mock_server

        resp = client.get("/queue/fakeid/cast")

    assert resp.status_code == 404
    assert "Job not found" in resp.json()["detail"]
