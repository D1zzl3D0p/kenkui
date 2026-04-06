"""Integration tests for the full TUI-Server flow using real HTTP."""

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import pytest

from kenkui import config


@pytest.fixture
def temp_config_dir():
    """Create a temporary config directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = config.CONFIG_DIR
        config.CONFIG_DIR = Path(tmpdir)
        yield Path(tmpdir)
        config.CONFIG_DIR = original_dir


@pytest.fixture
def server_process(temp_config_dir):
    """Start a real server process and yield the port."""
    port = 14001  # Use a unique port for testing

    # Start server process
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "kenkui.server.server",
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    max_attempts = 30
    for _ in range(max_attempts):
        try:
            response = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
            if response.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.5)
    else:
        proc.terminate()
        pytest.fail("Server failed to start within timeout")

    yield port

    # Cleanup
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture
def client(server_process):
    """Create an HTTP client connected to the test server."""
    port = server_process
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30.0) as c:
        yield c


class TestHealthEndpoint:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.8.0"


class TestQueueEndpoints:
    def test_get_empty_queue(self, client):
        # Clear any existing jobs first
        client.delete("/queue")

        response = client.get("/queue")
        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["pending_count"] == 0

    def test_add_job(self, client):
        response = client.post(
            "/queue",
            json={"ebook_path": "/tmp/test.epub", "voice": "alba"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["status"] == "pending"

    def test_get_job(self, client):
        # Add a job
        add_resp = client.post(
            "/queue",
            json={"ebook_path": "/tmp/test.epub", "voice": "alba"},
        )
        job_id = add_resp.json()["id"]

        # Get it
        response = client.get(f"/queue/{job_id}")
        assert response.status_code == 200
        assert response.json()["id"] == job_id

    def test_remove_job(self, client):
        # Add a job
        add_resp = client.post(
            "/queue",
            json={"ebook_path": "/tmp/test.epub", "voice": "alba"},
        )
        job_id = add_resp.json()["id"]

        # Remove it
        response = client.delete(f"/queue/{job_id}")
        assert response.status_code == 200

    def test_clear_queue(self, client):
        # Add jobs
        client.post("/queue", json={"ebook_path": "/tmp/test1.epub", "voice": "alba"})
        client.post("/queue", json={"ebook_path": "/tmp/test2.epub", "voice": "alba"})

        # Clear
        response = client.delete("/queue")
        assert response.status_code == 200

        # Verify empty
        queue = client.get("/queue").json()
        assert queue["items"] == []


class TestProcessingEndpoints:
    def test_start_processing(self, client):
        # Add a job
        client.post("/queue", json={"ebook_path": "/tmp/test.epub", "voice": "alba"})

        # Start processing
        response = client.post("/queue/start")
        assert response.status_code == 200

    def test_start_processing_no_pending(self, client):
        # Clear queue first
        client.delete("/queue")

        # When there's no pending job, the server still returns 200
        # but the processing thread immediately exits
        response = client.post("/queue/start")
        assert response.status_code == 200


class TestStatusEndpoint:
    def test_get_status_idle(self, client):
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["idle", "running"]


class TestConfigEndpoints:
    def test_get_config(self, client):
        response = client.get("/config")
        assert response.status_code == 200
        assert "config" in response.json()

    def test_update_config(self, client):
        new_config = {
            "workers": 8,
            "pause_line_ms": 500,
            "pause_chapter_ms": 1000,
            "m4b_bitrate": "128k",
            "keep_temp": False,
            "verbose": True,
            "temp": 0.7,
            "lsd_decode_steps": 1,
        }

        response = client.put("/config", json=new_config)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
