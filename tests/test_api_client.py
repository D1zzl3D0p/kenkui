"""Unit tests for the APIClient class."""

from unittest.mock import Mock, patch

import httpx
import pytest

from kenkui.api_client import (
    APIClient,
    JobInfo,
    QueueInfo,
    ServerStatus,
    get_client,
    reset_client,
)


def make_mock_response(json_data: dict, status_code: int = 200):
    """Create a mock httpx response with raise_for_status support."""
    response = Mock()
    response.json.return_value = json_data
    response.status_code = status_code
    response.raise_for_status = Mock()
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}",
            request=Mock(),
            response=response,
        )
    return response


@pytest.fixture
def client():
    """Create a fresh APIClient for each test."""
    reset_client()
    c = APIClient(host="127.0.0.1", port=45365)
    yield c
    c.close()
    reset_client()


class TestAPIClientInit:
    def test_client_initialization(self, client):
        assert client.base_url == "http://127.0.0.1:45365"

    def test_client_custom_host_port(self):
        c = APIClient(host="192.168.1.1", port=9000)
        assert c.base_url == "http://192.168.1.1:9000"
        c.close()


class TestHealthCheck:
    def test_health_check_success(self, client):
        mock_resp = make_mock_response({"status": "healthy", "version": "0.8.0"})

        with patch.object(client._client, "request", return_value=mock_resp):
            result = client.health_check()
            assert result == {"status": "healthy", "version": "0.8.0"}


class TestGetQueue:
    def test_get_queue_success(self, client):
        mock_resp = make_mock_response(
            {
                "items": [
                    {
                        "id": "test123",
                        "job": {"ebook_path": "/tmp/test.epub", "voice": "alba"},
                        "status": "pending",
                        "progress": 0.0,
                        "current_chapter": "",
                        "eta_seconds": 0,
                        "error_message": "",
                    }
                ],
                "current_item": None,
                "pending_count": 1,
                "completed_count": 0,
                "failed_count": 0,
            }
        )

        with patch.object(client._client, "request", return_value=mock_resp):
            result = client.get_queue()
            assert isinstance(result, QueueInfo)
            assert len(result.items) == 1
            assert result.items[0].id == "test123"
            assert result.pending_count == 1


class TestAddJob:
    def test_add_job_success(self, client):
        mock_resp = make_mock_response(
            {
                "id": "new123",
                "job": {"ebook_path": "/tmp/test.epub", "voice": "alba"},
                "status": "pending",
                "progress": 0.0,
                "current_chapter": "",
                "eta_seconds": 0,
                "error_message": "",
            }
        )

        with patch.object(client._client, "request", return_value=mock_resp):
            result = client.add_job(
                ebook_path="/tmp/test.epub",
                voice="alba",
            )
            assert isinstance(result, JobInfo)
            assert result.id == "new123"

    def test_add_job_with_chapter_selection(self, client):
        mock_resp = make_mock_response(
            {
                "id": "new123",
                "job": {"ebook_path": "/tmp/test.epub", "voice": "custom"},
                "status": "pending",
                "progress": 0.0,
                "current_chapter": "",
                "eta_seconds": 0,
                "error_message": "",
            }
        )

        chapter_sel = {"preset": {"value": "content-only"}, "included": [1, 2, 3]}

        with patch.object(client._client, "request", return_value=mock_resp):
            result = client.add_job(
                ebook_path="/tmp/test.epub",
                voice="custom",
                chapter_selection=chapter_sel,
            )
            assert result.job["voice"] == "custom"


class TestGetJob:
    def test_get_job_success(self, client):
        mock_resp = make_mock_response(
            {
                "id": "test123",
                "job": {"ebook_path": "/tmp/test.epub", "voice": "alba"},
                "status": "processing",
                "progress": 50.0,
                "current_chapter": "Chapter 1",
                "eta_seconds": 300,
                "error_message": "",
            }
        )

        with patch.object(client._client, "request", return_value=mock_resp):
            result = client.get_job("test123")
            assert isinstance(result, JobInfo)
            assert result.id == "test123"
            assert result.status == "processing"
            assert result.progress == 50.0


class TestRemoveJob:
    def test_remove_job_success(self, client):
        mock_resp = make_mock_response({"status": "removed", "job_id": "test123"})
        with patch.object(client._client, "request", return_value=mock_resp):
            result = client.remove_job("test123")
            assert result is True

    def test_remove_job_failed(self, client):
        error_resp = make_mock_response({}, status_code=400)
        with patch.object(
            client._client,
            "request",
            side_effect=httpx.HTTPStatusError(
                "400", request=Mock(), response=error_resp
            ),
        ):
            result = client.remove_job("test123")
            assert result is False


class TestStartStopProcessing:
    def test_start_processing(self, client):
        mock_resp = make_mock_response({"status": "started"})

        with patch.object(client._client, "request", return_value=mock_resp):
            result = client.start_processing()
            assert result == {"status": "started"}

    def test_stop_processing(self, client):
        mock_resp = make_mock_response({"status": "stopped"})

        with patch.object(client._client, "request", return_value=mock_resp):
            result = client.stop_processing()
            assert result == {"status": "stopped"}


class TestGetStatus:
    def test_get_status(self, client):
        mock_resp = make_mock_response(
            {
                "status": "running",
                "is_running": True,
                "current_job": "test123",
            }
        )

        with patch.object(client._client, "request", return_value=mock_resp):
            result = client.get_status()
            assert isinstance(result, ServerStatus)
            assert result.is_running is True
            assert result.current_job == "test123"


class TestConfig:
    def test_get_config(self, client):
        mock_resp = make_mock_response({"config": {"voice": "alba", "workers": 4}})

        with patch.object(client._client, "request", return_value=mock_resp):
            result = client.get_config()
            assert result == {"voice": "alba", "workers": 4}

    def test_update_config(self, client):
        mock_resp = make_mock_response(
            {
                "status": "updated",
                "config": {"voice": "alba", "workers": 8},
            }
        )

        with patch.object(client._client, "request", return_value=mock_resp):
            result = client.update_config({"voice": "alba", "workers": 8})
            assert result["status"] == "updated"


class TestClearQueue:
    def test_clear_queue(self, client):
        mock_resp = make_mock_response({"status": "cleared"})

        with patch.object(client._client, "request", return_value=mock_resp):
            result = client.clear_queue()
            assert result == {"status": "cleared"}


class TestSingleton:
    def test_get_client_returns_singleton(self):
        reset_client()
        c1 = get_client(host="127.0.0.1", port=45365)
        c2 = get_client(host="127.0.0.1", port=45365)
        assert c1 is c2
        c1.close()
        reset_client()

    def test_reset_client_clears_singleton(self):
        reset_client()
        c1 = get_client(host="127.0.0.1", port=45365)
        c1.close()
        reset_client()
        c2 = get_client(host="127.0.0.1", port=45365)
        assert c1 is not c2
        c2.close()
        reset_client()


class TestErrorHandling:
    def test_connection_error(self, client):
        with patch.object(
            client._client,
            "request",
            side_effect=httpx.ConnectError("Connection failed"),
        ), pytest.raises(httpx.ConnectError):
            client.health_check()

    def test_timeout_error(self, client):
        with patch.object(
            client._client, "request", side_effect=httpx.ReadTimeout("Timeout")
        ), pytest.raises(httpx.ReadTimeout):
            client.health_check()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
