"""Tests for CORS middleware and startup ready signal."""
import pytest
from fastapi.testclient import TestClient


def test_cors_allows_tauri_localhost():
    from kenkui.server.api import app
    client = TestClient(app)
    response = client.options(
        "/health",
        headers={
            "Origin": "tauri://localhost",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.headers.get("access-control-allow-origin") == "tauri://localhost"


def test_cors_allows_http_tauri_localhost():
    from kenkui.server.api import app
    client = TestClient(app)
    response = client.options(
        "/health",
        headers={
            "Origin": "http://tauri.localhost",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.headers.get("access-control-allow-origin") == "http://tauri.localhost"


def test_cors_reflects_in_get_response():
    from kenkui.server.api import app
    client = TestClient(app)
    response = client.get("/health", headers={"Origin": "tauri://localhost"})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "tauri://localhost"


def test_ready_signal_printed_on_startup(capsys):
    from kenkui.server.api import app
    with TestClient(app):
        captured = capsys.readouterr()
        assert "KENKUI_SERVER_READY" in captured.out
