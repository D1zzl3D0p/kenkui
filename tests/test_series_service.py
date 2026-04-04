"""Tests for series_service and the /series API routes."""
from __future__ import annotations

import tomli_w
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Service-layer tests
# ---------------------------------------------------------------------------

class TestListSeries:
    def test_returns_list_series_result(self, tmp_path, monkeypatch):
        import kenkui.series as _series_mod
        monkeypatch.setattr(_series_mod, "_series_dir_override", tmp_path)

        from kenkui.services.series_service import list_series, ListSeriesResult
        result = list_series()
        assert isinstance(result, ListSeriesResult)
        assert result.series == []
        assert result.total == 0

    def test_returns_entries_for_existing_manifests(self, tmp_path, monkeypatch):
        import kenkui.series as _series_mod
        monkeypatch.setattr(_series_mod, "_series_dir_override", tmp_path)

        # Write a valid manifest TOML
        data = {"name": "The Expanse", "updated_at": "", "characters": []}
        (tmp_path / "the-expanse.toml").write_bytes(tomli_w.dumps(data).encode())

        from kenkui.services.series_service import list_series, ListSeriesResult, SeriesEntry
        result = list_series()
        assert isinstance(result, ListSeriesResult)
        assert result.total == 1
        assert len(result.series) == 1
        assert isinstance(result.series[0], SeriesEntry)
        assert result.series[0].slug == "the-expanse"
        assert result.series[0].name == "The Expanse"


class TestLoadSeries:
    def test_returns_series_entry_with_correct_slug(self, tmp_path, monkeypatch):
        import kenkui.series as _series_mod
        monkeypatch.setattr(_series_mod, "_series_dir_override", tmp_path)

        data = {
            "name": "Wheel of Time",
            "updated_at": "2025-01-01T00:00:00+00:00",
            "characters": [
                {"canonical": "Rand", "aliases": ["the Dragon"], "voice": "alba", "gender": "he/him"},
            ],
        }
        (tmp_path / "wheel-of-time.toml").write_bytes(tomli_w.dumps(data).encode())

        from kenkui.services.series_service import load_series, SeriesEntry
        entry = load_series("wheel-of-time")
        assert isinstance(entry, SeriesEntry)
        assert entry.slug == "wheel-of-time"
        assert entry.name == "Wheel of Time"
        assert len(entry.characters) == 1
        assert entry.characters[0].canonical == "Rand"

    def test_raises_key_error_for_missing_slug(self, tmp_path, monkeypatch):
        import kenkui.series as _series_mod
        monkeypatch.setattr(_series_mod, "_series_dir_override", tmp_path)

        from kenkui.services.series_service import load_series
        with pytest.raises(KeyError):
            load_series("missing-series")


class TestSaveSeries:
    def test_save_calls_underlying_series_save(self, tmp_path, monkeypatch):
        import kenkui.series as _series_mod
        monkeypatch.setattr(_series_mod, "_series_dir_override", tmp_path)

        calls = []
        original_save = _series_mod.save_series

        def _mock_save(manifest):
            calls.append(manifest)
            return original_save(manifest)

        monkeypatch.setattr(_series_mod, "save_series", _mock_save)

        from kenkui.services.series_service import save_series, SeriesEntry
        entry = SeriesEntry(slug="test-series", name="Test Series", characters=[])
        save_series(entry)

        assert len(calls) == 1
        assert calls[0].slug == "test-series"
        assert calls[0].name == "Test Series"

    def test_save_persists_to_disk(self, tmp_path, monkeypatch):
        import kenkui.series as _series_mod
        monkeypatch.setattr(_series_mod, "_series_dir_override", tmp_path)

        from kenkui.services.series_service import save_series, load_series, SeriesEntry
        entry = SeriesEntry(slug="persisted", name="Persisted Series", characters=[])
        save_series(entry)

        loaded = load_series("persisted")
        assert loaded.name == "Persisted Series"


class TestDeleteSeries:
    def test_delete_existing_returns_true(self, tmp_path, monkeypatch):
        import kenkui.series as _series_mod
        monkeypatch.setattr(_series_mod, "_series_dir_override", tmp_path)

        data = {"name": "To Delete", "updated_at": "", "characters": []}
        (tmp_path / "to-delete.toml").write_bytes(tomli_w.dumps(data).encode())

        from kenkui.services.series_service import delete_series
        result = delete_series("to-delete")
        assert result is True
        assert not (tmp_path / "to-delete.toml").exists()

    def test_delete_missing_returns_false(self, tmp_path, monkeypatch):
        import kenkui.series as _series_mod
        monkeypatch.setattr(_series_mod, "_series_dir_override", tmp_path)

        from kenkui.services.series_service import delete_series
        result = delete_series("nonexistent")
        assert result is False


# ---------------------------------------------------------------------------
# API route tests
# ---------------------------------------------------------------------------

@pytest.fixture
def api_client(tmp_path, monkeypatch):
    """Return a TestClient with series dir isolated to tmp_path."""
    import kenkui.series as _series_mod
    monkeypatch.setattr(_series_mod, "_series_dir_override", tmp_path)

    from fastapi.testclient import TestClient
    from kenkui.server.api import app
    return TestClient(app)


class TestSeriesListRoute:
    def test_get_series_empty(self, api_client):
        response = api_client.get("/series")
        assert response.status_code == 200
        data = response.json()
        assert data["series"] == []
        assert data["total"] == 0

    def test_get_series_with_entries(self, tmp_path, monkeypatch):
        import kenkui.series as _series_mod
        monkeypatch.setattr(_series_mod, "_series_dir_override", tmp_path)

        raw = {"name": "Dune", "updated_at": "", "characters": []}
        (tmp_path / "dune.toml").write_bytes(tomli_w.dumps(raw).encode())

        from fastapi.testclient import TestClient
        from kenkui.server.api import app
        client = TestClient(app)

        response = client.get("/series")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["series"][0]["slug"] == "dune"


class TestSeriesGetRoute:
    def test_get_series_missing_returns_404(self, api_client):
        response = api_client.get("/series/missing")
        assert response.status_code == 404

    def test_get_series_existing_returns_200(self, tmp_path, monkeypatch):
        import kenkui.series as _series_mod
        monkeypatch.setattr(_series_mod, "_series_dir_override", tmp_path)

        raw = {
            "name": "Foundation",
            "updated_at": "",
            "characters": [
                {"canonical": "Hari Seldon", "aliases": [], "voice": "alba", "gender": "he/him"}
            ],
        }
        (tmp_path / "foundation.toml").write_bytes(tomli_w.dumps(raw).encode())

        from fastapi.testclient import TestClient
        from kenkui.server.api import app
        client = TestClient(app)

        response = client.get("/series/foundation")
        assert response.status_code == 200
        data = response.json()
        assert data["slug"] == "foundation"
        assert data["name"] == "Foundation"
        assert len(data["characters"]) == 1


class TestSeriesDeleteRoute:
    def test_delete_missing_returns_404(self, api_client):
        response = api_client.delete("/series/missing")
        assert response.status_code == 404

    def test_delete_existing_returns_200(self, tmp_path, monkeypatch):
        import kenkui.series as _series_mod
        monkeypatch.setattr(_series_mod, "_series_dir_override", tmp_path)

        raw = {"name": "Deleted", "updated_at": "", "characters": []}
        (tmp_path / "deleted.toml").write_bytes(tomli_w.dumps(raw).encode())

        from fastapi.testclient import TestClient
        from kenkui.server.api import app
        client = TestClient(app)

        response = client.delete("/series/deleted")
        assert response.status_code == 200
        assert not (tmp_path / "deleted.toml").exists()
