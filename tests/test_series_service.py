"""Tests for series_service."""
from __future__ import annotations

import pytest
import tomli_w

# ---------------------------------------------------------------------------
# Service-layer tests
# ---------------------------------------------------------------------------

class TestListSeries:
    def test_returns_list_series_result(self, tmp_path, monkeypatch):
        import kenkui.series as _series_mod
        monkeypatch.setattr(_series_mod, "_series_dir_override", tmp_path)

        from kenkui.services.series_service import ListSeriesResult, list_series
        result = list_series()
        assert isinstance(result, ListSeriesResult)
        assert result.series == []
        assert result.total == 0

    def test_returns_entries_for_existing_manifests(self, tmp_path, monkeypatch):
        import kenkui.series as _series_mod
        monkeypatch.setattr(_series_mod, "_series_dir_override", tmp_path)

        data = {"name": "The Expanse", "updated_at": "", "characters": []}
        (tmp_path / "the-expanse.toml").write_bytes(tomli_w.dumps(data).encode())

        from kenkui.services.series_service import ListSeriesResult, SeriesEntry, list_series
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

        from kenkui.services.series_service import SeriesEntry, load_series
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

        from kenkui.services.series_service import SeriesEntry, save_series
        entry = SeriesEntry(slug="test-series", name="Test Series", characters=[])
        save_series(entry)

        assert len(calls) == 1
        assert calls[0].slug == "test-series"
        assert calls[0].name == "Test Series"

    def test_save_persists_to_disk(self, tmp_path, monkeypatch):
        import kenkui.series as _series_mod
        monkeypatch.setattr(_series_mod, "_series_dir_override", tmp_path)

        from kenkui.services.series_service import SeriesEntry, load_series, save_series
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


class TestRosterCandidates:
    def test_list_roster_candidates_returns_result(self, tmp_path, monkeypatch):
        import json

        import kenkui.config as _config_mod
        import kenkui.series as _series_mod

        monkeypatch.setattr(_series_mod, "_series_dir_override", tmp_path / "series")
        monkeypatch.setattr(_config_mod, "CONFIG_DIR", tmp_path)
        cache_dir = tmp_path / "nlp_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "abc-roster.json").write_text(
            json.dumps({"book_hash": "abc", "roster": {"characters": []}, "characters": []}),
            encoding="utf-8",
        )

        from kenkui.services.series_service import RosterCandidateListResult, list_roster_candidates

        result = list_roster_candidates()
        assert isinstance(result, RosterCandidateListResult)
        assert result.total >= 1


class TestCreateAndMatchSeriesHelpers:
    def test_create_empty_series_persists(self, tmp_path, monkeypatch):
        import kenkui.series as _series_mod

        monkeypatch.setattr(_series_mod, "_series_dir_override", tmp_path)
        from kenkui.services.series_service import create_empty_series, load_series

        created = create_empty_series("Wheel of Time")
        assert created.slug == "wheel-of-time"
        loaded = load_series("wheel-of-time")
        assert loaded.name == "Wheel of Time"

    def test_match_series_characters_returns_inherited_voices(self, tmp_path, monkeypatch):
        import kenkui.series as _series_mod

        monkeypatch.setattr(_series_mod, "_series_dir_override", tmp_path)
        from kenkui.models import CharacterInfo, FastScanResult
        from kenkui.nlp.models import CharacterRecord, CharacterRoster
        from kenkui.services.series_service import (
            SeriesCharacterEntry,
            SeriesEntry,
            match_series_characters,
            save_series,
        )

        save_series(
            SeriesEntry(
                slug="wheel-of-time",
                name="Wheel of Time",
                characters=[SeriesCharacterEntry(canonical="Rand al'Thor", aliases=["Rand"], voice="alba", gender="he/him")],
            )
        )

        fast_result = FastScanResult(
            roster=CharacterRoster(characters=[CharacterRecord(slug="rand_althor", canonical_name="Rand al'Thor", aliases=["Rand"])]),
            characters=[CharacterInfo(character_id="Rand al'Thor", display_name="Rand al'Thor")],
            book_hash="abc",
        )

        result = match_series_characters("wheel-of-time", fast_result.to_dict())
        assert result.inherited_voices["Rand al'Thor"] == "alba"
        assert "Rand al'Thor" in result.pinned
