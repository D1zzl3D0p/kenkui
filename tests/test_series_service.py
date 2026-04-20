"""Tests for series_service and series API route helpers."""
from __future__ import annotations

import tomli_w
import pytest
from pathlib import Path
from unittest.mock import patch


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


class TestRosterCandidates:
    def test_list_roster_candidates_returns_result(self, tmp_path, monkeypatch):
        import json
        import kenkui.series as _series_mod
        import kenkui.config as _config_mod

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
        from kenkui.services.series_service import match_series_characters, save_series, SeriesCharacterEntry, SeriesEntry

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


class TestSeriesListRoute:
    def test_get_series_empty(self, tmp_path, monkeypatch):
        import kenkui.series as _series_mod

        monkeypatch.setattr(_series_mod, "_series_dir_override", tmp_path)
        from kenkui.server.api import list_series

        body = list_series().model_dump()
        assert body["series"] == []
        assert body["total"] == 0

    def test_get_series_with_entries(self, tmp_path, monkeypatch):
        import kenkui.series as _series_mod
        monkeypatch.setattr(_series_mod, "_series_dir_override", tmp_path)

        raw = {"name": "Dune", "updated_at": "", "characters": []}
        (tmp_path / "dune.toml").write_bytes(tomli_w.dumps(raw).encode())

        from kenkui.server.api import list_series

        body = list_series().model_dump()
        assert body["total"] == 1
        assert body["series"][0]["slug"] == "dune"


class TestSeriesGetRoute:
    def test_get_series_missing_returns_404(self, tmp_path, monkeypatch):
        import kenkui.series as _series_mod
        from fastapi import HTTPException

        monkeypatch.setattr(_series_mod, "_series_dir_override", tmp_path)
        from kenkui.server.api import get_series

        with pytest.raises(HTTPException) as exc:
            get_series("missing")
        assert exc.value.status_code == 404

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

        from kenkui.server.api import get_series

        body = get_series("foundation").model_dump()
        assert body["slug"] == "foundation"
        assert body["name"] == "Foundation"
        assert len(body["characters"]) == 1


class TestSeriesDeleteRoute:
    def test_delete_missing_returns_404(self, tmp_path, monkeypatch):
        import kenkui.series as _series_mod
        from fastapi import HTTPException

        monkeypatch.setattr(_series_mod, "_series_dir_override", tmp_path)
        from kenkui.server.api import delete_series

        with pytest.raises(HTTPException) as exc:
            delete_series("missing")
        assert exc.value.status_code == 404

    def test_delete_existing_returns_200(self, tmp_path, monkeypatch):
        import kenkui.series as _series_mod
        monkeypatch.setattr(_series_mod, "_series_dir_override", tmp_path)

        raw = {"name": "Deleted", "updated_at": "", "characters": []}
        (tmp_path / "deleted.toml").write_bytes(tomli_w.dumps(raw).encode())

        from kenkui.server.api import delete_series

        body = delete_series("deleted").model_dump()
        assert body["status"] == "ok"
        assert not (tmp_path / "deleted.toml").exists()


class TestSeriesCreationAndMatchRoutes:
    def test_list_roster_candidates_route(self):
        from kenkui.server.api import list_series_roster_candidates

        with patch(
            "kenkui.services.series_service.list_roster_candidates",
            return_value=type(
                "Result",
                (),
                {
                    "candidates": [
                        type(
                            "Candidate",
                            (),
                            {
                                "hash": "abc",
                                "title": "Book",
                                "path": "/tmp/book.epub",
                                "speaker_voices": {"Rand": "alba"},
                                "roster_path": "/tmp/abc-roster.json",
                            },
                        )()
                    ],
                    "total": 1,
                },
            )(),
        ):
            body = list_series_roster_candidates().model_dump()

        assert body["total"] == 1
        assert body["candidates"][0]["hash"] == "abc"

    def test_create_empty_series_route(self):
        from kenkui.server.api import CreateEmptySeriesRequest, create_empty_series

        with patch(
            "kenkui.services.series_service.create_empty_series",
            return_value=type(
                "Entry",
                (),
                {"slug": "wheel-of-time", "name": "Wheel of Time", "updated_at": "", "characters": []},
            )(),
        ):
            body = create_empty_series(CreateEmptySeriesRequest(name="Wheel of Time")).model_dump()

        assert body["slug"] == "wheel-of-time"
        assert body["name"] == "Wheel of Time"

    def test_create_series_from_candidate_route(self):
        from kenkui.server.api import CreateSeriesFromCandidateRequest, create_series_from_candidate

        with patch(
            "kenkui.services.series_service.build_series_from_candidate",
            return_value=type(
                "Entry",
                (),
                {
                    "slug": "wheel-of-time",
                    "name": "Wheel of Time",
                    "updated_at": "",
                    "characters": [
                        type(
                            "Character",
                            (),
                            {
                                "canonical": "Rand al'Thor",
                                "aliases": ["Rand"],
                                "voice": "alba",
                                "gender": "he/him",
                            },
                        )()
                    ],
                },
            )(),
        ):
            body = create_series_from_candidate(
                CreateSeriesFromCandidateRequest(
                    name="Wheel of Time",
                    roster_path="/tmp/abc-roster.json",
                )
            ).model_dump()

        assert body["slug"] == "wheel-of-time"
        assert body["characters"][0]["canonical"] == "Rand al'Thor"

    def test_match_series_route(self):
        from kenkui.server.api import SeriesMatchRequest, match_series

        with patch(
            "kenkui.services.series_service.match_series_characters",
            return_value=type(
                "MatchResult",
                (),
                {"inherited_voices": {"Rand al'Thor": "alba"}, "pinned": ["Rand al'Thor"]},
            )(),
        ):
            body = match_series("wheel-of-time", SeriesMatchRequest(fast_result={"characters": []})).model_dump()

        assert body["inherited_voices"]["Rand al'Thor"] == "alba"
        assert body["pinned"] == ["Rand al'Thor"]

    def test_match_series_route_missing_returns_404(self):
        from fastapi import HTTPException
        from kenkui.server.api import SeriesMatchRequest, match_series

        with patch("kenkui.services.series_service.match_series_characters", side_effect=KeyError("missing")):
            with pytest.raises(HTTPException) as exc:
                match_series("missing", SeriesMatchRequest(fast_result={"characters": []}))

        assert exc.value.status_code == 404
