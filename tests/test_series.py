from __future__ import annotations

from pathlib import Path

import pytest

from kenkui.series import (
    SeriesCharacter,
    SeriesManifest,
    load_series,
    save_series,
    list_series,
    slugify,
    series_dir,
)


class TestSlugify:
    def test_spaces_become_hyphens(self):
        assert slugify("Wheel of Time") == "wheel-of-time"

    def test_special_chars_stripped(self):
        assert slugify("Harry Potter & the Goblet") == "harry-potter-the-goblet"

    def test_already_clean(self):
        assert slugify("dune") == "dune"

    def test_leading_trailing_hyphens_stripped(self):
        assert slugify("  --Test Series--  ") == "test-series"


class TestSeriesManifestRoundTrip:
    def test_save_and_load(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kenkui.series._series_dir_override", tmp_path)
        manifest = SeriesManifest(
            name="Wheel of Time",
            slug="wheel-of-time",
            updated_at="",
            characters=[
                SeriesCharacter(
                    canonical="Matrim Cauthon",
                    aliases=["Mat", "Mat Cauthon"],
                    voice="jean",
                    gender="he/him",
                ),
            ],
        )
        save_series(manifest)
        loaded = load_series("wheel-of-time")
        assert loaded is not None
        assert loaded.name == "Wheel of Time"
        assert len(loaded.characters) == 1
        assert loaded.characters[0].canonical == "Matrim Cauthon"
        assert loaded.characters[0].voice == "jean"
        assert loaded.characters[0].aliases == ["Mat", "Mat Cauthon"]

    def test_load_nonexistent_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kenkui.series._series_dir_override", tmp_path)
        assert load_series("no-such-series") is None

    def test_save_sets_updated_at(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kenkui.series._series_dir_override", tmp_path)
        manifest = SeriesManifest(name="Dune", slug="dune", updated_at="", characters=[])
        save_series(manifest)
        loaded = load_series("dune")
        assert loaded.updated_at != ""

    def test_list_series_sorted_by_mtime(self, tmp_path, monkeypatch):
        import time
        monkeypatch.setattr("kenkui.series._series_dir_override", tmp_path)
        for slug in ("aaa", "bbb", "ccc"):
            m = SeriesManifest(name=slug, slug=slug, updated_at="", characters=[])
            save_series(m)
            time.sleep(0.05)  # ensure distinct mtimes on low-precision filesystems
        series = list_series()
        assert [s.slug for s in series] == ["ccc", "bbb", "aaa"]

    def test_list_series_empty_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kenkui.series._series_dir_override", tmp_path)
        assert list_series() == []
