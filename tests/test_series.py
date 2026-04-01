from __future__ import annotations

import json
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
    match_characters,
)
from kenkui.models import CharacterInfo, FastScanResult
from kenkui.nlp.models import AliasGroup, CharacterRoster


def _make_fast_result(canonical_aliases: dict[str, list[str]]) -> FastScanResult:
    """Build a minimal FastScanResult from a {canonical: [aliases]} dict."""
    groups = [
        AliasGroup(canonical=c, aliases=a) for c, a in canonical_aliases.items()
    ]
    roster = CharacterRoster(characters=groups)
    characters = [
        CharacterInfo(character_id=c, display_name=c)
        for c in canonical_aliases
    ]
    return FastScanResult(roster=roster, characters=characters, book_hash="test")


def _make_manifest(entries: list[tuple[str, list[str], str]]) -> SeriesManifest:
    """Build a SeriesManifest from [(canonical, aliases, voice)] tuples."""
    chars = [SeriesCharacter(canonical=c, aliases=a, voice=v) for c, a, v in entries]
    return SeriesManifest(name="Test", slug="test", updated_at="", characters=chars)


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


class TestMatchCharacters:
    def test_exact_canonical_match(self):
        roster = _make_fast_result({"Hermione Granger": ["Hermione"]})
        manifest = _make_manifest([("Hermione Granger", ["Hermione"], "cosette")])
        inherited, pinned = match_characters(roster.characters, roster, manifest)
        assert inherited["Hermione Granger"] == "cosette"
        assert "Hermione Granger" in pinned

    def test_alias_exact_match(self):
        roster = _make_fast_result({"Mat": []})
        manifest = _make_manifest([("Matrim Cauthon", ["Mat", "Mat Cauthon"], "jean")])
        inherited, pinned = match_characters(roster.characters, roster, manifest)
        assert inherited["Mat"] == "jean"

    def test_word_overlap_match(self):
        # "Mat" in new book; "Matrim Cauthon" in manifest (no shared alias yet)
        roster = _make_fast_result({"Mat": []})
        manifest = _make_manifest([("Matrim Cauthon", [], "jean")])
        # "mat" has 1 word; "matrim cauthon" has 2 words — overlap 1/1 = 1.0
        inherited, pinned = match_characters(roster.characters, roster, manifest)
        assert inherited.get("Mat") == "jean"

    def test_no_match_below_threshold(self):
        roster = _make_fast_result({"Alice": []})
        manifest = _make_manifest([("Robert Johnson", [], "alba")])
        inherited, pinned = match_characters(roster.characters, roster, manifest)
        assert "Alice" not in inherited
        assert len(pinned) == 0

    def test_unmatched_characters_excluded(self):
        roster = _make_fast_result({"Rand": [], "Newguy": []})
        manifest = _make_manifest([("Rand al'Thor", ["Rand"], "alba")])
        inherited, pinned = match_characters(roster.characters, roster, manifest)
        assert "Rand" in pinned
        assert "Newguy" not in pinned

    def test_empty_manifest(self):
        roster = _make_fast_result({"Alice": ["Ali"]})
        manifest = _make_manifest([])
        inherited, pinned = match_characters(roster.characters, roster, manifest)
        assert inherited == {}
        assert pinned == set()

    def test_short_prefix_does_not_match_unrelated_name(self):
        # "Ed" (2 chars) must not inherit a voice from "Edward Norton" — different character
        roster = _make_fast_result({"Ed": []})
        manifest = _make_manifest([("Edward Norton", [], "alba")])
        inherited, pinned = match_characters(roster.characters, roster, manifest)
        assert "Ed" not in inherited


from kenkui.series import (
    build_manifest_from_predecessor,
    update_manifest,
)


class TestBuildManifestFromPredecessor:
    def test_builds_from_roster_and_voices(self, tmp_path):
        roster = _make_fast_result({"Rand al'Thor": ["Rand"], "Mat": ["Matrim"]})
        roster_path = tmp_path / "abc123-roster.json"
        roster_path.write_text(
            json.dumps(roster.to_dict(), ensure_ascii=False), encoding="utf-8"
        )
        candidate = {
            "hash": "abc123",
            "title": "Eye of the World",
            "speaker_voices": {"Rand al'Thor": "alba", "Mat": "jean"},
            "roster_path": roster_path,
        }
        manifest = build_manifest_from_predecessor(candidate, "Wheel of Time")
        assert manifest.name == "Wheel of Time"
        assert manifest.slug == "wheel-of-time"
        rand = next(c for c in manifest.characters if c.canonical == "Rand al'Thor")
        assert rand.voice == "alba"
        assert "Rand" in rand.aliases

    def test_skips_chars_without_voice(self, tmp_path):
        roster = _make_fast_result({"Rand": [], "Background": []})
        roster_path = tmp_path / "abc123-roster.json"
        roster_path.write_text(json.dumps(roster.to_dict()), encoding="utf-8")
        candidate = {
            "hash": "abc123",
            "title": "Book",
            "speaker_voices": {"Rand": "alba"},
            "roster_path": roster_path,
        }
        manifest = build_manifest_from_predecessor(candidate, "Test")
        assert len(manifest.characters) == 1


class TestUpdateManifest:
    def test_new_character_appended(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kenkui.series._series_dir_override", tmp_path)
        manifest = _make_manifest([("Rand", [], "alba")])
        roster = _make_fast_result({"Rand": [], "Mat": []})
        speaker_voices = {"Rand": "alba", "Mat": "jean"}
        pinned = {"Rand"}
        updated = update_manifest(manifest, roster.characters, roster, speaker_voices, pinned)
        canonicals = [c.canonical for c in updated.characters]
        assert "Rand" in canonicals
        assert "Mat" in canonicals
        mat = next(c for c in updated.characters if c.canonical == "Mat")
        assert mat.voice == "jean"

    def test_modified_pinned_voice_updated(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kenkui.series._series_dir_override", tmp_path)
        manifest = _make_manifest([("Rand", [], "alba")])
        roster = _make_fast_result({"Rand": []})
        speaker_voices = {"Rand": "cosette"}
        pinned = {"Rand"}
        updated = update_manifest(manifest, roster.characters, roster, speaker_voices, pinned)
        rand = next(c for c in updated.characters if c.canonical == "Rand")
        assert rand.voice == "cosette"

    def test_duplicate_not_appended(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kenkui.series._series_dir_override", tmp_path)
        manifest = _make_manifest([("Rand al'Thor", ["Rand"], "alba")])
        roster = _make_fast_result({"Rand al'Thor": ["Rand"]})
        speaker_voices = {"Rand al'Thor": "alba"}
        pinned = {"Rand al'Thor"}
        updated = update_manifest(manifest, roster.characters, roster, speaker_voices, pinned)
        assert len(updated.characters) == 1
