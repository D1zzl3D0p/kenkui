"""Tests for wizard prompt execution."""

from unittest.mock import MagicMock

from kenkui.cli.add import _wizard_execute


def test_wizard_execute_returns_value_on_success():
    """_wizard_execute returns the result of prompt.execute() on success."""
    mock_prompt = MagicMock()
    mock_prompt.execute.return_value = "cosette"

    result = _wizard_execute(mock_prompt)
    assert result == "cosette"


class TestSeriesStepSkipped:
    """Series step is a no-op when mode is not multi."""

    def test_single_mode_skips_series(self):
        from kenkui.cli.add import _run_series_setup
        from kenkui.models import FastScanResult
        from kenkui.nlp.models import AliasGroup, CharacterRoster
        roster = FastScanResult(
            roster=CharacterRoster(characters=[]),
            characters=[],
            book_hash="test",
        )
        manifest, inherited, pinned = _run_series_setup(
            fast_result=roster,
            mode="single",
            prompts=[],
        )
        assert manifest is None
        assert inherited == {}
        assert pinned == set()


class TestSeriesStepNewSeries:
    """User picks 'new series' and names it."""

    def test_creates_manifest_from_candidate(self, tmp_path, monkeypatch):
        import json
        monkeypatch.setattr("kenkui.series._series_dir_override", tmp_path)

        from kenkui.cli.add import _run_series_setup
        from kenkui.models import CharacterInfo, FastScanResult
        from kenkui.nlp.models import AliasGroup, CharacterRoster

        roster = FastScanResult(
            roster=CharacterRoster(characters=[
                AliasGroup(canonical="Rand al'Thor", aliases=["Rand"]),
            ]),
            characters=[CharacterInfo(character_id="Rand al'Thor", display_name="Rand al'Thor")],
            book_hash="abc",
        )
        roster_path = tmp_path / "abc-roster.json"
        roster_path.write_text(json.dumps(roster.to_dict()), encoding="utf-8")

        # Simulate: yes to series, pick "new", pick candidate[0], name="Wheel of Time"
        prompts = ["yes", "new", 0, "Wheel of Time"]
        manifest, inherited, pinned = _run_series_setup(
            fast_result=roster,
            mode="multi",
            prompts=prompts,
            _roster_candidates=[{
                "hash": "abc",
                "title": "Eye of the World",
                "speaker_voices": {"Rand al'Thor": "alba"},
                "roster_path": roster_path,
                "path": "",
            }],
        )
        assert manifest is not None
        assert manifest.name == "Wheel of Time"


class TestConflictResolutionPinning:
    def test_pinned_char_not_reassigned(self):
        from kenkui.cli.add import _resolve_chapter_voice_conflicts
        from kenkui.models import CharacterInfo

        characters = [
            CharacterInfo(character_id="Rand", display_name="Rand", mention_count=100),
            CharacterInfo(character_id="Mat", display_name="Mat", mention_count=50),
        ]
        speaker_voices = {"Rand": "alba", "Mat": "alba"}
        pinned = {"Rand"}
        from kenkui.models import Chapter, Segment
        chapters = [
            Chapter(
                index=0,
                title="Ch1",
                paragraphs=[],
                segments=[
                    Segment(text="x", speaker="Rand"),
                    Segment(text="y", speaker="Mat"),
                ],
            )
        ]
        result, _ = _resolve_chapter_voice_conflicts(
            speaker_voices,
            characters,
            chapters,
            male_pool=["alba", "jean", "marius"],
            female_pool=["cosette", "fantine"],
            narrator_voice="cosette",
            pinned=pinned,
        )
        assert result["Rand"] == "alba"
        assert result["Mat"] != "alba"


class TestReviewShowsSeriesMarker:
    def test_inherited_label_shown(self):
        from kenkui.cli.add import _make_character_review_label
        from kenkui.models import CharacterInfo

        char = CharacterInfo(character_id="Rand", display_name="Rand al'Thor", mention_count=100)
        label = _make_character_review_label(
            char, voice="alba", pinned={"Rand"}, series_name="Wheel of Time"
        )
        assert "Wheel of Time" in label

    def test_non_inherited_no_marker(self):
        from kenkui.cli.add import _make_character_review_label
        from kenkui.models import CharacterInfo

        char = CharacterInfo(character_id="Newguy", display_name="New Guy", mention_count=10)
        label = _make_character_review_label(
            char, voice="jean", pinned=set(), series_name="Wheel of Time"
        )
        assert "Wheel of Time" not in label
