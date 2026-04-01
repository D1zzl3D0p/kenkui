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


class TestSeriesStepExistingSeries:
    def test_loads_existing_series_and_matches(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kenkui.series._series_dir_override", tmp_path)

        from kenkui.series import SeriesCharacter, SeriesManifest, save_series
        from kenkui.cli.add import _run_series_setup
        from kenkui.models import CharacterInfo, FastScanResult
        from kenkui.nlp.models import AliasGroup, CharacterRoster

        # Pre-save an existing series manifest
        manifest = SeriesManifest(
            name="Wheel of Time",
            slug="wheel-of-time",
            updated_at="",
            characters=[
                SeriesCharacter(canonical="Rand al'Thor", aliases=["Rand"], voice="alba", gender="he/him"),
            ],
        )
        save_series(manifest)

        roster = FastScanResult(
            roster=CharacterRoster(characters=[AliasGroup(canonical="Rand al'Thor", aliases=["Rand"])]),
            characters=[CharacterInfo(character_id="Rand al'Thor", display_name="Rand al'Thor")],
            book_hash="xyz",
        )

        # User says yes, picks "wheel-of-time" (existing slug)
        prompts = ["yes", "wheel-of-time"]
        result_manifest, inherited, pinned = _run_series_setup(
            fast_result=roster,
            mode="multi",
            prompts=prompts,
        )
        assert result_manifest is not None
        assert result_manifest.name == "Wheel of Time"
        assert "Rand al'Thor" in inherited
        assert inherited["Rand al'Thor"] == "alba"
        assert "Rand al'Thor" in pinned


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

    def test_pinned_conflict_recorded_in_unresolved(self):
        """When both chars in a conflict are pinned, the pair is returned in unresolved."""
        from kenkui.cli.add import _resolve_chapter_voice_conflicts
        from kenkui.models import CharacterInfo, Chapter, Segment

        characters = [
            CharacterInfo(character_id="Rand", display_name="Rand", mention_count=100),
            CharacterInfo(character_id="Mat", display_name="Mat", mention_count=50),
        ]
        speaker_voices = {"Rand": "alba", "Mat": "alba"}
        pinned = {"Rand", "Mat"}  # Both pinned — cannot reassign either
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
        result, unresolved = _resolve_chapter_voice_conflicts(
            speaker_voices,
            characters,
            chapters,
            male_pool=["alba", "jean"],
            female_pool=["cosette"],
            narrator_voice="cosette",
            pinned=pinned,
        )
        assert result["Rand"] == "alba"
        assert result["Mat"] == "alba"
        assert len(unresolved) == 1
        assert set(unresolved[0]) == {"Rand", "Mat"}

    def test_no_spare_voice_conflict_recorded_in_unresolved(self):
        """When no spare voice is available, conflict is recorded in unresolved."""
        from kenkui.cli.add import _resolve_chapter_voice_conflicts
        from kenkui.models import CharacterInfo, Chapter, Segment

        characters = [
            CharacterInfo(character_id="Rand", display_name="Rand", mention_count=100),
            CharacterInfo(character_id="Mat", display_name="Mat", mention_count=50),
        ]
        # Only one male voice available — no spare exists
        speaker_voices = {"Rand": "alba", "Mat": "alba"}
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
        result, unresolved = _resolve_chapter_voice_conflicts(
            speaker_voices,
            characters,
            chapters,
            male_pool=["alba"],  # Only one voice — no spare
            female_pool=["cosette"],
            narrator_voice="cosette",
            pinned=None,
        )
        assert len(unresolved) == 1
        assert set(unresolved[0]) == {"Rand", "Mat"}

    def test_same_conflict_in_multiple_chapters_deduped(self):
        """Same conflict pair in two chapters produces only one unresolved entry."""
        from kenkui.cli.add import _resolve_chapter_voice_conflicts
        from kenkui.models import CharacterInfo, Chapter, Segment

        characters = [
            CharacterInfo(character_id="Rand", display_name="Rand", mention_count=100),
            CharacterInfo(character_id="Mat", display_name="Mat", mention_count=50),
        ]
        # Same voice conflict in TWO chapters
        speaker_voices = {"Rand": "alba", "Mat": "alba"}
        pinned = {"Rand", "Mat"}  # both pinned — cannot reassign either
        chapters = [
            Chapter(
                index=0,
                title="Ch1",
                paragraphs=[],
                segments=[
                    Segment(text="x", speaker="Rand"),
                    Segment(text="y", speaker="Mat"),
                ],
            ),
            Chapter(
                index=1,
                title="Ch2",
                paragraphs=[],
                segments=[
                    Segment(text="x", speaker="Rand"),
                    Segment(text="y", speaker="Mat"),
                ],
            ),
        ]
        _, unresolved = _resolve_chapter_voice_conflicts(
            speaker_voices,
            characters,
            chapters,
            male_pool=["alba", "jean"],
            female_pool=["cosette"],
            narrator_voice="cosette",
            pinned=pinned,
        )
        # Should be exactly 1 conflict entry, not 2
        assert len(unresolved) == 1
        assert set(unresolved[0]) == {"Rand", "Mat"}

    def test_no_conflict_returns_empty_unresolved(self):
        """When no conflict exists, unresolved is empty."""
        from kenkui.cli.add import _resolve_chapter_voice_conflicts
        from kenkui.models import CharacterInfo, Chapter, Segment

        characters = [
            CharacterInfo(character_id="Rand", display_name="Rand", mention_count=100),
            CharacterInfo(character_id="Mat", display_name="Mat", mention_count=50),
        ]
        speaker_voices = {"Rand": "alba", "Mat": "jean"}
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
        result, unresolved = _resolve_chapter_voice_conflicts(
            speaker_voices,
            characters,
            chapters,
            male_pool=["alba", "jean"],
            female_pool=["cosette"],
            narrator_voice="cosette",
            pinned=None,
        )
        assert unresolved == []


class TestConflictWarningsInReview:
    def test_warnings_printed_for_unresolved_conflicts(self, capsys, monkeypatch):
        """_prompt_character_voice_review prints warnings for unresolved conflicts."""
        import sys
        from unittest.mock import MagicMock

        # Patch InquirerPy before importing
        mock_inquirer = MagicMock()
        mock_confirm = MagicMock()
        mock_confirm.execute.return_value = True  # Accept immediately
        mock_inquirer.confirm.return_value = mock_confirm
        mock_inquirerpy = MagicMock()
        mock_inquirerpy.inquirer = mock_inquirer

        monkeypatch.setitem(sys.modules, "InquirerPy", mock_inquirerpy)
        monkeypatch.setitem(sys.modules, "InquirerPy.inquirer", mock_inquirer)

        from kenkui.cli.add import _prompt_character_voice_review
        from kenkui.models import CharacterInfo

        characters = [
            CharacterInfo(character_id="Rand", display_name="Rand", mention_count=100),
        ]
        speaker_voices = {"Rand": "alba"}
        unresolved = [("Rand", "Mat")]

        # Capture rich console output
        from io import StringIO
        from rich.console import Console
        import kenkui.cli.add as add_mod
        buf = StringIO()
        orig_console = add_mod.console
        add_mod.console = Console(file=buf, highlight=False)
        try:
            _prompt_character_voice_review(
                speaker_voices,
                characters,
                narrator_voice="cosette",
                unresolved_conflicts=unresolved,
            )
        finally:
            add_mod.console = orig_console

        output = buf.getvalue()
        assert "Rand" in output
        assert "Mat" in output
        assert "same voice" in output

    def test_no_warnings_when_no_conflicts(self, capsys, monkeypatch):
        """_prompt_character_voice_review prints no warnings when unresolved_conflicts is empty."""
        import sys
        from unittest.mock import MagicMock

        mock_inquirer = MagicMock()
        mock_confirm = MagicMock()
        mock_confirm.execute.return_value = True
        mock_inquirer.confirm.return_value = mock_confirm
        mock_inquirerpy = MagicMock()
        mock_inquirerpy.inquirer = mock_inquirer

        monkeypatch.setitem(sys.modules, "InquirerPy", mock_inquirerpy)
        monkeypatch.setitem(sys.modules, "InquirerPy.inquirer", mock_inquirer)

        from kenkui.cli.add import _prompt_character_voice_review
        from kenkui.models import CharacterInfo

        characters = [
            CharacterInfo(character_id="Rand", display_name="Rand", mention_count=100),
        ]
        speaker_voices = {"Rand": "alba"}

        from io import StringIO
        from rich.console import Console
        import kenkui.cli.add as add_mod
        buf = StringIO()
        orig_console = add_mod.console
        add_mod.console = Console(file=buf, highlight=False)
        try:
            _prompt_character_voice_review(
                speaker_voices,
                characters,
                narrator_voice="cosette",
                unresolved_conflicts=[],
            )
        finally:
            add_mod.console = orig_console

        output = buf.getvalue()
        assert "same voice" not in output


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


class TestManifestWritebackOnConfirm:
    def _patch_inquirer(self, monkeypatch):
        """Monkeypatch InquirerPy.inquirer so _step_confirm doesn't fail on import."""
        import sys
        from unittest.mock import MagicMock
        mock_inquirer = MagicMock()
        mock_inquirerpy = MagicMock()
        mock_inquirerpy.inquirer = mock_inquirer
        monkeypatch.setitem(sys.modules, "InquirerPy", mock_inquirerpy)
        monkeypatch.setitem(sys.modules, "InquirerPy.inquirer", mock_inquirer)

    def test_manifest_saved_on_confirm(self, tmp_path, monkeypatch):
        import json
        self._patch_inquirer(monkeypatch)
        monkeypatch.setattr("kenkui.series._series_dir_override", tmp_path)

        from kenkui.series import SeriesCharacter, SeriesManifest, load_series
        from kenkui.models import CharacterInfo, FastScanResult
        from kenkui.nlp.models import AliasGroup, CharacterRoster
        from kenkui.cli.add import _step_confirm

        manifest = SeriesManifest(
            name="Wheel of Time", slug="wheel-of-time", updated_at="",
            characters=[SeriesCharacter(canonical="Rand", aliases=[], voice="alba")],
        )
        roster = FastScanResult(
            roster=CharacterRoster(characters=[AliasGroup(canonical="Rand", aliases=[])]),
            characters=[CharacterInfo(character_id="Rand", display_name="Rand")],
            book_hash="abc",
        )

        # Simulate confirm=True via monkeypatching _wizard_execute
        monkeypatch.setattr("kenkui.cli.add._wizard_execute", lambda p: True)

        state = {
            "_book_path": tmp_path / "book.epub",
            "_app_config": None,
            "_series_manifest": manifest,
            "_fast_result": roster,
            "_inherited_voices": {"Rand": "alba"},
            "_pinned": {"Rand"},
            "chapter_selection": {"preset": "content-only", "included": [], "excluded": []},
            "mode": "multi",
            "narration_mode": "multi",
            "voice": "alba",
            "speaker_voices": {"Rand": "alba", "NARRATOR": "jean"},
            "chapter_voices": {},
            "quality_overrides": {},
            "output_dir": str(tmp_path),
            "roster_cache_path": None,
        }

        result = _step_confirm(state)
        assert "_job_kwargs" in result

        saved = load_series("wheel-of-time")
        assert saved is not None
        rand_entry = next(c for c in saved.characters if c.canonical == "Rand")
        assert rand_entry.voice == "alba"

    def test_manifest_not_saved_on_cancel(self, tmp_path, monkeypatch):
        self._patch_inquirer(monkeypatch)
        monkeypatch.setattr("kenkui.series._series_dir_override", tmp_path)
        from kenkui.series import SeriesManifest, list_series
        from kenkui.cli.add import _step_confirm

        manifest = SeriesManifest(name="Dune", slug="dune", updated_at="", characters=[])

        monkeypatch.setattr("kenkui.cli.add._wizard_execute", lambda p: False)

        state = {
            "_book_path": tmp_path / "book.epub",
            "_app_config": None,
            "_series_manifest": manifest,
            "_fast_result": None,
            "_inherited_voices": {},
            "_pinned": set(),
            "chapter_selection": {"preset": "content-only", "included": [], "excluded": []},
            "mode": "single",
            "narration_mode": "single",
            "voice": "alba",
            "speaker_voices": {},
            "chapter_voices": {},
            "quality_overrides": {},
            "output_dir": str(tmp_path),
            "roster_cache_path": None,
        }

        result = _step_confirm(state)
        assert result.get("_cancelled")
        assert list_series() == []
