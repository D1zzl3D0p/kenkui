"""Tests for the confirmation screen, state initialization, and profile persistence."""
from __future__ import annotations
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import pytest

# ---------------------------------------------------------------------------
# Mock InquirerPy if not installed (CI / test-only environments)
# ---------------------------------------------------------------------------
_inquirerpy_available = True
try:
    from InquirerPy.base.control import Choice, Separator  # noqa: F401
except ImportError:
    _inquirerpy_available = False

    # Build minimal stubs so _build_confirmation_choices can work
    class _Choice:
        def __init__(self, *, value=None, name=""):
            self.value = value
            self.name = name

    class _Separator:
        def __init__(self, text=""):
            self.title = text
            self.name = text

    _mock_control = MagicMock()
    _mock_control.Choice = _Choice
    _mock_control.Separator = _Separator

    _mock_inquirerpy = MagicMock()
    sys.modules.setdefault("InquirerPy", _mock_inquirerpy)
    sys.modules.setdefault("InquirerPy.base", MagicMock())
    sys.modules.setdefault("InquirerPy.base.control", _mock_control)


class TestInitStateFromProfile:
    def _app_config(self, **kw):
        from kenkui.models import AppConfig
        cfg = AppConfig()
        for k, v in kw.items():
            setattr(cfg, k, v)
        return cfg

    def test_uses_profile_voice_over_default(self):
        from kenkui.cli.add import _init_state_from_profile
        cfg = self._app_config(default_voice="alba")
        state = _init_state_from_profile(Path("/tmp/t.epub"), cfg, {"voice": "cosette"})
        assert state["voice"] == "cosette"

    def test_falls_back_to_app_config_voice(self):
        from kenkui.cli.add import _init_state_from_profile
        cfg = self._app_config(default_voice="alba")
        state = _init_state_from_profile(Path("/tmp/t.epub"), cfg, {})
        assert state["voice"] == "alba"

    def test_uses_profile_chapter_preset(self):
        from kenkui.cli.add import _init_state_from_profile
        cfg = self._app_config(default_chapter_preset="content-only")
        state = _init_state_from_profile(Path("/tmp/t.epub"), cfg, {"chapter_preset": "all"})
        assert state["chapter_selection"]["preset"] == "all"

    def test_quality_overrides_from_profile(self):
        from kenkui.cli.add import _init_state_from_profile
        cfg = self._app_config()
        state = _init_state_from_profile(Path("/tmp/t.epub"), cfg, {"quality_overrides": {"temp": 0.9}})
        assert state["quality_overrides"] == {"temp": 0.9}

    def test_empty_quality_overrides_when_no_profile(self):
        from kenkui.cli.add import _init_state_from_profile
        cfg = self._app_config()
        state = _init_state_from_profile(Path("/tmp/t.epub"), cfg, {})
        assert state["quality_overrides"] == {}

    def test_shared_confirmation_state_helper_matches_cli_wrapper(self):
        from kenkui.cli.add import _init_state_from_profile
        from kenkui.services.confirmation_service import init_confirmation_state

        cfg = self._app_config(default_voice="alba")
        profile = {"voice": "cosette", "chapter_preset": "all"}
        path = Path("/tmp/t.epub")
        assert _init_state_from_profile(path, cfg, profile) == init_confirmation_state(path, cfg, profile)


class TestBuildConfirmationChoices:
    def _app_config(self, **kw):
        from kenkui.models import AppConfig
        cfg = AppConfig()
        for k, v in kw.items():
            setattr(cfg, k, v)
        return cfg

    def _state(self, cfg, **overrides):
        from kenkui.cli.add import _init_state_from_profile
        s = _init_state_from_profile(Path("/tmp/t.epub"), cfg, {})
        s.update(overrides)
        return s

    def test_submit_action_present(self):
        from kenkui.cli.add import _build_confirmation_choices
        cfg = self._app_config()
        choices = _build_confirmation_choices(self._state(cfg), cfg)
        values = [getattr(c, "value", None) for c in choices]
        assert "submit" in values

    def test_cancel_action_present(self):
        from kenkui.cli.add import _build_confirmation_choices
        cfg = self._app_config()
        choices = _build_confirmation_choices(self._state(cfg), cfg)
        values = [getattr(c, "value", None) for c in choices]
        assert "cancel" in values

    def test_voice_and_series_actions_present(self):
        from kenkui.cli.add import _build_confirmation_choices
        cfg = self._app_config()
        choices = _build_confirmation_choices(self._state(cfg), cfg)
        values = [getattr(c, "value", None) for c in choices]
        assert "voice" in values
        assert "series" in values

    def test_default_voice_shows_default_tag(self):
        from kenkui.cli.add import _build_confirmation_choices
        cfg = self._app_config(default_voice="alba")
        state = self._state(cfg, voice="alba", narration_mode="single")
        choices = _build_confirmation_choices(state, cfg)
        names = [getattr(c, "title", getattr(c, "name", str(c))) for c in choices]
        assert any("[DEFAULT]" in str(n) for n in names)

    def test_custom_voice_shows_custom_tag(self):
        from kenkui.cli.add import _build_confirmation_choices
        cfg = self._app_config(default_voice="alba")
        state = self._state(cfg, voice="cosette", narration_mode="single")
        choices = _build_confirmation_choices(state, cfg)
        names = [getattr(c, "title", getattr(c, "name", str(c))) for c in choices]
        assert any("[CUSTOM]" in str(n) for n in names)

    def test_series_choice_shows_current_series_name(self):
        from types import SimpleNamespace
        from kenkui.cli.add import _build_confirmation_choices

        cfg = self._app_config(default_voice="alba")
        state = self._state(
            cfg,
            series_slug="wheel-of-time",
            _series_manifest=SimpleNamespace(name="Wheel of Time", slug="wheel-of-time"),
        )
        choices = _build_confirmation_choices(state, cfg)
        names = [getattr(c, "title", getattr(c, "name", str(c))) for c in choices]
        assert any("Wheel of Time" in str(n) for n in names)

    def test_shared_summary_helper_matches_choice_lines(self):
        from kenkui.cli.add import _build_confirmation_choices
        from kenkui.services.confirmation_service import summarize_confirmation_state

        cfg = self._app_config(default_voice="alba")
        state = self._state(cfg, voice="cosette", narration_mode="single")
        choices = _build_confirmation_choices(state, cfg)
        names = [getattr(c, "title", getattr(c, "name", str(c))) for c in choices]
        summary = summarize_confirmation_state(state, cfg)
        assert any("cosette" in str(n) for n in names)
        assert any("[CUSTOM]" in str(n) for n in names)


class TestWorkflowStateHelpers:
    def _app_config(self, **kw):
        from kenkui.models import AppConfig
        cfg = AppConfig()
        for k, v in kw.items():
            setattr(cfg, k, v)
        return cfg

    def test_reset_voice_mode_restores_single_defaults(self):
        from kenkui.services.workflow_service import reset_voice_mode

        cfg = self._app_config(default_voice="alba")
        state = {
            "voice": "cosette",
            "narration_mode": "multi",
            "speaker_voices": {"Rand": "jean"},
            "chapter_voices": {"1": "alba"},
            "roster_cache_path": "/tmp/roster.json",
            "series_slug": "wheel-of-time",
            "_series_manifest": object(),
        }
        result = reset_voice_mode(state, cfg)
        assert result["voice"] == "alba"
        assert result["narration_mode"] == "single"
        assert result["speaker_voices"] == {}
        assert result["chapter_voices"] == {}
        assert result["roster_cache_path"] is None
        assert result["series_slug"] is None

    def test_apply_multi_voice_setup_sets_series_and_cache(self):
        from types import SimpleNamespace
        from kenkui.services.workflow_service import apply_multi_voice_setup

        state = {"narration_mode": "single", "chapter_voices": {"1": "alba"}}
        manifest = SimpleNamespace(slug="wheel-of-time")
        result = apply_multi_voice_setup(
            state,
            speaker_voices={"Rand": "jean"},
            roster_cache_path="/tmp/roster.json",
            manifest=manifest,
        )
        assert result["narration_mode"] == "multi"
        assert result["speaker_voices"] == {"Rand": "jean"}
        assert result["chapter_voices"] == {}
        assert result["series_slug"] == "wheel-of-time"

    def test_apply_chapter_voice_setup_clears_speaker_voices(self):
        from kenkui.services.workflow_service import apply_chapter_voice_setup

        state = {"speaker_voices": {"Rand": "jean"}}
        result = apply_chapter_voice_setup(state, {"1": "alba"})
        assert result["narration_mode"] == "single"
        assert result["chapter_voices"] == {"1": "alba"}
        assert result["speaker_voices"] == {}

    def test_describe_voice_mode(self):
        from kenkui.services.workflow_service import describe_voice_mode

        assert describe_voice_mode({"narration_mode": "single", "chapter_voices": {}}) == "single narrator"
        assert describe_voice_mode({"narration_mode": "multi", "chapter_voices": {}}) == "multi-voice NLP"
        assert describe_voice_mode({"narration_mode": "single", "chapter_voices": {"1": "alba"}}) == "chapter-voice"


class TestStateToJobKwargs:
    def test_produces_expected_keys(self):
        from kenkui.cli.add import _state_to_job_kwargs, _init_state_from_profile
        from kenkui.models import AppConfig
        cfg = AppConfig()
        state = _init_state_from_profile(Path("/tmp/t.epub"), cfg, {})
        kwargs = _state_to_job_kwargs(state)
        assert kwargs["ebook_path"] == "/tmp/t.epub"
        assert kwargs["voice"] == "alba"
        assert kwargs["narration_mode"] == "single"
        assert "chapter_selection" in kwargs
        assert "output_path" in kwargs


class TestRunConfirmationScreen:
    """End-to-end tests for _run_confirmation_screen using mocked InquirerPy."""

    def _app_config(self, **kw):
        from kenkui.models import AppConfig
        cfg = AppConfig()
        for k, v in kw.items():
            setattr(cfg, k, v)
        return cfg

    def test_submit_immediately_returns_job_kwargs(self, tmp_path):
        """When user selects Submit Job, screen returns a job-kwargs dict."""
        from kenkui.cli.add import _run_confirmation_screen
        from kenkui.cli import add_profile
        import argparse

        cfg = self._app_config(default_voice="alba")

        mock_client = MagicMock()
        mock_client.list_voices.return_value = {"voices": []}

        with patch.object(add_profile, "_profile_path", return_value=tmp_path / "p.toml"):
            with patch("kenkui.cli.add._wizard_execute") as mock_exec, \
                 patch("kenkui.cli.add._get_client", return_value=mock_client):
                mock_exec.return_value = "submit"

                args = argparse.Namespace(server_host="127.0.0.1", server_port=45365,
                                          output=None, voice=None, narration_mode=None,
                                          chapter_preset=None, headless=False)
                result = _run_confirmation_screen(tmp_path / "book.epub", cfg, args)

        assert result is not None
        assert "ebook_path" in result

    def test_chapters_submenu_then_submit(self, tmp_path):
        """Navigating into chapters submenu and back, then submitting, returns job kwargs."""
        from kenkui.cli.add import _run_confirmation_screen
        from kenkui.cli import add_profile
        import argparse

        cfg = self._app_config(default_voice="alba")

        mock_client = MagicMock()
        mock_client.list_voices.return_value = {"voices": []}

        with patch.object(add_profile, "_profile_path", return_value=tmp_path / "p.toml"):
            with patch("kenkui.cli.add._wizard_execute") as mock_exec, \
                 patch("kenkui.cli.add._get_client", return_value=mock_client):
                # Simulate: select "chapters" → (submenu returns "back") → select "submit"
                call_count = [0]
                def side_effect(prompt):
                    call_count[0] += 1
                    if call_count[0] == 1:
                        return "chapters"  # enter chapters submenu
                    elif call_count[0] == 2:
                        return "back"      # exit submenu
                    else:
                        return "submit"    # confirm
                mock_exec.side_effect = side_effect

                args = argparse.Namespace(server_host="127.0.0.1", server_port=45365,
                                          output=None, voice=None, narration_mode=None,
                                          chapter_preset=None, headless=False)
                result = _run_confirmation_screen(tmp_path / "book.epub", cfg, args)

        assert result is not None
        assert "ebook_path" in result

    def test_top_level_voice_action_updates_voice_then_submit(self, tmp_path):
        from kenkui.cli.add import _run_confirmation_screen
        from kenkui.cli import add_profile
        import argparse

        cfg = self._app_config(default_voice="alba")

        mock_client = MagicMock()
        mock_client.list_voices.return_value = {"voices": []}

        with patch.object(add_profile, "_profile_path", return_value=tmp_path / "p.toml"):
            with patch("kenkui.cli.add._wizard_execute") as mock_exec, \
                 patch("kenkui.cli.add._get_client", return_value=mock_client), \
                 patch("kenkui.cli.add._prompt_voice", return_value="cosette"):
                call_count = [0]

                def side_effect(prompt):
                    call_count[0] += 1
                    if call_count[0] == 1:
                        return "voice"
                    return "submit"

                mock_exec.side_effect = side_effect

                args = argparse.Namespace(server_host="127.0.0.1", server_port=45365,
                                          output=None, voice=None, narration_mode=None,
                                          chapter_preset=None, headless=False)
                result = _run_confirmation_screen(tmp_path / "book.epub", cfg, args)

        assert result is not None
        assert result["voice"] == "cosette"

    def test_top_level_series_action_can_clear_series_then_submit(self, tmp_path):
        from kenkui.cli.add import _run_confirmation_screen
        from kenkui.cli import add_profile
        import argparse

        cfg = self._app_config(default_voice="alba")
        profile = {"narration_mode": "multi"}

        mock_client = MagicMock()
        mock_client.list_voices.return_value = {"voices": []}

        with patch.object(add_profile, "_profile_path", return_value=tmp_path / "p.toml"):
            add_profile.save_last_profile(profile)

        with patch.object(add_profile, "_profile_path", return_value=tmp_path / "p.toml"):
            with patch("kenkui.cli.add._wizard_execute") as mock_exec, \
                 patch("kenkui.cli.add._get_client", return_value=mock_client), \
                 patch("kenkui.cli.add._init_state_from_profile") as mock_init:
                mock_init.return_value = {
                    "_book_path": tmp_path / "book.epub",
                    "_app_config": cfg,
                    "voice": "alba",
                    "narration_mode": "multi",
                    "job_nlp_provider": "anthropic",
                    "job_nlp_model": "claude-sonnet-4-6",
                    "chapter_selection": {"preset": "content-only", "included": [], "excluded": []},
                    "output_dir": str(tmp_path),
                    "quality_overrides": {},
                    "pp_overrides": {},
                    "speaker_voices": {"NARRATOR": "alba"},
                    "chapter_voices": {},
                    "roster_cache_path": None,
                    "series_slug": "wheel-of-time",
                    "_series_manifest": object(),
                }

                call_count = [0]

                def side_effect(prompt):
                    call_count[0] += 1
                    if call_count[0] == 1:
                        return "series"
                    if call_count[0] == 2:
                        return "clear"
                    return "submit"

                mock_exec.side_effect = side_effect

                args = argparse.Namespace(server_host="127.0.0.1", server_port=45365,
                                          output=None, voice=None, narration_mode=None,
                                          chapter_preset=None, headless=False)
                result = _run_confirmation_screen(tmp_path / "book.epub", cfg, args)

        assert result is not None
        assert result["series_slug"] is None


class TestProfilePersistence:
    def test_round_trip(self, tmp_path):
        from kenkui.cli import add_profile
        profile = {"voice": "alba", "narration_mode": "multi", "chapter_preset": "all"}
        with patch.object(add_profile, "_profile_path", return_value=tmp_path / "p.toml"):
            add_profile.save_last_profile(profile)
            loaded = add_profile.load_last_profile()
        assert loaded.get("voice") == "alba"
        assert loaded.get("narration_mode") == "multi"

    def test_load_missing_file_returns_empty(self, tmp_path):
        from kenkui.cli import add_profile
        with patch.object(add_profile, "_profile_path", return_value=tmp_path / "nope.toml"):
            assert add_profile.load_last_profile() == {}

    def test_save_handles_missing_tomli_w(self, tmp_path):
        """Fallback writer works when tomli_w is not installed."""
        import importlib
        import sys
        from kenkui.cli import add_profile

        # Force the ImportError path by temporarily hiding tomli_w
        real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__
        original_tomli_w = sys.modules.get("tomli_w")

        def mock_import(name, *args, **kwargs):
            if name == "tomli_w":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        with patch.object(add_profile, "_profile_path", return_value=tmp_path / "p.toml"):
            with patch("builtins.__import__", side_effect=mock_import):
                add_profile.save_last_profile({"voice": "cosette", "narration_mode": "single"})

        # Restore and load using the real tomllib
        with patch.object(add_profile, "_profile_path", return_value=tmp_path / "p.toml"):
            loaded = add_profile.load_last_profile()
        assert loaded.get("voice") == "cosette"
