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
