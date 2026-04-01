"""Tests for kenkui voices exclude/include/cast/audition commands."""
from argparse import Namespace
from unittest.mock import MagicMock


def _make_args(**kwargs):
    return Namespace(**kwargs)


class TestVoicesExclude:
    def test_exclude_adds_to_config(self, monkeypatch):
        from kenkui.models import AppConfig
        from kenkui.cli.voices import cmd_voices_exclude

        config = AppConfig()
        monkeypatch.setattr("kenkui.cli.voices.load_app_config", lambda *a, **kw: config)
        saved = {}
        monkeypatch.setattr("kenkui.cli.voices.save_app_config",
                            lambda cfg, path: saved.update({"cfg": cfg}))
        mock_reg = MagicMock()
        mock_reg.resolve.return_value = MagicMock()  # voice exists
        monkeypatch.setattr("kenkui.cli.voices.get_registry", lambda: mock_reg)

        cmd_voices_exclude(_make_args(voice="jean"))
        assert "jean" in saved["cfg"].excluded_voices

    def test_exclude_no_duplicate(self, monkeypatch):
        from kenkui.models import AppConfig
        from kenkui.cli.voices import cmd_voices_exclude

        config = AppConfig(excluded_voices=["jean"])
        monkeypatch.setattr("kenkui.cli.voices.load_app_config", lambda *a, **kw: config)
        saved = {}
        monkeypatch.setattr("kenkui.cli.voices.save_app_config",
                            lambda cfg, path: saved.update({"cfg": cfg}))
        mock_reg = MagicMock()
        mock_reg.resolve.return_value = MagicMock()
        monkeypatch.setattr("kenkui.cli.voices.get_registry", lambda: mock_reg)

        cmd_voices_exclude(_make_args(voice="jean"))
        assert saved == {}  # save not called

    def test_exclude_unknown_voice_warns_but_proceeds(self, monkeypatch):
        from kenkui.models import AppConfig
        from kenkui.cli.voices import cmd_voices_exclude

        config = AppConfig()
        monkeypatch.setattr("kenkui.cli.voices.load_app_config", lambda *a, **kw: config)
        saved = {}
        monkeypatch.setattr("kenkui.cli.voices.save_app_config",
                            lambda cfg, path: saved.update({"cfg": cfg}))
        mock_reg = MagicMock()
        mock_reg.resolve.return_value = None  # unknown voice
        monkeypatch.setattr("kenkui.cli.voices.get_registry", lambda: mock_reg)

        cmd_voices_exclude(_make_args(voice="nonexistent"))
        assert "nonexistent" in saved["cfg"].excluded_voices


class TestVoicesInclude:
    def test_include_removes_from_excluded(self, monkeypatch):
        from kenkui.models import AppConfig
        from kenkui.cli.voices import cmd_voices_include

        config = AppConfig(excluded_voices=["jean", "alba"])
        monkeypatch.setattr("kenkui.cli.voices.load_app_config", lambda *a, **kw: config)
        saved = {}
        monkeypatch.setattr("kenkui.cli.voices.save_app_config",
                            lambda cfg, path: saved.update({"cfg": cfg}))

        cmd_voices_include(_make_args(voice="jean"))
        assert "jean" not in saved["cfg"].excluded_voices
        assert "alba" in saved["cfg"].excluded_voices

    def test_include_not_in_list_skips_save(self, monkeypatch):
        from kenkui.models import AppConfig
        from kenkui.cli.voices import cmd_voices_include

        config = AppConfig(excluded_voices=[])
        monkeypatch.setattr("kenkui.cli.voices.load_app_config", lambda *a, **kw: config)
        saved = {}
        monkeypatch.setattr("kenkui.cli.voices.save_app_config",
                            lambda cfg, path: saved.update({"cfg": cfg}))

        cmd_voices_include(_make_args(voice="jean"))
        assert saved == {}  # save not called
