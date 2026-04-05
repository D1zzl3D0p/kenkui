"""Tests for the kenkui config command (thin-client rewrite)."""
from __future__ import annotations

import multiprocessing
import sys
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_PP = {
    "enabled": True,
    "noise_reduce": True,
    "noise_reduce_prop_decrease": 0.8,
    "highpass_hz": 80,
    "lowshelf_hz": 250,
    "lowshelf_db": -3.0,
    "presence_hz": 3500,
    "presence_db": 2.0,
    "deesser": True,
    "deesser_hz": 6500,
    "deesser_db": -4.0,
    "compressor_threshold_db": -18.0,
    "compressor_ratio": 3.0,
    "compressor_attack_ms": 5.0,
    "compressor_release_ms": 50.0,
    "limiter_threshold_db": -1.0,
    "autogain": True,
    "autogain_target_lufs": -23.0,
    "normalize": False,
    "normalize_target_db": -3.0,
    "normalize_lufs": None,
}

_DEFAULT_CONFIG = {
    "name": "default",
    "workers": 4,
    "verbose": False,
    "log_path": None,
    "keep_temp": False,
    "m4b_bitrate": "96k",
    "pause_line_ms": 800,
    "pause_chapter_ms": 2000,
    "pause_scene_break_ms": 4000,
    "temp": 0.7,
    "lsd_decode_steps": 1,
    "noise_clamp": None,
    "eos_threshold": -4.0,
    "frames_after_eos": None,
    "default_voice": "alba",
    "default_chapter_preset": "content-only",
    "default_output_dir": None,
    "nlp_model": "llama3.2",
    "excluded_voices": [],
    "post_processing": _DEFAULT_PP,
}

_VOICES_RESPONSE = {
    "voices": [
        {
            "name": "alba",
            "source": "builtin",
            "gender": "Female",
            "accent": "British",
            "dataset": None,
            "speaker_id": None,
            "excluded": False,
            "description": "alba (builtin)",
        },
        {
            "name": "jean",
            "source": "compiled",
            "gender": "Male",
            "accent": "American",
            "dataset": None,
            "speaker_id": None,
            "excluded": False,
            "description": "jean (compiled)",
        },
    ],
    "total": 2,
}


def _make_client(
    *,
    get_config=None,
    update_config=None,
    list_voices=None,
):
    """Build a mock APIClient context manager."""
    instance = MagicMock()
    instance.get_config.return_value = get_config if get_config is not None else dict(_DEFAULT_CONFIG)
    instance.update_config.return_value = {"status": "updated", "config": {}}
    instance.list_voices.return_value = list_voices if list_voices is not None else _VOICES_RESPONSE
    cm = MagicMock()
    cm.__enter__.return_value = instance
    cm.__exit__.return_value = False
    return cm


def _make_args(path="default"):
    return Namespace(path=path)


def _stub_inquirerpy(monkeypatch, answers: list):
    """Stub InquirerPy so that prompt.execute() pops answers in order.

    The code does ``from InquirerPy import inquirer`` then calls
    ``inquirer.number(...).execute()``.  We need ``inquirer`` (the attribute
    on the InquirerPy module mock) to have those methods set up correctly.
    """
    answer_iter = iter(answers)

    class _FakePrompt:
        def execute(self):
            return next(answer_iter)

    def _make_fake(*args, **kwargs):
        return _FakePrompt()

    # Build the inquirer sub-object that the code imports via
    # ``from InquirerPy import inquirer``
    mock_inquirer = MagicMock()
    mock_inquirer.number.side_effect = _make_fake
    mock_inquirer.text.side_effect = _make_fake
    mock_inquirer.fuzzy.side_effect = _make_fake
    mock_inquirer.select.side_effect = _make_fake
    mock_inquirer.confirm.side_effect = _make_fake

    # Top-level InquirerPy module; ``from InquirerPy import inquirer``
    # resolves to mock_inq.inquirer.
    mock_inq = MagicMock()
    mock_inq.inquirer = mock_inquirer

    monkeypatch.setitem(sys.modules, "InquirerPy", mock_inq)
    monkeypatch.setitem(sys.modules, "InquirerPy.inquirer", mock_inquirer)
    monkeypatch.setitem(sys.modules, "InquirerPy.validator", MagicMock())
    return mock_inquirer


# ---------------------------------------------------------------------------
# Tests: cmd_config
# ---------------------------------------------------------------------------


class TestCmdConfig:
    """Tests that cmd_config talks only to APIClient (no local file / registry access)."""

    # Default answers covering all prompts when enable_pp=False:
    # workers, default_output_dir, default_voice, default_chapter_preset,
    # m4b_bitrate, pause_line_ms, pause_chapter_ms, temp, lsd_decode_steps,
    # eos_threshold, frames_after_eos, nlp_model, enable_pp, confirm_save
    _DEFAULT_ANSWERS = [
        4,              # workers
        "",             # default_output_dir (blank -> None)
        "alba",         # default_voice
        "content-only", # default_chapter_preset
        "96k",          # m4b_bitrate
        800,            # pause_line_ms
        2000,           # pause_chapter_ms
        0.7,            # temp
        1,              # lsd_decode_steps
        -4.0,           # eos_threshold
        0,              # frames_after_eos (0 -> None/auto)
        "llama3.2",     # nlp_model
        False,          # enable_pp -> skips pp fields
        True,           # confirm save
    ]

    _CANCEL_ANSWERS = [
        4, "", "alba", "content-only", "96k",
        800, 2000, 0.7, 1, -4.0, 0, "llama3.2",
        False,  # enable_pp
        False,  # cancel
    ]

    def _run_wizard(self, monkeypatch, api_cm, answers=None):
        """Drive cmd_config end-to-end with stubbed prompts."""
        from kenkui.cli.config import cmd_config

        if answers is None:
            answers = list(self._DEFAULT_ANSWERS)

        _stub_inquirerpy(monkeypatch, answers)

        with patch("kenkui.cli.config.APIClient", return_value=api_cm):
            result = cmd_config(_make_args())
        return result

    def test_calls_get_config_on_start(self, monkeypatch):
        """cmd_config must fetch current config from API."""
        api_cm = _make_client()

        self._run_wizard(monkeypatch, api_cm)

        api_cm.__enter__.return_value.get_config.assert_called_once()

    def test_calls_list_voices_for_choices(self, monkeypatch):
        """cmd_config must call list_voices to populate the voice fuzzy prompt."""
        api_cm = _make_client()

        self._run_wizard(monkeypatch, api_cm)

        api_cm.__enter__.return_value.list_voices.assert_called_once()

    def test_calls_update_config_on_confirm(self, monkeypatch):
        """When user confirms, update_config must be called once."""
        api_cm = _make_client()

        self._run_wizard(monkeypatch, api_cm)

        api_cm.__enter__.return_value.update_config.assert_called_once()

    def test_no_update_config_on_cancel(self, monkeypatch):
        """When user cancels (confirm=False), update_config must NOT be called."""
        api_cm = _make_client()

        self._run_wizard(monkeypatch, api_cm, answers=list(self._CANCEL_ANSWERS))

        api_cm.__enter__.return_value.update_config.assert_not_called()

    def test_update_config_payload_has_correct_keys(self, monkeypatch):
        """The dict passed to update_config must contain expected top-level keys."""
        api_cm = _make_client()

        self._run_wizard(monkeypatch, api_cm)

        call_args = api_cm.__enter__.return_value.update_config.call_args
        payload = call_args[0][0]  # first positional argument

        assert "workers" in payload
        assert "default_voice" in payload
        assert "m4b_bitrate" in payload
        assert "post_processing" in payload
        assert "nlp_model" in payload

    def test_voice_choices_built_from_api_voices(self, monkeypatch):
        """Voice list from API is used to populate the fuzzy prompt choices."""
        voices_data = {
            "voices": [
                {
                    "name": "cosette",
                    "source": "builtin",
                    "gender": "Female",
                    "accent": "French",
                    "dataset": None,
                    "speaker_id": None,
                    "excluded": False,
                    "description": "cosette (builtin)",
                },
            ],
            "total": 1,
        }
        api_cm = _make_client(list_voices=voices_data)

        answers = [
            4, "", "cosette", "content-only", "96k",
            800, 2000, 0.7, 1, -4.0, 0, "llama3.2",
            False,  # enable_pp
            True,   # confirm
        ]
        self._run_wizard(monkeypatch, api_cm, answers=answers)

        api_cm.__enter__.return_value.list_voices.assert_called_once()

    def test_returns_zero_on_success(self, monkeypatch):
        """cmd_config should return 0 after a successful save."""
        api_cm = _make_client()
        result = self._run_wizard(monkeypatch, api_cm)
        assert result == 0

    def test_returns_zero_on_cancel(self, monkeypatch):
        """cmd_config should return 0 even when user cancels."""
        api_cm = _make_client()
        result = self._run_wizard(monkeypatch, api_cm, answers=list(self._CANCEL_ANSWERS))
        assert result == 0

    def test_no_business_logic_imports(self):
        """config.py must not import get_registry, load_app_config, or save_app_config."""
        import ast
        import pathlib

        src = pathlib.Path(__file__).parent.parent / "src/kenkui/cli/config.py"
        tree = ast.parse(src.read_text())
        forbidden = {"get_registry", "load_app_config", "save_app_config"}
        found = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in forbidden:
                        found.add(alias.name)
                    if alias.asname and alias.asname in forbidden:
                        found.add(alias.asname)

        assert not found, f"Forbidden business-logic symbols found in config.py: {found}"

    def test_no_appconfig_model_import(self):
        """config.py must not import AppConfig or PostProcessingConfig from models."""
        import ast
        import pathlib

        src = pathlib.Path(__file__).parent.parent / "src/kenkui/cli/config.py"
        tree = ast.parse(src.read_text())
        forbidden = {"AppConfig", "PostProcessingConfig"}
        found = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in forbidden:
                        found.add(alias.name)

        assert not found, f"Forbidden model imports found in config.py: {found}"

    def test_post_processing_enabled_path(self, monkeypatch):
        """When enable_pp=True, all pp prompts are visited and pp dict is in payload."""
        api_cm = _make_client()
        answers = [
            4, "", "alba", "content-only", "96k",
            800, 2000, 0.7, 1, -4.0, 0, "llama3.2",
            True,   # enable_pp -> enter pp block
            True,   # noise_reduce
            0.8,    # noise_reduce_prop_decrease
            80,     # highpass_hz
            250,    # lowshelf_hz
            -3.0,   # lowshelf_db
            3500,   # presence_hz
            2.0,    # presence_db
            True,   # deesser
            True,   # autogain
            -23.0,  # autogain_target_lufs
            -18.0,  # comp_thresh
            3.0,    # comp_ratio
            -1.0,   # limiter_thresh
            False,  # normalize
            True,   # confirm
        ]

        self._run_wizard(monkeypatch, api_cm, answers=answers)

        call_args = api_cm.__enter__.return_value.update_config.call_args
        payload = call_args[0][0]
        pp = payload["post_processing"]
        assert pp["enabled"] is True
        assert pp["noise_reduce"] is True

    def test_server_not_running_returns_error(self, capsys):
        """When server is not running, cmd_config returns 1 with a helpful message."""
        import httpx
        from kenkui.cli.config import cmd_config

        with patch("kenkui.cli.config.APIClient") as MockClient:
            MockClient.return_value.__enter__.side_effect = httpx.ConnectError("refused")
            result = cmd_config(Namespace())
        assert result == 1
        captured = capsys.readouterr()
        assert "connect" in captured.out.lower() or "server" in captured.out.lower()

    def test_post_processing_disabled_preserves_existing_fields(self, monkeypatch):
        """When enable_pp=False, payload has enabled=False and retains other pp keys from server."""
        # Server config has custom pp fields
        custom_pp = {
            **_DEFAULT_PP,
            "highpass_hz": 120,
            "compressor_ratio": 5.0,
        }
        cfg = {**_DEFAULT_CONFIG, "post_processing": custom_pp}
        api_cm = _make_client(get_config=cfg)

        # Wizard answers: enable_pp=False (skips all pp prompts), then confirm
        answers = [
            4, "", "alba", "content-only", "96k",
            800, 2000, 0.7, 1, -4.0, 0, "llama3.2",
            False,  # enable_pp -> skip pp block
            True,   # confirm
        ]
        self._run_wizard(monkeypatch, api_cm, answers=answers)

        call_args = api_cm.__enter__.return_value.update_config.call_args
        payload = call_args[0][0]
        pp = payload["post_processing"]
        assert pp["enabled"] is False
        # Existing fields from server config must be preserved
        assert pp["highpass_hz"] == 120
        assert pp["compressor_ratio"] == 5.0
