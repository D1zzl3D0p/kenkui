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


class TestVoicesCast:
    def _make_item(self, name, speaker_voices, job_id="abc12345"):
        from kenkui.models import QueueItem, JobConfig, NarrationMode, JobStatus
        job = JobConfig(
            ebook_path="/books/test.epub",
            name=name,
            narration_mode=NarrationMode.MULTI,
            speaker_voices=speaker_voices,
        )
        return QueueItem(id=job_id, job=job, status=JobStatus.COMPLETED)

    def test_exact_title_match_shows_table(self, monkeypatch):
        from kenkui.cli.voices import cmd_voices_cast
        from argparse import Namespace

        item = self._make_item(
            "Wheel of Time",
            {"Rand al'Thor": "alba", "Mat": "jean", "NARRATOR": "cosette"},
        )
        mock_qm = MagicMock()
        mock_qm.completed_items = [item]
        monkeypatch.setattr("kenkui.cli.voices.QueueManager", lambda: mock_qm)

        # Should not raise
        cmd_voices_cast(Namespace(title="Wheel of Time"))

    def test_fuzzy_title_match_case_insensitive(self, monkeypatch):
        from kenkui.cli.voices import cmd_voices_cast
        from argparse import Namespace

        item = self._make_item("Wheel of Time", {"Rand": "alba"})
        mock_qm = MagicMock()
        mock_qm.completed_items = [item]
        monkeypatch.setattr("kenkui.cli.voices.QueueManager", lambda: mock_qm)

        # lowercase input must still match "Wheel of Time"
        cmd_voices_cast(Namespace(title="wheel of time"))

    def test_no_matches_prints_warning(self, monkeypatch):
        from kenkui.cli.voices import cmd_voices_cast
        from argparse import Namespace

        mock_qm = MagicMock()
        mock_qm.completed_items = []
        monkeypatch.setattr("kenkui.cli.voices.QueueManager", lambda: mock_qm)

        # Should not raise
        cmd_voices_cast(Namespace(title="Unknown Book"))

    def test_disambiguation_when_multiple_match(self, monkeypatch):
        from kenkui.cli.voices import cmd_voices_cast
        from argparse import Namespace

        item1 = self._make_item("Wheel of Time Book 1", {"Rand": "alba"}, "id1")
        item2 = self._make_item("Wheel of Time Book 2", {"Rand": "jean"}, "id2")
        mock_qm = MagicMock()
        mock_qm.completed_items = [item1, item2]
        monkeypatch.setattr("kenkui.cli.voices.QueueManager", lambda: mock_qm)

        # Should not crash; print disambiguation
        cmd_voices_cast(Namespace(title="Wheel of Time"))

    def test_narrator_sorted_last(self):
        from kenkui.cli.voices import _sort_cast

        voices = {"NARRATOR": "cosette", "Rand": "alba", "Mat": "jean"}
        keys = [k for k, _ in _sort_cast(voices)]
        assert keys[-1] == "NARRATOR"


class TestVoicesAudition:
    def _make_args(self, voice="jean", text=None):
        from argparse import Namespace
        return Namespace(voice=voice, text=text)

    def test_happy_path_saves_and_opens(self, tmp_path, monkeypatch):
        from kenkui.cli.voices import cmd_voices_audition
        from pydub import AudioSegment

        mock_model = MagicMock()
        mock_model.get_state_for_audio_prompt.return_value = object()

        monkeypatch.setattr("kenkui.cli.voices._get_or_load_model",
                            lambda *a, **kw: mock_model)
        monkeypatch.setattr("kenkui.cli.voices.load_voice", lambda v: v)
        monkeypatch.setattr("kenkui.cli.voices._render_text",
                            lambda *a, **kw: AudioSegment.silent(duration=500))
        monkeypatch.setattr("kenkui.cli.voices.load_app_config", lambda *a, **kw: MagicMock(
            temp=0.7, lsd_decode_steps=1, noise_clamp=None, eos_threshold=-4.0,
        ))
        monkeypatch.setattr("kenkui.cli.voices.get_registry",
                            lambda: MagicMock(resolve=lambda v: MagicMock()))

        out_path = tmp_path / "kenkui-jean-preview.wav"
        monkeypatch.setattr("kenkui.cli.voices._preview_path", lambda v: out_path)

        opened = []
        monkeypatch.setattr("subprocess.Popen", lambda cmd: opened.append(cmd))

        cmd_voices_audition(self._make_args())
        assert out_path.exists()
        assert len(opened) == 1

    def test_synthesis_failure_exits(self, monkeypatch):
        from kenkui.cli.voices import cmd_voices_audition
        import pytest

        mock_model = MagicMock()
        mock_model.get_state_for_audio_prompt.return_value = object()

        monkeypatch.setattr("kenkui.cli.voices._get_or_load_model",
                            lambda *a, **kw: mock_model)
        monkeypatch.setattr("kenkui.cli.voices.load_voice", lambda v: v)
        monkeypatch.setattr("kenkui.cli.voices._render_text",
                            lambda *a, **kw: None)  # simulate failure
        monkeypatch.setattr("kenkui.cli.voices.load_app_config", lambda *a, **kw: MagicMock(
            temp=0.7, lsd_decode_steps=1, noise_clamp=None, eos_threshold=-4.0,
        ))
        monkeypatch.setattr("kenkui.cli.voices.get_registry",
                            lambda: MagicMock(resolve=lambda v: MagicMock()))
        monkeypatch.setattr("kenkui.cli.voices._preview_path",
                            lambda v: MagicMock(__str__=lambda s: "/tmp/x.wav"))

        with pytest.raises(SystemExit) as exc:
            cmd_voices_audition(self._make_args())
        assert exc.value.code == 1

    def test_custom_text_used(self, tmp_path, monkeypatch):
        from kenkui.cli.voices import cmd_voices_audition, DEFAULT_AUDITION_TEXT
        from pydub import AudioSegment

        captured = {}

        def fake_render(model, voice_state, text, **kwargs):
            captured["text"] = text
            return AudioSegment.silent(duration=100)

        mock_model = MagicMock()
        mock_model.get_state_for_audio_prompt.return_value = object()

        monkeypatch.setattr("kenkui.cli.voices._get_or_load_model",
                            lambda *a, **kw: mock_model)
        monkeypatch.setattr("kenkui.cli.voices.load_voice", lambda v: v)
        monkeypatch.setattr("kenkui.cli.voices._render_text", fake_render)
        monkeypatch.setattr("kenkui.cli.voices.load_app_config", lambda *a, **kw: MagicMock(
            temp=0.7, lsd_decode_steps=1, noise_clamp=None, eos_threshold=-4.0,
        ))
        monkeypatch.setattr("kenkui.cli.voices.get_registry",
                            lambda: MagicMock(resolve=lambda v: MagicMock()))
        out_path = tmp_path / "preview.wav"
        monkeypatch.setattr("kenkui.cli.voices._preview_path", lambda v: out_path)
        monkeypatch.setattr("subprocess.Popen", lambda cmd: None)

        cmd_voices_audition(self._make_args(text="Hello world"))
        assert captured["text"] == "Hello world"
        assert captured["text"] != DEFAULT_AUDITION_TEXT

    def test_system_player_platform(self, monkeypatch):
        import sys as _sys
        from kenkui.cli.voices import _player_command

        monkeypatch.setattr(_sys, "platform", "darwin")
        assert _player_command() == "open"

        monkeypatch.setattr(_sys, "platform", "linux")
        assert _player_command() == "xdg-open"
