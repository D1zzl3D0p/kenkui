"""Tests for kenkui voices exclude/include/cast/audition commands."""
from argparse import Namespace
from unittest.mock import MagicMock, patch


def _make_args(**kwargs):
    return Namespace(**kwargs)


def _make_api_client(
    *,
    list_voices=None,
    exclude_voice=None,
    include_voice=None,
    get_queue_cast=None,
    audition_voice=None,
    poll_task=None,
    get_audition_wav_url=None,
):
    """Build a mock APIClient context manager."""
    instance = MagicMock()
    instance.list_voices.return_value = list_voices or {"voices": [], "total": 0}
    instance.exclude_voice.return_value = exclude_voice or {"status": "excluded"}
    instance.include_voice.return_value = include_voice or {"status": "included"}
    instance.get_queue_cast.return_value = get_queue_cast or {
        "job_id": "abc", "book_name": "Test", "narration_mode": "multi", "cast": []
    }
    instance.audition_voice.return_value = audition_voice or {"task_id": "task-1"}
    instance.poll_task.return_value = poll_task or {"status": "completed"}
    instance.get_audition_wav_url.return_value = get_audition_wav_url or "http://localhost/voices/audition/task-1.wav"
    cm = MagicMock()
    cm.__enter__.return_value = instance
    cm.__exit__.return_value = False
    return cm


class TestVoicesExclude:
    def test_exclude_adds_to_config(self):
        from kenkui.cli.voices import cmd_voices_exclude

        api_cm = _make_api_client(exclude_voice={"status": "excluded"})
        with patch("kenkui.cli.voices.APIClient", return_value=api_cm):
            cmd_voices_exclude(_make_args(voice="jean"))
        api_cm.__enter__.return_value.exclude_voice.assert_called_once_with("jean")

    def test_exclude_no_duplicate(self):
        from kenkui.cli.voices import cmd_voices_exclude

        api_cm = _make_api_client(exclude_voice={"status": "already_excluded", "warning": "Voice 'jean' is already excluded."})
        with patch("kenkui.cli.voices.APIClient", return_value=api_cm):
            cmd_voices_exclude(_make_args(voice="jean"))
        # Should still call the API — server decides idempotency
        api_cm.__enter__.return_value.exclude_voice.assert_called_once_with("jean")

    def test_exclude_unknown_voice_warns_but_proceeds(self):
        from kenkui.cli.voices import cmd_voices_exclude

        api_cm = _make_api_client(exclude_voice={"status": "excluded", "warning": "Voice not found in registry."})
        with patch("kenkui.cli.voices.APIClient", return_value=api_cm):
            cmd_voices_exclude(_make_args(voice="nonexistent"))
        api_cm.__enter__.return_value.exclude_voice.assert_called_once_with("nonexistent")


class TestVoicesInclude:
    def test_include_removes_from_excluded(self):
        from kenkui.cli.voices import cmd_voices_include

        api_cm = _make_api_client(include_voice={"status": "included"})
        with patch("kenkui.cli.voices.APIClient", return_value=api_cm):
            cmd_voices_include(_make_args(voice="jean"))
        api_cm.__enter__.return_value.include_voice.assert_called_once_with("jean")

    def test_include_not_in_list_skips_save(self):
        from kenkui.cli.voices import cmd_voices_include

        api_cm = _make_api_client(include_voice={"status": "not_excluded", "warning": "Voice 'jean' was not excluded."})
        with patch("kenkui.cli.voices.APIClient", return_value=api_cm):
            cmd_voices_include(_make_args(voice="jean"))
        # API is called regardless; server handles the not-excluded case
        api_cm.__enter__.return_value.include_voice.assert_called_once_with("jean")


class TestVoicesCast:
    def test_cast_shows_table_for_job(self):
        from kenkui.cli.voices import cmd_voices_cast

        cast_data = {
            "job_id": "abc12345",
            "book_name": "Wheel of Time",
            "narration_mode": "multi",
            "cast": [
                {"character_id": "Rand al'Thor", "display_name": "Rand al'Thor", "voice_name": "alba", "quote_count": 10, "mention_count": 20, "gender_pronoun": "he/him"},
                {"character_id": "Mat", "display_name": "Mat", "voice_name": "jean", "quote_count": 5, "mention_count": 8, "gender_pronoun": None},
                {"character_id": "NARRATOR", "display_name": "Narrator", "voice_name": "cosette", "quote_count": 0, "mention_count": 0, "gender_pronoun": None},
            ],
        }
        api_cm = _make_api_client(get_queue_cast=cast_data)
        with patch("kenkui.cli.voices.APIClient", return_value=api_cm):
            cmd_voices_cast(Namespace(job_id="abc12345"))
        api_cm.__enter__.return_value.get_queue_cast.assert_called_once_with("abc12345")

    def test_cast_empty_cast_prints_message(self):
        from kenkui.cli.voices import cmd_voices_cast

        cast_data = {
            "job_id": "abc12345",
            "book_name": "Empty Book",
            "narration_mode": "single",
            "cast": [],
        }
        api_cm = _make_api_client(get_queue_cast=cast_data)
        with patch("kenkui.cli.voices.APIClient", return_value=api_cm):
            cmd_voices_cast(Namespace(job_id="abc12345"))

    def test_narrator_sorted_last(self):
        from kenkui.cli.voices import _sort_cast

        voices = {"NARRATOR": "cosette", "Rand": "alba", "Mat": "jean"}
        keys = [k for k, _ in _sort_cast(voices)]
        assert keys[-1] == "NARRATOR"


class TestVoicesAudition:
    def _make_args(self, voice="jean", text=None):
        from argparse import Namespace
        return Namespace(voice=voice, text=text)

    def test_happy_path_calls_api_and_plays(self, monkeypatch):
        from kenkui.cli.voices import cmd_voices_audition

        wav_bytes = b"RIFF\x00\x00\x00\x00WAVEfmt "  # minimal wav header stub

        api_cm = _make_api_client(
            audition_voice={"task_id": "task-1"},
            poll_task={"status": "completed"},
            get_audition_wav_url="http://localhost/voices/audition/task-1.wav",
        )

        opened = []
        import httpx

        monkeypatch.setattr(httpx, "get", lambda url: MagicMock(content=wav_bytes))
        monkeypatch.setattr("subprocess.run", lambda cmd, **kw: opened.append(cmd))

        with patch("kenkui.cli.voices.APIClient", return_value=api_cm):
            cmd_voices_audition(self._make_args())

        api_cm.__enter__.return_value.audition_voice.assert_called_once()
        api_cm.__enter__.return_value.poll_task.assert_called_once()
        assert len(opened) == 1

    def test_synthesis_failure_prints_error(self, monkeypatch):
        from kenkui.cli.voices import cmd_voices_audition

        api_cm = _make_api_client(
            audition_voice={"task_id": "task-fail"},
            poll_task={"status": "failed", "error": "synthesis error"},
        )

        with patch("kenkui.cli.voices.APIClient", return_value=api_cm):
            # Should return early without playing — not raise SystemExit
            cmd_voices_audition(self._make_args())

        api_cm.__enter__.return_value.get_audition_wav_url.assert_not_called()

    def test_custom_text_passed_to_api(self, monkeypatch):
        from kenkui.cli.voices import cmd_voices_audition

        wav_bytes = b"RIFF\x00\x00\x00\x00WAVEfmt "
        api_cm = _make_api_client(
            audition_voice={"task_id": "task-1"},
            poll_task={"status": "completed"},
        )

        import httpx
        monkeypatch.setattr(httpx, "get", lambda url: MagicMock(content=wav_bytes))
        monkeypatch.setattr("subprocess.run", lambda cmd, **kw: None)

        with patch("kenkui.cli.voices.APIClient", return_value=api_cm):
            cmd_voices_audition(self._make_args(text="Hello world"))

        call_kwargs = api_cm.__enter__.return_value.audition_voice.call_args
        assert call_kwargs[1].get("text") == "Hello world" or (
            len(call_kwargs[0]) > 1 and call_kwargs[0][1] == "Hello world"
        )

    def test_system_player_platform(self, monkeypatch):
        import sys as _sys
        from kenkui.cli.voices import _player_command

        monkeypatch.setattr(_sys, "platform", "darwin")
        assert _player_command() == "open"

        monkeypatch.setattr(_sys, "platform", "linux")
        assert _player_command() == "xdg-open"


class TestVoicesTUI:
    """Tests for the interactive voice management TUI."""

    @staticmethod
    def _stub_inquirerpy(monkeypatch):
        """Stub InquirerPy in sys.modules so lazy imports inside TUI functions succeed.

        _tui_execute is monkeypatched separately to drive the prompts.
        The inquirer object returned here is a MagicMock — its .select/.fuzzy/.etc
        calls produce MagicMock objects that _tui_execute ignores entirely.
        """
        import sys
        from unittest.mock import MagicMock
        mock_inq = MagicMock()
        monkeypatch.setitem(sys.modules, "InquirerPy", mock_inq)
        monkeypatch.setitem(sys.modules, "InquirerPy.inquirer", mock_inq.inquirer)

    # Autouse fixture: stubs InquirerPy for every test in this class.
    import pytest

    @pytest.fixture(autouse=True)
    def _stub_inq(self, monkeypatch):
        self._stub_inquirerpy(monkeypatch)

    def _voices_list_response(self, names):
        return {
            "voices": [
                {"name": n, "source": "builtin", "gender": "Male", "accent": "American",
                 "dataset": None, "speaker_id": None, "excluded": False, "description": f"{n} (builtin)"}
                for n in names
            ],
            "total": len(names),
        }

    def test_tui_exit_immediately(self, monkeypatch):
        """Main menu: selecting Exit breaks the loop cleanly."""
        from kenkui.cli.voices import cmd_voices_tui
        from argparse import Namespace

        prompts = iter(["exit"])
        monkeypatch.setattr(
            "kenkui.cli.voices._tui_execute",
            lambda p: next(prompts),
        )
        cmd_voices_tui(Namespace())  # should not raise

    def test_tui_browse_back_then_exit(self, monkeypatch):
        """Browse → Back → Exit exercises the browse return path."""
        from kenkui.cli.voices import cmd_voices_tui
        from argparse import Namespace

        api_cm = _make_api_client(list_voices=self._voices_list_response(["alba", "jean"]))
        monkeypatch.setattr("kenkui.cli.voices.APIClient", lambda **kw: api_cm)

        # Main: browse → browse list: __back__ → main: exit
        prompts = iter(["browse", "__back__", "exit"])
        monkeypatch.setattr(
            "kenkui.cli.voices._tui_execute",
            lambda p: next(prompts),
        )
        cmd_voices_tui(Namespace())

    def test_tui_browse_audition_returns_to_browse(self, monkeypatch):
        """Browse → select voice → Audition → returns to browse list."""
        from kenkui.cli.voices import cmd_voices_tui
        from argparse import Namespace

        wav_bytes = b"RIFF\x00\x00\x00\x00WAVEfmt "
        import httpx
        monkeypatch.setattr(httpx, "get", lambda url: MagicMock(content=wav_bytes))
        monkeypatch.setattr("subprocess.run", lambda cmd, **kw: None)

        api_cm = _make_api_client(
            list_voices=self._voices_list_response(["jean"]),
            audition_voice={"task_id": "t1"},
            poll_task={"status": "completed"},
            get_audition_wav_url="http://localhost/voices/audition/t1.wav",
        )
        # Provide get_voice for the detail view
        api_cm.__enter__.return_value.get_voice.return_value = {
            "name": "jean", "source": "builtin", "gender": "Male",
            "accent": "American", "dataset": None, "speaker_id": None,
            "excluded": False, "description": "jean (builtin)",
        }
        monkeypatch.setattr("kenkui.cli.voices.APIClient", lambda **kw: api_cm)

        # main: browse → voice list: jean → action: audition →
        # back in browse (loop): __back__ → main: exit
        prompts = iter(["browse", "jean", "audition", "__back__", "exit"])
        monkeypatch.setattr(
            "kenkui.cli.voices._tui_execute",
            lambda p: next(prompts),
        )
        cmd_voices_tui(Namespace())

    def test_tui_browse_toggle_exclude(self, monkeypatch, tmp_path):
        """Browse → voice → Exclude from pool → Back → exit."""
        from kenkui.cli.voices import cmd_voices_tui
        from argparse import Namespace

        exclude_calls = []
        api_cm = _make_api_client(
            list_voices=self._voices_list_response(["jean"]),
            exclude_voice={"status": "excluded"},
        )
        api_cm.__enter__.return_value.get_voice.return_value = {
            "name": "jean", "source": "builtin", "gender": "Male",
            "accent": "American", "dataset": None, "speaker_id": None,
            "excluded": False, "description": "jean (builtin)",
        }
        orig_exclude = api_cm.__enter__.return_value.exclude_voice
        orig_exclude.side_effect = lambda name: exclude_calls.append(name) or {"status": "excluded"}
        monkeypatch.setattr("kenkui.cli.voices.APIClient", lambda **kw: api_cm)

        # main: browse → voice: jean → toggle → back → browse: __back__ → exit
        prompts = iter(["browse", "jean", "toggle", "back", "__back__", "exit"])
        monkeypatch.setattr(
            "kenkui.cli.voices._tui_execute",
            lambda p: next(prompts),
        )
        cmd_voices_tui(Namespace())
        assert "jean" in exclude_calls

    def test_tui_pool_no_exclusions(self, monkeypatch):
        """Pool menu shows a message and returns when nothing is excluded."""
        from kenkui.cli.voices import cmd_voices_tui
        from argparse import Namespace

        api_cm = _make_api_client(list_voices={"voices": [], "total": 0})
        monkeypatch.setattr("kenkui.cli.voices.APIClient", lambda **kw: api_cm)

        # main: pool → main: exit
        prompts = iter(["pool", "exit"])
        monkeypatch.setattr(
            "kenkui.cli.voices._tui_execute",
            lambda p: next(prompts),
        )
        cmd_voices_tui(Namespace())

    def test_tui_cast_lookup(self, monkeypatch):
        """Cast lookup fires cmd_voices_cast with the typed job_id."""
        from kenkui.cli.voices import cmd_voices_tui
        from argparse import Namespace

        cast_calls = []
        monkeypatch.setattr(
            "kenkui.cli.voices.cmd_voices_cast",
            lambda args: cast_calls.append(args.job_id),
        )

        # main: cast → (text prompt returns "job-123") → main: exit
        prompts = iter(["cast", "job-123", "exit"])
        monkeypatch.setattr(
            "kenkui.cli.voices._tui_execute",
            lambda p: next(prompts),
        )
        cmd_voices_tui(Namespace())
        assert cast_calls == ["job-123"]
