"""Tests for the kenkui CLI entry point."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TEST_EPUB = Path("src/kenkui/samples/Les Miserables - Victor Hugo.epub")
EBOOK_EXTS = {".epub", ".mobi", ".fb2", ".azw", ".azw3", ".azw4"}


class TestHelpAndVersion:
    def test_help_exits_0(self):
        r = subprocess.run(
            [sys.executable, "-m", "kenkui", "--help"],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0
        assert "kenkui" in r.stdout.lower()

    def test_help_shows_key_args(self):
        r = subprocess.run(
            [sys.executable, "-m", "kenkui", "--help"],
            capture_output=True,
            text=True,
        )
        # Voices and presets are now discovered inside the wizard, not via flags.
        for flag in ("--config", "--output", "--verbose"):
            assert flag in r.stdout, f"{flag} missing from --help"

    def test_version_exits_0(self):
        r = subprocess.run(
            [sys.executable, "-m", "kenkui", "--version"],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0
        assert "0." in r.stdout or "0." in r.stderr

    def test_subcommands_shown_in_help(self):
        r = subprocess.run(
            [sys.executable, "-m", "kenkui", "--help"],
            capture_output=True,
            text=True,
        )
        for cmd in ("add", "queue", "config"):
            assert cmd in r.stdout, f"Sub-command '{cmd}' missing from --help"

    def test_add_help_exits_0(self):
        r = subprocess.run(
            [sys.executable, "-m", "kenkui", "add", "--help"],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0

    def test_queue_help_exits_0(self):
        r = subprocess.run(
            [sys.executable, "-m", "kenkui", "queue", "--help"],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0

    def test_config_help_exits_0(self):
        r = subprocess.run(
            [sys.executable, "-m", "kenkui", "config", "--help"],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0


class TestModeDetection:
    """Verify parser correctly handles the bare shorthand and sub-commands."""

    def test_ebook_extension_set(self):
        from kenkui.__main__ import EBOOK_EXTENSIONS

        for ext in (".epub", ".mobi", ".fb2", ".azw3"):
            assert ext in EBOOK_EXTENSIONS

    def test_bare_book_path_parsed(self):
        """A bare ebook path is stored in args.book (uses bare parser)."""
        from kenkui.__main__ import _build_bare_parser

        parser = _build_bare_parser()
        args = parser.parse_args([str(TEST_EPUB)])
        assert args.book is not None
        assert args.book.suffix.lower() == ".epub"

    def test_bare_no_subcommand_no_book(self):
        """With no args, the subcommand parser should succeed (prints help)."""
        from kenkui.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args([])
        # No book positional in the subparser parser — command should be None
        assert getattr(args, "command", None) is None

    def test_add_subcommand_book_parsed(self):
        from kenkui.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["add", str(TEST_EPUB)])
        assert args.command == "add"
        assert args.book is not None

    def test_add_subcommand_config_flag(self):
        from kenkui.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["add", str(TEST_EPUB), "-c", "myconfig"])
        assert args.config == "myconfig"

    def test_queue_subcommand_parsed(self):
        from kenkui.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["queue"])
        assert args.command == "queue"

    def test_queue_live_flag(self):
        from kenkui.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["queue", "--live"])
        assert args.live is True

    def test_queue_start_subcommand(self):
        from kenkui.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["queue", "start"])
        assert args.command == "queue"
        assert args.queue_command == "start"

    def test_queue_start_live(self):
        from kenkui.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["queue", "start", "--live"])
        assert args.queue_command == "start"
        assert args.live is True

    def test_queue_stop_subcommand(self):
        from kenkui.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["queue", "stop"])
        assert args.queue_command == "stop"

    def test_config_subcommand_path(self):
        from kenkui.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["config", "/tmp/my-config.toml"])
        assert args.command == "config"
        assert args.path == "/tmp/my-config.toml"

    def test_nonexistent_file_exits_nonzero(self):
        r = subprocess.run(
            [sys.executable, "-m", "kenkui", "/nonexistent/path/book.epub"],
            capture_output=True,
            text=True,
        )
        assert r.returncode != 0

    def test_add_missing_book_exits_nonzero(self):
        """'kenkui add' without a book path should error."""
        r = subprocess.run(
            [sys.executable, "-m", "kenkui", "add"],
            capture_output=True,
            text=True,
        )
        assert r.returncode != 0


class TestLogging:
    def test_verbose_flag_accepted(self):
        """--verbose should not cause a parse error."""
        r = subprocess.run(
            [sys.executable, "-m", "kenkui", "--verbose", "--help"],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0
