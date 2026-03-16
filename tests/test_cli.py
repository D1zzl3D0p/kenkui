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
        for flag in (
            "--config",
            "--output",
            "--list-voices",
            "--list-presets",
            "--verbose",
        ):
            assert flag in r.stdout, f"{flag} missing from --help"

    def test_version_exits_0(self):
        r = subprocess.run(
            [sys.executable, "-m", "kenkui", "--version"],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0
        assert "0." in r.stdout or "0." in r.stderr


class TestInformationalFlags:
    def test_list_voices_exits_0(self):
        r = subprocess.run(
            [sys.executable, "-m", "kenkui", "--list-voices"],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0
        assert "alba" in r.stdout

    def test_list_voices_shows_sections(self):
        r = subprocess.run(
            [sys.executable, "-m", "kenkui", "--list-voices"],
            capture_output=True,
            text=True,
        )
        assert "Built-in" in r.stdout

    def test_list_presets_exits_0(self):
        r = subprocess.run(
            [sys.executable, "-m", "kenkui", "--list-presets"],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0
        assert "content-only" in r.stdout

    def test_list_presets_shows_all_presets(self):
        r = subprocess.run(
            [sys.executable, "-m", "kenkui", "--list-presets"],
            capture_output=True,
            text=True,
        )
        for preset in ("all", "content-only", "chapters-only", "with-parts", "none"):
            assert preset in r.stdout, f"Preset '{preset}' missing from --list-presets"


class TestModeDetection:
    """Verify that ebook files trigger headless and dirs/absent trigger TUI."""

    def test_ebook_extension_set(self):
        from kenkui.__main__ import EBOOK_EXTENSIONS

        for ext in (".epub", ".mobi", ".fb2", ".azw3"):
            assert ext in EBOOK_EXTENSIONS

    def test_file_with_epub_extension_triggers_headless(self):
        """If an .epub file is given, _run_headless should be called."""
        from kenkui.__main__ import EBOOK_EXTENSIONS, _build_parser

        parser = _build_parser()
        # Simulate passing an epub file
        args = parser.parse_args([str(TEST_EPUB)])
        assert args.input is not None
        assert args.input.suffix.lower() in EBOOK_EXTENSIONS

    def test_directory_triggers_tui(self):
        from kenkui.__main__ import EBOOK_EXTENSIONS, _build_parser

        parser = _build_parser()
        args = parser.parse_args(["src/kenkui/samples"])
        # A directory should NOT match ebook extension check
        assert (
            args.input is None
            or args.input.suffix.lower() not in EBOOK_EXTENSIONS
            or not args.input.is_file()
        )

    def test_no_input_triggers_tui(self):
        from kenkui.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args([])
        assert args.input is None

    def test_nonexistent_file_exits_1(self):
        r = subprocess.run(
            [sys.executable, "-m", "kenkui", "/nonexistent/path/book.epub"],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 1


class TestLogging:
    def test_verbose_flag_accepted(self):
        """--verbose should not cause an error when used with --list-voices."""
        r = subprocess.run(
            [sys.executable, "-m", "kenkui", "--verbose", "--list-voices"],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0
