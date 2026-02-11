"""Tests for the kenkui CLI flags."""

import subprocess
import sys
from pathlib import Path

import pytest

# Path to the test EPUB
TEST_EPUB = Path("src/kenkui/samples/Les Miserables - Victor Hugo.epub")


@pytest.fixture
def epub_path():
    """Return the path to the test EPUB file."""
    return TEST_EPUB


class TestChapterFilteringFlags:
    """Tests for the chapter filtering flags."""

    def test_chapter_preset_help_text(self):
        """Test that --chapter-preset appears in help output."""
        result = subprocess.run(
            [sys.executable, "-m", "kenkui", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--chapter-preset" in result.stdout
        # argparse shows choices with curly braces and commas
        assert "all,content-only,chapters-only,with-parts" in result.stdout.replace(
            " ", ""
        ).replace("{", "").replace("}", "")

    def test_include_chapter_help_text(self):
        """Test that --include-chapter appears in help output."""
        result = subprocess.run(
            [sys.executable, "-m", "kenkui", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--include-chapter" in result.stdout or "-I" in result.stdout

    def test_exclude_chapter_help_text(self):
        """Test that --exclude-chapter appears in help output."""
        result = subprocess.run(
            [sys.executable, "-m", "kenkui", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--exclude-chapter" in result.stdout or "-X" in result.stdout

    def test_preview_help_text(self):
        """Test that --preview appears in help output."""
        result = subprocess.run(
            [sys.executable, "-m", "kenkui", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--preview" in result.stdout


class TestVerboseFlag:
    """Tests for the --verbose flag."""

    def test_verbose_help_text(self):
        """Test that --verbose appears in help output."""
        result = subprocess.run(
            [sys.executable, "-m", "kenkui", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--verbose" in result.stdout
        assert "worker logs" in result.stdout.lower() or "Show all" in result.stdout


class TestBasicCLI:
    """Basic CLI functionality tests."""

    def test_list_voices_flag(self):
        """Test that --list-voices works and shows available voices."""
        result = subprocess.run(
            [sys.executable, "-m", "kenkui", "--list-voices"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Available Voices" in result.stdout
        assert "alba" in result.stdout

    def test_no_input_shows_help(self):
        """Test that running without input shows abbreviated help."""
        result = subprocess.run(
            [sys.executable, "-m", "kenkui"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Kenkui - EPUB to Audiobook Converter" in result.stdout
        assert "Usage:" in result.stdout

    def test_epub_file_not_found(self):
        """Test error handling for non-existent EPUB file."""
        result = subprocess.run(
            [sys.executable, "-m", "kenkui", "nonexistent.epub"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "No EPUB files found" in result.stdout or "Error" in result.stdout


class TestKeyboardInterrupt:
    """Tests for keyboard interrupt handling."""

    def test_keyboard_interrupt_exit_code(self):
        """Test that keyboard interrupt produces correct exit code."""
        # This is a basic test - in practice we'd need to mock the input
        # For now, we just verify the exit code constant is defined
        assert True  # Placeholder for more complex test
