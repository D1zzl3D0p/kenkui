"""Tests for kenkui.services.nlp_service."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kenkui.services.nlp_service import fast_scan, full_analysis


# ---------------------------------------------------------------------------
# fast_scan tests
# ---------------------------------------------------------------------------


def test_fast_scan_raises_for_missing_file(tmp_path):
    """FileNotFoundError when the ebook path does not exist."""
    missing = str(tmp_path / "nonexistent.epub")
    with pytest.raises(FileNotFoundError):
        fast_scan(missing)


def test_fast_scan_delegates_to_run_fast_scan(tmp_path):
    """fast_scan should call run_fast_scan with chapters and the correct nlp_model."""
    fake_epub = tmp_path / "book.epub"
    fake_epub.write_bytes(b"fake")

    fake_chapters = [MagicMock(), MagicMock()]
    mock_reader = MagicMock()
    mock_reader.get_chapters.return_value = fake_chapters

    mock_result = MagicMock()

    with (
        patch("kenkui.services.nlp_service.get_reader", return_value=mock_reader) as mock_get_reader,
        patch("kenkui.services.nlp_service.run_fast_scan", return_value=mock_result) as mock_run,
    ):
        result = fast_scan(str(fake_epub), nlp_model="llama3.2")

    mock_get_reader.assert_called_once_with(Path(str(fake_epub)))
    mock_reader.get_chapters.assert_called_once()

    call_args = mock_run.call_args
    assert call_args[0][0] is fake_chapters              # chapters positional arg
    assert call_args[0][1] == Path(str(fake_epub))       # book_path positional arg
    assert call_args[0][2] == "llama3.2"                 # nlp_model positional arg
    assert result is mock_result


def test_fast_scan_uses_config_nlp_model(tmp_path):
    """When nlp_model=None, fast_scan should read nlp_model from AppConfig."""
    fake_epub = tmp_path / "book.epub"
    fake_epub.write_bytes(b"fake")

    fake_chapters = [MagicMock()]
    mock_reader = MagicMock()
    mock_reader.get_chapters.return_value = fake_chapters

    mock_config = MagicMock()
    mock_config.nlp_model = "mistral"

    mock_result = MagicMock()

    with (
        patch("kenkui.services.nlp_service.get_reader", return_value=mock_reader),
        patch("kenkui.services.nlp_service.run_fast_scan", return_value=mock_result) as mock_run,
        patch("kenkui.services.nlp_service.load_app_config", return_value=mock_config) as mock_cfg,
    ):
        result = fast_scan(str(fake_epub), nlp_model=None)

    mock_cfg.assert_called_once_with(None)
    call_args = mock_run.call_args
    assert call_args[0][2] == "mistral"
    assert result is mock_result


def test_fast_scan_progress_callback_receives_int_and_str(tmp_path):
    """Progress callback should receive (int, str) tuples with increasing percents."""
    fake_epub = tmp_path / "book.epub"
    fake_epub.write_bytes(b"fake")

    mock_reader = MagicMock()
    mock_reader.get_chapters.return_value = [MagicMock()]

    received: list[tuple[int, str]] = []

    def _cb(pct: int, msg: str) -> None:
        received.append((pct, msg))

    # Simulate run_fast_scan calling the string-only adapter 3 times
    def _fake_run_fast_scan(chapters, book_path, nlp_model, progress_callback=None, **kwargs):
        if progress_callback:
            progress_callback("Loading spaCy…")
            progress_callback("Building roster…")
            progress_callback("Counting mentions…")
        return MagicMock()

    with (
        patch("kenkui.services.nlp_service.get_reader", return_value=mock_reader),
        patch("kenkui.services.nlp_service.run_fast_scan", side_effect=_fake_run_fast_scan),
    ):
        fast_scan(str(fake_epub), nlp_model="llama3.2", progress_callback=_cb)

    # Should have received: 0 (Parsing ebook), 10 (Starting NLP scan),
    # 25, 40, 55 (3 adapter calls: 10+15, 25+15, 40+15), 100 (Scan complete)
    assert len(received) == 6

    percents = [p for p, _ in received]
    messages = [m for _, m in received]

    assert percents[0] == 0
    assert messages[0] == "Parsing ebook"
    assert percents[1] == 10
    assert messages[1] == "Starting NLP scan"
    # Adapter bumps: 10+15=25, 25+15=40, 40+15=55
    assert percents[2] == 25
    assert percents[3] == 40
    assert percents[4] == 55
    assert percents[5] == 100
    assert messages[5] == "Scan complete"

    # All percents are ints
    for p, _ in received:
        assert isinstance(p, int)
    # All messages are strings
    for _, m in received:
        assert isinstance(m, str)

    # Percents are non-decreasing
    for i in range(1, len(percents)):
        assert percents[i] >= percents[i - 1]


# ---------------------------------------------------------------------------
# full_analysis tests
# ---------------------------------------------------------------------------


def test_full_analysis_raises_for_missing_file(tmp_path):
    """FileNotFoundError when the ebook path does not exist."""
    missing = str(tmp_path / "nonexistent.epub")
    with pytest.raises(FileNotFoundError):
        full_analysis(missing)


def test_full_analysis_delegates_to_run_analysis(tmp_path):
    """full_analysis should call run_analysis with chapters and the correct nlp_model."""
    fake_epub = tmp_path / "book.epub"
    fake_epub.write_bytes(b"fake")

    fake_chapters = [MagicMock(), MagicMock()]
    mock_reader = MagicMock()
    mock_reader.get_chapters.return_value = fake_chapters

    mock_result = MagicMock()

    with (
        patch("kenkui.services.nlp_service.get_reader", return_value=mock_reader) as mock_get_reader,
        patch("kenkui.services.nlp_service.run_analysis", return_value=mock_result) as mock_run,
    ):
        result = full_analysis(str(fake_epub), nlp_model="llama3.2")

    mock_get_reader.assert_called_once_with(Path(str(fake_epub)))
    mock_reader.get_chapters.assert_called_once()

    call_args = mock_run.call_args
    assert call_args[0][0] is fake_chapters              # chapters positional arg
    assert call_args[0][1] == Path(str(fake_epub))       # book_path positional arg
    assert call_args[0][2] == "llama3.2"                 # nlp_model positional arg
    assert result is mock_result


def test_full_analysis_uses_config_nlp_model(tmp_path):
    """When nlp_model=None, full_analysis should read nlp_model from AppConfig."""
    fake_epub = tmp_path / "book.epub"
    fake_epub.write_bytes(b"fake")

    fake_chapters = [MagicMock()]
    mock_reader = MagicMock()
    mock_reader.get_chapters.return_value = fake_chapters

    mock_config = MagicMock()
    mock_config.nlp_model = "mistral"

    mock_result = MagicMock()

    with (
        patch("kenkui.services.nlp_service.get_reader", return_value=mock_reader),
        patch("kenkui.services.nlp_service.run_analysis", return_value=mock_result) as mock_run,
        patch("kenkui.services.nlp_service.load_app_config", return_value=mock_config) as mock_cfg,
    ):
        result = full_analysis(str(fake_epub), nlp_model=None)

    mock_cfg.assert_called_once_with(None)
    call_args = mock_run.call_args
    assert call_args[0][2] == "mistral"
    assert result is mock_result


def test_full_analysis_progress_callback_receives_int_and_str(tmp_path):
    """Progress callback should receive (int, str) tuples with increasing percents."""
    fake_epub = tmp_path / "book.epub"
    fake_epub.write_bytes(b"fake")

    mock_reader = MagicMock()
    mock_reader.get_chapters.return_value = [MagicMock()]

    received: list[tuple[int, str]] = []

    def _cb(pct: int, msg: str) -> None:
        received.append((pct, msg))

    # Simulate run_analysis calling the string-only adapter 3 times
    def _fake_run_analysis(chapters, book_path, nlp_model, progress_callback=None, **kwargs):
        if progress_callback:
            progress_callback("Extracting quotes…")
            progress_callback("Clustering entities…")
            progress_callback("Attributing speakers…")
        return MagicMock()

    with (
        patch("kenkui.services.nlp_service.get_reader", return_value=mock_reader),
        patch("kenkui.services.nlp_service.run_analysis", side_effect=_fake_run_analysis),
    ):
        full_analysis(str(fake_epub), nlp_model="llama3.2", progress_callback=_cb)

    # Should have received: 0 (Parsing ebook), 5 (Starting NLP analysis),
    # 13, 21, 29 (3 adapter calls: 5+8, 13+8, 21+8), 100 (Analysis complete)
    assert len(received) == 6

    percents = [p for p, _ in received]
    messages = [m for _, m in received]

    assert percents[0] == 0
    assert messages[0] == "Parsing ebook"
    assert percents[1] == 5
    assert messages[1] == "Starting NLP analysis"
    # Adapter bumps: start=5, bump=8 → 5+8=13, 13+8=21, 21+8=29
    assert percents[2] == 13
    assert percents[3] == 21
    assert percents[4] == 29
    assert percents[5] == 100
    assert messages[5] == "Analysis complete"

    # All percents are ints
    for p, _ in received:
        assert isinstance(p, int)
    # All messages are strings
    for _, m in received:
        assert isinstance(m, str)

    # Percents are non-decreasing
    for i in range(1, len(percents)):
        assert percents[i] >= percents[i - 1]


@pytest.mark.skip(reason="Requires real spaCy / Ollama — integration-only")
def test_full_analysis_real_pipeline():
    """Placeholder: full pipeline test requiring spaCy + Ollama."""
    # Would test: real epub file → full NLP pipeline → NLPResult with characters and quotes
