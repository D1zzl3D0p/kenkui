"""Tests for kenkui.services.nlp_service."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kenkui.models import Chapter
from kenkui.services.nlp_service import fast_scan, full_analysis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_nlp_record(slug="jane_eyre", canonical="Jane Eyre", mentions=50):
    """Return a mock kenkui.nlp.models.CharacterRecord (Pydantic)."""
    rec = MagicMock()
    rec.slug = slug
    rec.canonical_name = canonical
    rec.aliases = ["Jane"]
    rec.titles = []
    rec.gender = "she/her"
    rec.role = "protagonist"
    rec.description = ""
    rec.chapters = [0, 1]
    rec.first_appearance = ("jane_eyre_v1", 0)
    rec.last_appearance = ("jane_eyre_v1", 35)
    rec.mention_count = mentions
    rec.quote_count = 0
    return rec


def _make_mock_roster(records=None):
    roster = MagicMock()
    roster.characters = records if records is not None else [_make_mock_nlp_record()]
    return roster


# ---------------------------------------------------------------------------
# fast_scan tests
# ---------------------------------------------------------------------------


def test_fast_scan_raises_for_missing_file(tmp_path):
    """FileNotFoundError when the ebook path does not exist."""
    missing = str(tmp_path / "nonexistent.epub")
    with pytest.raises(FileNotFoundError):
        fast_scan(missing)


def test_fast_scan_calls_provider_build_roster(tmp_path):
    """fast_scan should call provider.build_roster with the parsed chapters."""
    fake_epub = tmp_path / "book.epub"
    fake_epub.write_bytes(b"fake")

    fake_chapters = [Chapter(index=0, title="Ch 1", paragraphs=["Hello"]),
                     Chapter(index=1, title="Ch 2", paragraphs=["World"])]
    mock_reader = MagicMock()
    mock_reader.get_chapters.return_value = fake_chapters

    mock_roster = _make_mock_roster()
    mock_provider = MagicMock()
    mock_provider.build_roster.return_value = mock_roster

    with (
        patch("kenkui.services.nlp_service.get_reader", return_value=mock_reader),
        patch("kenkui.services.nlp_service.get_provider", return_value=mock_provider),
        patch("kenkui.services.nlp_service.get_cached_roster", return_value=None),
        patch("kenkui.services.nlp_service.cache_roster"),
        patch("kenkui.services.nlp_service.book_hash", return_value="abc123"),
    ):
        result = fast_scan(str(fake_epub), nlp_model="llama3.2")

    mock_provider.build_roster.assert_called_once()
    assert mock_provider.build_roster.call_args[0][0] is fake_chapters


def test_fast_scan_uses_config_nlp_model(tmp_path):
    """When nlp_model=None, fast_scan passes the config (unmodified) to get_provider."""
    fake_epub = tmp_path / "book.epub"
    fake_epub.write_bytes(b"fake")

    mock_reader = MagicMock()
    mock_reader.get_chapters.return_value = [Chapter(index=0, title="Ch", paragraphs=["t"])]

    mock_config = MagicMock()
    mock_config.nlp_model = "mistral"
    mock_config.nlp_provider = "ollama"

    mock_roster = _make_mock_roster()
    mock_provider = MagicMock()
    mock_provider.build_roster.return_value = mock_roster

    with (
        patch("kenkui.services.nlp_service.get_reader", return_value=mock_reader),
        patch("kenkui.services.nlp_service.get_provider", return_value=mock_provider) as mock_get_provider,
        patch("kenkui.services.nlp_service.get_cached_roster", return_value=None),
        patch("kenkui.services.nlp_service.cache_roster"),
        patch("kenkui.services.nlp_service.book_hash", return_value="abc123"),
        patch("kenkui.services.nlp_service.load_app_config", return_value=mock_config) as mock_cfg,
    ):
        fast_scan(str(fake_epub), nlp_model=None)

    mock_cfg.assert_called_once_with(None)
    # Config is passed unchanged (no nlp_model override) to get_provider
    mock_get_provider.assert_called_once_with(mock_config)


def test_fast_scan_progress_callback_receives_int_and_str(tmp_path):
    """Progress callback should receive (int, str) tuples with increasing percents."""
    fake_epub = tmp_path / "book.epub"
    fake_epub.write_bytes(b"fake")

    mock_reader = MagicMock()
    mock_reader.get_chapters.return_value = [Chapter(index=0, title="Ch", paragraphs=["t"])]

    received: list[tuple[int, str]] = []

    def _cb(pct: int, msg: str) -> None:
        received.append((pct, msg))

    mock_roster = _make_mock_roster()

    def _fake_build_roster(chapters, series_roster=None, progress_callback=None):
        if progress_callback:
            progress_callback("Loading spaCy…")
            progress_callback("Building roster…")
            progress_callback("Counting mentions…")
        return mock_roster

    mock_provider = MagicMock()
    mock_provider.build_roster.side_effect = _fake_build_roster

    with (
        patch("kenkui.services.nlp_service.get_reader", return_value=mock_reader),
        patch("kenkui.services.nlp_service.get_provider", return_value=mock_provider),
        patch("kenkui.services.nlp_service.get_cached_roster", return_value=None),
        patch("kenkui.services.nlp_service.cache_roster"),
        patch("kenkui.services.nlp_service.book_hash", return_value="abc123"),
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
    assert percents[2] == 25
    assert percents[3] == 40
    assert percents[4] == 55
    assert percents[5] == 100
    assert messages[5] == "Scan complete"

    for p, _ in received:
        assert isinstance(p, int)
    for _, m in received:
        assert isinstance(m, str)
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


def test_full_analysis_calls_provider_build_roster_and_attribute_chapter(tmp_path):
    """full_analysis calls build_roster once then attribute_chapter once per chapter."""
    fake_epub = tmp_path / "book.epub"
    fake_epub.write_bytes(b"fake")

    fake_chapters = [
        Chapter(index=0, title="Ch 1", paragraphs=["text1"]),
        Chapter(index=1, title="Ch 2", paragraphs=["text2"]),
    ]
    mock_reader = MagicMock()
    mock_reader.get_chapters.return_value = fake_chapters

    mock_roster = _make_mock_roster()
    mock_attr_result = MagicMock()
    mock_attr_result.attributions = []

    mock_provider = MagicMock()
    mock_provider.build_roster.return_value = mock_roster
    mock_provider.attribute_chapter.return_value = mock_attr_result

    with (
        patch("kenkui.services.nlp_service.get_reader", return_value=mock_reader),
        patch("kenkui.services.nlp_service.get_provider", return_value=mock_provider),
        patch("kenkui.services.nlp_service.get_cached_result", return_value=None),
        patch("kenkui.services.nlp_service.cache_result"),
        patch("kenkui.services.nlp_service.book_hash", return_value="abc123"),
        patch("kenkui.services.nlp_service._attribution_to_segments", return_value=[]),
    ):
        result = full_analysis(str(fake_epub), nlp_model="llama3.2")

    mock_provider.build_roster.assert_called_once()
    assert mock_provider.attribute_chapter.call_count == 2


def test_full_analysis_uses_config_nlp_model(tmp_path):
    """When nlp_model=None, full_analysis passes the config to get_provider."""
    fake_epub = tmp_path / "book.epub"
    fake_epub.write_bytes(b"fake")

    mock_reader = MagicMock()
    mock_reader.get_chapters.return_value = [Chapter(index=0, title="Ch", paragraphs=["t"])]

    mock_config = MagicMock()
    mock_config.nlp_model = "mistral"
    mock_config.nlp_provider = "ollama"

    mock_roster = _make_mock_roster()
    mock_attr_result = MagicMock()
    mock_attr_result.attributions = []

    mock_provider = MagicMock()
    mock_provider.build_roster.return_value = mock_roster
    mock_provider.attribute_chapter.return_value = mock_attr_result

    with (
        patch("kenkui.services.nlp_service.get_reader", return_value=mock_reader),
        patch("kenkui.services.nlp_service.get_provider", return_value=mock_provider) as mock_get_provider,
        patch("kenkui.services.nlp_service.get_cached_result", return_value=None),
        patch("kenkui.services.nlp_service.cache_result"),
        patch("kenkui.services.nlp_service.book_hash", return_value="abc123"),
        patch("kenkui.services.nlp_service._attribution_to_segments", return_value=[]),
        patch("kenkui.services.nlp_service.load_app_config", return_value=mock_config) as mock_cfg,
    ):
        full_analysis(str(fake_epub), nlp_model=None)

    mock_cfg.assert_called_once_with(None)
    mock_get_provider.assert_called_once_with(mock_config)


def test_full_analysis_progress_callback_receives_int_and_str(tmp_path):
    """Progress callback should receive (int, str) tuples with increasing percents."""
    fake_epub = tmp_path / "book.epub"
    fake_epub.write_bytes(b"fake")

    mock_reader = MagicMock()
    mock_reader.get_chapters.return_value = [Chapter(index=0, title="Ch", paragraphs=["t"])]

    received: list[tuple[int, str]] = []

    def _cb(pct: int, msg: str) -> None:
        received.append((pct, msg))

    mock_roster = _make_mock_roster()

    def _fake_build_roster(chapters, series_roster=None, progress_callback=None):
        if progress_callback:
            progress_callback("Extracting quotes…")
            progress_callback("Clustering entities…")
            progress_callback("Attributing speakers…")
        return mock_roster

    mock_attr_result = MagicMock()
    mock_attr_result.attributions = []

    mock_provider = MagicMock()
    mock_provider.build_roster.side_effect = _fake_build_roster
    mock_provider.attribute_chapter.return_value = mock_attr_result

    with (
        patch("kenkui.services.nlp_service.get_reader", return_value=mock_reader),
        patch("kenkui.services.nlp_service.get_provider", return_value=mock_provider),
        patch("kenkui.services.nlp_service.get_cached_result", return_value=None),
        patch("kenkui.services.nlp_service.cache_result"),
        patch("kenkui.services.nlp_service.book_hash", return_value="abc123"),
        patch("kenkui.services.nlp_service._attribution_to_segments", return_value=[]),
    ):
        full_analysis(str(fake_epub), nlp_model="llama3.2", progress_callback=_cb)

    percents = [p for p, _ in received]
    messages = [m for _, m in received]

    assert percents[0] == 0
    assert messages[0] == "Parsing ebook"
    assert percents[1] == 5
    assert messages[1] == "Starting NLP analysis"
    # 3 roster adapter bumps: 5+8=13, 13+8=21, 21+8=29
    assert percents[2] == 13
    assert percents[3] == 21
    assert percents[4] == 29
    assert messages[-1] == "Analysis complete"
    assert percents[-1] == 100

    for p, _ in received:
        assert isinstance(p, int)
    for _, m in received:
        assert isinstance(m, str)
    for i in range(1, len(percents)):
        assert percents[i] >= percents[i - 1]


@pytest.mark.skip(reason="Requires real spaCy / Ollama — integration-only")
def test_full_analysis_real_pipeline():
    """Placeholder: full pipeline test requiring spaCy + Ollama."""
    # Would test: real epub file → full NLP pipeline → NLPResult with characters and quotes
