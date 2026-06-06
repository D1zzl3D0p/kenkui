"""Tests for kenkui.services.nlp_service."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kenkui.models import Chapter
from kenkui.services.nlp_service import attribute_only, fast_scan, full_analysis

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


def _make_mock_pipeline(roster=None, attr_result=None):
    """Return a mock NLPPipeline with mocked extract() and _attribution."""
    mock_roster = roster or _make_mock_roster()
    mock_attr_result = attr_result or MagicMock(attributions=[])

    pipeline = MagicMock()
    pipeline.extract.return_value = mock_roster
    pipeline._attribution = MagicMock()
    pipeline._attribution.attribute_chapter.return_value = mock_attr_result
    pipeline.attribute.return_value = MagicMock(
        characters=[],
        chapters=[],
        book_hash="abc123",
    )
    return pipeline


# ---------------------------------------------------------------------------
# fast_scan tests
# ---------------------------------------------------------------------------


def test_fast_scan_raises_for_missing_file(tmp_path):
    """FileNotFoundError when the ebook path does not exist."""
    missing = str(tmp_path / "nonexistent.epub")
    with pytest.raises(FileNotFoundError):
        fast_scan(missing)


def test_fast_scan_calls_pipeline_extract(tmp_path):
    """fast_scan should call pipeline.extract() with the parsed chapters."""
    fake_epub = tmp_path / "book.epub"
    fake_epub.write_bytes(b"fake")

    fake_chapters = [Chapter(index=0, title="Ch 1", paragraphs=["Hello"]),
                     Chapter(index=1, title="Ch 2", paragraphs=["World"])]
    mock_reader = MagicMock()
    mock_reader.get_chapters.return_value = fake_chapters

    mock_pipeline = _make_mock_pipeline()

    with (
        patch("kenkui.services.nlp_service.get_reader", return_value=mock_reader),
        patch("kenkui.services.nlp_service.NLPPipeline", return_value=mock_pipeline),
        patch("kenkui.services.nlp_service.NLPConfig"),
        patch("kenkui.services.nlp_service.get_cached_roster", return_value=None),
        patch("kenkui.services.nlp_service.cache_roster"),
        patch("kenkui.services.nlp_service.book_hash", return_value="abc123"),
    ):
        result = fast_scan(str(fake_epub), nlp_model="llama3.2")

    mock_pipeline.extract.assert_called_once()
    call_kwargs = mock_pipeline.extract.call_args
    assert call_kwargs.kwargs["book_path"] == Path(str(fake_epub))
    assert call_kwargs.kwargs["chapters"] is fake_chapters
    assert call_kwargs.kwargs["use_cache"] is False


def test_fast_scan_uses_config_nlp_model(tmp_path):
    """When nlp_model=None, fast_scan passes the unmodified config to NLPConfig.from_app_config."""
    fake_epub = tmp_path / "book.epub"
    fake_epub.write_bytes(b"fake")

    mock_reader = MagicMock()
    mock_reader.get_chapters.return_value = [Chapter(index=0, title="Ch", paragraphs=["t"])]

    mock_config = MagicMock()
    mock_config.nlp_model = "mistral"
    mock_config.nlp_provider = "ollama"
    mock_config.nlp_attribution_provider = ""

    mock_pipeline = _make_mock_pipeline()

    with (
        patch("kenkui.services.nlp_service.get_reader", return_value=mock_reader),
        patch("kenkui.services.nlp_service.NLPPipeline", return_value=mock_pipeline),
        patch("kenkui.services.nlp_service.NLPConfig") as mock_nlp_cfg_cls,
        patch("kenkui.services.nlp_service.get_cached_roster", return_value=None),
        patch("kenkui.services.nlp_service.cache_roster"),
        patch("kenkui.services.nlp_service.book_hash", return_value="abc123"),
        patch("kenkui.services.nlp_service.load_app_config", return_value=mock_config) as mock_cfg,
    ):
        fast_scan(str(fake_epub), nlp_model=None)

    mock_cfg.assert_called_once_with(None)
    # NLPConfig.from_app_config should be called with the unmodified config
    mock_nlp_cfg_cls.from_app_config.assert_called_once_with(mock_config)


def test_fast_scan_progress_callback_receives_int_and_str(tmp_path):
    """Progress callback should receive (int, str) tuples at key milestones."""
    fake_epub = tmp_path / "book.epub"
    fake_epub.write_bytes(b"fake")

    mock_reader = MagicMock()
    mock_reader.get_chapters.return_value = [Chapter(index=0, title="Ch", paragraphs=["t"])]

    received: list[tuple[int, str]] = []

    def _cb(pct: int, msg: str) -> None:
        received.append((pct, msg))

    mock_pipeline = _make_mock_pipeline()

    with (
        patch("kenkui.services.nlp_service.get_reader", return_value=mock_reader),
        patch("kenkui.services.nlp_service.NLPPipeline", return_value=mock_pipeline),
        patch("kenkui.services.nlp_service.NLPConfig"),
        patch("kenkui.services.nlp_service.get_cached_roster", return_value=None),
        patch("kenkui.services.nlp_service.cache_roster"),
        patch("kenkui.services.nlp_service.book_hash", return_value="abc123"),
    ):
        fast_scan(str(fake_epub), nlp_model="llama3.2", progress_callback=_cb)

    percents = [p for p, _ in received]
    messages = [m for _, m in received]

    # Must receive: 0 (Starting extraction), ..., 100 (Extraction complete)
    assert percents[0] == 0
    assert messages[0] == "Starting extraction"
    assert percents[-1] == 100
    assert messages[-1] == "Extraction complete"

    for p, _ in received:
        assert isinstance(p, int)
    for _, m in received:
        assert isinstance(m, str)
    for i in range(1, len(percents)):
        assert percents[i] >= percents[i - 1]


def test_fast_scan_progress_event_callback_uses_chapter_units(tmp_path):
    fake_epub = tmp_path / "book.epub"
    fake_epub.write_bytes(b"fake")

    fake_chapters = [
        Chapter(index=0, title="Ch 1", paragraphs=["text1"]),
        Chapter(index=1, title="Ch 2", paragraphs=["text2"]),
    ]
    mock_reader = MagicMock()
    mock_reader.get_chapters.return_value = fake_chapters

    events = []
    mock_pipeline = _make_mock_pipeline()

    with (
        patch("kenkui.services.nlp_service.get_reader", return_value=mock_reader),
        patch("kenkui.services.nlp_service.NLPPipeline", return_value=mock_pipeline),
        patch("kenkui.services.nlp_service.NLPConfig"),
        patch("kenkui.services.nlp_service.get_cached_roster", return_value=None),
        patch("kenkui.services.nlp_service.cache_roster"),
        patch("kenkui.services.nlp_service.book_hash", return_value="abc123"),
    ):
        fast_scan(str(fake_epub), nlp_model="llama3.2", progress_event_callback=events.append)

    assert [event.stage for event in events] == [
        "nlp_extraction",
        "nlp_extraction",
        "nlp_extraction",
        "nlp_extraction",
    ]
    assert all(event.unit == "chapters" for event in events)
    assert [event.status for event in events] == ["started", "advanced", "advanced", "completed"]
    assert [event.completed_units for event in events] == [0, 1, 1, 2]
    assert all(event.total_units == 2 for event in events)


# ---------------------------------------------------------------------------
# full_analysis tests
# ---------------------------------------------------------------------------


def test_full_analysis_raises_for_missing_file(tmp_path):
    """FileNotFoundError when the ebook path does not exist."""
    missing = str(tmp_path / "nonexistent.epub")
    with pytest.raises(FileNotFoundError):
        full_analysis(missing)


def test_full_analysis_calls_pipeline_extract_and_attribute_chapter(tmp_path):
    """full_analysis calls pipeline.extract once then _attribution.attribute_chapter once per chapter."""
    fake_epub = tmp_path / "book.epub"
    fake_epub.write_bytes(b"fake")

    fake_chapters = [
        Chapter(index=0, title="Ch 1", paragraphs=["text1"]),
        Chapter(index=1, title="Ch 2", paragraphs=["text2"]),
    ]
    mock_reader = MagicMock()
    mock_reader.get_chapters.return_value = fake_chapters

    mock_pipeline = _make_mock_pipeline()

    with (
        patch("kenkui.services.nlp_service.get_reader", return_value=mock_reader),
        patch("kenkui.services.nlp_service.NLPPipeline", return_value=mock_pipeline),
        patch("kenkui.services.nlp_service.NLPConfig"),
        patch("kenkui.services.nlp_service.get_cached_result", return_value=None),
        patch("kenkui.services.nlp_service.cache_result"),
        patch("kenkui.services.nlp_service.book_hash", return_value="abc123"),
        patch("kenkui.services.nlp_service._attribution_to_segments", return_value=[]),
    ):
        result = full_analysis(str(fake_epub), nlp_model="llama3.2")

    mock_pipeline.extract.assert_called_once()
    assert mock_pipeline._attribution.attribute_chapter.call_count == 2


def test_full_analysis_reuses_cached_roster_and_still_attributes(tmp_path):
    fake_epub = tmp_path / "book.epub"
    fake_epub.write_bytes(b"fake")

    fake_chapters = [Chapter(index=0, title="Ch 1", paragraphs=['"Hi," Jane said.'])]
    mock_reader = MagicMock()
    mock_reader.get_chapters.return_value = fake_chapters

    cached_roster = _make_mock_roster()
    cached_fast = MagicMock(roster=cached_roster)
    mock_pipeline = _make_mock_pipeline(roster=_make_mock_roster())

    with (
        patch("kenkui.services.nlp_service.get_reader", return_value=mock_reader),
        patch("kenkui.services.nlp_service.NLPPipeline", return_value=mock_pipeline),
        patch("kenkui.services.nlp_service.NLPConfig"),
        patch("kenkui.services.nlp_service.get_cached_result", return_value=None),
        patch("kenkui.services.nlp_service.get_cached_roster", return_value=cached_fast),
        patch("kenkui.services.nlp_service.cache_result"),
        patch("kenkui.services.nlp_service.book_hash", return_value="abc123"),
        patch("kenkui.services.nlp_service._attribution_to_segments", return_value=[]),
    ):
        full_analysis(str(fake_epub), nlp_model="llama3.2")

    mock_pipeline.extract.assert_not_called()
    mock_pipeline._attribution.attribute_chapter.assert_called_once()
    assert mock_pipeline._attribution.attribute_chapter.call_args.args[1] is cached_roster


def test_full_analysis_uses_config_nlp_model(tmp_path):
    """When nlp_model=None, full_analysis passes the config to NLPConfig.from_app_config."""
    fake_epub = tmp_path / "book.epub"
    fake_epub.write_bytes(b"fake")

    mock_reader = MagicMock()
    mock_reader.get_chapters.return_value = [Chapter(index=0, title="Ch", paragraphs=["t"])]

    mock_config = MagicMock()
    mock_config.nlp_model = "mistral"
    mock_config.nlp_provider = "ollama"
    mock_config.nlp_attribution_provider = ""

    mock_pipeline = _make_mock_pipeline()

    with (
        patch("kenkui.services.nlp_service.get_reader", return_value=mock_reader),
        patch("kenkui.services.nlp_service.NLPPipeline", return_value=mock_pipeline),
        patch("kenkui.services.nlp_service.NLPConfig") as mock_nlp_cfg_cls,
        patch("kenkui.services.nlp_service.get_cached_result", return_value=None),
        patch("kenkui.services.nlp_service.cache_result"),
        patch("kenkui.services.nlp_service.book_hash", return_value="abc123"),
        patch("kenkui.services.nlp_service._attribution_to_segments", return_value=[]),
        patch("kenkui.services.nlp_service.load_app_config", return_value=mock_config) as mock_cfg,
    ):
        full_analysis(str(fake_epub), nlp_model=None)

    mock_cfg.assert_called_once_with(None)
    mock_nlp_cfg_cls.from_app_config.assert_called_once_with(mock_config)


def test_full_analysis_progress_callback_receives_int_and_str(tmp_path):
    """Progress callback should receive (int, str) tuples at key milestones."""
    fake_epub = tmp_path / "book.epub"
    fake_epub.write_bytes(b"fake")

    mock_reader = MagicMock()
    mock_reader.get_chapters.return_value = [Chapter(index=0, title="Ch", paragraphs=["t"])]

    received: list[tuple[int, str]] = []

    def _cb(pct: int, msg: str) -> None:
        received.append((pct, msg))

    mock_pipeline = _make_mock_pipeline()

    with (
        patch("kenkui.services.nlp_service.get_reader", return_value=mock_reader),
        patch("kenkui.services.nlp_service.NLPPipeline", return_value=mock_pipeline),
        patch("kenkui.services.nlp_service.NLPConfig"),
        patch("kenkui.services.nlp_service.get_cached_result", return_value=None),
        patch("kenkui.services.nlp_service.cache_result"),
        patch("kenkui.services.nlp_service.book_hash", return_value="abc123"),
        patch("kenkui.services.nlp_service._attribution_to_segments", return_value=[]),
    ):
        full_analysis(str(fake_epub), nlp_model="llama3.2", extraction_progress_callback=_cb)

    percents = [p for p, _ in received]
    messages = [m for _, m in received]

    assert percents[0] == 0
    assert messages[0] == "Starting extraction"
    assert messages[-1] == "Extraction complete"
    assert percents[-1] == 100

    for p, _ in received:
        assert isinstance(p, int)
    for _, m in received:
        assert isinstance(m, str)
    for i in range(1, len(percents)):
        assert percents[i] >= percents[i - 1]


def test_full_analysis_attribution_progress_event_callback_uses_chapter_units(tmp_path):
    fake_epub = tmp_path / "book.epub"
    fake_epub.write_bytes(b"fake")

    fake_chapters = [
        Chapter(index=0, title="Ch 1", paragraphs=["text1"]),
        Chapter(index=1, title="Ch 2", paragraphs=["text2"]),
    ]
    mock_reader = MagicMock()
    mock_reader.get_chapters.return_value = fake_chapters

    events = []
    mock_pipeline = _make_mock_pipeline()

    with (
        patch("kenkui.services.nlp_service.get_reader", return_value=mock_reader),
        patch("kenkui.services.nlp_service.NLPPipeline", return_value=mock_pipeline),
        patch("kenkui.services.nlp_service.NLPConfig"),
        patch("kenkui.services.nlp_service.get_cached_result", return_value=None),
        patch("kenkui.services.nlp_service.get_cached_roster", return_value=None),
        patch("kenkui.services.nlp_service.cache_result"),
        patch("kenkui.services.nlp_service.book_hash", return_value="abc123"),
        patch("kenkui.services.nlp_service._attribution_to_segments", return_value=[]),
    ):
        full_analysis(
            str(fake_epub),
            nlp_model="llama3.2",
            attribution_progress_event_callback=events.append,
        )

    assert [event.status for event in events] == ["started", "advanced", "advanced", "completed"]
    assert [event.completed_units for event in events] == [0, 1, 2, 2]
    assert all(event.stage == "nlp_attribution" for event in events)
    assert all(event.total_units == 2 for event in events)


def test_attribute_only_progress_event_callback_uses_chapter_units(tmp_path):
    fake_epub = tmp_path / "book.epub"
    fake_epub.write_bytes(b"fake")

    fake_chapters = [
        Chapter(index=0, title="Ch 1", paragraphs=["text1"]),
        Chapter(index=1, title="Ch 2", paragraphs=["text2"]),
    ]
    events = []
    mock_pipeline = _make_mock_pipeline()

    with (
        patch("kenkui.services.nlp_service.NLPPipeline", return_value=mock_pipeline),
        patch("kenkui.services.nlp_service.NLPConfig"),
        patch("kenkui.services.nlp_service.cache_result"),
        patch("kenkui.services.nlp_service.book_hash", return_value="abc123"),
        patch("kenkui.services.nlp_service._attribution_to_segments", return_value=[]),
    ):
        attribute_only(
            roster=_make_mock_roster(),
            chapters=fake_chapters,
            ebook_path=str(fake_epub),
            nlp_model="llama3.2",
            progress_event_callback=events.append,
        )

    assert [event.status for event in events] == ["started", "advanced", "advanced", "completed"]
    assert [event.completed_units for event in events] == [0, 1, 2, 2]
    assert all(event.stage == "nlp_attribution" for event in events)
    assert all(event.unit == "chapters" for event in events)


@pytest.mark.skip(reason="Requires real spaCy / Ollama — integration-only")
def test_full_analysis_real_pipeline():
    """Placeholder: full pipeline test requiring spaCy + Ollama."""
    # Would test: real epub file → full NLP pipeline → NLPResult with characters and quotes
