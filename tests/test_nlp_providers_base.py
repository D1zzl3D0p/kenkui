"""Tests for provider protocol definitions and local execution wrappers."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from kenkui.nlp.providers._base import AttributionProvider, ExtractionProvider
from kenkui.nlp.providers.local import LocalAttributionProvider, LocalExtractionProvider


def _make_extraction_mock():
    m = MagicMock()
    m.build_roster = MagicMock(return_value=MagicMock())
    return m


def _make_attribution_mock():
    m = MagicMock()
    m.attribute_chapter = MagicMock(return_value=MagicMock())
    return m


class TestExtractionProviderProtocol:
    def test_mock_satisfies_protocol(self):
        mock = _make_extraction_mock()
        assert isinstance(mock, ExtractionProvider)

    def test_object_without_build_roster_fails(self):
        class NoMethod:
            pass
        assert not isinstance(NoMethod(), ExtractionProvider)


class TestAttributionProviderProtocol:
    def test_mock_satisfies_protocol(self):
        mock = _make_attribution_mock()
        assert isinstance(mock, AttributionProvider)

    def test_object_without_attribute_chapter_fails(self):
        class NoMethod:
            pass
        assert not isinstance(NoMethod(), AttributionProvider)


class TestLocalExtractionProvider:
    def test_delegates_build_roster_to_adapter(self):
        adapter = _make_extraction_mock()
        chapters = [MagicMock()]
        series_roster = MagicMock()
        progress_cb = MagicMock()

        provider = LocalExtractionProvider(adapter)
        provider.build_roster(chapters, series_roster=series_roster, progress_callback=progress_cb)

        adapter.build_roster.assert_called_once_with(
            chapters,
            series_roster=series_roster,
            progress_callback=progress_cb,
            step_callback=None,
            book_path=None,
        )

    def test_passes_book_path(self):
        adapter = _make_extraction_mock()
        provider = LocalExtractionProvider(adapter)
        path = Path("/tmp/book.epub")
        provider.build_roster([], book_path=path)
        adapter.build_roster.assert_called_once_with([], series_roster=None, progress_callback=None, step_callback=None, book_path=path)

    def test_returns_adapter_result(self):
        adapter = _make_extraction_mock()
        expected = MagicMock()
        adapter.build_roster.return_value = expected
        provider = LocalExtractionProvider(adapter)
        result = provider.build_roster([])
        assert result is expected

    def test_satisfies_extraction_provider_protocol(self):
        adapter = _make_extraction_mock()
        provider = LocalExtractionProvider(adapter)
        assert isinstance(provider, ExtractionProvider)


class TestLocalAttributionProvider:
    def test_delegates_attribute_chapter_to_adapter(self):
        adapter = _make_attribution_mock()
        chapter = MagicMock()
        roster = MagicMock()
        progress_cb = MagicMock()

        provider = LocalAttributionProvider(adapter)
        provider.attribute_chapter(chapter, roster, progress_callback=progress_cb)

        adapter.attribute_chapter.assert_called_once_with(chapter, roster, progress_callback=progress_cb)

    def test_returns_adapter_result(self):
        adapter = _make_attribution_mock()
        expected = MagicMock()
        adapter.attribute_chapter.return_value = expected
        provider = LocalAttributionProvider(adapter)
        result = provider.attribute_chapter(MagicMock(), MagicMock())
        assert result is expected

    def test_satisfies_attribution_provider_protocol(self):
        adapter = _make_attribution_mock()
        provider = LocalAttributionProvider(adapter)
        assert isinstance(provider, AttributionProvider)
