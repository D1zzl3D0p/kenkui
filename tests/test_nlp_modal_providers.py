"""Tests for ModalExtractionProvider and ModalAttributionProvider.

Modal is mocked via sys.modules so the real 'modal' package is not required.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from kenkui.nlp.providers.modal import ModalAttributionProvider, ModalExtractionProvider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_modal():
    """Inject a mock modal module so tests run without modal installed."""
    m = MagicMock()
    sys.modules["modal"] = m
    yield m
    del sys.modules["modal"]


@pytest.fixture()
def no_modal():
    """Ensure modal is NOT available for testing the not-installed path."""
    original = sys.modules.pop("modal", None)
    yield
    if original is not None:
        sys.modules["modal"] = original


# ---------------------------------------------------------------------------
# ModalExtractionProvider
# ---------------------------------------------------------------------------


def test_modal_extraction_raises_if_modal_not_installed(no_modal):
    adapter = MagicMock()
    provider = ModalExtractionProvider(adapter)
    with pytest.raises(NotImplementedError, match="pip install modal"):
        provider.build_roster(chapters=[])


def test_modal_extraction_delegates_to_adapter_when_modal_available(mock_modal):
    roster_sentinel = MagicMock()
    adapter = MagicMock()
    adapter.build_roster.return_value = roster_sentinel

    chapter = MagicMock()
    series_roster = MagicMock()
    callback = MagicMock()
    book_path = MagicMock()

    provider = ModalExtractionProvider(adapter)
    result = provider.build_roster(
        chapters=[chapter],
        series_roster=series_roster,
        progress_callback=callback,
        book_path=book_path,
    )

    adapter.build_roster.assert_called_once_with(
        [chapter],
        series_roster=series_roster,
        progress_callback=callback,
        step_callback=None,
        book_path=book_path,
    )
    assert result is roster_sentinel


# ---------------------------------------------------------------------------
# ModalAttributionProvider
# ---------------------------------------------------------------------------


def test_modal_attribution_raises_if_modal_not_installed(no_modal):
    adapter = MagicMock()
    provider = ModalAttributionProvider(adapter)
    with pytest.raises(NotImplementedError, match="pip install modal"):
        provider.attribute_chapter(chapter=MagicMock(), roster=MagicMock())


def test_modal_attribution_delegates_to_adapter_when_modal_available(mock_modal):
    attribution_sentinel = MagicMock()
    adapter = MagicMock()
    adapter.attribute_chapter.return_value = attribution_sentinel

    chapter = MagicMock()
    roster = MagicMock()
    callback = MagicMock()

    provider = ModalAttributionProvider(adapter)
    result = provider.attribute_chapter(
        chapter=chapter,
        roster=roster,
        progress_callback=callback,
    )

    adapter.attribute_chapter.assert_called_once_with(
        chapter,
        roster,
        progress_callback=callback,
    )
    assert result is attribution_sentinel
