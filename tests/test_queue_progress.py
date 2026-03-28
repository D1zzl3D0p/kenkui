"""Tests for phase-aware progress display in queue.py."""
import pytest
from unittest.mock import MagicMock


def _make_active_item(current_chapter: str, progress: float = 50.0):
    item = MagicMock()
    item.current_chapter = current_chapter
    item.progress = progress
    item.job = {"name": "Test Book"}
    item.id = "test-id"
    return item


def _make_queue_info(current_chapter: str, progress: float = 50.0):
    qi = MagicMock()
    qi.current_item = _make_active_item(current_chapter, progress)
    return qi


def test_attribution_phase_shows_phase_label():
    from kenkui.cli.queue import _build_progress_bar
    qi = _make_queue_info("[Attribution] chunk 3/12")
    prog = _build_progress_bar(qi)
    assert prog is not None
    # The task description should contain the phase indicator
    tasks = list(prog.tasks)
    assert len(tasks) == 1
    assert "Attribution" in tasks[0].description or "attribution" in tasks[0].description.lower()


def test_tts_phase_shows_chapter():
    from kenkui.cli.queue import _build_progress_bar
    qi = _make_queue_info("Chapter 5: The Great Hunt")
    prog = _build_progress_bar(qi)
    assert prog is not None
    tasks = list(prog.tasks)
    assert "Chapter 5" in tasks[0].description


def test_attribution_progress_preserved():
    from kenkui.cli.queue import _build_progress_bar
    qi = _make_queue_info("[Attribution] chunk 3/12", progress=8.0)
    prog = _build_progress_bar(qi)
    tasks = list(prog.tasks)
    assert tasks[0].completed == 8.0


def test_no_active_item_returns_none():
    from kenkui.cli.queue import _build_progress_bar
    qi = MagicMock()
    qi.current_item = None
    assert _build_progress_bar(qi) is None
