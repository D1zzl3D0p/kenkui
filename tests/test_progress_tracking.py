"""Focused unit tests for ChapterProgressTracker.

These pin the progress-callback sequence and error/log collection semantics
that previously lived inline (and untestable) inside
``AudioBuilder._process_chapters``.
"""

from __future__ import annotations

import logging


def _make_tracker(total_chars=100):
    from kenkui.progress_tracking import ChapterProgressTracker

    events: list[dict] = []

    def emit(stage, status, message, **kwargs):
        events.append({"stage": stage, "status": status, "message": message, **kwargs})

    return ChapterProgressTracker(emit, total_chars), events


def test_start_message_emits_message_event_and_records_state():
    tracker, events = _make_tracker()
    # ("START", pid, title, total, total_chars, is_first, index)
    tracker.process_message(("START", 7, "Chapter 1", 3, 50, True, 0))

    assert tracker.current_chapter == "Chapter 1"
    assert 7 in tracker.worker_state
    assert tracker.worker_state[7]["status"] == "started"
    assert events == [
        {
            "stage": "tts_synthesis",
            "status": "message",
            "message": "Chapter 1",
            "completed_units": 0,
            "total_units": 100,
            "unit": "chars",
            "active_chapters": tracker.active_chapter_progress(),
        }
    ]


def test_update_accumulates_batches_and_chars_and_emits_advanced():
    tracker, events = _make_tracker()
    tracker.process_message(("START", 7, "Chapter 1", 3, 50, True, 0))
    events.clear()
    # ("UPDATE", pid, batch_delta, ?, ?, chars)
    tracker.process_message(("UPDATE", 7, 2, 1, 1, 40))

    assert tracker.completed_batches == 2
    assert tracker.completed_tts_units == 40
    assert tracker.worker_state[7]["current"] == 2
    assert tracker.worker_state[7]["status"] == "advanced"
    assert events[-1]["status"] == "advanced"
    assert events[-1]["completed_units"] == 40


def test_update_clamps_completed_units_to_total_chars():
    tracker, _ = _make_tracker(total_chars=10)
    tracker.process_message(("START", 7, "C", 3, 50, True, 0))
    tracker.process_message(("UPDATE", 7, 1, 1, 1, 999))
    assert tracker.completed_tts_units == 10


def test_update_ignores_negative_chars():
    tracker, _ = _make_tracker()
    tracker.process_message(("START", 7, "C", 3, 50, True, 0))
    tracker.process_message(("UPDATE", 7, 1, 1, 1, -5))
    assert tracker.completed_tts_units == 0


def test_done_marks_completed_emits_then_removes_state():
    tracker, events = _make_tracker()
    tracker.process_message(("START", 7, "Chapter 1", 3, 50, True, 0))
    events.clear()
    tracker.process_message(("DONE", 7))

    # The completed chapter is reflected in the emit that fires *before* removal.
    assert events[-1]["status"] == "advanced"
    completed = events[-1]["active_chapters"]
    assert completed[0].status == "completed"
    assert completed[0].completed_units == completed[0].total_units == 3
    # ...and then state is dropped.
    assert 7 not in tracker.worker_state


def test_error_collects_error_and_emits_failed():
    tracker, events = _make_tracker()
    tracker.process_message(("START", 7, "Chapter 1", 3, 50, True, 0))
    events.clear()
    tracker.process_message(("ERROR", 7, "Chapter 1", "boom", "tb-here"))

    assert events[-1]["status"] == "failed"
    assert events[-1]["message"] == "boom"
    assert tracker.worker_state[7]["status"] == "failed"
    assert tracker.worker_errors == [
        {"pid": 7, "chapter": "Chapter 1", "message": "boom", "traceback": "tb-here"}
    ]


def test_log_ring_buffer_caps_at_twenty():
    tracker, _ = _make_tracker()
    for i in range(25):
        tracker.process_message(("LOG", 7, f"line {i}"))
    assert len(tracker.worker_logs) == 20
    assert tracker.worker_logs[0] == "[7] line 5"
    assert tracker.worker_logs[-1] == "[7] line 24"


def test_active_chapter_progress_sorted_by_index():
    tracker, _ = _make_tracker()
    tracker.process_message(("START", 1, "Two", 3, 50, False, 2))
    tracker.process_message(("START", 2, "One", 3, 50, True, 0))
    titles = [cp.title for cp in tracker.active_chapter_progress()]
    assert titles == ["One", "Two"]


def test_finalize_completed_marks_remaining_and_clears():
    tracker, events = _make_tracker()
    tracker.process_message(("START", 7, "Chapter 1", 3, 50, True, 0))
    events.clear()
    tracker.finalize_completed()

    assert tracker.worker_state == {}
    assert events[-1]["active_chapters"][0].status == "completed"


def test_malformed_message_raises_indexerror_for_caller_to_handle():
    tracker, _ = _make_tracker()
    # UPDATE for an unknown pid with a too-short tuple: msg[2] missing.
    try:
        tracker.process_message(("UPDATE",))
    except (IndexError, KeyError, ValueError, TypeError):
        pass
    else:  # pragma: no cover
        raise AssertionError("expected a parse error to propagate")


def test_log_errors_emits_logging(caplog):
    tracker, _ = _make_tracker()
    tracker.process_message(("ERROR", 7, "Chapter 1", "boom", "tb-here"))
    with caplog.at_level(logging.ERROR):
        tracker.log_errors()
    joined = "\n".join(r.message for r in caplog.records)
    assert "Worker errors encountered" in joined
    assert "boom" in joined
    assert "tb-here" in joined
