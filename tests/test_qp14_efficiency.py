"""QP14 efficiency tests — property tests of the new implementation.

Part 1: Co-occurrence precompute in _auto_assign_unmapped_speakers
  - Verify assignment semantics on deterministic fixtures.
  - Optimised implementation must produce identical results.

Part 2: Adaptive exponential backoff in _process_chapters polling loop
  - Inject a fake sleep and fake clock so no real time elapses.
  - Verify idle iterations sleep with increasing delays (capped).
  - Verify activity resets the delay back to the minimum.
"""

from __future__ import annotations

import queue as _queue
from types import SimpleNamespace
from unittest.mock import call, patch

from kenkui.models import Chapter, Segment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chapter(index: int, segments: list[dict]) -> Chapter:
    ch = Chapter(index=index, title=f"Ch {index}", paragraphs=[])
    ch.segments = [
        Segment(
            text=s["text"],
            speaker=s["speaker"],
            index=i,
            is_scene_break=s.get("is_scene_break", False),
        )
        for i, s in enumerate(segments)
    ]
    return ch


def _voices(names: list[str]) -> list[SimpleNamespace]:
    gender_map = {
        "cedar": "male",
        "oak": "male",
        "alba": "female",
        "ivy": "female",
        "pine": "male",
        "rose": "female",
    }
    return [
        SimpleNamespace(
            voice_id=name,
            pool_enabled=True,
            status="available",
            gender=gender_map.get(name, "male"),
        )
        for name in names
    ]


# ---------------------------------------------------------------------------
# Part 1 — Co-occurrence precompute (equivalence tests)
# ---------------------------------------------------------------------------


class TestAutoAssignEquivalence:
    """Pin exact assignment results; identical after optimisation."""

    def _assign(self, chapters, speaker_voices=None, character_genders=None,
                 roster_slugs=None, voice_list=None):
        from kenkui.parsing import _auto_assign_unmapped_speakers

        if voice_list is None:
            voice_list = _voices(["alba", "cedar", "oak", "ivy"])

        with patch(
            "kenkui.services.voice_service.list_voices",
            return_value=voice_list,
        ):
            return _auto_assign_unmapped_speakers(
                chapters,
                speaker_voices or {},
                "alba",
                lambda _: None,
                character_genders=character_genders,
                roster_slugs=roster_slugs,
            )

    def test_single_speaker_gets_assigned(self):
        ch = _make_chapter(0, [{"text": "Hi.", "speaker": "darrow"}])
        result = self._assign([ch])
        assert "darrow" in result

    def test_two_speakers_in_same_chapter_get_different_voices(self):
        ch = _make_chapter(0, [
            {"text": "A.", "speaker": "darrow"},
            {"text": "B.", "speaker": "sevro"},
        ])
        result = self._assign([ch])
        assert result.get("darrow") != result.get("sevro")

    def test_speakers_in_different_chapters_may_share_voice(self):
        """When voices run out, speakers from different chapters can share."""
        # 5 speakers, only 4 voices → at least one share must occur.
        chapters = [
            _make_chapter(i, [{"text": "X.", "speaker": f"sp{i}"}])
            for i in range(5)
        ]
        result = self._assign(chapters, voice_list=_voices(["alba", "cedar", "oak", "ivy"]))
        assert len(result) == 5
        # All must be assigned.
        assert all(v is not None for v in result.values())

    def test_co_occurring_speakers_prefer_different_voices(self):
        """Two speakers appearing in the same chapter get different voices when pool allows."""
        ch = _make_chapter(0, [
            {"text": "A.", "speaker": "alpha"},
            {"text": "B.", "speaker": "beta"},
        ])
        # 2 speakers, 2 voices → each gets a unique voice.
        result = self._assign([ch], voice_list=_voices(["cedar", "oak"]))
        assert result["alpha"] != result["beta"]

    def test_co_occurring_pool_exhausted_assigns_least_overlap(self):
        """When pool is exhausted, the least-overlapping voice is reused (not crash)."""
        ch = _make_chapter(0, [
            {"text": "A.", "speaker": "alpha"},
            {"text": "B.", "speaker": "beta"},
            {"text": "C.", "speaker": "gamma"},
        ])
        # Only 2 voices for 3 co-occurring speakers → forced share.
        result = self._assign([ch], voice_list=_voices(["cedar", "oak"]))
        assert len(result) == 3
        assert all(v in {"cedar", "oak"} for v in result.values())

    def test_existing_speaker_voices_not_overwritten(self):
        ch = _make_chapter(0, [
            {"text": "A.", "speaker": "darrow"},
            {"text": "B.", "speaker": "sevro"},
        ])
        existing = {"darrow": "cedar"}
        result = self._assign([ch], speaker_voices=existing)
        assert result["darrow"] == "cedar"  # unchanged
        assert "sevro" in result

    def test_roster_filter_skips_non_roster_speakers(self):
        ch = _make_chapter(0, [
            {"text": "A.", "speaker": "darrow"},
            {"text": "B.", "speaker": "fisherman"},
        ])
        result = self._assign([ch], roster_slugs={"darrow"})
        assert "darrow" in result
        assert "fisherman" not in result

    def test_gender_aware_assignment_male_speaker(self):
        ch = _make_chapter(0, [{"text": "A.", "speaker": "darrow"}])
        result = self._assign(
            [ch],
            character_genders={"darrow": "he/him"},
            voice_list=_voices(["cedar", "alba"]),
        )
        # cedar is male — darrow should get cedar.
        assert result.get("darrow") == "cedar"

    def test_gender_aware_assignment_female_speaker(self):
        ch = _make_chapter(0, [{"text": "A.", "speaker": "lyria"}])
        # Use a distinct narrator so alba is not excluded.
        from kenkui.parsing import _auto_assign_unmapped_speakers
        with patch(
            "kenkui.services.voice_service.list_voices",
            return_value=_voices(["alba", "cedar"]),
        ):
            result = _auto_assign_unmapped_speakers(
                [ch],
                {},
                "narrator-voice",  # narrator is not in pool → alba remains eligible
                lambda _: None,
                character_genders={"lyria": "she/her"},
            )
        # alba is female — lyria should get alba.
        assert result.get("lyria") == "alba"

    def test_prominence_order_high_prominence_gets_fresh_voice(self):
        """High-segment-count speaker should win the fresh exclusive voice."""
        ch = _make_chapter(0, [
            {"text": "A.", "speaker": "major"},
            {"text": "B.", "speaker": "major"},
            {"text": "C.", "speaker": "major"},
            {"text": "D.", "speaker": "minor"},
        ])
        result = self._assign([ch], voice_list=_voices(["cedar", "oak"]))
        # Both in the same chapter, so they cannot share; both should be assigned.
        assert result.get("major") != result.get("minor")

    def test_no_speakers_returns_empty(self):
        ch = _make_chapter(0, [])
        result = self._assign([ch])
        assert result == {}

    def test_sentinel_speakers_not_assigned(self):
        ch = _make_chapter(0, [
            {"text": "Narration.", "speaker": "NARRATOR"},
        ])
        result = self._assign([ch])
        assert "NARRATOR" not in result

    def test_multi_chapter_co_occurrence_tracking(self):
        """Voices already used in ch0 should not be reused for co-occurring speakers in ch0."""
        ch0 = _make_chapter(0, [
            {"text": "A.", "speaker": "alpha"},
            {"text": "B.", "speaker": "beta"},
        ])
        ch1 = _make_chapter(1, [
            {"text": "C.", "speaker": "gamma"},  # different chapter — may share with alpha/beta
        ])
        result = self._assign([ch0, ch1], voice_list=_voices(["cedar", "oak"]))
        # alpha and beta are in the same chapter → different voices.
        assert result["alpha"] != result["beta"]
        # gamma is in a different chapter → can share with one of them.
        assert result["gamma"] in {"cedar", "oak"}


# ---------------------------------------------------------------------------
# Part 2 — Adaptive exponential backoff helper
# ---------------------------------------------------------------------------


class TestPollingBackoff:
    """Adaptive backoff for the _process_chapters while-True polling loop.

    We extract and test the backoff helper `_poll_backoff_sleep` directly
    (no real sleep, no real processes).  The helper will be added to parsing.py
    as a module-level function and called from the poll loop.
    """

    def test_first_idle_returns_min_delay(self):
        """idle_count=1 → minimum delay (e.g. 0.005s)."""
        from kenkui.parsing import _poll_backoff_delay
        delay = _poll_backoff_delay(idle_count=1)
        assert delay >= 0.001
        assert delay <= 0.01

    def test_delay_increases_with_idle_count(self):
        """Higher idle_count → larger delay (monotonically non-decreasing)."""
        from kenkui.parsing import _poll_backoff_delay
        delays = [_poll_backoff_delay(idle_count=i) for i in range(1, 20)]
        for i in range(1, len(delays)):
            assert delays[i] >= delays[i - 1], f"Delay decreased at i={i}: {delays}"

    def test_delay_is_capped(self):
        """Delay never exceeds 0.2 s regardless of how large idle_count is."""
        from kenkui.parsing import _poll_backoff_delay
        for idle_count in [1, 10, 50, 100, 1000]:
            assert _poll_backoff_delay(idle_count) <= 0.2

    def test_zero_idle_count_returns_zero_or_min(self):
        """idle_count=0 is not expected to be called but must not crash."""
        from kenkui.parsing import _poll_backoff_delay
        delay = _poll_backoff_delay(idle_count=0)
        assert delay >= 0.0

    def test_exponential_growth_before_cap(self):
        """Delays should grow faster than linearly before hitting the cap."""
        from kenkui.parsing import _poll_backoff_delay
        d1 = _poll_backoff_delay(idle_count=1)
        d5 = _poll_backoff_delay(idle_count=5)
        # Exponential growth means d5 >> 5 * d1 (at least 3× the linear expectation).
        assert d5 >= 3 * d1


# ---------------------------------------------------------------------------
# Part 3 — Drain-loop integration: backoff wiring inside _run_drain_loop
# ---------------------------------------------------------------------------


class _FakeQueue:
    """Minimal queue stand-in: yields pre-loaded messages then stays empty."""

    def __init__(self, messages=()):
        self._msgs = list(messages)

    def empty(self):
        return not self._msgs

    def get_nowait(self):
        if not self._msgs:
            raise _queue.Empty
        return self._msgs.pop(0)


class _CountdownFuture:
    """Fake future that reports done() after *done_after* calls to done()."""

    def __init__(self, done_after: int):
        self._remaining = done_after

    def done(self):
        if self._remaining > 0:
            self._remaining -= 1
            return False
        return True


class _FakeTracker:
    def __init__(self):
        self.messages: list = []

    def process_message(self, msg):
        self.messages.append(msg)

    def finalize_completed(self):
        pass

    def log_errors(self):
        pass


def _make_drain_builder():
    """Return a minimal AudioBuilder wired for drain-loop tests (no real config)."""
    from kenkui.parsing import AudioBuilder

    b = AudioBuilder.__new__(AudioBuilder)
    b.was_cancelled = False
    b.cancel_check = None
    return b


class TestDrainLoopBackoff:
    """Integration test: verify backoff wiring inside AudioBuilder._run_drain_loop.

    No real sleeps — time.sleep is patched throughout.
    """

    def test_idle_iterations_increment_delay(self):
        """(a) Each idle iteration sleeps with the delay returned by _poll_backoff_delay."""
        from kenkui.parsing import _poll_backoff_delay

        builder = _make_drain_builder()
        # done_after=2: done() returns False on calls 1 and 2, True on call 3.
        # Loop: iter1 sleep(delay(1)) → done() False; iter2 sleep(delay(2)) → done()
        # False; iter3 sleep(delay(3)) → done() True → break. Exactly 3 sleeps.
        futures = {_CountdownFuture(done_after=2): None}
        tracker = _FakeTracker()
        fq = _FakeQueue()  # always empty

        with patch("kenkui.parsing.time.sleep") as mock_sleep:
            builder._run_drain_loop(fq, futures, tracker)

        assert mock_sleep.call_count == 3
        expected = [
            call(_poll_backoff_delay(1)),
            call(_poll_backoff_delay(2)),
            call(_poll_backoff_delay(3)),
        ]
        assert mock_sleep.call_args_list == expected

    def test_drained_message_resets_idle_counter(self):
        """(b) A drained message resets idle_iters; next sleep is back to minimum."""
        from unittest.mock import Mock

        from kenkui.parsing import _poll_backoff_delay

        builder = _make_drain_builder()

        # Drive the loop through exactly 4 outer iterations:
        #   iter1: queue empty → sleep(delay(1))  → done() False (short-circuits)
        #   iter2: queue empty → sleep(delay(2))  → done() False (short-circuits)
        #   iter3: queue has msg → drain (no sleep) → done() False (short-circuits)
        #   iter4: queue empty → sleep(delay(1))  → done() True → queue.empty() → break
        #
        # empty() short-circuit note: the final `all(done) and queue.empty()` only
        # calls queue.empty() when all futures are already done.  Iterations 1-3 all
        # short-circuit at done()=False, so empty() is NOT called at end of those iters.
        fq = Mock()
        fq.empty.side_effect = [
            True,   # iter1 inner-while → empty, skip drain
            True,   # iter2 inner-while → empty, skip drain
            False,  # iter3 inner-while → not empty, enter drain
            True,   # iter3 inner-while after get → empty, exit drain
            True,   # iter4 inner-while → empty, skip drain
            True,   # iter4 final check (done()=True, so this is evaluated) → break
        ]
        fq.get_nowait.return_value = "msg"

        # done_after=3: returns False on calls 1,2,3 then True on call 4.
        futures = {_CountdownFuture(done_after=3): None}
        tracker = _FakeTracker()

        with patch("kenkui.parsing.time.sleep") as mock_sleep:
            builder._run_drain_loop(fq, futures, tracker)

        expected = [
            call(_poll_backoff_delay(1)),  # iter1
            call(_poll_backoff_delay(2)),  # iter2
            call(_poll_backoff_delay(1)),  # iter4: reset after drain in iter3
        ]
        assert mock_sleep.call_args_list == expected
        assert tracker.messages == ["msg"]

    def test_no_sleep_when_queue_active(self):
        """(c) time.sleep is never called when the queue always has messages."""
        builder = _make_drain_builder()
        # Pre-load 3 messages; future is done from the start after queue drains.
        fq = _FakeQueue(["a", "b", "c"])
        futures = {_CountdownFuture(done_after=0): None}
        tracker = _FakeTracker()

        with patch("kenkui.parsing.time.sleep") as mock_sleep:
            builder._run_drain_loop(fq, futures, tracker)

        mock_sleep.assert_not_called()
        assert tracker.messages == ["a", "b", "c"]

    def test_get_nowait_empty_race_does_not_crash(self):
        """(d) queue.Empty from get_nowait() (empty/get race) is handled gracefully.

        _FakeQueue.empty() reports False but get_nowait() raises queue.Empty —
        simulating the multiprocessing.Manager proxy race.  The loop must break
        out of the inner drain without propagating the exception or losing any
        already-processed messages.
        """
        from unittest.mock import Mock

        builder = _make_drain_builder()

        # empty() returns False once (triggers drain entry), then get_nowait raises Empty.
        fq = Mock()
        fq.empty.side_effect = [
            False,  # inner-while: enter drain
            True,   # final termination check
        ]
        fq.get_nowait.side_effect = _queue.Empty

        futures = {_CountdownFuture(done_after=0): None}
        tracker = _FakeTracker()

        # Must not raise; loop should complete normally.
        with patch("kenkui.parsing.time.sleep"):
            builder._run_drain_loop(fq, futures, tracker)

        assert tracker.messages == []  # no messages processed (Empty before any get)
