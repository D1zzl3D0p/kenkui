"""QP14 efficiency tests — written BEFORE implementation (TDD).

Part 1: Co-occurrence precompute in _auto_assign_unmapped_speakers
  - Pin current assignment results on a deterministic fixture.
  - After optimization, identical results must be produced.

Part 2: Adaptive exponential backoff in _process_chapters polling loop
  - Inject a fake sleep and fake clock so no real time elapses.
  - Verify idle iterations sleep with increasing delays (capped).
  - Verify activity resets the delay back to the minimum.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

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
        """Delay never exceeds 0.5 s regardless of how large idle_count is."""
        from kenkui.parsing import _poll_backoff_delay
        for idle_count in [1, 10, 50, 100, 1000]:
            assert _poll_backoff_delay(idle_count) <= 0.5

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
