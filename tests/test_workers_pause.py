"""Tests for _pause_for_segment — logarithmic inter-segment silence helper."""

from __future__ import annotations

import math

import pytest
from pydub import AudioSegment


def _pause(pause_line_ms: int, text: str) -> AudioSegment:
    from kenkui.workers import _pause_for_segment
    return _pause_for_segment(pause_line_ms, text)


class TestPauseForSegment:
    BASE = 1000  # pause_line_ms

    def test_no_breaks_is_half_base(self):
        """Text with no \\n\\n → pause_line_ms // 2."""
        seg = _pause(self.BASE, "Just a sentence.")
        assert seg.duration_seconds == pytest.approx(0.5, abs=0.05)

    def test_one_break_equals_base(self):
        """Text with one \\n\\n → exactly pause_line_ms."""
        seg = _pause(self.BASE, "Para one.\n\nPara two.")
        assert seg.duration_seconds == pytest.approx(1.0, abs=0.05)

    def test_two_breaks_approx_1_5s(self):
        """Text with two \\n\\n → ≈ 1.5 × pause_line_ms (user spec)."""
        text = "A\n\nB\n\nC"
        seg = _pause(self.BASE, text)
        # 1 + 0.7 * ln(2) ≈ 1.485
        expected = self.BASE * (1 + 0.7 * math.log(2))
        assert seg.duration_seconds == pytest.approx(expected / 1000, abs=0.1)

    def test_three_breaks_approx_1_8s(self):
        """Text with three \\n\\n → ≈ 1.77 × pause_line_ms."""
        text = "A\n\nB\n\nC\n\nD"
        seg = _pause(self.BASE, text)
        expected = self.BASE * (1 + 0.7 * math.log(3))
        assert seg.duration_seconds == pytest.approx(expected / 1000, abs=0.1)

    def test_monotonically_increasing(self):
        """More breaks → longer pause."""
        texts = ["x", "a\n\nb", "a\n\nb\n\nc", "a\n\nb\n\nc\n\nd"]
        durations = [_pause(self.BASE, t).duration_seconds for t in texts]
        for i in range(len(durations) - 1):
            assert durations[i] < durations[i + 1]

    def test_returns_audio_segment(self):
        seg = _pause(500, "text")
        assert isinstance(seg, AudioSegment)

    def test_different_base_scales_proportionally(self):
        """Halving base halves the resulting pause."""
        seg_1000 = _pause(1000, "A\n\nB")
        seg_500 = _pause(500, "A\n\nB")
        assert seg_500.duration_seconds == pytest.approx(
            seg_1000.duration_seconds / 2, abs=0.02
        )
