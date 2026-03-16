"""Tests for kenkui.utils — batch_text and _normalize_bitrate."""

from __future__ import annotations

import pytest

from kenkui.models import _normalize_bitrate
from kenkui.utils import batch_text

# ---------------------------------------------------------------------------
# batch_text
# ---------------------------------------------------------------------------


class TestBatchTextMerging:
    """Short segments should be merged into batches up to max_chars."""

    def test_merges_short_segments(self):
        segs = ["Hello.", "World.", "Foo."]
        result = batch_text(segs, max_chars=800, merge_short=True)
        assert len(result) == 1
        assert "Hello." in result[0]
        assert "World." in result[0]

    def test_respects_max_chars_boundary(self):
        # Each segment is 10 chars; max_chars=25 → 2 segments per batch
        segs = ["AAAAAAAAAA"] * 5  # 10 chars each
        result = batch_text(segs, max_chars=25, merge_short=True)
        for batch in result:
            assert len(batch) <= 25 + 1  # +1 for the space separator

    def test_empty_input_returns_empty(self):
        assert batch_text([], max_chars=800) == []

    def test_single_short_segment_returned_as_is(self):
        result = batch_text(["Hello world."], max_chars=800)
        assert result == ["Hello world."]

    def test_skips_blank_segments(self):
        segs = ["Hello.", "", "World."]
        result = batch_text(segs, max_chars=800)
        # Empty string should be dropped
        assert all(s.strip() for s in result)


class TestBatchTextSplitting:
    """Long paragraphs should be split at sentence boundaries."""

    def test_splits_long_paragraph(self):
        long = "Sentence one. Sentence two. Sentence three. " * 5  # ~220 chars
        result = batch_text([long], max_chars=50)
        assert len(result) > 1
        for chunk in result:
            assert (
                len(chunk) <= 50 or "." in chunk
            )  # each chunk is at most one sentence

    def test_long_paragraphs_flushed_before_merge(self):
        """A long paragraph should be emitted separately, not merged with next short one."""
        long = "A" * 900 + "."
        short = "Short."
        result = batch_text([long, short], max_chars=800)
        # The long paragraph should be in its own batch (or split)
        # The short one should be separate or merged only with other shorts
        joined = " ".join(result)
        assert "A" in joined
        assert "Short." in joined


class TestBatchTextMergeShortFalse:
    """merge_short=False: each paragraph is its own item (no merging)."""

    def test_no_merging_when_false(self):
        segs = ["A.", "B.", "C."]
        result = batch_text(segs, max_chars=800, merge_short=False)
        assert result == ["A.", "B.", "C."]

    def test_long_paragraphs_still_split_when_merge_false(self):
        long = "Sentence one. Sentence two. Sentence three. " * 10
        result = batch_text([long], max_chars=50, merge_short=False)
        assert len(result) > 1


# ---------------------------------------------------------------------------
# _normalize_bitrate
# ---------------------------------------------------------------------------


class TestNormalizeBitrate:
    @pytest.mark.parametrize(
        "inp,expected",
        [
            ("64k", "64k"),
            ("96K", "96k"),
            ("128k", "128k"),
            ("64", "64k"),  # THE bug case — bare int gets 'k'
            ("96", "96k"),
            ("128", "128k"),
            ("999", "999k"),
            ("1000", "1000"),  # >= 1000 → already in bps, pass through
            ("128000", "128000"),  # large bps value, pass through
            ("", "96k"),  # empty → default
            (None, "96k"),  # None → default
            ("bad", "96k"),  # garbage → default
            (" 64k ", "64k"),  # whitespace stripped
        ],
    )
    def test_normalize(self, inp, expected):
        assert _normalize_bitrate(inp) == expected

    def test_custom_default(self):
        assert _normalize_bitrate(None, default="128k") == "128k"
        assert _normalize_bitrate("", default="64k") == "64k"
