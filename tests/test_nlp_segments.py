"""Tests for _split_paragraph_by_quotes and _merge_consecutive_segments.

These functions live in kenkui.nlp and are the core of per-quotation voice
assignment.  Tests are written before the implementation (TDD).
"""

from __future__ import annotations

import pytest

from kenkui.models import Segment


# ---------------------------------------------------------------------------
# Import helpers under test (will fail until implemented)
# ---------------------------------------------------------------------------


def _import_helpers():
    from kenkui.nlp import _merge_consecutive_segments, _split_paragraph_by_quotes
    return _split_paragraph_by_quotes, _merge_consecutive_segments


# ---------------------------------------------------------------------------
# _split_paragraph_by_quotes
# ---------------------------------------------------------------------------


class TestSplitParagraphByQuotes:
    """Tests for _split_paragraph_by_quotes(para, para_quotes) -> list[tuple[str,str]]"""

    def _make_quote(self, qid, text, para_index=0):
        """Return a (Quote, AttributionItem)-like pair as simple objects."""
        from kenkui.nlp.models import AttributionItem, Quote
        q = Quote(id=qid, text=text, para_index=para_index, char_offset=0)
        a = AttributionItem(quote_id=qid, speaker="Alice", emotion="neutral")
        return q, a

    def _split(self, para, para_quotes):
        fn, _ = _import_helpers()
        return fn(para, para_quotes)

    def test_narrator_only_paragraph_no_quotes(self):
        """A paragraph with no quotes should return a single NARRATOR span."""
        result = self._split("She walked down the hall.", [])
        assert result == [("She walked down the hall.", "NARRATOR")]

    def test_single_attributed_quote(self):
        """A quoted span with attribution → character span; surrounding text → NARRATOR."""
        para = '"Hello," she said.'
        q, a = self._make_quote(0, '"Hello,"')
        result = self._split(para, [(q, a)])
        # Should have Alice for the quote, NARRATOR for " she said."
        speakers = [s for _, s in result]
        assert "Alice" in speakers
        assert "NARRATOR" in speakers
        # Alice span must contain the quote text
        alice_texts = [t for t, s in result if s == "Alice"]
        assert any("Hello" in t for t in alice_texts)

    def test_narrator_before_and_after_quote(self):
        """Narrator text before the quote and after should both be NARRATOR spans."""
        para = 'She asked, "How are you?" and waited.'
        q, a = self._make_quote(0, '"How are you?"')
        result = self._split(para, [(q, a)])
        speakers = [s for _, s in result]
        assert speakers.count("NARRATOR") >= 1
        assert "Alice" in speakers

    def test_two_quotes_different_speakers(self):
        """Two attributed quotes with different speakers produce separate character spans."""
        from kenkui.nlp.models import AttributionItem, Quote
        para = '"Hi," said Alice. "Hello," said Bob.'
        q1 = Quote(id=0, text='"Hi,"', para_index=0, char_offset=0)
        a1 = AttributionItem(quote_id=0, speaker="Alice", emotion="neutral")
        q2 = Quote(id=1, text='"Hello,"', para_index=0, char_offset=0)
        a2 = AttributionItem(quote_id=1, speaker="Bob", emotion="neutral")
        fn, _ = _import_helpers()
        result = fn(para, [(q1, a1), (q2, a2)])
        speakers = [s for _, s in result]
        assert "Alice" in speakers
        assert "Bob" in speakers

    def test_unmatched_quote_falls_back_to_narrator(self):
        """A quote in the paragraph that has no attribution should be NARRATOR."""
        from kenkui.nlp.models import AttributionItem, Quote
        para = '"Mystery line." She stared.'
        # No quotes in para_quotes → everything NARRATOR
        fn, _ = _import_helpers()
        result = fn(para, [])
        # With no attributions, should return the whole paragraph as NARRATOR
        speakers = [s for _, s in result]
        assert all(s == "NARRATOR" for s in speakers)

    def test_empty_paragraph_returns_narrator(self):
        fn, _ = _import_helpers()
        result = fn("", [])
        assert result == [("", "NARRATOR")]

    def test_full_coverage_no_whitespace_lost(self):
        """Concatenating all spans should reconstruct the original paragraph."""
        from kenkui.nlp.models import AttributionItem, Quote
        para = 'He said, "Let\'s go," and stood up.'
        q = Quote(id=0, text='"Let\'s go,"', para_index=0, char_offset=0)
        a = AttributionItem(quote_id=0, speaker="Bob", emotion="neutral")
        fn, _ = _import_helpers()
        result = fn(para, [(q, a)])
        reconstructed = "".join(t for t, _ in result)
        assert reconstructed == para


# ---------------------------------------------------------------------------
# _merge_consecutive_segments
# ---------------------------------------------------------------------------


class TestMergeConsecutiveSegments:
    """Tests for _merge_consecutive_segments(segments) -> list[Segment]"""

    def _merge(self, segments):
        _, fn = _import_helpers()
        return fn(segments)

    def test_empty_list(self):
        assert self._merge([]) == []

    def test_single_segment_unchanged(self):
        segs = [Segment(text="Hello", speaker="Alice", index=0)]
        result = self._merge(segs)
        assert len(result) == 1
        assert result[0].text == "Hello"
        assert result[0].speaker == "Alice"

    def test_no_consecutive_same_speaker(self):
        """Alternating speakers → no merging."""
        segs = [
            Segment(text="Hello", speaker="Alice", index=0),
            Segment(text="Hi", speaker="NARRATOR", index=1),
            Segment(text="Goodbye", speaker="Bob", index=2),
        ]
        result = self._merge(segs)
        assert len(result) == 3

    def test_merge_two_narrator_segments(self):
        """Two consecutive NARRATOR segments should be merged."""
        segs = [
            Segment(text="Para one.", speaker="NARRATOR", index=0),
            Segment(text="Para two.", speaker="NARRATOR", index=1),
        ]
        result = self._merge(segs)
        assert len(result) == 1
        assert result[0].speaker == "NARRATOR"
        assert "Para one." in result[0].text
        assert "Para two." in result[0].text

    def test_merge_two_character_segments(self):
        """Two consecutive same-character segments should be merged."""
        segs = [
            Segment(text='"Hello,"', speaker="Alice", index=0),
            Segment(text='"How are you?"', speaker="Alice", index=1),
        ]
        result = self._merge(segs)
        assert len(result) == 1
        assert result[0].speaker == "Alice"
        assert "Hello" in result[0].text
        assert "How are you?" in result[0].text

    def test_interleaved_no_merge(self):
        """A-N-A pattern should NOT merge across the NARRATOR span."""
        segs = [
            Segment(text='"Hello,"', speaker="Alice", index=0),
            Segment(text="she said,", speaker="NARRATOR", index=1),
            Segment(text='"goodbye."', speaker="Alice", index=2),
        ]
        result = self._merge(segs)
        assert len(result) == 3

    def test_indices_rewritten_after_merge(self):
        """Merged segment list must have contiguous 0-based indices."""
        segs = [
            Segment(text="A", speaker="NARRATOR", index=0),
            Segment(text="B", speaker="NARRATOR", index=1),
            Segment(text="C", speaker="Alice", index=2),
        ]
        result = self._merge(segs)
        assert [s.index for s in result] == list(range(len(result)))

    def test_narrator_join_uses_double_newline(self):
        """Merged NARRATOR spans should be separated by '\\n\\n'."""
        segs = [
            Segment(text="First paragraph.", speaker="NARRATOR", index=0),
            Segment(text="Second paragraph.", speaker="NARRATOR", index=1),
        ]
        result = self._merge(segs)
        assert "\n\n" in result[0].text

    def test_character_join_uses_space(self):
        """Merged character spans should be separated by a space."""
        segs = [
            Segment(text='"Line one."', speaker="Alice", index=0),
            Segment(text='"Line two."', speaker="Alice", index=1),
        ]
        result = self._merge(segs)
        assert " " in result[0].text
