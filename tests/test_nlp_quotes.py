"""Tests for kenkui.nlp.quotes — regex dialogue extractor."""

from __future__ import annotations

import pytest

from kenkui.nlp.quotes import extract_quotes
from kenkui.nlp.models import Quote


class TestExtractQuotesEmpty:
    def test_empty_list(self):
        assert extract_quotes([]) == []

    def test_no_quotes_in_paragraph(self):
        assert extract_quotes(["The sun rose slowly over the hills."]) == []

    def test_multiple_paragraphs_no_quotes(self):
        assert extract_quotes(["Rain fell.", "Wind howled.", "Silence followed."]) == []


class TestExtractQuotesStraight:
    def test_single_straight_quote(self):
        quotes = extract_quotes(['"Hello," she said.'])
        assert len(quotes) == 1
        assert '"Hello,"' in quotes[0].text or "Hello" in quotes[0].text

    def test_quote_id_starts_at_zero(self):
        quotes = extract_quotes(['"First."', '"Second."'])
        assert quotes[0].id == 0
        assert quotes[1].id == 1

    def test_ids_sequential(self):
        paras = ['"A."', '"B."', '"C."']
        quotes = extract_quotes(paras)
        assert [q.id for q in quotes] == [0, 1, 2]


class TestExtractQuotesCurly:
    def test_curly_open_close(self):
        quotes = extract_quotes(['\u201cGood morning,\u201d he said.'])
        assert len(quotes) == 1
        assert "Good morning" in quotes[0].text

    def test_mixed_straight_and_curly(self):
        paras = ['"Straight quote."', '\u201cCurly quote.\u201d']
        quotes = extract_quotes(paras)
        assert len(quotes) == 2


class TestExtractQuotesParaIndex:
    def test_para_index_correct(self):
        paras = [
            "No quote here.",
            '"First dialogue."',
            "More narration.",
            '"Second dialogue."',
        ]
        quotes = extract_quotes(paras)
        assert len(quotes) == 2
        assert quotes[0].para_index == 1
        assert quotes[1].para_index == 3

    def test_two_quotes_same_paragraph(self):
        paras = ['"Hello." "Goodbye."']
        quotes = extract_quotes(paras)
        assert len(quotes) == 2
        assert quotes[0].para_index == 0
        assert quotes[1].para_index == 0

    def test_quote_in_first_paragraph(self):
        quotes = extract_quotes(['"Yes," said Alice.'])
        assert quotes[0].para_index == 0


class TestExtractQuotesCharOffset:
    def test_offset_zero_for_first_para_start(self):
        quotes = extract_quotes(['"Hello."'])
        assert quotes[0].char_offset == 0

    def test_offset_increases_across_paragraphs(self):
        paras = ['"A."', '"B."']
        quotes = extract_quotes(paras)
        # Second paragraph starts at len('"A."') + 2 ("\n\n" separator)
        assert quotes[1].char_offset == len(paras[0]) + 2

    def test_offset_monotone(self):
        paras = ['"First."', 'Middle text. "Second."', '"Third."']
        quotes = extract_quotes(paras)
        offsets = [q.char_offset for q in quotes]
        assert offsets == sorted(offsets)


class TestExtractQuotesMultiline:
    def test_quote_spanning_newline_within_paragraph(self):
        # A quote that contains a newline (within a single paragraph string)
        para = '"He said\nthis slowly."'
        quotes = extract_quotes([para])
        assert len(quotes) == 1
        assert "He said" in quotes[0].text


class TestExtractQuotesOverlap:
    """Regression tests: italic span wrapping a dialogue quote must not produce
    two Quote objects (one dialogue, one italic) for the same text region."""

    def test_italic_wrapping_dialogue_produces_one_quote(self):
        # \x02"Hello"\x03 — italic marker around a dialogue quote
        para = '\x02"Hello"\x03'
        quotes = extract_quotes([para])
        assert len(quotes) == 1, (
            f"Expected exactly 1 Quote for overlapping italic+dialogue, got {len(quotes)}: {quotes}"
        )

    def test_italic_wrapping_dialogue_kind_is_italic(self):
        # The italic match starts at 0 and the dialogue match starts at 1,
        # so italic wins (it sorts first by start position).
        para = '\x02"Hello"\x03'
        quotes = extract_quotes([para])
        assert quotes[0].kind == "italic"

    def test_non_overlapping_italic_and_dialogue_produce_two_quotes(self):
        # Italic span then separate dialogue: both should be present.
        para = '\x02inner\x03 then "dialogue"'
        quotes = extract_quotes([para])
        assert len(quotes) == 2
        kinds = {q.kind for q in quotes}
        assert "italic" in kinds
        assert "dialogue" in kinds


class TestSplitParagraphByQuotesOverlap:
    """Regression tests: _split_paragraph_by_quotes must not emit two spans
    for the same overlapping italic+dialogue region."""

    def _split(self, para, para_quotes=None):
        from kenkui.nlp import _split_paragraph_by_quotes
        return _split_paragraph_by_quotes(para, para_quotes or [])

    def test_italic_wrapping_dialogue_produces_one_span(self):
        para = '\x02"Hello"\x03'
        spans = self._split(para)
        # Should be exactly one non-empty span (no duplicates)
        non_empty = [(t, s) for t, s in spans if t]
        assert len(non_empty) == 1, (
            f"Expected 1 span for overlapping italic+dialogue, got {len(non_empty)}: {non_empty}"
        )

    def test_non_overlapping_regions_both_present(self):
        para = '\x02inner\x03 then "dialogue"'
        spans = self._split(para)
        texts = [t for t, _ in spans if t]
        combined = "".join(texts)
        # Both regions must appear in the output
        assert "inner" in combined
        assert "dialogue" in combined
