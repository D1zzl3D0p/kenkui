"""Tests for italic span extraction as dialogue/monologue."""
from __future__ import annotations

# STX and ETX markers inserted by the EPUB reader around <em>/<i> content.
_STX = "\x02"
_ETX = "\x03"


class TestItalicExtraction:
    """Tests for extract_quotes() handling of italic spans."""

    def test_long_italic_internal_monologue(self):
        from kenkui.nlp.quotes import extract_quotes
        para = f"She thought, {_STX}I cannot believe this is happening.{_ETX}"
        quotes = extract_quotes([para])
        assert len(quotes) == 1
        assert quotes[0].kind == "italic"
        assert "I cannot believe" in quotes[0].text

    def test_short_one_word_italic_extracted(self):
        from kenkui.nlp.quotes import extract_quotes
        para = f"He felt {_STX}wrong{_ETX} about it."
        quotes = extract_quotes([para])
        assert len(quotes) == 1
        assert quotes[0].kind == "italic"

    def test_short_two_word_italic_extracted(self):
        from kenkui.nlp.quotes import extract_quotes
        para = f"She whispered {_STX}come closer{_ETX}."
        quotes = extract_quotes([para])
        assert len(quotes) == 1
        assert quotes[0].kind == "italic"

    def test_stx_etx_markers_stripped_from_text(self):
        from kenkui.nlp.quotes import extract_quotes
        para = f"{_STX}Inner thought.{_ETX}"
        quotes = extract_quotes([para])
        assert len(quotes) == 1
        assert _STX not in quotes[0].text
        assert _ETX not in quotes[0].text
        assert quotes[0].text == "Inner thought."

    def test_multiple_italic_spans_in_one_paragraph(self):
        from kenkui.nlp.quotes import extract_quotes
        para = f"She thought {_STX}yes{_ETX} then {_STX}no{_ETX} and finally {_STX}maybe{_ETX}."
        quotes = extract_quotes([para])
        italic_quotes = [q for q in quotes if q.kind == "italic"]
        assert len(italic_quotes) == 3

    def test_italic_para_index_set_correctly(self):
        from kenkui.nlp.quotes import extract_quotes
        paras = [
            "Normal paragraph.",
            f"She thought {_STX}this is the second paragraph{_ETX}.",
        ]
        quotes = extract_quotes(paras)
        assert len(quotes) == 1
        assert quotes[0].para_index == 1

    def test_italic_at_start_of_paragraph(self):
        from kenkui.nlp.quotes import extract_quotes
        para = f"{_STX}Why is this happening?{_ETX} she wondered."
        quotes = extract_quotes([para])
        assert len(quotes) == 1
        assert quotes[0].kind == "italic"
        assert quotes[0].char_offset == 0

    def test_no_italic_markers_returns_empty(self):
        from kenkui.nlp.quotes import extract_quotes
        para = "No special markers here."
        quotes = extract_quotes([para])
        assert quotes == []


class TestItalicOverlapWithDialogue:
    """Regression tests: italic wrapping dialogue produces one quote."""

    def test_italic_wrapping_dialogue_produces_one_italic_quote(self):
        """When <em> wraps a quoted phrase, we get one 'italic' quote, not two."""
        from kenkui.nlp.quotes import extract_quotes
        # epub reader produces: \x02"actual quote"\x03
        para = f'{_STX}"She said something profound."{_ETX}'
        quotes = extract_quotes([para])
        assert len(quotes) == 1
        assert quotes[0].kind == "italic"

    def test_adjacent_italic_and_dialogue_produce_two_quotes(self):
        """Non-overlapping italic and dialogue spans produce two separate quotes."""
        from kenkui.nlp.quotes import extract_quotes
        para = f'{_STX}She thought{_ETX} then said "Hello."'
        quotes = extract_quotes([para])
        assert len(quotes) == 2
        kinds = {q.kind for q in quotes}
        assert "italic" in kinds
        assert "dialogue" in kinds


class TestItalicAttributionKind:
    """Tests that italic kind is preserved in attribution input."""

    def test_italic_quote_kind_field_is_italic(self):
        from kenkui.nlp.quotes import extract_quotes
        para = f"He knew {_STX}something was wrong{_ETX}."
        quotes = extract_quotes([para])
        assert quotes[0].kind == "italic"

    def test_italic_quote_has_id(self):
        from kenkui.nlp.quotes import extract_quotes
        para = f"{_STX}A thought.{_ETX}"
        quotes = extract_quotes([para])
        assert quotes[0].id is not None
        assert isinstance(quotes[0].id, int)
