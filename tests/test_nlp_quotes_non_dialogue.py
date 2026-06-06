"""Tests for short scare-quote (non-dialogue) extraction and attribution."""
from __future__ import annotations

from unittest.mock import MagicMock

from kenkui.nlp.models import AttributionItem, AttributionResult


class TestScarequoteExtraction:
    """Scare quotes ARE extracted by extract_quotes() — the regex can't distinguish.
    The LLM attribution step handles NARRATOR assignment based on context.
    """

    def test_single_word_scare_quote_extracted(self):
        from kenkui.nlp.quotes import extract_quotes
        para = 'My "friend" tried to help.'
        quotes = extract_quotes([para])
        assert len(quotes) == 1
        assert quotes[0].kind == "dialogue"
        assert "friend" in quotes[0].text

    def test_two_word_scare_quote_extracted(self):
        from kenkui.nlp.quotes import extract_quotes
        para = 'He called it "the thing".'
        quotes = extract_quotes([para])
        assert len(quotes) == 1
        assert "the thing" in quotes[0].text

    def test_three_word_scare_quote_extracted(self):
        from kenkui.nlp.quotes import extract_quotes
        para = 'A "so-called expert" spoke.'
        quotes = extract_quotes([para])
        assert len(quotes) == 1

    def test_scare_quote_mid_sentence(self):
        from kenkui.nlp.quotes import extract_quotes
        para = 'She gave her "blessing" reluctantly.'
        quotes = extract_quotes([para])
        assert len(quotes) == 1

    def test_scare_quote_alongside_real_dialogue(self):
        """Paragraph with a scare quote AND real dialogue should produce 2 quotes."""
        from kenkui.nlp.quotes import extract_quotes
        para = 'She used "air quotes" then said "I mean it literally."'
        quotes = extract_quotes([para])
        assert len(quotes) == 2

    def test_curly_quote_scare_quote_extracted(self):
        """Curly/smart quotes are also extracted."""
        from kenkui.nlp.quotes import extract_quotes
        para = 'My \u201cfriend\u201d tried to help.'
        quotes = extract_quotes([para])
        assert len(quotes) == 1


class TestScarequoteAttribution:
    """Tests that the LLM attribution prompt correctly handles scare quotes.
    We mock the LLM to return NARRATOR for scare-quote contexts and verify
    the attribution pipeline respects it.
    """

    def _make_mock_llm(self, speaker: str = "NARRATOR", emotion: str = "neutral", confidence: int = 4):
        """Return a mock LLM that always returns an AttributionResult for the requested quote IDs.

        The mock inspects the quotes_json in the prompt to discover which quote_ids
        are being requested, then returns an AttributionResult for each of them.
        """
        import re as _re

        mock = MagicMock()

        def fake_generate(prompt: str, schema=None, **kwargs):
            # Parse the quote IDs from the prompt JSON payload
            ids = _re.findall(r'"quote_id":\s*(\d+)', prompt)
            items = [
                AttributionItem(
                    quote_id=int(qid),
                    speaker=speaker,
                    emotion=emotion,
                    confidence=confidence,
                )
                for qid in ids
            ]
            return AttributionResult(attributions=items)

        mock.generate.side_effect = fake_generate
        return mock

    def test_scare_quote_attributed_to_narrator_by_llm(self):
        """When LLM returns NARRATOR for a scare quote, attribution result is NARRATOR."""
        from kenkui.nlp.attribution import attribute_all_chunks
        from kenkui.nlp.chunker import chunk_paragraphs
        from kenkui.nlp.quotes import extract_quotes

        paras = ['My "friend" tried to help.']
        quotes = extract_quotes(paras)
        assert len(quotes) == 1

        chunks = chunk_paragraphs(paras, quotes)
        mock_llm = self._make_mock_llm(speaker="NARRATOR")

        result = attribute_all_chunks(chunks, quotes, roster_names=[], llm=mock_llm)
        assert result[quotes[0].id].speaker == "NARRATOR"

    def test_real_dialogue_attributed_to_character(self):
        """Real dialogue in same paragraph gets attributed to character, not NARRATOR."""
        from kenkui.nlp.attribution import attribute_all_chunks
        from kenkui.nlp.chunker import chunk_paragraphs
        from kenkui.nlp.quotes import extract_quotes

        paras = ['"Hello," she said.']
        quotes = extract_quotes(paras)
        assert len(quotes) == 1

        chunks = chunk_paragraphs(paras, quotes)
        mock_llm = self._make_mock_llm(speaker="Alice", confidence=4)

        result = attribute_all_chunks(chunks, quotes, roster_names=["Alice"], llm=mock_llm)
        assert result[quotes[0].id].speaker == "Alice"
