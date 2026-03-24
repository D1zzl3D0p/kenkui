"""Tests for kenkui.nlp.attribution — LLM speaker attribution engine.

All tests use a mock LLMClient so no Ollama instance is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from kenkui.nlp.attribution import _attribute_chunk, attribute_all_chunks
from kenkui.nlp.chunker import Chunk
from kenkui.nlp.models import AttributionItem, AttributionResult, Quote
from kenkui.nlp import _normalize_speaker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_quote(qid: int, para_index: int = 0, text: str = "") -> Quote:
    return Quote(
        id=qid,
        text=text or f'"Quote {qid}."',
        para_index=para_index,
        char_offset=qid * 20,
    )


def _make_chunk(para_indices: list[int], quote_ids: list[int]) -> Chunk:
    text = "\n\n".join(f"Paragraph {i}." for i in para_indices)
    return Chunk(text=text, para_indices=para_indices, quote_ids=quote_ids)


def _mock_llm(attributions: list[dict]) -> MagicMock:
    """Return a LLMClient mock whose generate() returns AttributionResult."""
    llm = MagicMock()
    llm.generate.return_value = AttributionResult(
        attributions=[AttributionItem(**a) for a in attributions]
    )
    return llm


# ---------------------------------------------------------------------------
# _attribute_chunk
# ---------------------------------------------------------------------------


class TestAttributeChunk:
    def test_returns_attribution_result(self):
        chunk = _make_chunk([0], [0])
        quote = _make_quote(0)
        llm = _mock_llm([{"quote_id": 0, "speaker": "Alice", "emotion": "neutral"}])
        result = _attribute_chunk(chunk, [quote], ["Alice", "NARRATOR"], [], llm)
        assert isinstance(result, AttributionResult)
        assert result.attributions[0].speaker == "Alice"

    def test_empty_chunk_quotes_returns_empty(self):
        chunk = _make_chunk([0], [])
        llm = MagicMock()
        result = _attribute_chunk(chunk, [], ["Alice"], [], llm)
        assert result.attributions == []
        llm.generate.assert_not_called()

    def test_missing_quote_id_filled_with_unknown(self):
        """LLM returns attribution for quote 0 but skips quote 1 — quote 1 → Unknown."""
        chunk = _make_chunk([0], [0, 1])
        quotes = [_make_quote(0), _make_quote(1)]
        llm = _mock_llm([{"quote_id": 0, "speaker": "Alice", "emotion": "happy"}])
        result = _attribute_chunk(chunk, quotes, ["Alice"], [], llm)
        ids = {a.quote_id: a for a in result.attributions}
        assert ids[1].speaker == "Unknown"
        assert ids[1].emotion == "neutral"

    def test_llm_failure_defaults_all_unknown(self):
        """ValidationError from LLM → all quotes default to Unknown."""
        chunk = _make_chunk([0], [0, 1])
        quotes = [_make_quote(0), _make_quote(1)]
        llm = MagicMock()
        llm.generate.side_effect = Exception("network error")
        result = _attribute_chunk(chunk, quotes, ["Alice"], [], llm)
        assert all(a.speaker == "Unknown" for a in result.attributions)
        assert all(a.emotion == "neutral" for a in result.attributions)

    def test_last_speakers_passed_to_prompt(self):
        """Verify last_speakers appears in the prompt sent to the LLM."""
        chunk = _make_chunk([0], [0])
        quote = _make_quote(0)
        llm = _mock_llm([{"quote_id": 0, "speaker": "Bob", "emotion": "sad"}])
        _attribute_chunk(chunk, [quote], ["Bob"], ["Alice", "Bob"], llm)
        call_args = llm.generate.call_args
        prompt = call_args[0][0]  # first positional arg
        assert "Alice" in prompt
        assert "Bob" in prompt


# ---------------------------------------------------------------------------
# attribute_all_chunks
# ---------------------------------------------------------------------------


class TestAttributeAllChunks:
    def test_all_quote_ids_in_result(self):
        quotes = [_make_quote(i, i) for i in range(3)]
        chunks = [_make_chunk([i], [i]) for i in range(3)]
        llm = MagicMock()
        llm.generate.side_effect = [
            AttributionResult(attributions=[
                AttributionItem(quote_id=i, speaker="Alice", emotion="neutral")
            ])
            for i in range(3)
        ]
        result = attribute_all_chunks(chunks, quotes, ["Alice"], llm)
        assert set(result.keys()) == {0, 1, 2}

    def test_last_speakers_threaded_between_chunks(self):
        """The speaker from chunk N should appear in chunk N+1's prompt."""
        quotes = [_make_quote(0, 0), _make_quote(1, 1)]
        chunks = [_make_chunk([0], [0]), _make_chunk([1], [1])]

        call_prompts: list[str] = []

        def capture_generate(prompt, schema):
            call_prompts.append(prompt)
            qid = 0 if len(call_prompts) == 1 else 1
            return AttributionResult(attributions=[
                AttributionItem(quote_id=qid, speaker="Gandalf", emotion="neutral")
            ])

        llm = MagicMock()
        llm.generate.side_effect = capture_generate

        attribute_all_chunks(chunks, quotes, ["Gandalf"], llm)

        # Second prompt should mention "Gandalf" as a recent speaker
        assert "Gandalf" in call_prompts[1]

    def test_empty_chunks_returns_empty(self):
        result = attribute_all_chunks([], [], ["Alice"], MagicMock())
        assert result == {}

    def test_unknown_not_carried_as_last_speaker(self):
        """'Unknown' attributions should not pollute last_speakers state."""
        quotes = [_make_quote(0, 0), _make_quote(1, 1)]
        chunks = [_make_chunk([0], [0]), _make_chunk([1], [1])]

        call_prompts: list[str] = []

        def capture(prompt, schema):
            call_prompts.append(prompt)
            qid = 0 if len(call_prompts) == 1 else 1
            return AttributionResult(attributions=[
                AttributionItem(quote_id=qid, speaker="Unknown", emotion="neutral")
            ])

        llm = MagicMock()
        llm.generate.side_effect = capture
        attribute_all_chunks(chunks, quotes, ["Alice"], llm)

        # "Unknown" should NOT appear in the second chunk's last_speakers section
        # (the prompt says "Recent speakers: (start of chapter)" or similar)
        second_prompt = call_prompts[1]
        # The last_speakers list should not contain "Unknown"
        # We check by looking at what comes after "Recent speakers"
        idx = second_prompt.find("Recent speakers")
        if idx != -1:
            snippet = second_prompt[idx:idx + 80]
            assert "Unknown" not in snippet

    def test_roster_aliases_appear_in_prompt(self):
        """roster_aliases kwarg causes alias info to appear in the LLM prompt."""
        chunk = _make_chunk([0], [0])
        quote = _make_quote(0)
        llm = _mock_llm([{"quote_id": 0, "speaker": "Tiffany Aching", "emotion": "neutral"}])
        roster_aliases = {"Tiffany Aching": ["Tiffany Aching", "Tiffany", "Miss Aching"]}
        _attribute_chunk(
            chunk, [quote], ["Tiffany Aching", "NARRATOR", "Unknown"],
            [], llm, roster_aliases=roster_aliases,
        )
        prompt = llm.generate.call_args[0][0]
        assert "Tiffany" in prompt
        assert "Miss Aching" in prompt


# ---------------------------------------------------------------------------
# _normalize_speaker
# ---------------------------------------------------------------------------


class TestNormalizeSpeaker:
    def _alias_map(self):
        return {
            "tiffany aching": "Tiffany Aching",
            "tiffany": "Tiffany Aching",
            "miss aching": "Tiffany Aching",
            "granny weatherwax": "Esmerelda Weatherwax",
            "granny": "Esmerelda Weatherwax",
            "esmerelda weatherwax": "Esmerelda Weatherwax",
        }

    def _canonicals(self):
        return ["Tiffany Aching", "Esmerelda Weatherwax"]

    def test_exact_lowercase_match(self):
        assert _normalize_speaker(
            "Tiffany", self._alias_map(), self._canonicals()
        ) == "Tiffany Aching"

    def test_canonical_passthrough(self):
        assert _normalize_speaker(
            "Tiffany Aching", self._alias_map(), self._canonicals()
        ) == "Tiffany Aching"

    def test_narrator_passthrough(self):
        assert _normalize_speaker(
            "NARRATOR", self._alias_map(), self._canonicals()
        ) == "NARRATOR"

    def test_unknown_passthrough(self):
        assert _normalize_speaker(
            "Unknown", self._alias_map(), self._canonicals()
        ) == "Unknown"

    def test_fuzzy_word_overlap(self):
        # "Weatherwax" is not in alias_map but overlaps with "Esmerelda Weatherwax"
        result = _normalize_speaker(
            "Weatherwax", {}, ["Tiffany Aching", "Esmerelda Weatherwax"]
        )
        assert result == "Esmerelda Weatherwax"

    def test_no_match_returns_original(self):
        result = _normalize_speaker("Zzyx", {}, ["Tiffany Aching"])
        assert result == "Zzyx"
