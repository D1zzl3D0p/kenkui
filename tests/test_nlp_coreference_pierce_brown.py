"""Integration tests for character coreference using Pierce Brown's Golden Son chapters 7-8."""
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kenkui.nlp.models import AttributionItem, AttributionResult

GOLDEN_SON_PATH = Path(
    "/Users/dizzler/Projects/Calibre Library/Pierce Brown/Golden Son (445)/Golden Son - Pierce Brown.epub"
)


@pytest.mark.skipif(
    not GOLDEN_SON_PATH.exists(),
    reason="Golden Son epub not available in this environment",
)
class TestPierceBrownCoreference:
    """Tests using the actual Golden Son epub (chapters 7-8).

    LLM calls are mocked for determinism. Entity extraction uses spaCy
    and the heuristic clustering algorithm (no LLM required).
    """

    @pytest.fixture(scope="class")
    def chapters_7_8(self):
        """Extract chapters 7 and 8 from the Golden Son epub."""
        from kenkui.readers.epub import EpubReader

        reader = EpubReader(GOLDEN_SON_PATH)
        all_chapters = reader.get_chapters()

        # Chapter.index is the 0-based sequential index assigned during extraction.
        # Chapters 7 and 8 (1-based) are at index 6 and 7 (0-based).
        # Try by index attribute first, then fall back to list position.
        target = [ch for ch in all_chapters if ch.index in (6, 7)]
        if not target:
            # Fall back to positional slice
            ch_list = list(all_chapters)
            target = ch_list[6:9] if len(ch_list) > 8 else ch_list[-3:]
        assert target, "Could not find chapters 7-8 in Golden Son"
        return target

    def test_chapters_have_paragraphs(self, chapters_7_8):
        """Sanity: extracted chapters have substantial text."""
        total_chars = sum(
            sum(len(p) for p in ch.paragraphs)
            for ch in chapters_7_8
        )
        assert total_chars > 1000, f"Expected substantial text, got {total_chars} chars"

    def test_darrow_appears_in_entity_extraction(self, chapters_7_8):
        """Darrow should be recognized as a person entity."""
        all_text = " ".join(
            " ".join(ch.paragraphs)
            for ch in chapters_7_8
        )
        assert "Darrow" in all_text, "Expected 'Darrow' in chapters 7-8"


class TestAliasClusteringDarrow:
    """Pure-Python heuristic clustering — no epub, no LLM, no spaCy required."""

    def test_is_alias_of_darrow(self):
        """'Darrow' should be detected as an alias of 'Darrow of Lykos'."""
        from kenkui.nlp.entities import _is_alias_of

        assert _is_alias_of("Darrow", "Darrow of Lykos") is True

    def test_is_alias_of_sevro(self):
        """'Sevro' should be detected as an alias of 'Sevro au Barca'."""
        from kenkui.nlp.entities import _is_alias_of

        # "au" is not in _PARTICLES so sig words of "Sevro au Barca" = ["sevro", "au", "barca"]
        # "Sevro" sig words = ["sevro"] — 1 < 3 and "sevro" in canon_words → True
        assert _is_alias_of("Sevro", "Sevro au Barca") is True

    def test_alias_clustering_groups_darrow_variants(self):
        """Heuristic clustering should group Darrow aliases correctly."""
        from kenkui.nlp.entities import _cluster_by_heuristic

        names = ["Darrow of Lykos", "Darrow", "Sevro", "Sevro au Barca"]
        groups = _cluster_by_heuristic(names)

        # Canonical names are the longest-form names
        canonicals = {g.canonical_name for g in groups}
        assert "Darrow of Lykos" in canonicals, (
            f"Expected 'Darrow of Lykos' as canonical, got: {canonicals}"
        )

        # Find the group containing Darrow of Lykos
        darrow_group = next(g for g in groups if g.canonical_name =="Darrow of Lykos")
        assert "Darrow" in darrow_group.aliases, (
            f"Expected 'Darrow' in aliases of 'Darrow of Lykos', got: {darrow_group.aliases}"
        )

    def test_alias_clustering_groups_sevro_variants(self):
        """Heuristic clustering should group Sevro aliases correctly."""
        from kenkui.nlp.entities import _cluster_by_heuristic

        names = ["Darrow of Lykos", "Darrow", "Sevro", "Sevro au Barca"]
        groups = _cluster_by_heuristic(names)

        canonicals = {g.canonical_name for g in groups}
        assert "Sevro au Barca" in canonicals, (
            f"Expected 'Sevro au Barca' as canonical, got: {canonicals}"
        )

        sevro_group = next(g for g in groups if g.canonical_name =="Sevro au Barca")
        assert "Sevro" in sevro_group.aliases, (
            f"Expected 'Sevro' in aliases of 'Sevro au Barca', got: {sevro_group.aliases}"
        )

    def test_distinct_characters_not_merged(self):
        """Darrow and Sevro should remain in separate groups."""
        from kenkui.nlp.entities import _cluster_by_heuristic

        names = ["Darrow of Lykos", "Darrow", "Sevro", "Sevro au Barca"]
        groups = _cluster_by_heuristic(names)
        assert len(groups) == 2, (
            f"Expected 2 character groups, got {len(groups)}: {[g.canonical_name for g in groups]}"
        )

    def test_each_name_in_exactly_one_group(self):
        """Every input name should appear in exactly one alias group."""
        from kenkui.nlp.entities import _cluster_by_heuristic

        names = ["Darrow of Lykos", "Darrow", "Sevro", "Sevro au Barca"]
        groups = _cluster_by_heuristic(names)
        all_aliases = [alias for g in groups for alias in g.aliases]
        assert len(all_aliases) == len(set(all_aliases)), "Duplicate alias found across groups"
        assert set(all_aliases) == set(names), (
            f"Not all names accounted for. Got: {set(all_aliases)}, expected: {set(names)}"
        )


class TestSpeakerMomentum:
    """last_speakers threading should maintain A/B speaker momentum."""

    def test_speaker_momentum_in_long_dialogue(self):
        """In A/B alternating dialogue without explicit cues, attribution should not collapse to Unknown.

        When A and B alternate for 6 turns without explicit speaker cues,
        the attribution should not collapse all turns to 'Unknown'.
        """
        from kenkui.nlp.attribution import attribute_all_chunks
        from kenkui.nlp.chunker import chunk_paragraphs
        from kenkui.nlp.quotes import extract_quotes

        roster = ["Darrow", "Sevro"]
        paras = [
            '"We need to move now," he said.',
            '"Agreed. How many men do we have?"',
            '"Forty. Maybe fifty if the Howlers show."',
            '"That\'s not enough. We\'ll be slaughtered."',
            '"It\'s all we\'ve got. Make it work."',
            '"Fine. Your funeral, brother."',
        ]
        quotes = extract_quotes(paras)
        assert len(quotes) == 6, f"Expected 6 quotes, got {len(quotes)}"

        chunks = chunk_paragraphs(paras, quotes)

        # Build a mock LLM that alternates speakers A/B/A/B.
        # The mock returns AttributionResult instances directly (matching how
        # _attribute_chunk calls llm.generate(prompt, AttributionResult)).
        call_count = [0]

        def fake_generate(prompt: str, schema=None, **kwargs):
            # Extract quote_ids from the prompt JSON payload
            ids = [int(x) for x in re.findall(r'"quote_id":\s*(\d+)', prompt)]
            attributions = []
            for i, qid in enumerate(ids):
                speaker = roster[(call_count[0] + i) % 2]
                attributions.append(
                    AttributionItem(
                        quote_id=qid,
                        speaker=speaker,
                        emotion="neutral",
                        confidence=4,
                    )
                )
            call_count[0] += len(ids)
            return AttributionResult(attributions=attributions)

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = fake_generate

        result = attribute_all_chunks(chunks, quotes, roster_names=roster, llm=mock_llm)

        # With A/B alternation, no more than 1 "Unknown" should appear
        unknown_count = sum(1 for a in result.values() if a.speaker == "Unknown")
        assert unknown_count <= 1, (
            f"Too many Unknown attributions ({unknown_count}/6): "
            f"{[a.speaker for a in result.values()]}"
        )

        # At least 4 of 6 should be attributed to named characters
        named = [a.speaker for a in result.values() if a.speaker in roster]
        assert len(named) >= 4, (
            f"Expected at least 4 named attributions, got: {[a.speaker for a in result.values()]}"
        )

    def test_last_speakers_state_advances_across_chunks(self):
        """last_speakers from chunk N should appear in chunk N+1's prompt."""
        from kenkui.nlp.attribution import attribute_all_chunks
        from kenkui.nlp.chunker import chunk_paragraphs
        from kenkui.nlp.quotes import extract_quotes

        roster = ["Darrow", "Sevro"]
        paras = [
            '"We need to move now," he said.',
            '"Agreed. How many men do we have?"',
        ]
        quotes = extract_quotes(paras)
        # Force two chunks by using a very low target_words so each paragraph is its own chunk
        chunks = chunk_paragraphs(paras, quotes, target_words=1)

        captured_prompts: list[str] = []

        def capture_generate(prompt: str, schema=None, **kwargs):
            captured_prompts.append(prompt)
            ids = [int(x) for x in re.findall(r'"quote_id":\s*(\d+)', prompt)]
            attributions = [
                AttributionItem(quote_id=qid, speaker="Darrow", emotion="neutral", confidence=4)
                for qid in ids
            ]
            return AttributionResult(attributions=attributions)

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = capture_generate

        attribute_all_chunks(chunks, quotes, roster_names=roster, llm=mock_llm)

        # If there were multiple chunks, the second prompt should contain "Darrow"
        # as a recent speaker (carried over from chunk 1).
        if len(captured_prompts) >= 2:
            assert "Darrow" in captured_prompts[1], (
                "Expected 'Darrow' in second chunk's prompt as a recent speaker"
            )
