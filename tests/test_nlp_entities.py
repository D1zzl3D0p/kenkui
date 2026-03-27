"""Tests for kenkui.nlp.entities — alias clustering heuristic.

The heuristic functions (_significant_words, _is_alias_of, _cluster_by_heuristic,
build_roster) are all pure Python and need no external dependencies.

Tests that call extract_person_names() require spaCy's en_core_web_sm and are
skipped automatically when the model is not installed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from kenkui.nlp.entities import (
    _cluster_by_heuristic,
    _filter_roster_hallucinations,
    _is_alias_of,
    _sample_text_for_roster,
    _significant_words,
    build_roster,
    build_roster_with_llm,
    extract_person_names,
    infer_gender_pronouns,
)
from kenkui.nlp.models import AliasGroup, CharacterRoster


# ---------------------------------------------------------------------------
# _significant_words
# ---------------------------------------------------------------------------


class TestSignificantWords:
    def test_plain_name(self):
        assert _significant_words("Harry Potter") == ["harry", "potter"]

    def test_strips_title_mr(self):
        assert _significant_words("Mr. Potter") == ["potter"]

    def test_strips_title_dr(self):
        assert _significant_words("Dr Smith") == ["smith"]

    def test_strips_particle_von(self):
        assert _significant_words("Ludwig von Mises") == ["ludwig", "mises"]

    def test_strips_trailing_period(self):
        assert _significant_words("Mrs.") == []

    def test_single_short_word_excluded(self):
        # Single-char words are stripped (length guard > 1)
        result = _significant_words("A B C")
        assert result == []

    def test_mixed_case_lowercased(self):
        assert _significant_words("HERMIONE GRANGER") == ["hermione", "granger"]

    def test_apostrophe_stripped(self):
        assert _significant_words("O'Brien") == ["o'brien".strip(".,'-")]


# ---------------------------------------------------------------------------
# _is_alias_of
# ---------------------------------------------------------------------------


class TestIsAliasOf:
    def test_first_name_is_alias_of_full_name(self):
        assert _is_alias_of("Harry", "Harry Potter") is True

    def test_surname_with_title_is_alias(self):
        assert _is_alias_of("Mr. Potter", "Harry Potter") is True

    def test_bare_surname_is_alias(self):
        assert _is_alias_of("Potter", "Harry Potter") is True

    def test_full_name_not_alias_of_itself(self):
        # Same word count — guard prevents merging equal-length names
        assert _is_alias_of("Harry Potter", "Harry Potter") is False

    def test_different_surname_not_alias(self):
        assert _is_alias_of("Ron", "Harry Potter") is False

    def test_hermione_aliases(self):
        assert _is_alias_of("Hermione", "Hermione Granger") is True

    def test_does_not_merge_different_first_names(self):
        # "Tom" should not be an alias of "Tom Sawyer" when we test against
        # "Tom Robinson" — but this function only tests pairs, so we check
        # that "Tom" IS an alias of both (the clustering handles the rest).
        assert _is_alias_of("Tom", "Tom Sawyer") is True
        assert _is_alias_of("Tom", "Tom Robinson") is True

    def test_empty_candidate_returns_false(self):
        assert _is_alias_of("", "Harry Potter") is False

    def test_empty_canonical_returns_false(self):
        assert _is_alias_of("Harry", "") is False

    def test_longer_candidate_not_alias(self):
        # "Harry James Potter" should NOT be an alias of "Harry Potter"
        # (it has MORE significant words).
        assert _is_alias_of("Harry James Potter", "Harry Potter") is False


# ---------------------------------------------------------------------------
# _cluster_by_heuristic
# ---------------------------------------------------------------------------


class TestClusterByHeuristic:
    def test_empty_input(self):
        assert _cluster_by_heuristic([]) == []

    def test_single_name(self):
        groups = _cluster_by_heuristic(["Harry Potter"])
        assert len(groups) == 1
        assert groups[0].canonical == "Harry Potter"
        assert groups[0].aliases == ["Harry Potter"]

    def test_harry_potter_cluster(self):
        names = ["Harry", "Harry Potter", "Mr. Potter", "Ron", "Ron Weasley"]
        groups = _cluster_by_heuristic(names)
        canonicals = {g.canonical for g in groups}
        assert "Harry Potter" in canonicals
        assert "Ron Weasley" in canonicals

        hp = next(g for g in groups if g.canonical == "Harry Potter")
        rw = next(g for g in groups if g.canonical == "Ron Weasley")

        assert "Harry" in hp.aliases
        assert "Mr. Potter" in hp.aliases
        assert "Ron" in rw.aliases

    def test_hermione_cluster(self):
        names = ["Hermione", "Hermione Granger", "Albus Dumbledore", "Dumbledore"]
        groups = _cluster_by_heuristic(names)
        canonicals = {g.canonical for g in groups}
        assert "Hermione Granger" in canonicals
        assert "Albus Dumbledore" in canonicals

        hg = next(g for g in groups if g.canonical == "Hermione Granger")
        ad = next(g for g in groups if g.canonical == "Albus Dumbledore")
        assert "Hermione" in hg.aliases
        assert "Dumbledore" in ad.aliases

    def test_duplicates_deduplicated(self):
        groups = _cluster_by_heuristic(["Alice", "Alice", "Alice Wonderland"])
        assert len(groups) == 1
        assert groups[0].canonical == "Alice Wonderland"

    def test_distinct_names_not_merged(self):
        names = ["Alice", "Bob"]
        groups = _cluster_by_heuristic(names)
        assert len(groups) == 2

    def test_every_name_appears_in_exactly_one_group(self):
        names = [
            "Harry", "Harry Potter", "Ron", "Ron Weasley",
            "Hermione", "Hermione Granger", "Dumbledore", "Albus Dumbledore",
        ]
        groups = _cluster_by_heuristic(names)
        all_aliases = [alias for g in groups for alias in g.aliases]
        # No duplicate aliases
        assert len(all_aliases) == len(set(all_aliases))
        # Every input name is represented
        assert set(all_aliases) == set(names)


# ---------------------------------------------------------------------------
# build_roster (requires spaCy)
# ---------------------------------------------------------------------------

_SPACY_AVAILABLE = False
try:
    import spacy
    if spacy.util.is_package("en_core_web_sm"):
        _SPACY_AVAILABLE = True
except ImportError:
    pass

spacy_required = pytest.mark.skipif(
    not _SPACY_AVAILABLE,
    reason="spaCy en_core_web_sm not installed",
)


# ---------------------------------------------------------------------------
# infer_gender_pronouns
# ---------------------------------------------------------------------------


class TestInferGenderPronouns:
    def test_she_pronouns(self):
        text = "Tiffany Aching walked in. She smiled at him. She said nothing."
        assert infer_gender_pronouns("Tiffany Aching", ["Tiffany Aching", "Tiffany"], text) == "she/her"

    def test_he_pronouns(self):
        text = "Harry Potter arrived. He drew his wand. He looked around."
        assert infer_gender_pronouns("Harry Potter", ["Harry Potter", "Harry"], text) == "he/him"

    def test_they_pronouns(self):
        text = "River Song appeared. They grinned. Their plan was working."
        assert infer_gender_pronouns("River Song", ["River Song"], text) == "they/them"

    def test_alias_match(self):
        # Text uses an alias ("Tiffany") rather than the canonical name.
        text = "Tiffany stepped forward. She raised her hand."
        assert infer_gender_pronouns("Tiffany Aching", ["Tiffany Aching", "Tiffany"], text) == "she/her"

    def test_no_pronouns_returns_empty(self):
        text = "The mountain loomed. Rocks fell. Nothing moved."
        assert infer_gender_pronouns("Harry Potter", ["Harry Potter"], text) == ""

    def test_name_not_in_text_returns_empty(self):
        text = "He walked. She talked. They argued."
        assert infer_gender_pronouns("Hermione Granger", ["Hermione Granger"], text) == ""

    def test_majority_wins(self):
        # 3 she vs 1 he — should return she/her
        text = (
            "Alice laughed. She went home. She ate dinner. She slept. "
            "Alice came back. He waved at her."  # 'he' is 1 occurrence
        )
        assert infer_gender_pronouns("Alice", ["Alice"], text) == "she/her"

    def test_empty_text_returns_empty(self):
        assert infer_gender_pronouns("Alice", ["Alice"], "") == ""

    def test_empty_aliases_uses_canonical(self):
        text = "Bob arrived. He said hello."
        assert infer_gender_pronouns("Bob", ["Bob"], text) == "he/him"


@spacy_required
class TestBuildRoster:
    @pytest.fixture(scope="class")
    def nlp(self):
        import spacy
        return spacy.load("en_core_web_sm")

    def test_returns_character_roster(self, nlp):
        text = "Harry Potter went to Hogwarts. Hermione Granger followed Harry."
        result = build_roster(text, nlp)
        assert isinstance(result, CharacterRoster)

    def test_empty_text_no_crash(self, nlp):
        result = build_roster("", nlp)
        assert result.characters == []

    def test_no_persons_no_characters(self, nlp):
        result = build_roster("The weather was cold. Rain fell.", nlp)
        assert result.characters == []

    def test_known_name_clustered(self, nlp):
        text = (
            "Harry Potter walked in. Harry smiled. "
            "Ron Weasley followed Ron."
        )
        result = build_roster(text, nlp)
        canonicals = {g.canonical for g in result.characters}
        # At minimum, both canonical forms should exist
        assert any("Harry" in c for c in canonicals)


# ---------------------------------------------------------------------------
# _sample_text_for_roster
# ---------------------------------------------------------------------------


class TestSampleTextForRoster:
    def test_short_text_returned_unchanged(self):
        text = "Harry Potter walked in.\n\nRon Weasley followed."
        result = _sample_text_for_roster(text, target_words=4000)
        assert result == text

    def test_long_text_under_word_limit(self):
        # Build a ~50,000-word text by repeating a paragraph
        para = "Harry Potter walked down the corridor. " * 20  # ~120 words
        text = "\n\n".join([para] * 400)  # ~48,000 words
        result = _sample_text_for_roster(text, target_words=4000)
        word_count = len(result.split())
        # Allow some slack for separator words; budget is 4000 across 5 buckets
        assert word_count <= 5500

    def test_long_text_has_separator(self):
        para = "Some text about characters. " * 20
        text = "\n\n".join([para] * 400)
        result = _sample_text_for_roster(text, target_words=4000)
        assert "[...]" in result

    def test_empty_text_no_crash(self):
        result = _sample_text_for_roster("", target_words=4000)
        assert result == ""


# ---------------------------------------------------------------------------
# _filter_roster_hallucinations
# ---------------------------------------------------------------------------


class TestFilterRosterHallucinations:
    def _roster(self, *entries):
        return CharacterRoster(characters=[
            AliasGroup(canonical=c, aliases=list(a)) for c, a in entries
        ])

    def test_verbatim_aliases_kept(self):
        text = "Mr. Potter sat down. Harry looked around. Harry Potter smiled."
        roster = self._roster(
            ("Harry Potter", ["Harry Potter", "Harry", "Mr. Potter"])
        )
        result = _filter_roster_hallucinations(roster, text)
        all_aliases = [a for g in result.characters for a in g.aliases]
        assert "Harry" in all_aliases
        assert "Mr. Potter" in all_aliases
        assert "Harry Potter" in all_aliases

    def test_hallucinated_alias_removed(self):
        text = "Harry walked in."
        roster = self._roster(
            ("Harry", ["Harry", "Hermione Granger"])  # Hermione not in text
        )
        result = _filter_roster_hallucinations(roster, text)
        all_aliases = [a for g in result.characters for a in g.aliases]
        assert "Hermione Granger" not in all_aliases

    def test_entire_entry_dropped_when_no_aliases_survive(self):
        text = "Harry walked in."
        roster = self._roster(
            ("Harry", ["Harry"]),
            ("Gandalf the Grey", ["Gandalf the Grey", "Gandalf"])  # not in text
        )
        result = _filter_roster_hallucinations(roster, text)
        canonicals = {g.canonical for g in result.characters}
        assert not any("Gandalf" in c for c in canonicals)

    def test_hallucinated_canonical_promotes_longest_survivor(self):
        text = "Harry walked in."
        # Canonical "Harry James Potter" is not in text; alias "Harry" is
        roster = self._roster(
            ("Harry James Potter", ["Harry James Potter", "Harry"])
        )
        result = _filter_roster_hallucinations(roster, text)
        all_aliases = [a for g in result.characters for a in g.aliases]
        assert "Harry" in all_aliases
        assert "Harry James Potter" not in all_aliases

    def test_empty_roster_returns_empty(self):
        result = _filter_roster_hallucinations(CharacterRoster(characters=[]), "some text")
        assert result.characters == []

    def test_case_insensitive_matching(self):
        text = "HARRY POTTER arrived."
        roster = self._roster(("Harry Potter", ["Harry Potter"]))
        result = _filter_roster_hallucinations(roster, text)
        assert len(result.characters) == 1

    def test_short_alias_removed(self):
        text = "A walked in."
        roster = self._roster(("A", ["A"]))
        result = _filter_roster_hallucinations(roster, text)
        # Single-char alias stripped (len < 2)
        assert result.characters == []


# ---------------------------------------------------------------------------
# build_roster_with_llm
# ---------------------------------------------------------------------------


def _mock_nlp_no_names():
    """spaCy nlp mock that returns zero PERSON entities."""
    nlp = MagicMock()
    doc = MagicMock()
    doc.ents = []
    nlp.return_value = doc
    return nlp


def _mock_llm_roster(characters: list[dict]) -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = CharacterRoster(
        characters=[AliasGroup(**c) for c in characters]
    )
    return llm


class TestBuildRosterWithLLM:
    def test_happy_path_returns_correct_canonicals(self):
        text = "Harry Potter smiled. Harry nodded. Mr. Potter left. Ron Weasley followed Ron."
        llm = _mock_llm_roster([
            {"canonical": "Harry Potter", "aliases": ["Harry Potter", "Harry", "Mr. Potter"]},
            {"canonical": "Ron Weasley", "aliases": ["Ron Weasley", "Ron"]},
        ])
        result = build_roster_with_llm(text, _mock_nlp_no_names(), llm)
        canonicals = {g.canonical for g in result.characters}
        assert "Harry Potter" in canonicals
        assert "Ron Weasley" in canonicals

    def test_aliases_resolved_into_canonical(self):
        text = "Harry Potter smiled. Harry nodded. Mr. Potter left."
        llm = _mock_llm_roster([
            {"canonical": "Harry Potter", "aliases": ["Harry Potter", "Harry", "Mr. Potter"]},
        ])
        result = build_roster_with_llm(text, _mock_nlp_no_names(), llm)
        assert len(result.characters) == 1
        hp = result.characters[0]
        assert hp.canonical == "Harry Potter"
        assert "Harry" in hp.aliases
        assert "Mr. Potter" in hp.aliases

    def test_llm_failure_falls_back_to_heuristic(self):
        text = "Harry Potter walked in. Harry smiled."
        nlp = MagicMock()
        doc = MagicMock()
        ent = MagicMock()
        ent.label_ = "PERSON"
        ent.text = "Harry Potter"
        doc.ents = [ent]
        nlp.return_value = doc

        llm = MagicMock()
        llm.generate.side_effect = Exception("ollama unreachable")

        result = build_roster_with_llm(text, nlp, llm)
        assert isinstance(result, CharacterRoster)
        # Fallback to heuristic; spaCy returned "Harry Potter" so at least 1 character
        assert len(result.characters) >= 1

    def test_hallucinated_character_removed(self):
        text = "Harry walked in."
        llm = _mock_llm_roster([
            {"canonical": "Harry", "aliases": ["Harry"]},
            {"canonical": "Gandalf", "aliases": ["Gandalf", "The Grey"]},  # not in text
        ])
        result = build_roster_with_llm(text, _mock_nlp_no_names(), llm)
        canonicals = {g.canonical for g in result.characters}
        assert not any("Gandalf" in c for c in canonicals)

    def test_duplicate_llm_entries_merged(self):
        text = "Harry Potter smiled. Harry nodded."
        # LLM returns "Harry Potter" and "Harry" as separate entries
        llm = _mock_llm_roster([
            {"canonical": "Harry Potter", "aliases": ["Harry Potter"]},
            {"canonical": "Harry", "aliases": ["Harry"]},
        ])
        result = build_roster_with_llm(text, _mock_nlp_no_names(), llm)
        # Heuristic should collapse these into a single canonical
        assert len(result.characters) == 1
        assert result.characters[0].canonical == "Harry Potter"

    def test_empty_llm_roster_falls_back_to_heuristic(self):
        text = "Harry Potter walked in."
        nlp = MagicMock()
        doc = MagicMock()
        ent = MagicMock()
        ent.label_ = "PERSON"
        ent.text = "Harry Potter"
        doc.ents = [ent]
        nlp.return_value = doc

        llm = _mock_llm_roster([])  # empty roster
        result = build_roster_with_llm(text, nlp, llm)
        assert isinstance(result, CharacterRoster)
        # Falls back to heuristic; spaCy found "Harry Potter"
        canonicals = {g.canonical for g in result.characters}
        assert "Harry Potter" in canonicals


class TestDeduplicateRosterWithLLM:
    def test_merges_nickname_into_full_name(self):
        from kenkui.nlp.entities import deduplicate_roster_with_llm
        from kenkui.nlp.models import AliasGroup, CanonicalMergeResult, CanonicalMergeEntry, CharacterRoster

        roster = CharacterRoster(characters=[
            AliasGroup(canonical="Matrim Cauthon", aliases=["Matrim Cauthon", "Matrim"]),
            AliasGroup(canonical="Mat", aliases=["Mat"]),
            AliasGroup(canonical="Perrin Aybara", aliases=["Perrin Aybara", "Perrin"]),
        ])
        llm = MagicMock()
        llm.generate.return_value = CanonicalMergeResult(merges=[
            CanonicalMergeEntry(canonical="Matrim Cauthon", duplicates=["Mat"]),
        ])

        result = deduplicate_roster_with_llm(roster, llm)

        canonicals = {g.canonical for g in result.characters}
        assert "Mat" not in canonicals
        assert "Matrim Cauthon" in canonicals
        assert "Perrin Aybara" in canonicals
        merged = next(g for g in result.characters if g.canonical == "Matrim Cauthon")
        assert "Mat" in merged.aliases

    def test_preserves_gender_from_absorbed_entry(self):
        from kenkui.nlp.entities import deduplicate_roster_with_llm
        from kenkui.nlp.models import AliasGroup, CanonicalMergeResult, CanonicalMergeEntry, CharacterRoster

        roster = CharacterRoster(characters=[
            AliasGroup(canonical="Matrim Cauthon", aliases=["Matrim Cauthon"], gender=""),
            AliasGroup(canonical="Mat", aliases=["Mat"], gender="he/him"),
        ])
        llm = MagicMock()
        llm.generate.return_value = CanonicalMergeResult(merges=[
            CanonicalMergeEntry(canonical="Matrim Cauthon", duplicates=["Mat"]),
        ])

        result = deduplicate_roster_with_llm(roster, llm)
        survivor = next(g for g in result.characters if g.canonical == "Matrim Cauthon")
        assert survivor.gender == "he/him"

    def test_llm_error_returns_roster_unchanged(self):
        from kenkui.nlp.entities import deduplicate_roster_with_llm
        from kenkui.nlp.models import AliasGroup, CharacterRoster

        roster = CharacterRoster(characters=[
            AliasGroup(canonical="Alice", aliases=["Alice"]),
        ])
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("ollama down")

        result = deduplicate_roster_with_llm(roster, llm)
        assert len(result.characters) == 1

    def test_unknown_duplicate_canonical_ignored(self):
        from kenkui.nlp.entities import deduplicate_roster_with_llm
        from kenkui.nlp.models import AliasGroup, CanonicalMergeResult, CanonicalMergeEntry, CharacterRoster

        roster = CharacterRoster(characters=[
            AliasGroup(canonical="Alice", aliases=["Alice"]),
        ])
        llm = MagicMock()
        # LLM hallucinates a name not in the roster
        llm.generate.return_value = CanonicalMergeResult(merges=[
            CanonicalMergeEntry(canonical="Alice", duplicates=["Alicia"]),
        ])

        result = deduplicate_roster_with_llm(roster, llm)
        assert len(result.characters) == 1  # unchanged


class TestResolveEpithetsWithLLM:
    def test_adds_epithet_as_alias(self):
        from kenkui.nlp.entities import resolve_epithets_with_llm
        from kenkui.nlp.models import AliasGroup, CharacterRoster, EpithetResolutionResult, EpithetMapping

        roster = CharacterRoster(characters=[
            AliasGroup(canonical="Rand al'Thor", aliases=["Rand al'Thor", "Rand"]),
        ])
        llm = MagicMock()
        llm.generate.return_value = EpithetResolutionResult(mappings=[
            EpithetMapping(epithet="the Dragon Reborn", canonical_name="Rand al'Thor"),
        ])

        result = resolve_epithets_with_llm(roster, ["the Dragon Reborn"], llm)
        rand = result.characters[0]
        assert "the Dragon Reborn" in rand.aliases

    def test_empty_phrases_skips_llm(self):
        from kenkui.nlp.entities import resolve_epithets_with_llm
        from kenkui.nlp.models import AliasGroup, CharacterRoster

        roster = CharacterRoster(characters=[AliasGroup(canonical="Alice", aliases=["Alice"])])
        llm = MagicMock()

        result = resolve_epithets_with_llm(roster, [], llm)
        llm.generate.assert_not_called()
        assert result is roster

    def test_unknown_canonical_in_mapping_ignored(self):
        from kenkui.nlp.entities import resolve_epithets_with_llm
        from kenkui.nlp.models import AliasGroup, CharacterRoster, EpithetResolutionResult, EpithetMapping

        roster = CharacterRoster(characters=[AliasGroup(canonical="Alice", aliases=["Alice"])])
        llm = MagicMock()
        llm.generate.return_value = EpithetResolutionResult(mappings=[
            EpithetMapping(epithet="the chosen one", canonical_name="Bob"),  # Bob not in roster
        ])

        result = resolve_epithets_with_llm(roster, ["the chosen one"], llm)
        assert "the chosen one" not in result.characters[0].aliases


class TestNormalizeCanonicalNamesWithLLM:
    def test_strips_appositive_suffix(self):
        from kenkui.nlp.entities import normalize_canonical_names_with_llm
        from kenkui.nlp.models import AliasGroup, CharacterRoster, NameNormalizationResult, NameNormalizationEntry

        roster = CharacterRoster(characters=[
            AliasGroup(canonical="Rand al'Thor, Dragon Reborn", aliases=["Rand al'Thor, Dragon Reborn"]),
        ])
        llm = MagicMock()
        llm.generate.return_value = NameNormalizationResult(names=[
            NameNormalizationEntry(original="Rand al'Thor, Dragon Reborn", simplified="Rand al'Thor"),
        ])

        result = normalize_canonical_names_with_llm(roster, llm)
        assert result.characters[0].canonical == "Rand al'Thor"
        assert "Rand al'Thor, Dragon Reborn" in result.characters[0].aliases

    def test_unchanged_name_not_modified(self):
        from kenkui.nlp.entities import normalize_canonical_names_with_llm
        from kenkui.nlp.models import AliasGroup, CharacterRoster, NameNormalizationResult, NameNormalizationEntry

        roster = CharacterRoster(characters=[
            AliasGroup(canonical="Harry Potter", aliases=["Harry Potter"]),
        ])
        llm = MagicMock()
        llm.generate.return_value = NameNormalizationResult(names=[
            NameNormalizationEntry(original="Harry Potter", simplified="Harry Potter"),
        ])

        result = normalize_canonical_names_with_llm(roster, llm)
        assert result.characters[0].canonical == "Harry Potter"
        # No duplicate alias added for unchanged names
        assert result.characters[0].aliases.count("Harry Potter") == 1

    def test_llm_error_returns_unchanged(self):
        from kenkui.nlp.entities import normalize_canonical_names_with_llm
        from kenkui.nlp.models import AliasGroup, CharacterRoster

        roster = CharacterRoster(characters=[AliasGroup(canonical="Alice", aliases=["Alice"])])
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("timeout")

        result = normalize_canonical_names_with_llm(roster, llm)
        assert result.characters[0].canonical == "Alice"
