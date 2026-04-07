"""Tests for kenkui.nlp.booknlp_roster.

All BookNLP I/O is mocked — no booknlp installation required.  Tests verify
the JSON-parsing logic, skipping rules, and integration with the three-tier
fallback in build_roster_with_llm().
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from kenkui.nlp.models import CharacterRoster


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_book_json(characters: list[dict]) -> str:
    """Serialise a minimal .book JSON structure."""
    return json.dumps({"characters": characters})


def _char(char_id: int, proper: list[dict], pronoun: list[dict] | None = None) -> dict:
    """Build a minimal BookNLP character entry."""
    return {
        "id": char_id,
        "mentions": {
            "proper": proper,
            "pronoun": pronoun or [],
        },
        "g": {"argmax": "she/her"},
        "count": sum(p.get("c", 0) for p in proper),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_booknlp_module():
    """Provide a mock BookNLP class and patch the import."""
    mock_bnlp_instance = MagicMock()
    mock_bnlp_class = MagicMock(return_value=mock_bnlp_instance)
    mock_module = MagicMock()
    mock_module.BookNLP = mock_bnlp_class
    return mock_bnlp_instance, mock_bnlp_class, mock_module


# ---------------------------------------------------------------------------
# build_roster_from_booknlp
# ---------------------------------------------------------------------------


class TestBuildRosterFromBookNLP:
    def test_returns_none_when_booknlp_missing(self):
        from kenkui.nlp.booknlp_roster import build_roster_from_booknlp

        with patch.dict("sys.modules", {"booknlp": None, "booknlp.booknlp": None}):
            result = build_roster_from_booknlp("some text")
        assert result is None

    def test_happy_path_returns_roster(self, mock_booknlp_module, tmp_path):
        from kenkui.nlp.booknlp_roster import build_roster_from_booknlp

        bnlp_instance, bnlp_class, bnlp_mod = mock_booknlp_module

        book_data = _make_book_json([
            _char(1, [{"n": "Tiffany Aching", "c": 45}, {"n": "Tiffany", "c": 150}]),
            _char(2, [{"n": "Granny Weatherwax", "c": 30}, {"n": "Mistress Weatherwax", "c": 10}]),
        ])

        def fake_process(input_path, output_dir, book_id):
            (tmp_path / f"{book_id}.book").write_text(book_data, encoding="utf-8")

        bnlp_instance.process.side_effect = fake_process

        with patch("kenkui.nlp.booknlp_roster.tempfile.TemporaryDirectory") as mock_td:
            mock_td.return_value.__enter__.return_value = str(tmp_path)
            mock_td.return_value.__exit__.return_value = False
            with patch.dict("sys.modules", {"booknlp": bnlp_mod, "booknlp.booknlp": bnlp_mod}):
                result = build_roster_from_booknlp("Tiffany Aching walked in. Granny Weatherwax followed.")

        assert result is not None
        assert isinstance(result.roster, CharacterRoster)
        canonicals = {g.canonical for g in result.roster.characters}
        assert any("Tiffany" in c for c in canonicals)
        assert any("Weatherwax" in c for c in canonicals)

    def test_skips_pronoun_only_characters(self, mock_booknlp_module, tmp_path):
        from kenkui.nlp.booknlp_roster import build_roster_from_booknlp

        bnlp_instance, bnlp_class, bnlp_mod = mock_booknlp_module

        # One character with proper names, one with empty proper (pronoun-only)
        book_data = _make_book_json([
            _char(1, [{"n": "Tiffany", "c": 50}]),
            _char(2, [], pronoun=[{"n": "she", "c": 10}]),  # no proper names → skip
        ])

        def fake_process(input_path, output_dir, book_id):
            (tmp_path / f"{book_id}.book").write_text(book_data, encoding="utf-8")

        bnlp_instance.process.side_effect = fake_process

        with patch("kenkui.nlp.booknlp_roster.tempfile.TemporaryDirectory") as mock_td:
            mock_td.return_value.__enter__.return_value = str(tmp_path)
            mock_td.return_value.__exit__.return_value = False
            with patch.dict("sys.modules", {"booknlp": bnlp_mod, "booknlp.booknlp": bnlp_mod}):
                result = build_roster_from_booknlp("Tiffany walked in.")

        assert result is not None
        assert len(result.roster.characters) == 1

    def test_all_proper_name_forms_appear_as_aliases(self, mock_booknlp_module, tmp_path):
        from kenkui.nlp.booknlp_roster import build_roster_from_booknlp

        bnlp_instance, bnlp_class, bnlp_mod = mock_booknlp_module

        book_data = _make_book_json([
            _char(1, [
                {"n": "Tiffany Aching", "c": 45},
                {"n": "Tiffany", "c": 150},
                {"n": "Miss Aching", "c": 5},
            ]),
        ])

        def fake_process(input_path, output_dir, book_id):
            (tmp_path / f"{book_id}.book").write_text(book_data, encoding="utf-8")

        bnlp_instance.process.side_effect = fake_process

        text = "Tiffany Aching walked in. Tiffany smiled. Miss Aching nodded."
        with patch("kenkui.nlp.booknlp_roster.tempfile.TemporaryDirectory") as mock_td:
            mock_td.return_value.__enter__.return_value = str(tmp_path)
            mock_td.return_value.__exit__.return_value = False
            with patch.dict("sys.modules", {"booknlp": bnlp_mod, "booknlp.booknlp": bnlp_mod}):
                result = build_roster_from_booknlp(text)

        assert result is not None
        all_aliases = {a for g in result.roster.characters for a in g.aliases}
        assert "Tiffany Aching" in all_aliases
        assert "Tiffany" in all_aliases
        assert "Miss Aching" in all_aliases

    def test_heuristic_picks_most_specific_canonical(self, mock_booknlp_module, tmp_path):
        from kenkui.nlp.booknlp_roster import build_roster_from_booknlp

        bnlp_instance, bnlp_class, bnlp_mod = mock_booknlp_module

        # BookNLP may list high-freq short form first; heuristic should pick full name
        book_data = _make_book_json([
            _char(1, [{"n": "Tiffany", "c": 150}, {"n": "Tiffany Aching", "c": 45}]),
        ])

        def fake_process(input_path, output_dir, book_id):
            (tmp_path / f"{book_id}.book").write_text(book_data, encoding="utf-8")

        bnlp_instance.process.side_effect = fake_process

        text = "Tiffany Aching walked in. Tiffany smiled."
        with patch("kenkui.nlp.booknlp_roster.tempfile.TemporaryDirectory") as mock_td:
            mock_td.return_value.__enter__.return_value = str(tmp_path)
            mock_td.return_value.__exit__.return_value = False
            with patch.dict("sys.modules", {"booknlp": bnlp_mod, "booknlp.booknlp": bnlp_mod}):
                result = build_roster_from_booknlp(text)

        assert result is not None
        # _cluster_by_heuristic sorts longest first → "Tiffany Aching" is canonical
        assert result.roster.characters[0].canonical == "Tiffany Aching"

    def test_empty_characters_returns_empty_roster(self, mock_booknlp_module, tmp_path):
        from kenkui.nlp.booknlp_roster import build_roster_from_booknlp

        bnlp_instance, bnlp_class, bnlp_mod = mock_booknlp_module

        book_data = _make_book_json([])

        def fake_process(input_path, output_dir, book_id):
            (tmp_path / f"{book_id}.book").write_text(book_data, encoding="utf-8")

        bnlp_instance.process.side_effect = fake_process

        with patch("kenkui.nlp.booknlp_roster.tempfile.TemporaryDirectory") as mock_td:
            mock_td.return_value.__enter__.return_value = str(tmp_path)
            mock_td.return_value.__exit__.return_value = False
            with patch.dict("sys.modules", {"booknlp": bnlp_mod, "booknlp.booknlp": bnlp_mod}):
                result = build_roster_from_booknlp("some text")

        assert result is not None
        assert result.roster.characters == []

    def test_booknlp_process_failure_returns_none(self, mock_booknlp_module, tmp_path):
        from kenkui.nlp.booknlp_roster import build_roster_from_booknlp

        bnlp_instance, bnlp_class, bnlp_mod = mock_booknlp_module
        bnlp_instance.process.side_effect = RuntimeError("BookNLP internal error")

        with patch("kenkui.nlp.booknlp_roster.tempfile.TemporaryDirectory") as mock_td:
            mock_td.return_value.__enter__.return_value = str(tmp_path)
            mock_td.return_value.__exit__.return_value = False
            with patch.dict("sys.modules", {"booknlp": bnlp_mod, "booknlp.booknlp": bnlp_mod}):
                result = build_roster_from_booknlp("some text")

        assert result is None


# ---------------------------------------------------------------------------
# Integration: BookNLP as Tier 1 of build_roster_with_llm
# ---------------------------------------------------------------------------


class TestBuildRosterWithLLMBookNLPTier:
    def test_booknlp_tier_returns_roster(self, mock_booknlp_module, tmp_path):
        """When BookNLP returns a roster, the result is a CharacterRoster."""
        from kenkui.nlp.entities import build_roster_with_llm
        from kenkui.nlp.models import AliasGroup, CanonicalMergeResult, NameNormalizationResult, NameNormalizationEntry, EpithetResolutionResult

        bnlp_instance, bnlp_class, bnlp_mod = mock_booknlp_module

        book_data = _make_book_json([
            _char(1, [{"n": "Tiffany Aching", "c": 50}]),
        ])

        def fake_process(input_path, output_dir, book_id):
            (tmp_path / f"{book_id}.book").write_text(book_data, encoding="utf-8")

        bnlp_instance.process.side_effect = fake_process

        nlp = MagicMock()
        llm = MagicMock()
        # LLM cleanup passes return empty/no-op results
        def llm_generate(prompt, schema):
            if schema.__name__ == "CanonicalMergeResult":
                return CanonicalMergeResult(merges=[])
            if schema.__name__ == "EpithetResolutionResult":
                return EpithetResolutionResult(mappings=[])
            if schema.__name__ == "NameNormalizationResult":
                return NameNormalizationResult(names=[
                    NameNormalizationEntry(original="Tiffany Aching", simplified="Tiffany Aching"),
                ])
            return CharacterRoster(characters=[AliasGroup(canonical="Tiffany Aching", aliases=["Tiffany Aching"])])
        llm.generate.side_effect = llm_generate

        with patch("kenkui.nlp.booknlp_roster.tempfile.TemporaryDirectory") as mock_td:
            mock_td.return_value.__enter__.return_value = str(tmp_path)
            mock_td.return_value.__exit__.return_value = False
            with patch.dict("sys.modules", {"booknlp": bnlp_mod, "booknlp.booknlp": bnlp_mod}):
                result = build_roster_with_llm("Tiffany Aching walked in.", nlp, llm)

        assert isinstance(result, CharacterRoster)
        assert any("Tiffany" in g.canonical for g in result.characters)

    def test_llm_tier_used_when_booknlp_missing(self):
        """When BookNLP is missing, the LLM tier must be attempted."""
        from kenkui.nlp.entities import build_roster_with_llm
        from kenkui.nlp.models import AliasGroup, CanonicalMergeResult, NameNormalizationResult, NameNormalizationEntry

        nlp = MagicMock()
        doc = MagicMock()
        doc.ents = []
        nlp.return_value = doc

        llm = MagicMock()
        call_count = [0]
        def llm_generate(prompt, schema):
            call_count[0] += 1
            if schema.__name__ == "CanonicalMergeResult":
                return CanonicalMergeResult(merges=[])
            if schema.__name__ == "NameNormalizationResult":
                return NameNormalizationResult(names=[
                    NameNormalizationEntry(original="Tiffany Aching", simplified="Tiffany Aching"),
                ])
            return CharacterRoster(
                characters=[AliasGroup(canonical="Tiffany Aching", aliases=["Tiffany Aching"])]
            )
        llm.generate.side_effect = llm_generate

        with patch.dict("sys.modules", {"booknlp": None, "booknlp.booknlp": None}):
            result = build_roster_with_llm("Tiffany Aching walked in.", nlp, llm)

        assert isinstance(result, CharacterRoster)
        # LLM should have been called at least once (for roster extraction + cleanup passes)
        assert call_count[0] >= 1


class TestBookNLPGenderPassthrough:
    def test_gender_argmax_propagates_to_alias_group(self, mock_booknlp_module, tmp_path):
        from kenkui.nlp.booknlp_roster import build_roster_from_booknlp

        bnlp_instance, bnlp_class, bnlp_mod = mock_booknlp_module

        book_data = _make_book_json([
            _char(1, [{"n": "Tiffany Aching", "c": 50}]),  # _char sets g.argmax="she/her"
            _char(2, [{"n": "Rob Anybody", "c": 30}]),
        ])
        # Override gender for second char
        book_data_dict = json.loads(book_data)
        book_data_dict["characters"][1]["g"]["argmax"] = "he/him"
        book_data = json.dumps(book_data_dict)

        def fake_process(input_path, output_dir, book_id):
            (tmp_path / f"{book_id}.book").write_text(book_data, encoding="utf-8")

        bnlp_instance.process.side_effect = fake_process

        with patch("kenkui.nlp.booknlp_roster.tempfile.TemporaryDirectory") as mock_td:
            mock_td.return_value.__enter__.return_value = str(tmp_path)
            mock_td.return_value.__exit__.return_value = False
            with patch.dict("sys.modules", {"booknlp": bnlp_mod, "booknlp.booknlp": bnlp_mod}):
                data = build_roster_from_booknlp("Tiffany Aching walked. Rob Anybody followed.")

        assert data is not None
        groups = {g.canonical: g for g in data.roster.characters}
        tiffany = next((g for g in data.roster.characters if "Tiffany" in g.canonical), None)
        rob = next((g for g in data.roster.characters if "Rob" in g.canonical), None)
        assert tiffany is not None and tiffany.gender == "she/her"
        assert rob is not None and rob.gender == "he/him"

    def test_unknown_gender_yields_empty_string(self, mock_booknlp_module, tmp_path):
        from kenkui.nlp.booknlp_roster import build_roster_from_booknlp

        bnlp_instance, bnlp_class, bnlp_mod = mock_booknlp_module

        book_data_dict = json.loads(_make_book_json([
            _char(1, [{"n": "Granny Weatherwax", "c": 30}]),
        ]))
        book_data_dict["characters"][0]["g"]["argmax"] = "unknown"
        book_data = json.dumps(book_data_dict)

        def fake_process(input_path, output_dir, book_id):
            (tmp_path / f"{book_id}.book").write_text(book_data, encoding="utf-8")

        bnlp_instance.process.side_effect = fake_process

        with patch("kenkui.nlp.booknlp_roster.tempfile.TemporaryDirectory") as mock_td:
            mock_td.return_value.__enter__.return_value = str(tmp_path)
            mock_td.return_value.__exit__.return_value = False
            with patch.dict("sys.modules", {"booknlp": bnlp_mod, "booknlp.booknlp": bnlp_mod}):
                data = build_roster_from_booknlp("Granny Weatherwax walked in.")

        assert data is not None
        assert data.roster.characters[0].gender == ""


class TestBookNLPCommonPhrases:
    def test_high_freq_common_mentions_collected(self, mock_booknlp_module, tmp_path):
        from kenkui.nlp.booknlp_roster import build_roster_from_booknlp

        bnlp_instance, bnlp_class, bnlp_mod = mock_booknlp_module

        book_data_dict = json.loads(_make_book_json([
            _char(1, [{"n": "Rand al'Thor", "c": 50}]),
        ]))
        book_data_dict["characters"][0]["mentions"]["common"] = [
            {"n": "the Dragon Reborn", "c": 30},
            {"n": "the Dragon", "c": 15},
            {"n": "young man", "c": 3},  # below threshold — should be excluded
        ]
        book_data = json.dumps(book_data_dict)

        def fake_process(input_path, output_dir, book_id):
            (tmp_path / f"{book_id}.book").write_text(book_data, encoding="utf-8")

        bnlp_instance.process.side_effect = fake_process

        with patch("kenkui.nlp.booknlp_roster.tempfile.TemporaryDirectory") as mock_td:
            mock_td.return_value.__enter__.return_value = str(tmp_path)
            mock_td.return_value.__exit__.return_value = False
            with patch.dict("sys.modules", {"booknlp": bnlp_mod, "booknlp.booknlp": bnlp_mod}):
                data = build_roster_from_booknlp("Rand al'Thor walked in.")

        assert data is not None
        assert "the Dragon Reborn" in data.common_phrases
        assert "the Dragon" in data.common_phrases
        assert "young man" not in data.common_phrases  # below _MIN_COMMON_FREQ=5

    def test_empty_common_mentions_yields_empty_list(self, mock_booknlp_module, tmp_path):
        from kenkui.nlp.booknlp_roster import build_roster_from_booknlp

        bnlp_instance, bnlp_class, bnlp_mod = mock_booknlp_module

        book_data = _make_book_json([_char(1, [{"n": "Tiffany", "c": 50}])])

        def fake_process(input_path, output_dir, book_id):
            (tmp_path / f"{book_id}.book").write_text(book_data, encoding="utf-8")

        bnlp_instance.process.side_effect = fake_process

        with patch("kenkui.nlp.booknlp_roster.tempfile.TemporaryDirectory") as mock_td:
            mock_td.return_value.__enter__.return_value = str(tmp_path)
            mock_td.return_value.__exit__.return_value = False
            with patch.dict("sys.modules", {"booknlp": bnlp_mod, "booknlp.booknlp": bnlp_mod}):
                data = build_roster_from_booknlp("Tiffany walked in.")

        assert data is not None
        assert data.common_phrases == []


# ---------------------------------------------------------------------------
# _normalize_gender_pronoun
# ---------------------------------------------------------------------------


from kenkui.nlp import _normalize_gender_pronoun  # noqa: E402


@pytest.mark.parametrize(
    "value,expected",
    [
        ("M", "he/him"),
        ("male", "he/him"),
        ("He/Him", "he/him"),
        ("F", "she/her"),
        ("female", "she/her"),
        ("they", "they/them"),
        ("nonbinary", "they/them"),
        ("unknown", ""),
        ("", ""),
        ("N", ""),
    ],
)
class TestNormalizeGenderPronoun:
    def test_mapping(self, value: str, expected: str) -> None:
        assert _normalize_gender_pronoun(value) == expected
