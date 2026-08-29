"""Deriving a character roster with spaCy instead of a language model."""

from __future__ import annotations

import builtins
import json
import sys
from collections import Counter
from typing import TYPE_CHECKING

import pytest

import kenkui as kk
from kenkui._characters import (
    _chapter_roster,
    resolve_attribution,
    spacy_roster,
)
from kenkui._characters.quotes import extract_spans
from kenkui._domain.casting import CharacterProfile

if TYPE_CHECKING:
    from pathlib import Path

spacy = pytest.importorskip("spacy", reason="the spacy extra is not installed")

# One passage carrying every signal the inference reads: speech verbs, direct
# address inside dialogue, possessed body parts, and gendered pronouns. Names
# recur, because MIN_MENTIONS deliberately ignores anything seen once or twice
# -- below that a parse error and a walk-on are indistinguishable.
PASSAGE = """
Tam al'Thor set down his axe. "The road is long," he said.
"You should rest, Tam," Egwene replied. Her voice was soft.
Tam al'Thor shook his head. "There is no time," he answered.
Egwene frowned at Tam. "Then I will walk with you," Egwene said.
Tam al'Thor smiled. "As you like," he said.
Tam shouted for the horse. His hands were steady on the reins.
Egwene followed Tam down the road, and Tam did not look back.
"""

# The same signals, but with the two people kept apart, as ordinary prose keeps
# them. The gender window reads pronouns near a name, so a dense two-hander
# where every line touches both characters is exactly the case it abstains on.
SEPARATED = """
Egwene walked alone through the village. She carried her basket carefully.
Her hair was dark. She had walked this road since she was a girl.
"It is a long way," Egwene said. She did not mind the walk.
She stopped at the well. Her hands were cold, and she rubbed them together.
Egwene thought of her mother. She missed her.
"""


def roster_of(text: str, *, pipeline: str = "en_core_web_lg") -> tuple:
    """Run inference over a single synthetic chapter."""
    chapter = kk.ChapterInspection(
        id="ch-1", index=0, title="One", speech_characters=len(text), text=text
    )
    spans = {"ch-1": extract_spans("ch-1", text)}
    return spacy_roster.infer_roster((chapter,), spans, pipeline=pipeline)


class TestModelIdParsing:
    """`infer_characters` selects spaCy by model id, not by a separate flag."""

    @pytest.mark.parametrize(
        ("model_id", "expected"),
        [
            ("spacy", spacy_roster.DEFAULT_PIPELINE),
            # The obvious spelling a caller reaches for first.
            ("spaCy", spacy_roster.DEFAULT_PIPELINE),
            ("SPACY", spacy_roster.DEFAULT_PIPELINE),
            ("spacy:en_core_web_sm", "en_core_web_sm"),
            ("spacy:en_core_web_trf", "en_core_web_trf"),
        ],
    )
    def test_spacy_ids_name_a_pipeline(self, model_id: str, expected: str) -> None:
        """A spaCy scheme resolves to the pipeline it names."""
        assert spacy_roster.pipeline_for(model_id) == expected

    @pytest.mark.parametrize(
        "model_id",
        ["openrouter/deepseek/deepseek-chat", "gpt-4o-mini", "spacyish", ""],
    )
    def test_other_ids_are_left_to_the_model_boundary(self, model_id: str) -> None:
        """Anything else is a model id, and never touches this module."""
        assert spacy_roster.pipeline_for(model_id) is None

    def test_a_scheme_without_a_pipeline_is_refused(self, tmp_path: Path) -> None:
        """`spacy:` names nothing, and must fail where the caller wrote it."""
        with pytest.raises(kk.ValidationError) as caught:
            kk.book(tmp_path / "x.epub").infer_characters("spacy:")
        assert caught.value.code == kk.ErrorCode.INVALID_MODEL


class TestInference:
    """What the dependency parse actually recovers from a passage."""

    def test_speaking_characters_are_found(self) -> None:
        """The nsubj of a speech verb is a speaker."""
        characters, _ = roster_of(PASSAGE)
        ids = {character.id for character in characters}
        assert "tam-al-thor" in ids
        assert "egwene" in ids

    def test_short_and_full_forms_fold_into_one_character(self) -> None:
        """One man under two surface forms must not get two voices."""
        characters, _ = roster_of(PASSAGE)
        tams = [c for c in characters if "tam" in c.id]
        assert len(tams) == 1
        assert tams[0].id == "tam-al-thor"
        assert set(tams[0].aliases) == {"Tam", "Tam al'Thor"}

    def test_gender_is_voted_from_surrounding_pronouns(self) -> None:
        """Pronouns near a name vote on how that character is heard."""
        characters, _ = roster_of(SEPARATED)
        by_id = {character.id: character for character in characters}
        assert by_id["egwene"].gender == "feminine"

    @pytest.mark.parametrize(
        ("votes", "expected"),
        [
            ({"feminine": 12, "masculine": 2}, "feminine"),
            ({"masculine": 9, "feminine": 4}, "masculine"),
            # Too few votes to mean anything.
            ({"feminine": 2}, None),
            # A majority, but not a margin: a two-hander where every line
            # touches both characters looks like this.
            ({"feminine": 16, "masculine": 14}, None),
            ({}, None),
        ],
    )
    def test_gender_is_claimed_only_on_a_clear_margin(
        self, votes: dict[str, int], expected: str | None
    ) -> None:
        """No clear majority means no claim, not a coin flip.

        `casting.candidates` answers an unsourced gender by offering the whole
        pool. A wrong guess is worse than none: it silently restricts a
        character to voices that sound wrong for them.
        """
        assert spacy_roster._vote_gender(Counter(votes)) == expected  # noqa: SLF001

    def test_pronouns_never_become_characters(self) -> None:
        """A pronoun as an id would collapse every speaker of that gender into one."""
        characters, _ = roster_of(PASSAGE)
        ids = {character.id for character in characters}
        assert not ids & {"he", "she", "her", "his", "i"}

    def test_chapter_ids_are_recorded(self) -> None:
        """Casting forbids two speakers in one chapter sharing a voice."""
        characters, _ = roster_of(PASSAGE)
        assert all(c.chapter_ids == ("ch-1",) for c in characters)

    def test_prose_without_people_yields_an_empty_roster(self) -> None:
        """Description with no speakers produces nothing to cast."""
        characters, narrator = roster_of(
            "The rain fell on the empty road. Nothing moved for a long while."
        )
        assert characters == ()
        assert narrator is None

    def test_inference_is_deterministic(self) -> None:
        """Two runs must agree: the plan fingerprint depends on it."""
        first, _ = roster_of(PASSAGE)
        second, _ = roster_of(PASSAGE)
        assert first == second


class TestFirstPersonNarrator:
    """A narrator who says "I said" governs no speech verb and is invisible."""

    def test_a_named_but_unspeaking_character_is_returned_as_narrator(
        self,
    ) -> None:
        """Addressed far more often than seen to speak: that is the narrator."""
        text = (
            '"Darrow, you must run," Eo said. "I will not," I said.\n'
            '"Darrow, listen to me," she pleaded. "There is no time," I said.\n'
            '"Please, Darrow," Eo whispered. "I am staying," I said.\n'
            '"Darrow!" she cried. "No," I said.\n'
        )
        _, narrator = roster_of(text)
        assert narrator == "darrow"


class TestFailureModes:
    """Both ways spaCy can be unavailable are stable public failures."""

    def test_missing_package_is_a_model_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An uninstalled optional dependency is a stable public failure."""
        real_import = builtins.__import__

        def _refuse(name: str, *args: object, **kwargs: object) -> object:
            if name == "spacy" or name.startswith("spacy."):
                raise ModuleNotFoundError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "spacy", raising=False)
        monkeypatch.setattr(builtins, "__import__", _refuse)

        with pytest.raises(kk.ModelError) as caught:
            roster_of(PASSAGE)
        assert caught.value.code == kk.ErrorCode.SPACY_PACKAGE_MISSING

    def test_missing_pipeline_is_a_model_error(self) -> None:
        """A pipeline that was never downloaded fails the same way."""
        with pytest.raises(kk.ModelError) as caught:
            roster_of(PASSAGE, pipeline="en_core_web_does_not_exist")
        assert caught.value.code == kk.ErrorCode.SPACY_PIPELINE_MISSING


class TestPipelineIntegration:
    """The roster pass reaches no model at all when spaCy is selected."""

    def test_spacy_roster_makes_no_roster_calls(self) -> None:
        """Attribution still runs; the roster half of the spend disappears."""
        client = _CountingClient()
        record = resolve_attribution(
            _inspection(PASSAGE),
            "digest-spacy",
            "fake/model",
            roster_model_id="spacy:en_core_web_lg",
            client=client,
        )
        assert client.roster_prompts == 0
        assert client.attribution_prompts > 0
        assert record.characters

    def test_the_model_roster_still_asks(self) -> None:
        """The default path is unchanged: a model id still drives a roster call."""
        client = _CountingClient()
        resolve_attribution(
            _inspection(PASSAGE),
            "digest-model",
            "fake/model",
            roster_model_id="fake/roster-model",
            client=client,
        )
        assert client.roster_prompts > 0

    def test_infer_characters_accepts_the_spacy_id(self, tmp_path: Path) -> None:
        """The whole point: `infer_characters("spaCy")` is now valid intent."""
        pipeline = kk.book(tmp_path / "x.epub").infer_characters("spaCy")
        assert any(
            getattr(operation, "model_id", "") == "spaCy"
            for operation in pipeline.operations
        )


class _CountingClient:
    """Separates roster prompts from attribution prompts, and answers both."""

    def __init__(self) -> None:
        self.roster_prompts = 0
        self.attribution_prompts = 0

    def complete(self, model: str, prompt: str) -> str:
        """Answer a roster or an attribution prompt, counting which."""
        assert model
        if "List the speaking characters" in prompt:
            self.roster_prompts += 1
            return json.dumps(
                {"characters": [{"id": "tam-al-thor", "name": "Tam al'Thor"}]}
            )
        self.attribution_prompts += 1
        # Every quote in the passage goes to one speaker: `_measured` drops a
        # character nobody was attributed to, so an empty answer would leave
        # the record characterless whatever the roster found.
        return json.dumps(
            {
                "attributions": [
                    {"quote_id": index, "speaker": "Tam al'Thor"}
                    for index in range(1, 12)
                ]
            }
        )


def _inspection(text: str) -> kk.BookInspection:
    """One chapter, wrapped as the inspection resolve_attribution expects."""
    return kk.BookInspection(
        metadata=kk.BookMetadata(title="T", author="A"),
        chapters=(
            kk.ChapterInspection(
                id="ch-1",
                index=0,
                title="One",
                speech_characters=len(text),
                text=text,
            ),
        ),
    )


class TestChapterRoster:
    """Attribution is offered only the characters a chapter can contain.

    The roster the prompt carries used to be the whole book's, because the
    model roster cannot know where anyone appears until attribution has
    already run. A spaCy roster does know: it read every chapter to build the
    roster in the first place, and records the chapters each name was seen in.
    """

    def _profile(
        self, character_id: str, chapters: tuple[str, ...]
    ) -> CharacterProfile:
        return CharacterProfile(
            id=character_id,
            display_name=character_id.title(),
            gender=None,
            spoken_characters=0,
            chapter_ids=chapters,
        )

    def test_absent_characters_are_dropped(self) -> None:
        """A name never seen in this chapter is not a candidate for its quotes."""
        roster = (
            self._profile("darrow", ("ch-1", "ch-2")),
            self._profile("sevro", ("ch-2",)),
        )
        kept = _chapter_roster(roster, "ch-1", None)
        assert {character.id for character in kept} == {"darrow"}


    def test_the_narrator_survives(self) -> None:
        """The narrator is marked in the prompt block and must be in it."""
        roster = (
            self._profile("darrow", ("ch-9",)),
            self._profile("sevro", ("ch-1",)),
        )
        kept = _chapter_roster(roster, "ch-1", "darrow")
        assert {character.id for character in kept} == {"darrow", "sevro"}

    def test_an_unplaced_roster_is_passed_through(self) -> None:
        """The model roster leaves chapter_ids empty until after attribution.

        Filtering on an empty field would hand every chapter an empty roster
        and narrate the entire book, so that path must stay untouched.
        """
        roster = (self._profile("darrow", ()), self._profile("sevro", ()))
        kept = _chapter_roster(roster, "ch-1", None)
        assert kept == roster

    def test_a_chapter_matching_nobody_falls_back_to_the_book(self) -> None:
        """Better a wide roster than a chapter that cannot name anyone at all.

        Reached when a chapter's speakers are all referred to by role rather
        than by name. Passing nothing would skip the model call and narrate
        every line; passing the book lets it mint the roles it finds.
        """
        roster = (self._profile("darrow", ("ch-9",)),)
        kept = _chapter_roster(roster, "ch-1", None)
        assert kept == roster
