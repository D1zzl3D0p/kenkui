"""Gendering a speaker from the pronoun in their own dialogue tag."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import kenkui as kk
from kenkui._characters import dialogue_tags
from kenkui._domain.casting import CharacterProfile
from kenkui._domain.planning import SpeakerSpan

if TYPE_CHECKING:
    import pytest


def _chapter(text: str) -> kk.ChapterInspection:
    return kk.ChapterInspection(
        id="ch-1", index=0, title="One", speech_characters=len(text), text=text
    )


class TestTagVotes:
    """What the tag beside a quote says about whoever was attributed to it."""

    def test_a_trailing_pronoun_tag_genders_the_speaker(self) -> None:
        """`"..." she said` states the gender of whoever spoke the quote."""
        text = '"I will not go," she said.'
        spans = (SpeakerSpan("ch-1", 0, 16, "ellie"),)

        votes = dialogue_tags.tag_genders((_chapter(text),), spans)

        assert votes["ellie"]["feminine"] == 1

    def test_a_leading_pronoun_tag_genders_the_speaker(self) -> None:
        """English puts the tag before the quote as readily as after it."""
        text = 'He said, "I will not go."'
        spans = (SpeakerSpan("ch-1", 9, len(text), "daniel"),)

        votes = dialogue_tags.tag_genders((_chapter(text),), spans)

        assert votes["daniel"]["masculine"] == 1

    def test_a_named_tag_casts_no_gender_vote(self) -> None:
        """`"..." Ellie said` names the speaker but says nothing about gender."""
        text = '"I will not go," Ellie said.'
        spans = (SpeakerSpan("ch-1", 0, 16, "ellie"),)

        votes = dialogue_tags.tag_genders((_chapter(text),), spans)

        assert votes.get("ellie", Counter()) == Counter()

    def test_narration_casts_no_vote(self) -> None:
        """A span with no speaker has no one to gender."""
        text = '"I will not go," she said.'
        spans = (SpeakerSpan("ch-1", 0, len(text), None),)

        assert dialogue_tags.tag_genders((_chapter(text),), spans) == {}


class TestPrecedence:
    """A confident tag outranks the roster; an unconfident one does not."""

    def test_a_confident_tag_vote_supersedes_the_roster(self) -> None:
        """The tag's pronoun is the speaker's; the roster's was merely nearby."""
        characters = (
            CharacterProfile("ellie", "Ellie", "masculine", 100, ("ch-1",)),
            CharacterProfile("ahdi", "Ahdi", None, 100, ("ch-1",)),
        )
        votes = {
            "ellie": Counter({"feminine": 9, "masculine": 1}),
            "ahdi": Counter({"masculine": 4}),
        }

        applied = {c.id: c.gender for c in dialogue_tags.apply(characters, votes)}

        assert applied == {"ellie": "feminine", "ahdi": "masculine"}

    def test_an_unconfident_tag_vote_leaves_the_roster_alone(self) -> None:
        """Below the threshold this signal is too sparse to overturn a book."""
        characters = (
            CharacterProfile("vera", "Aunt Vera", "feminine", 100, ("ch-1",)),
        )
        votes = {"vera": Counter({"masculine": 2})}

        applied = {c.id: c.gender for c in dialogue_tags.apply(characters, votes)}

        assert applied == {"vera": "feminine"}

    def test_a_split_tag_vote_decides_nothing(self) -> None:
        """No margin means the roster's answer stands, right or wrong."""
        characters = (CharacterProfile("chris", "Chris", "feminine", 100, ("ch-1",)),)
        votes = {"chris": Counter({"masculine": 5, "feminine": 4})}

        applied = {c.id: c.gender for c in dialogue_tags.apply(characters, votes)}

        assert applied == {"chris": "feminine"}

    def test_every_other_field_survives_the_rewrite(self) -> None:
        """Only gender is decided here; volume and aliases are already measured."""
        spoken = 4200
        character = CharacterProfile(
            "ellie", "Ellie", None, spoken, ("ch-1", "ch-2"), aliases=("Ellie", "El")
        )

        (applied,) = dialogue_tags.apply(
            (character,), {"ellie": Counter({"feminine": 5})}
        )

        assert applied.gender == "feminine"
        assert applied.spoken_characters == spoken
        assert applied.chapter_ids == ("ch-1", "ch-2")
        assert applied.aliases == ("Ellie", "El")

    def test_a_conflict_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """A tag disagreeing with the roster means a quote went to the wrong mouth."""
        characters = (CharacterProfile("ellie", "Ellie", "masculine", 100, ("ch-1",)),)

        with caplog.at_level("WARNING"):
            dialogue_tags.apply(characters, {"ellie": Counter({"feminine": 6})})

        assert "dialogue_tag_gender_conflict" in caplog.text
