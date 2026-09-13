"""Deriving a character roster with spaCy instead of a language model."""
# ruff: noqa: D101, D102, D103, PLR2004

from __future__ import annotations

import builtins
import json
import sys
from collections import Counter
from typing import TYPE_CHECKING

import pytest

import kenkui as kk
from kenkui import _characters
from kenkui._characters import (
    _chapter_roster,
    resolve_attribution,
    spacy_roster,
)
from kenkui._characters.entries import RosterEntry
from kenkui._domain.casting import CharacterProfile
from kenkui._domain.grid import build_grid, dialogue_ranges
from kenkui._domain.planning import SpeakerSpan

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path
    from types import ModuleType

    from kenkui._characters.store import AttributionRecord

pytestmark = pytest.mark.spacy
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

SPOUSES = """
Count Fenring bowed low. "My dear Baron," Count Fenring said. He smiled thinly.
Lady Fenring laughed. "You are cruel," Lady Fenring said. She turned away.
"Enough," Count Fenring said. He waved a hand.
"As you wish," Lady Fenring said. She sat down.
Count Fenring rose. He left the room.
Count Fenring sighed. He was tired.
Count Fenring frowned, and Lady Fenring watched him.
"""

PROMOTED = """
Captain Nefud saluted. "Yes, my Lord," Captain Nefud said. He waited.
Lieutenant Nefud had once said the same. "At once," Nefud said. Nefud turned.
"It is done," Captain Nefud said.
"""

GROUPS = """
The Fremen came at dawn. The Fremen fought on Arrakis. "Go," Stilgar said.
The Fremen said nothing. They lived on Arrakis. From Arrakis came spice.
"Stay," Stilgar said. Stilgar waited. The Fremen watched Stilgar.
"""


def signals_of(text: str) -> spacy_roster._Signals:
    chapter = kk.ChapterInspection(
        id="ch-1", index=0, title="One", speech_characters=len(text), text=text
    )
    return spacy_roster.collect_signals(
        (chapter,), {"ch-1": dialogue_ranges(build_grid(chapter))}
    )


class TestSignals:
    """What one read of the book records, and how titled forms fold."""

    def test_a_couple_sharing_a_surname_are_two_candidates(self) -> None:
        signals = signals_of(SPOUSES)
        assert signals.mentions["Count Fenring"] >= 3
        assert signals.mentions["Lady Fenring"] >= 3
        assert "Fenring" not in signals.mentions

    def test_a_man_promoted_keeps_one_name(self) -> None:
        signals = signals_of(PROMOTED)
        assert signals.mentions["Nefud"] >= 5
        assert "Captain Nefud" not in signals.mentions

    def test_group_and_place_evidence_is_recorded(self) -> None:
        signals = signals_of(GROUPS)
        assert signals.the_det["Fremen"] >= 2
        assert signals.prep["Arrakis"]["on"] >= 1

    def test_rename_moves_every_tally(self) -> None:
        signals = spacy_roster._Signals()  # noqa: SLF001
        signals.mentions["A"] = 2
        signals.gender["A"]["masculine"] = 3
        signals.chapters["A"].add("ch-1")
        signals.title_hosts["count"]["A"] = 1
        signals.rename("A", "B")
        assert signals.mentions["B"] == 2
        assert signals.gender["B"]["masculine"] == 3
        assert signals.chapters["B"] == {"ch-1"}
        assert signals.title_hosts["count"]["B"] == 1
        assert "A" not in signals.mentions


def signals_with(
    mentions: Mapping[str, int],
    *,
    gender: Mapping[str, Mapping[str, int]] | None = None,
    title_hosts: Mapping[str, Mapping[str, int]] | None = None,
) -> spacy_roster._Signals:
    signals = spacy_roster._Signals()  # noqa: SLF001
    signals.mentions.update(mentions)
    signals.agency.update(dict.fromkeys(mentions, 1))
    for name, votes in (gender or {}).items():
        signals.gender[name].update(votes)
    for title, hosts in (title_hosts or {}).items():
        signals.title_hosts[title].update(hosts)
    return signals


def fold(
    signals: spacy_roster._Signals, *, fallback: bool = True
) -> tuple[dict[str, str], dict[str, str]]:
    return spacy_roster._fold_names(  # noqa: SLF001
        signals, set(signals.mentions), fallback=fallback
    )


class TestFold:
    """Which names are one person, decided by rule."""

    def test_title_and_surname_do_not_absorb_the_family(self) -> None:
        canonical, _ = fold(
            signals_with(
                {"Mr Elliot": 40, "Anne Elliot": 30, "Walter Elliot": 20, "Anne": 400}
            )
        )
        assert canonical["Mr Elliot"] == "Mr Elliot"
        assert canonical["Anne"] == "Anne Elliot"

    def test_a_single_claimant_owns_a_titled_short_form(self) -> None:
        canonical, _ = fold(
            signals_with({"Iakin Nefud": 3, "Captain Nefud": 5, "Nefud": 50})
        )
        assert canonical["Captain Nefud"] == "Iakin Nefud"
        assert canonical["Nefud"] == "Iakin Nefud"

    def test_a_wife_never_folds_into_her_husband(self) -> None:
        signals = signals_with(
            {"Perrin Aybara": 20, "Perrin": 900, "Mistress Aybara": 5},
            gender={"Perrin": {"masculine": 90}, "Perrin Aybara": {"masculine": 5}},
        )
        canonical, _ = fold(signals)
        assert canonical["Mistress Aybara"] == "Mistress Aybara"
        assert canonical["Perrin"] == "Perrin Aybara"

    def test_tie_rule_ignores_tiny_claimants_in_the_fallback(self) -> None:
        paul = signals_with({"Paul": 1678, "Paul Atreides": 18, "Paul Muad'Dib": 14})
        assert fold(paul)[0]["Paul"] == "Paul"
        seldon = signals_with({"Seldon": 77, "Hari Seldon": 45, "Raven Seldon": 4})
        assert fold(seldon)[0]["Seldon"] == "Hari Seldon"

    def test_a_real_tie_drops_the_bare_name_in_the_fallback(self) -> None:
        charles = signals_with(
            {"Charles": 111, "Charles Hayter": 30, "Charles Musgrove": 40}
        )
        canonical, removed = fold(charles)
        assert "Charles" not in canonical
        assert "Charles" in removed

    def test_the_identity_path_keeps_ambiguous_names_for_the_model(self) -> None:
        charles = signals_with(
            {"Charles": 111, "Charles Hayter": 30, "Charles Musgrove": 40}
        )
        assert fold(charles, fallback=False)[0]["Charles"] == "Charles"

    def test_bare_titles_in_the_fallback(self) -> None:
        signals = signals_with(
            {
                "Baron": 500,
                "Vladimir Harkonnen": 16,
                "Duke": 480,
                "Leto Atreides": 15,
                "Paul Atreides": 18,
                "Mayor": 70,
            },
            title_hosts={
                "baron": {"Vladimir Harkonnen": 8},
                "duke": {"Leto Atreides": 50, "Paul Atreides": 5},
            },
        )
        canonical, removed = fold(signals)
        assert canonical["Baron"] == "Vladimir Harkonnen"
        assert "Duke" in removed
        assert canonical["Mayor"] == "Mayor"

    def test_bare_titles_on_the_identity_path_are_left_for_the_model(self) -> None:
        signals = signals_with(
            {"Baron": 500, "Vladimir Harkonnen": 16},
            title_hosts={"baron": {"Vladimir Harkonnen": 8}},
        )
        assert fold(signals, fallback=False)[0]["Baron"] == "Baron"

    def test_couple_stays_apart_end_to_end(self) -> None:
        characters, _ = roster_of(SPOUSES)
        by_name = {character.display_name: character for character in characters}
        assert "Count Fenring" in by_name
        assert "Lady Fenring" in by_name
        assert by_name["Lady Fenring"].gender == "feminine"


class TestFallbackFilters:
    """Offline-only filters for groups, places, and forms of address."""

    def test_a_group_that_never_speaks_is_removed(self) -> None:
        signals = signals_with({"Fremen": 100, "Stilgar": 80})
        signals.the_det["Fremen"] = 60
        signals.speech["Stilgar"] = 30
        names = {
            entry.display_name
            for entry in spacy_roster._base_entries(signals, "", fallback=True)  # noqa: SLF001
        }
        assert names == {"Stilgar"}

    def test_an_epithet_character_who_speaks_is_kept(self) -> None:
        signals = signals_with({"Dragon": 368})
        signals.the_det["Dragon"] = 360
        signals.speech["Dragon"] = 102
        names = {
            entry.display_name
            for entry in spacy_roster._base_entries(signals, "", fallback=True)  # noqa: SLF001
        }
        assert names == {"Dragon"}

    def test_a_place_is_removed_but_of_does_not_count(self) -> None:
        signals = signals_with({"Caladan": 52, "Muad'Dib": 165})
        signals.prep["Caladan"].update({"on": 20, "from": 5})
        signals.prep["Muad'Dib"].update({"of": 60})
        names = {
            entry.display_name
            for entry in spacy_roster._base_entries(signals, "", fallback=True)  # noqa: SLF001
        }
        assert names == {"Muad'Dib"}

    def test_a_form_of_address_is_removed_but_a_nickname_is_kept(self) -> None:
        signals = signals_with({"Sire": 47, "Nieshka": 39})
        signals.vocative.update({"Sire": 44, "Nieshka": 39})
        text = "Yes, sire. No, sire. Nieshka laughed."
        names = {
            entry.display_name
            for entry in spacy_roster._base_entries(signals, text, fallback=True)  # noqa: SLF001
        }
        assert names == {"Nieshka"}

    def test_the_identity_path_skips_the_filters(self) -> None:
        signals = signals_with({"Fremen": 100})
        signals.the_det["Fremen"] = 60
        names = {
            entry.display_name
            for entry in spacy_roster._base_entries(signals, "", fallback=False)  # noqa: SLF001
        }
        assert names == {"Fremen"}


class _MergeAll:
    def resolve(
        self,
        entries: Sequence[RosterEntry],
        text: str,
        mentions: Mapping[str, int],
    ) -> tuple[RosterEntry, ...]:
        del text, mentions
        head = max(entries, key=lambda entry: entry.mentions)
        return (
            RosterEntry(
                head.display_name,
                frozenset().union(*(entry.aliases for entry in entries)),
                sum(entry.mentions for entry in entries),
                sum(entry.evidence for entry in entries),
            ),
        )


class _Fails:
    def resolve(
        self,
        entries: Sequence[RosterEntry],
        text: str,
        mentions: Mapping[str, int],
    ) -> None:
        del entries, text, mentions


class TestIdentityWiring:
    def test_the_resolver_reshapes_the_roster(self) -> None:
        chapter = kk.ChapterInspection(
            id="ch-1",
            index=0,
            title="One",
            speech_characters=len(PASSAGE),
            text=PASSAGE,
        )
        characters, _ = spacy_roster.infer_roster(
            (chapter,),
            {"ch-1": dialogue_ranges(build_grid(chapter))},
            identity=_MergeAll(),
        )
        assert len(characters) == 1

    def test_a_failed_pass_falls_back_to_the_rules(self) -> None:
        chapter = kk.ChapterInspection(
            id="ch-1",
            index=0,
            title="One",
            speech_characters=len(PASSAGE),
            text=PASSAGE,
        )
        spans = {"ch-1": dialogue_ranges(build_grid(chapter))}
        with_failure, _ = spacy_roster.infer_roster(
            (chapter,), spans, identity=_Fails()
        )
        offline, _ = spacy_roster.infer_roster((chapter,), spans)
        assert with_failure == offline


class TestSpelling:
    """One name, however the typesetter spelled it."""

    def test_apostrophe_variants_clean_to_one_name(self) -> None:
        clean = spacy_roster._clean  # noqa: SLF001
        assert clean("Muad\u2018Dib") == clean("Muad\u2019Dib") == "Muad'Dib"

    def test_hyphenated_names_stay_whole(self) -> None:
        nlp = spacy_roster._load(spacy_roster.DEFAULT_PIPELINE)  # noqa: SLF001
        doc = nlp('"Enough," said Feyd-Rautha Harkonnen, and Feyd-Rautha smiled.')
        spans = [s.text for s in spacy_roster._proper_noun_spans(doc)]  # noqa: SLF001
        assert "Feyd-Rautha Harkonnen" in spans
        assert "Rautha Harkonnen" not in spans

    def test_a_spaced_dash_still_separates_names(self) -> None:
        nlp = spacy_roster._load(spacy_roster.DEFAULT_PIPELINE)  # noqa: SLF001
        doc = nlp("Paul - Jessica watched him - said nothing.")
        spans = [s.text for s in spacy_roster._proper_noun_spans(doc)]  # noqa: SLF001
        assert "Paul - Jessica" not in spans


def roster_of(
    text: str, *, pipeline: str = "en_core_web_lg"
) -> tuple[tuple[CharacterProfile, ...], str | None]:
    """Run inference over a single synthetic chapter."""
    chapter = kk.ChapterInspection(
        id="ch-1", index=0, title="One", speech_characters=len(text), text=text
    )
    dialogue = {"ch-1": dialogue_ranges(build_grid(chapter))}
    return spacy_roster.infer_roster((chapter,), dialogue, pipeline=pipeline)


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

    def test_gender_resolves_in_a_dense_two_hander(self) -> None:
        """Both people share every line, so a wide window cancels them out.

        A symmetric window counts the other character's pronouns as evidence
        about this one, and in a scene like this the two tallies converge until
        neither holds a margin. Both then abstain -- and an unsourced gender is
        answered with the whole voice pool, which is how a woman draws a
        masculine voice.

        The nearest following pronoun refers to the name it follows, so it is
        unmoved by who else is in the scene.
        """
        two_hander = """
Tam al'Thor set down his axe. Egwene watched her brother closely.
Tam al'Thor shook his head. Egwene folded her hands in her lap.
Tam al'Thor raised his voice. Egwene lowered her eyes.
Tam al'Thor took his horse. Egwene gathered her basket.
"""

        characters, _ = roster_of(two_hander)
        by_id = {character.id: character for character in characters}

        assert by_id["egwene"].gender == "feminine"
        assert by_id["tam-al-thor"].gender == "masculine"

    def test_a_gendered_honorific_outranks_nearby_pronouns(self) -> None:
        """The honorific Aunt outranks nearby pronouns about someone else."""
        text = (
            "Aunt Vera set down the tray. He had left the door open again.\n"
            "Aunt Vera frowned at him. He said nothing at all.\n"
            "Aunt Vera poured the tea. He watched her hands.\n"
            "Aunt Vera sighed. He shrugged at Aunt Vera and looked away.\n"
        )

        characters, _ = roster_of(text)
        by_id = {character.id: character for character in characters}

        assert by_id["aunt-vera"].gender == "feminine"

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
        assert spacy_roster._majority(Counter(votes)) == expected  # noqa: SLF001

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

        def _refuse(
            name: str,
            globals: Mapping[str, object] | None = None,  # noqa: A002
            locals: Mapping[str, object] | None = None,  # noqa: A002
            fromlist: Sequence[str] = (),
            level: int = 0,
        ) -> ModuleType:
            if name == "spacy" or name.startswith("spacy."):
                raise ModuleNotFoundError(name)
            return real_import(name, globals, locals, fromlist, level)

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


class TestStaleRosterRefresh:
    """A stored record must not replay genders derived by older inference."""

    def _inspection(self, text: str) -> kk.BookInspection:
        return kk.BookInspection(
            kk.BookMetadata("T", "A", cover_available=False),
            (kk.ChapterInspection("ch-1", 0, "One", len(text), text),),
        )

    def _record(self, text: str, gender: str | None) -> AttributionRecord:
        return _characters.store.AttributionRecord(
            attribution_id="a" * 64,
            book_id="b" * 64,
            model_id="spacy",
            prompt_version="characters-v5",
            params={},
            characters=(CharacterProfile("egwene", "Egwene", gender, 40, ("ch-1",)),),
            spans=(SpeakerSpan("ch-1", 0, len(text), "egwene"),),
        )

    def test_a_stale_gender_is_re_derived_from_the_current_roster(self) -> None:
        """The cache key cannot see that gender inference changed.

        It keys on the parser, the normalizer, the prompt and the models --
        none of which move when the roster's own logic is fixed. A record
        written by the old inference would otherwise re-cast the whole book
        the old way, which is the defect the fix was for.
        """
        stale = self._record(SEPARATED, None)
        inspection = self._inspection(SEPARATED)

        refreshed = _characters._with_current_roster(stale, inspection, "spacy")  # noqa: SLF001

        assert refreshed.characters[0].gender == "feminine"

    def test_the_expensive_spans_are_never_re_bought(self) -> None:
        """Gender never enters the attribution prompt, so the spans still hold."""
        stale = self._record(SEPARATED, None)
        inspection = self._inspection(SEPARATED)

        refreshed = _characters._with_current_roster(stale, inspection, "spacy")  # noqa: SLF001

        assert refreshed.spans == stale.spans

    def test_the_refreshed_genders_reach_the_store(self) -> None:
        """A refresh that only lives in memory leaves the store contradicting itself.

        The record returned here decides the cast, so the render is correct
        either way. But the row on disk still says what the old inference said,
        so anything reading the store -- `list_castings`, a series' pins, an
        operator inspecting it -- sees a gender the render did not use.
        """
        text = SEPARATED
        inspection = self._inspection(text)
        stale = self._record(text, None)
        _characters.store.write_attribution(stale)

        _characters._with_current_roster(stale, inspection, "spacy")  # noqa: SLF001

        stored = _characters.store.read_attribution(stale.attribution_id)
        assert stored is not None
        assert stored.characters[0].gender == "feminine"

    def test_an_unchanged_roster_is_not_rewritten(self) -> None:
        """Nothing to correct means no reason to rewrite every span row."""
        text = SEPARATED
        inspection = self._inspection(text)
        current = self._record(text, "feminine")
        _characters.store.write_attribution(current)
        before = _characters.store.read_attribution(current.attribution_id)

        _characters._with_current_roster(current, inspection, "spacy")  # noqa: SLF001

        assert _characters.store.read_attribution(current.attribution_id) == before

    def test_a_model_roster_is_left_alone(self) -> None:
        """Refreshing an LLM roster would cost what this exists to avoid."""
        stale = self._record(SEPARATED, None)
        inspection = self._inspection(SEPARATED)

        untouched = _characters._with_current_roster(  # noqa: SLF001
            stale, inspection, "openrouter/deepseek/deepseek-v4-flash"
        )

        assert untouched == stale
