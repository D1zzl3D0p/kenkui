"""First-person narration is found by its dialogue tags, or not at all."""
# ruff: noqa: RUF001 - the typographic quotes are the data under test;
# writing them as escapes would make every case unreadable.

from __future__ import annotations

import json
import zipfile
from typing import TYPE_CHECKING

import kenkui as kk
from kenkui._characters import _measured, _roster_for, resolve_attribution
from kenkui._characters.attribution import _resolve
from kenkui._characters.narration import first_person_tags, is_first_person
from kenkui._characters.quotes import extract_spans
from kenkui._domain.planning import SpeakerSpan

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

FIRST = '"I will not," I said. "You know that." She turned away.'
THIRD = '"I will not," Anne said. "You know that." She turned away.'


def _epub_with(tmp_path: Path, body: str) -> Path:
    """Write a one-chapter EPUB containing *body*."""
    path = tmp_path / "book.epub"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?><container version="1.0" '
            'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="c.opf" '
            'media-type="application/oebps-package+xml"/></rootfiles></container>',
        )
        archive.writestr(
            "c.opf",
            '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" '
            'version="3.0" unique-identifier="i"><metadata '
            'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:identifier '
            'id="i">x</dc:identifier><dc:title>T</dc:title>'
            "<dc:language>en</dc:language></metadata><manifest>"
            '<item id="a" href="a.xhtml" media-type="application/xhtml+xml"/>'
            '</manifest><spine><itemref idref="a"/></spine></package>',
        )
        archive.writestr(
            "a.xhtml",
            '<html xmlns="http://www.w3.org/1999/xhtml"><body><p>'
            + body
            + "</p></body></html>",
        )
    return path


def _ends(text: str) -> list[int]:
    return [s.end for s in extract_spans("ch", text) if s.is_dialogue]


def test_first_person_tag_is_counted() -> None:
    """`"..." I said` is the plain first-person tag this module exists for."""
    assert first_person_tags(FIRST, _ends(FIRST)) == 1


def test_third_person_tag_is_not_counted() -> None:
    """A named speaker must not be mistaken for the narrator."""
    assert first_person_tags(THIRD, _ends(THIRD)) == 0


def test_inverted_tag_is_counted() -> None:
    """`said I` is the other order English puts a first-person tag in."""
    text = '"I will not," said I. "You know that."'
    assert first_person_tags(text, _ends(text)) == 1


def test_a_book_needs_several_tags_to_qualify() -> None:
    """One match is noise; the minimum is what turns a book first-person."""
    assert is_first_person(FIRST, _ends(FIRST)) is False
    doubled = FIRST * 3
    assert is_first_person(doubled, _ends(doubled)) is True


def test_curly_single_interrupted_tag_is_not_counted() -> None:
    """The scan must not run through a closing curly single quote.

    quotes.py treats the curly single pair as a dialogue delimiter for
    British prose, so the tag here is "she said" -- the "I said" that
    follows belongs to the next, separate quote.
    """
    text = "‘Well,’ she said, ‘I said nothing.’"
    assert first_person_tags(text, _ends(text)) == 0


def test_curly_single_back_to_back_quotes_are_not_counted() -> None:
    """A closing quote immediately followed by another must not be bridged."""
    text = "‘Stop it.’ ‘I told you already,’ he shouted at her sister."
    assert first_person_tags(text, _ends(text)) == 0


def test_curly_single_first_person_tag_is_still_counted() -> None:
    """The fix for the bridging bug must not over-correct into a miss."""
    text = "‘I will not,’ I said."
    assert first_person_tags(text, _ends(text)) == 1


def test_curly_apostrophe_contraction_before_the_verb_is_counted() -> None:
    """A professionally typeset contraction must not read as a closing quote.

    EPUBs typeset every apostrophe as U+2019, the same glyph that closes
    British dialogue. `I'd` here is a contraction, not a second quote, and
    excluding U+2019 outright -- the fix for the bridging bug -- must not
    cost this tag.
    """
    text = '"Stop it," I’d said.'
    assert first_person_tags(text, _ends(text)) == 1


def test_curly_apostrophe_contraction_after_the_verb_is_counted() -> None:
    """A contraction elsewhere in the window must not block the real tag."""
    text = '"Stop it," I said, though I’d rather not.'
    assert first_person_tags(text, _ends(text)) == 1


class NarratedClient:
    """A model that names a narrator when asked, and attributes to them."""

    def __init__(self) -> None:
        """Track roster prompts so the test can inspect what was asked."""
        self.roster_prompts: list[str] = []

    def complete(self, model: str, prompt: str) -> str:
        """Answer a roster prompt with a named narrator, else attribute to them."""
        assert model
        if "List the speaking characters" in prompt:
            self.roster_prompts.append(prompt)
            return json.dumps(
                {
                    "characters": [
                        {"id": "nieshka", "name": "Nieshka", "gender": "feminine"}
                    ],
                    "narrator": "nieshka",
                }
            )
        return json.dumps({"attributions": [{"quote_id": 0, "speaker": "narrator"}]})


def test_roster_prompt_asks_for_a_narrator_when_first_person(tmp_path: Path) -> None:
    """A first-person chapter's roster prompt asks who narrates it."""
    text = '"I will not," I said. "You know that." ' * 3
    book = kk.epub(_epub_with(tmp_path, text))
    client = NarratedClient()
    record = resolve_attribution(
        book.inspect(), "b" * 64, "m/x", client=client, roster_model_id="m/x"
    )
    assert any("narrator" in prompt for prompt in client.roster_prompts)
    assert any(span.character_id == "nieshka" for span in record.spans)


def test_bare_narrator_resolves_to_the_narrating_character() -> None:
    """The bare word "narrator" maps to the book's narrating character."""
    assert _resolve("narrator", frozenset({"nieshka"}), "nieshka") == "nieshka"


def test_bare_narrator_is_unknown_without_a_narrator() -> None:
    """Without a narrator, the bare word "narrator" resolves to nobody."""
    assert _resolve("narrator", frozenset({"nieshka"}), None) is None


def test_a_pronoun_is_still_refused() -> None:
    """A narrator on the book does not open the door to pronoun answers."""
    assert _resolve("she", frozenset({"nieshka"}), "nieshka") is None


class _RosterClient:
    """Answers every roster prompt with a fixed payload."""

    def __init__(self, payload: Mapping[str, object]) -> None:
        """Store the payload to hand back regardless of prompt content."""
        self.payload = payload

    def complete(self, model: str, prompt: str) -> str:
        """Return the fixed payload, ignoring the prompt entirely."""
        assert model
        assert prompt
        return json.dumps(self.payload)


def test_roster_for_rejects_a_narrator_not_on_its_own_roster() -> None:
    """A narrator id the model did not also list cannot be vouched for."""
    payload = {
        "characters": [{"id": "nieshka", "name": "Nieshka"}],
        "narrator": "someone-else",
    }
    _roster, narrator = _roster_for(
        "text", "m/x", _RosterClient(payload), first_person=True
    )
    assert narrator is None


def test_roster_for_narrator_absent_or_null() -> None:
    """A model that omits or nulls the narrator key names nobody."""
    absent = {"characters": [{"id": "nieshka", "name": "Nieshka"}]}
    null = {"characters": [{"id": "nieshka", "name": "Nieshka"}], "narrator": None}
    for payload in (absent, null):
        _roster, narrator = _roster_for(
            "text", "m/x", _RosterClient(payload), first_person=True
        )
        assert narrator is None


def test_roster_for_ignores_a_claimed_narrator_outside_first_person() -> None:
    """A narrator is only ever asked for, and only ever accepted, in first person."""
    payload = {
        "characters": [{"id": "nieshka", "name": "Nieshka"}],
        "narrator": "nieshka",
    }
    _roster, narrator = _roster_for(
        "text", "m/x", _RosterClient(payload), first_person=False
    )
    assert narrator is None


def test_narrator_orphaned_by_roster_merge_is_never_attributed() -> None:
    """A narrator vouched by one chapter's roster can be folded by identity.

    Chapter one vouches "nieshka" against its own roster and votes them
    narrator. Chapter two names the same person by their full name under a
    different id, and merge_rosters folds the bare form into it -- exactly
    the short-form folding merge_rosters exists to do -- leaving "nieshka"
    absent from the merged roster. The orphaned id must never reach a span:
    every span.character_id is either None or present in record.characters.
    """
    ch1 = '"I will not," I said. "You know that." ' * 3
    ch2 = 'Nieshka Fullname entered the room. "Hello," she said.'

    class FoldingClient:
        """Names a narrator in chapter one that chapter two's roster folds away."""

        def complete(self, model: str, prompt: str) -> str:
            """Route by prompt kind and, for rosters, by which chapter it is."""
            assert model
            if "List the speaking characters" in prompt:
                if "I will not" in prompt:
                    return json.dumps(
                        {
                            "characters": [{"id": "nieshka", "name": "Nieshka"}],
                            "narrator": "nieshka",
                        }
                    )
                return json.dumps(
                    {
                        "characters": [
                            {"id": "nieshka-fullname", "name": "Nieshka Fullname"}
                        ]
                    }
                )
            return json.dumps(
                {"attributions": [{"quote_id": 0, "speaker": "narrator"}]}
            )

    chapter1 = kk.ChapterInspection("ch1", 0, "One", len(ch1), ch1)
    chapter2 = kk.ChapterInspection("ch2", 1, "Two", len(ch2), ch2)
    metadata = kk.BookMetadata("T", "A", cover_available=False)
    book = kk.BookInspection(metadata, (chapter1, chapter2))

    record = resolve_attribution(
        book, "c" * 64, "m/x", client=FoldingClient(), roster_model_id="m/x"
    )
    known = {character.id for character in record.characters}
    # The fold really happened, or the invariant below would pass vacuously.
    assert "nieshka" not in known
    assert all(
        span.character_id is None or span.character_id in known for span in record.spans
    )


def test_a_role_answer_is_scoped_to_its_chapter() -> None:
    """A role the text identifies but never names is scoped to its chapter."""
    resolved = _resolve("guard", frozenset({"rand"}), None, chapter_id="ch-1")
    assert resolved == "role:guard@ch-1"


def test_the_same_role_in_two_chapters_is_two_characters() -> None:
    """The same role word in different chapters mints two distinct ids."""
    first = _resolve("guard", frozenset(), None, chapter_id="ch-1")
    second = _resolve("guard", frozenset(), None, chapter_id="ch-2")
    assert first != second


def test_a_roster_character_still_wins_over_a_role() -> None:
    """A roster id shaped like a role resolves to the character, not a role."""
    assert _resolve("rand", frozenset({"rand"}), None, chapter_id="ch-1") == "rand"


def test_a_name_outside_the_roster_becomes_a_role() -> None:
    """A speaker the roster missed still gets a voice of their own.

    Previously the vocabulary was a closed 34-word list, so a speaker the
    text names plainly -- "a militsya officer", "Professor Rochambeaux" --
    had no legal token and resolved to unknown, which is read in the
    narrator's voice. An open vocabulary casts a hallucinated name too; a
    wrong-but-distinct voice is the better failure.
    """
    assert (
        _resolve("Rochambeaux", frozenset({"dhatt"}), None, chapter_id="ch13")
        == "role:rochambeaux@ch13"
    )
    assert (
        _resolve("officer", frozenset(), None, chapter_id="ch13") == "role:officer@ch13"
    )


def test_a_pronoun_is_never_minted_as_a_role() -> None:
    """role:he@ch13 would collapse every male speaker into one voice."""
    assert _resolve("he", frozenset(), None, chapter_id="ch13") is None
    assert _resolve("She", frozenset(), None, chapter_id="ch13") is None
    assert _resolve("they", frozenset(), None, chapter_id="ch13") is None


def test_unknown_is_still_unknown() -> None:
    """The reserved word means the model declined, not that it named someone."""
    assert _resolve("unknown", frozenset(), None, chapter_id="ch13") is None


def test_no_chapter_to_scope_to_means_no_role() -> None:
    """A role is only meaningful scoped to the chapter that minted it."""
    assert _resolve("Rochambeaux", frozenset(), None, chapter_id=None) is None


def test_a_role_id_synthesised_by_measured_satisfies_the_invariant() -> None:
    """A role id minted during attribution appears in record.characters too.

    Every span.character_id is either None or present in record.characters --
    the invariant tasks 4 and 5 established. Role ids are minted during
    attribution and are never listed on any chapter's roster, so _measured
    must synthesise a CharacterProfile for each one it finds in the spans, or
    an orphaned role id would reach a span with nothing behind it.
    """
    text = '"Halt!" the guard said.'

    class RoleClient:
        """Rosters one unrelated character; attributes the quote to a role."""

        def complete(self, model: str, prompt: str) -> str:
            """Route by prompt kind: roster names "rand", attribution "guard"."""
            assert model
            if "List the speaking characters" in prompt:
                return json.dumps({"characters": [{"id": "rand", "name": "Rand"}]})
            return json.dumps({"attributions": [{"quote_id": 0, "speaker": "guard"}]})

    chapter = kk.ChapterInspection("ch1", 0, "One", len(text), text)
    metadata = kk.BookMetadata("T", "A", cover_available=False)
    book = kk.BookInspection(metadata, (chapter,))

    record = resolve_attribution(
        book, "e" * 64, "m/x", client=RoleClient(), roster_model_id="m/x"
    )
    known = {character.id for character in record.characters}
    assert "role:guard@ch1" in known
    assert any(span.character_id == "role:guard@ch1" for span in record.spans)
    assert all(
        span.character_id is None or span.character_id in known for span in record.spans
    )


def test_gendered_role_words_carry_their_own_gender() -> None:
    """The role woman is feminine because the text said so.

    Seven of the fifteen largest ungendered entries in a real run were role
    words that state a gender outright. Synthesising them as unknown sent
    each one to the whole voice pool, which is a coin flip on a speaker the
    text had already identified.
    """
    spans = (
        SpeakerSpan("ch3", 0, 10, "role:woman@ch3"),
        SpeakerSpan("ch3", 10, 20, "role:old-man@ch3"),
        SpeakerSpan("ch3", 20, 30, "role:young-woman@ch3"),
        SpeakerSpan("ch3", 30, 40, "role:innkeeper@ch3"),
    )
    profiles = {character.id: character for character in _measured((), spans)}
    assert profiles["role:woman@ch3"].gender == "feminine"
    assert profiles["role:old-man@ch3"].gender == "masculine"
    assert profiles["role:young-woman@ch3"].gender == "feminine"


def test_an_ungendered_role_word_stays_unknown() -> None:
    """An innkeeper may be anyone; casting them from a pool would be a guess."""
    spans = (SpeakerSpan("ch3", 0, 10, "role:innkeeper@ch3"),)
    profiles = {character.id: character for character in _measured((), spans)}
    assert profiles["role:innkeeper@ch3"].gender is None


def test_present_tense_first_person_tags_are_counted() -> None:
    """A narrator writing now, not remembering.

    Measured on one such novel: "I say" 161 times against "I said" 6. Matching
    only the past tense read the whole book as third-person and left its
    narrator uncast.
    """
    for tag in ("“Go,” I say.", "“Why?” I ask.", "“Fine,” I reply."):
        end = tag.index("”") + 1
        assert first_person_tags(tag, [end]) == 1, tag


def test_past_tense_first_person_tags_still_count() -> None:
    """What the widening was widened from, pinned so it cannot be dropped."""
    for tag in ("“Go,” I said.", "“Why?” I asked."):
        end = tag.index("”") + 1
        assert first_person_tags(tag, [end]) == 1, tag


def test_third_person_present_is_not_a_first_person_tag() -> None:
    """Widening to bare verbs must not make "he says" first person."""
    for tag in ("“Go,” he says.", "“Now,” she asks.", "“Fine,” Darrow says."):
        end = tag.index("”") + 1
        assert first_person_tags(tag, [end]) == 0, tag


def test_a_present_tense_chapter_reads_as_first_person() -> None:
    """Enough present-tense tags turn the chapter, exactly as past ones do."""
    text = "“Go,” I say. “Now,” I say. “Please,” I ask."
    ends = [index + 1 for index, char in enumerate(text) if char == "”"]
    assert is_first_person(text, ends)
