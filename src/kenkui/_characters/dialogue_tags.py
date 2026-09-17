"""Gender a speaker from the pronoun in their own dialogue tag.

Attribution has already decided who spoke each quote. A tag like `"..." she
said` therefore states that speaker's gender directly, and with none of the
proximity noise the roster's pronoun signal has to tolerate: the pronoun is the
tag's grammatical subject, and the tag belongs to the quote.

The yield depends entirely on how an author writes tags, and varies far more
than it might seem. Counted over three books: Dune 698 pronoun tags, Red Rising
156, and The Subtle Art of Folding Space just 3 -- that last one carries its
dialogue on noun phrases ("the woman says") and subordinate clauses ("As Ellie
says this") almost exclusively.

So this is a cross-check rather than a primary signal. It is nearly free, it is
almost error-free where it fires, and it concentrates on the characters who
speak most, which are the ones a cast must get right. But a book can offer
almost none of it, and `spacy_roster` has to stand on its own.

Regex rather than a second parse: `spacy_roster` has already put the book
through spaCy by the time attribution runs, and re-parsing it to read a handful
of two-word tags would cost more than the signal is worth.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

from kenkui._characters.narration import _VERBS
from kenkui.observability import get_logger, log_event

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from kenkui._domain.casting import CharacterProfile
    from kenkui._domain.planning import SpeakerSpan
    from kenkui.inspection import ChapterInspection

_LOGGER = get_logger(__name__)

# Votes required to overturn a known gender, and the margin the winner must
# hold. An unknown can use a single unopposed tag: minor speakers may only
# have one line, and the alternative is selecting from the entire voice pool.
_TAG_MINIMUM = 3
_TAG_MARGIN = 2
# How much narration either side of a quote can hold its tag. A tag sits
# immediately against the quote; anything further away is a new sentence.
_TAG_WINDOW = 120

# "she said" and "said she" -- the two orders English puts a tag in. Anchored to
# the quote edge so only an adjacent tag matches: `_AFTER` allows nothing but
# whitespace and a comma before the pronoun, and `_BEFORE` nothing but
# whitespace and punctuation after the verb.
_ADVERB = r"(?:(?:quietly|softly|loudly|sharply|gently|firmly|finally)\s+)?"
_SUBJECT = r"(?:she|he)"
_TAG = rf"(?:{_SUBJECT}\s+{_ADVERB}(?:{_VERBS})|(?:{_VERBS})\s+{_SUBJECT})"
_AFTER = re.compile(
    rf"^[\s,]*(?P<tag>{_TAG})\b",
    re.IGNORECASE,
)
_BEFORE = re.compile(
    rf"\b(?P<tag>{_TAG})\b[\s,:]*$",
    re.IGNORECASE,
)


def _tag_gender(before: str, after: str) -> str | None:
    """Return the gender a quote's adjacent tag states, or None when it has none.

    The trailing tag is consulted first because it is the commoner form and the
    less ambiguous: text before a quote may end in a tag belonging to an
    earlier quote, while text after one begins at the tag or has none.
    """
    for pattern, text in ((_AFTER, after), (_BEFORE, before)):
        match = pattern.search(text)
        if match is not None:
            return (
                "feminine"
                if "she" in match.group("tag").lower().split()
                else "masculine"
            )
    return None


def tag_genders(
    chapters: Sequence[ChapterInspection], spans: Sequence[SpeakerSpan]
) -> dict[str, Counter[str]]:
    """Tally gender votes per character from the tags beside their own quotes."""
    text_by_id = {chapter.id: chapter.text for chapter in chapters}
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    for span in spans:
        if span.character_id is None:
            continue  # narration has no speaker to gender
        text = text_by_id.get(span.chapter_id)
        if text is None:
            continue
        gender = _tag_gender(
            text[max(0, span.start - _TAG_WINDOW) : span.start],
            text[span.end : span.end + _TAG_WINDOW],
        )
        if gender is not None:
            votes[span.character_id][gender] += 1
    return dict(votes)


def _confident(tally: Counter[str]) -> str | None:
    """Return the tag-sourced gender when the vote is clear, else None."""
    top = tally.most_common(1)
    if not top or top[0][1] < _TAG_MINIMUM:
        return None
    winner, count = top[0]
    other = tally["masculine" if winner == "feminine" else "feminine"]
    return winner if count >= _TAG_MARGIN * other else None


def apply(
    characters: Sequence[CharacterProfile], votes: Mapping[str, Counter[str]]
) -> tuple[CharacterProfile, ...]:
    """Let a confident tag vote decide, superseding whatever the roster read.

    The roster's pronoun signal is proximity -- a guess that the nearest pronoun
    refers to this name. A dialogue tag's pronoun *is* the speaker's, so where
    the two disagree the tag is the better evidence, and filling only the gaps
    the roster left would discard the better answer in the cases that matter.

    The cost of that is real and worth naming: a mis-attributed quote can now
    turn a correct gender wrong, where gap-filling could only help. The
    threshold above is the guard, and the disagreement is logged rather than
    swallowed -- the only way a tag is wrong is that a quote went to the wrong
    mouth, and that character's lines are already in the wrong voice if so. The
    log line names the bug rather than the symptom.
    """
    decided: list[CharacterProfile] = []
    for character in characters:
        gender = character.gender
        tally = votes.get(character.id) or Counter()
        answer = _confident(tally)
        if answer is None and tally["masculine"] and tally["feminine"]:
            log_event(
                _LOGGER,
                "dialogue_tag_gender_ambiguous",
                level=logging.WARNING,
                context={
                    "boundary": "characters",
                    "character": character.id,
                    "feminine_votes": tally["feminine"],
                    "masculine_votes": tally["masculine"],
                },
            )
        if gender is None and len(+tally) == 1:
            answer = tally.most_common(1)[0][0]
        if answer is not None:
            if gender is not None and gender != answer:
                log_event(
                    _LOGGER,
                    "dialogue_tag_gender_conflict",
                    level=logging.WARNING,
                    # LogContext is str | int | bool, so the tally is flattened
                    # rather than passed as a mapping.
                    context={
                        "boundary": "characters",
                        "character": character.id,
                        "roster": gender,
                        "tags": answer,
                        "feminine_votes": tally["feminine"],
                        "masculine_votes": tally["masculine"],
                    },
                )
            gender = answer
        decided.append(
            type(character)(
                id=character.id,
                display_name=character.display_name,
                gender=gender,
                spoken_characters=character.spoken_characters,
                chapter_ids=character.chapter_ids,
                aliases=character.aliases,
            )
        )
    return tuple(decided)
