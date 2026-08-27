"""Assign a speaker to each dialogue span.

Extraction already decided *where* speech is. This decides *who*, and only for
dialogue: narration is never sent to a model, which keeps the prompt small and
removes a whole class of wrong answer.

Every failure lands on "unknown", which renders in the unknown voice. An
unattributed line sounds like narration, which is unremarkable; a wrongly
attributed line is audibly wrong. The asymmetry is deliberate throughout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kenkui._characters.infer import PRONOUNS, UNKNOWN, slugify
from kenkui._characters.llm import complete_json
from kenkui._characters.prompts import (
    ATTRIBUTION_PROMPT,
    CONTINUITY_SPEAKERS,
)
from kenkui._characters.quotes import TextSpan, extract_spans
from kenkui._domain.planning import SpeakerSpan
from kenkui.errors import ModelError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from kenkui._characters.llm import Client
    from kenkui._domain.casting import CharacterProfile
    from kenkui.inspection import ChapterInspection

_SCHEMA: Mapping[str, type] = {"attributions": list}

NARRATOR = "narrator"


@dataclass(frozen=True, slots=True)
class AttributionCoverage:
    """How one chapter's quotes were accounted for.

    "The model said unknown" and "the model never mentioned this quote id"
    are different failures with different fixes. Both resolved to None, so a
    truncated or malformed response was indistinguishable from ordinary
    model caution and could only be found by reading the book.
    """

    quotes: int
    answered: int
    unknown: int
    dropped: int


def _escaped(text: str) -> str:
    """Neutralise braces so book text survives str.format unchanged."""
    return text.replace("{", "{{").replace("}", "}}")


def _roster_block(
    characters: Sequence[CharacterProfile], narrator_id: str | None = None
) -> str:
    return "\n".join(
        f'- {character.id}  (appears as "{character.display_name}")'
        + ("  [narrates this book]" if character.id == narrator_id else "")
        for character in characters
    )


def _resolve(  # noqa: PLR0911 - one branch per resolution rule, kept flat.
    speaker: object,
    known: frozenset[str],
    narrator_id: str | None = None,
    *,
    chapter_id: str | None = None,
) -> str | None:
    """Return a known character id, a scoped role id, or None meaning unknown.

    A returned pronoun is rejected outright: the prompt forbids it, and a
    pronoun that slipped through would merge unrelated speakers into one voice.

    The bare word "narrator" is the one exception, and only when the book has
    one. Models shorten an id they are asked to reproduce, and a namespaced
    narrator id loses the answers it shortens; accepting the short word keeps
    them while leaving one id per person, so a narrating character's dialogue
    and their narration stay one voice.
    """
    if not isinstance(speaker, str):
        return None
    candidate = speaker.strip().lower()
    if not candidate or candidate == UNKNOWN:
        return None
    if candidate == NARRATOR:
        # Gated on membership too: a narrator vouched against one chapter's
        # roster can be folded away by merge_rosters, and an orphaned id
        # must never reach a span whatever the caller passed.
        return narrator_id if narrator_id in known else None
    if candidate in PRONOUNS:
        return None
    if candidate in known:
        return candidate
    if chapter_id is not None:
        # Any answer that is not on the roster is a speaker the roster
        # missed, not a non-answer. A closed vocabulary could not anticipate
        # "officer" or "Professor Rochambeaux", and every speaker it failed
        # to name resolved to unknown -- which is read in the narrator's
        # voice, so a first-person book renders both halves of a
        # conversation as one person.
        #
        # Scoped to the chapter because chapter 40's officer is not chapter
        # 12's. A speaker recurring across chapters therefore gets a voice
        # per chapter; these are overwhelmingly one-scene parts, and a
        # wrong-but-distinct voice beats collapsing into the narrator.
        #
        # Pronouns are refused above, so no minted role can merge unrelated
        # speakers. A hallucinated name does become a voice, which is the
        # accepted cost of the open vocabulary.
        slug = slugify(candidate)
        return f"role:{slug}@{chapter_id}" if slug else None
    return None


def attribute_chapter(  # noqa: PLR0913 - one call site, all inputs explicit.
    chapter: ChapterInspection,
    characters: Sequence[CharacterProfile],
    model_id: str,
    *,
    client: Client | None = None,
    recent: Sequence[str] = (),
    spans: tuple[TextSpan, ...] | None = None,
    narrator_id: str | None = None,
) -> tuple[tuple[SpeakerSpan, ...], tuple[str, ...], AttributionCoverage]:
    """Return one chapter's speaker spans and the speakers that ended it.

    The trailing speakers feed the next chapter's prompt, so a conversation
    running across a chapter boundary keeps its alternation.

    ``spans`` accepts an already-extracted partition, so a caller that had to
    look for dialogue before deciding to call does not pay for a second scan.
    Extraction is pure, so supplying it changes nothing but the cost.
    """
    if spans is None:
        spans = extract_spans(chapter.id, chapter.text)
    dialogue = [span for span in spans if span.is_dialogue]
    if not dialogue or not characters:
        # Nothing to attribute, so nothing is worth a model call.
        return (
            _all_narrated(spans),
            tuple(recent),
            AttributionCoverage(len(dialogue), 0, 0, 0),
        )

    quotes = [
        {"quote_id": index, "text": chapter.text[span.start : span.end]}
        for index, span in enumerate(dialogue)
    ]
    prompt = ATTRIBUTION_PROMPT.format(
        roster=_escaped(_roster_block(characters, narrator_id)),
        recent=_escaped(", ".join(recent) or "(start of book)"),
        passage=_escaped(chapter.text),
        quotes=_escaped(json.dumps(quotes, ensure_ascii=False, indent=2)),
    )
    known = frozenset(character.id for character in characters)
    answers = _answers(
        model_id, prompt, client, known, narrator_id, chapter_id=chapter.id
    )

    # A quote's id is its position among the dialogue spans, so pairing them
    # back up is a zip. A model that skipped an id leaves None, which is
    # unknown: gaps need no special case.
    speakers = [answers.get(index) for index in range(len(dialogue))]
    # A key the model never returned is a dropped quote; a key present with
    # no resolution is one it declined. Only the first is a defect.
    returned = set(answers)
    coverage = AttributionCoverage(
        quotes=len(dialogue),
        answered=sum(1 for value in answers.values() if value is not None),
        unknown=sum(1 for value in answers.values() if value is None),
        dropped=sum(1 for index in range(len(dialogue)) if index not in returned),
    )
    by_start = dict(zip((span.start for span in dialogue), speakers, strict=True))

    resolved = tuple(
        SpeakerSpan(
            span.chapter_id,
            span.start,
            span.end,
            by_start.get(span.start) if span.is_dialogue else None,
        )
        for span in spans
    )
    named = [speaker for speaker in speakers if speaker is not None]
    # A chapter where nobody could be placed carries the previous chapter's
    # speakers forward rather than resetting continuity to nothing.
    trailing = tuple(named[-CONTINUITY_SPEAKERS:]) if named else tuple(recent)
    return resolved, trailing, coverage


def _all_narrated(spans: Sequence[TextSpan]) -> tuple[SpeakerSpan, ...]:
    """Carry every span through as narration, speaker unassigned."""
    return tuple(
        SpeakerSpan(span.chapter_id, span.start, span.end, None) for span in spans
    )


def _answers(  # noqa: PLR0913 - one call site, all inputs explicit.
    model_id: str,
    prompt: str,
    client: Client | None,
    known: frozenset[str],
    narrator_id: str | None = None,
    *,
    chapter_id: str | None = None,
) -> dict[int, str | None]:
    """Return quote index to resolved speaker, tolerating a bad response.

    A model that fails or answers unusably leaves every quote unknown rather
    than stopping the render: the book still reads, in one voice for the lines
    it could not place.
    """
    try:
        payload = complete_json(model_id, prompt, _SCHEMA, client=client)
    except ModelError:
        return {}
    answers: dict[int, str | None] = {}
    for item in payload["attributions"]:
        if not isinstance(item, dict):
            continue
        quote_id = item.get("quote_id")
        if isinstance(quote_id, int) and not isinstance(quote_id, bool):
            answers[quote_id] = _resolve(
                item.get("speaker"), known, narrator_id, chapter_id=chapter_id
            )
    return answers
