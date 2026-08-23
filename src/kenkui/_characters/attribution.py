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
from typing import TYPE_CHECKING

from kenkui._characters.infer import PRONOUNS, UNKNOWN
from kenkui._characters.llm import complete_json
from kenkui._characters.prompts import ATTRIBUTION_PROMPT, CONTINUITY_SPEAKERS
from kenkui._characters.quotes import TextSpan, extract_spans
from kenkui._domain.planning import SpeakerSpan
from kenkui.errors import ModelError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from kenkui._characters.llm import Client
    from kenkui._domain.casting import CharacterProfile
    from kenkui.inspection import ChapterInspection

_SCHEMA: Mapping[str, type] = {"attributions": list}


def _escaped(text: str) -> str:
    """Neutralise braces so book text survives str.format unchanged."""
    return text.replace("{", "{{").replace("}", "}}")


def _roster_block(characters: Sequence[CharacterProfile]) -> str:
    return "\n".join(
        f'- {character.id}  (appears as "{character.display_name}")'
        for character in characters
    )


def _resolve(speaker: object, known: frozenset[str]) -> str | None:
    """Return a known character id, or None meaning unknown.

    A returned pronoun is rejected outright: the prompt forbids it, and a
    pronoun that slipped through would merge unrelated speakers into one voice.
    """
    if not isinstance(speaker, str):
        return None
    candidate = speaker.strip().lower()
    if not candidate or candidate == UNKNOWN or candidate in PRONOUNS:
        return None
    return candidate if candidate in known else None


def attribute_chapter(
    chapter: ChapterInspection,
    characters: Sequence[CharacterProfile],
    model_id: str,
    *,
    client: Client | None = None,
    recent: Sequence[str] = (),
) -> tuple[tuple[SpeakerSpan, ...], tuple[str, ...]]:
    """Return one chapter's speaker spans and the speakers that ended it.

    The trailing speakers feed the next chapter's prompt, so a conversation
    running across a chapter boundary keeps its alternation.
    """
    spans = extract_spans(chapter.id, chapter.text)
    dialogue = [span for span in spans if span.is_dialogue]
    if not dialogue or not characters:
        # Nothing to attribute, so nothing is worth a model call.
        return _all_narrated(spans), tuple(recent)

    quotes = [
        {"quote_id": index, "text": chapter.text[span.start : span.end]}
        for index, span in enumerate(dialogue)
    ]
    prompt = ATTRIBUTION_PROMPT.format(
        roster=_escaped(_roster_block(characters)),
        recent=_escaped(", ".join(recent) or "(start of book)"),
        passage=_escaped(chapter.text),
        quotes=_escaped(json.dumps(quotes, ensure_ascii=False, indent=2)),
    )
    known = frozenset(character.id for character in characters)
    answers = _answers(model_id, prompt, client, known)

    # A quote's id is its position among the dialogue spans, so pairing them
    # back up is a zip. A model that skipped an id leaves None, which is
    # unknown: gaps need no special case.
    speakers = [answers.get(index) for index in range(len(dialogue))]
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
    return resolved, trailing


def _all_narrated(spans: Sequence[TextSpan]) -> tuple[SpeakerSpan, ...]:
    """Carry every span through as narration, speaker unassigned."""
    return tuple(
        SpeakerSpan(span.chapter_id, span.start, span.end, None) for span in spans
    )


def _answers(
    model_id: str, prompt: str, client: Client | None, known: frozenset[str]
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
            answers[quote_id] = _resolve(item.get("speaker"), known)
    return answers
