"""Character inference, dialogue attribution, and their durable store.

The only part of Kenkui that calls a language model. Everything here runs in
the parent process during resolution, never inside a spawned render worker, so
the render path's offline posture is untouched: workers deny sockets outright.

`resolve_attribution` is the single entry point. It returns a stored result
when one exists and derives one otherwise, exactly as voice resolution returns
a provisioned asset or fetches it. The pure planner receives the finished
value and never reaches a model or the store itself.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

from kenkui._characters import store
from kenkui._characters.attribution import (
    AttributionCoverage,
    attribute_chapter,
)
from kenkui._characters.infer import merge_rosters, normalise_roster, slugify
from kenkui._characters.llm import complete_json
from kenkui._characters.narration import is_first_person
from kenkui._characters.prompts import (
    PROMPT_VERSION,
    ROLE_GENDERS,
    ROSTER_PROMPT,
)
from kenkui._characters.quotes import extract_spans
from kenkui._characters.store import AttributionRecord
from kenkui._domain.casting import (
    CastingOutcome,
    CastingRequest,
    CharacterProfile,
    solve,
)
from kenkui._domain.planning import (
    NORMALIZATION_SCHEMA_VERSION,
    PARSER_SCHEMA_VERSION,
)
from kenkui.errors import ModelError
from kenkui.observability import get_logger, log_event

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from kenkui._characters.llm import Client
    from kenkui._domain.planning import SpeakerSpan
    from kenkui.cancellation import CancellationToken
    from kenkui.inspection import BookInspection

__all__ = ["AttributionRecord", "resolve_attribution", "store"]

# Fixed, and part of the store key: a different temperature is a different
# derivation, not a cache hit.
PARAMS: Mapping[str, object] = {"temperature": 0.0}

_LOGGER = get_logger(__name__)

_ROSTER_SCHEMA: Mapping[str, type] = {"characters": list}

# Above this share of a chapter dropped, the response is the problem
# rather than the passage.
_DROPPED_WARN_RATIO = 0.05


def _roster_for(
    chapter_text: str,
    model_id: str,
    client: Client | None,
    *,
    first_person: bool,
) -> tuple[tuple[CharacterProfile, ...], str | None]:
    """Infer one chapter's characters, and who narrates it when asked."""
    escaped = chapter_text.replace("{", "{{").replace("}", "}}")
    try:
        payload = complete_json(
            model_id,
            ROSTER_PROMPT.format(passage=escaped),
            _ROSTER_SCHEMA,
            client=client,
        )
    except ModelError:
        return (), None
    roster = normalise_roster(payload["characters"])
    narrator = None
    if first_person:
        claimed = payload.get("narrator")
        if isinstance(claimed, str) and claimed.strip():
            candidate = slugify(claimed)
            # Only a character the model also listed. A narrator who is not
            # on the roster cannot be cast, and inventing an entry here would
            # put an unvouched id into the plan fingerprint.
            if any(character.id == candidate for character in roster):
                narrator = candidate
    return roster, narrator


def _log_coverage(chapter_id: str, coverage: AttributionCoverage) -> None:
    """Record how one chapter's quotes were accounted for.

    A dropped quote is a defect in the response, not caution: it means the
    model never mentioned that id at all. Kept separate from "unknown" so a
    chapter answering badly is visible without reading the book, and logged
    for an operator rather than raised, since the caller cannot act on it.
    """
    if not coverage.quotes:
        return
    log_event(
        _LOGGER,
        "attribution_coverage",
        level=(
            logging.WARNING
            if coverage.dropped > coverage.quotes * _DROPPED_WARN_RATIO
            else logging.INFO
        ),
        context={
            "boundary": "attribution",
            "chapter_id": chapter_id,
            "quotes": coverage.quotes,
            "answered": coverage.answered,
            "unknown": coverage.unknown,
            "dropped": coverage.dropped,
        },
    )


def _measured(
    characters: Sequence[CharacterProfile],
    spans: Sequence[SpeakerSpan],
) -> tuple[CharacterProfile, ...]:
    """Fill in each character's speech volume and the chapters they speak in.

    Casting weights by spoken characters and forbids two speakers in one
    chapter from sharing a voice, so both fields have to be real rather than
    the zeros inference leaves behind.
    """
    volume: dict[str, int] = {}
    chapters: dict[str, list[str]] = {}
    for span in spans:
        if span.character_id is None:
            continue
        volume[span.character_id] = volume.get(span.character_id, 0) + (
            span.end - span.start
        )
        seen = chapters.setdefault(span.character_id, [])
        if span.chapter_id not in seen:
            seen.append(span.chapter_id)
    named = tuple(
        type(character)(
            id=character.id,
            display_name=character.display_name,
            gender=character.gender,
            spoken_characters=volume.get(character.id, 0),
            chapter_ids=tuple(chapters.get(character.id, ())),
        )
        for character in characters
        # A character nobody attributed anything to cannot be cast, and would
        # otherwise consume a voice from a pool that is not deep enough to
        # waste one.
        if character.id in volume
    )
    # Roles are minted during attribution, never listed on any chapter's
    # roster, so they need synthesising here or the invariant that every
    # span.character_id is either None or present in characters would break.
    roles = {
        span.character_id
        for span in spans
        if span.character_id is not None and span.character_id.startswith("role:")
    }
    synthesised = tuple(
        CharacterProfile(
            id=role,
            display_name=role.removeprefix("role:").split("@")[0].replace("-", " "),
            gender=ROLE_GENDERS.get(role.removeprefix("role:").split("@")[0]),
            spoken_characters=volume.get(role, 0),
            chapter_ids=tuple(chapters.get(role, ())),
        )
        for role in sorted(roles)
    )
    return (*named, *synthesised)


def resolve_cast(record: AttributionRecord, request: CastingRequest) -> CastingOutcome:
    """Solve one cast and store it beneath the attribution it was solved from.

    Solving is pure and cheap, so the row is not a cache: it is what makes the
    cast a thing the caller can name later. ``list_castings`` and
    ``remove_casting`` are public verbs, and without this write the first
    always answers empty and the second is always a no-op.

    The write is allowed to fail loud, matching the store's own rule: losing a
    cast silently means a later render re-casts the book without saying so.
    """
    outcome = solve(request)
    store.write_cast(
        store.CastRecord(
            cast_id=store.cast_key(
                record.attribution_id,
                request.method,
                request.explicit,
                request.narrator_voice_id,
                request.unknown_voice_id,
            ),
            attribution_id=record.attribution_id,
            method=request.method,
            narrator_voice_id=request.narrator_voice_id,
            unknown_voice_id=request.unknown_voice_id,
            assignments=tuple(
                (character_id, voice_id, character_id in request.explicit)
                for character_id, voice_id in sorted(outcome.assignments.items())
            ),
        )
    )
    return outcome


def resolve_attribution(  # noqa: PLR0913 - one call site, all inputs explicit.
    inspection: BookInspection,
    source_hash: str,
    model_id: str,
    *,
    roster_model_id: str | None = None,
    client: Client | None = None,
    cancel: CancellationToken | None = None,
) -> AttributionRecord:
    """Return stored attribution for this book and model, else derive it.

    Called only from the shell. Cancellation is checked between chapters
    because a long book is a long sequence of model calls, and the only other
    check happens once before rendering starts.

    ``roster_model_id`` names the model that infers characters, which the
    caller may choose independently of the one that attributes quotes. It
    keys the record too: a roster derived by a different model is different
    material, not a cache hit.
    """
    roster_model = roster_model_id or model_id
    key = store.attribution_key(
        source_hash,
        model_id,
        PROMPT_VERSION,
        PARAMS,
        tuple(chapter.id for chapter in inspection.chapters),
        (PARSER_SCHEMA_VERSION, NORMALIZATION_SCHEMA_VERSION),
        roster_model_id=roster_model,
    )
    cached = store.read_attribution(key)
    if cached is not None:
        return cached

    # Extracted once and reused: both passes below need the same partition,
    # and scanning a 600k-character book twice for it is pure waste.
    extracted = {
        chapter.id: extract_spans(chapter.id, chapter.text)
        for chapter in inspection.chapters
    }

    rosters: list[tuple[CharacterProfile, ...]] = []
    narrators: Counter[str] = Counter()
    for chapter in inspection.chapters:
        if cancel is not None:
            cancel.raise_if_cancelled()
        # A chapter with no quoted speech has no speaker to attribute, so its
        # roster is never consulted. Front matter and purely descriptive
        # chapters are common enough that asking about them is real spend.
        spans_here = extracted[chapter.id]
        if not any(span.is_dialogue for span in spans_here):
            continue
        ends = [span.end for span in spans_here if span.is_dialogue]
        roster, narrator = _roster_for(
            chapter.text,
            roster_model,
            client,
            first_person=is_first_person(chapter.text, ends),
        )
        rosters.append(roster)
        if narrator is not None:
            narrators[narrator] += 1
    characters = merge_rosters(tuple(rosters))
    # One narrator per book: chapters that disagree are outvoted rather than
    # producing a second narrating character.
    narrator_id = narrators.most_common(1)[0][0] if narrators else None
    # Vouched against the chapter roster that named them, not the merged
    # one: alias folding can rename or drop that exact id. A narrator who
    # did not survive the fold cannot be marked in the prompt, or the id
    # would reach a span with no character behind it.
    if narrator_id is not None and not any(
        character.id == narrator_id for character in characters
    ):
        narrator_id = None

    spans: list[SpeakerSpan] = []
    recent: tuple[str, ...] = ()
    for chapter in inspection.chapters:
        if cancel is not None:
            cancel.raise_if_cancelled()
        chapter_spans, recent, coverage = attribute_chapter(
            chapter,
            characters,
            model_id,
            client=client,
            recent=recent,
            spans=extracted[chapter.id],
            narrator_id=narrator_id,
        )
        _log_coverage(chapter.id, coverage)
        spans.extend(chapter_spans)

    record = AttributionRecord(
        attribution_id=key,
        book_id=source_hash,
        model_id=model_id,
        prompt_version=PROMPT_VERSION,
        params=PARAMS,
        characters=_measured(characters, spans),
        spans=tuple(spans),
    )
    store.write_attribution(record)
    return record
