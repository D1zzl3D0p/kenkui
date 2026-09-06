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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import TYPE_CHECKING

from kenkui._characters import dialogue_tags, spacy_roster, store
from kenkui._characters.attribution import (
    AttributionCoverage,
    attribute_chapter,
)
from kenkui._characters.infer import (
    ROLE_PREFIX,
    merge_rosters,
    normalise_roster,
    slugify,
)
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
    from kenkui._characters.quotes import TextSpan
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
# Chapters are independent model calls; this bounds how many are in flight.
# The upstream provider queues rather than refusing (no 429s at 24), so the
# bound trades tail latency for throughput instead of rate-limit safety.
_ATTRIBUTION_CONCURRENCY = 12


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
            aliases=character.aliases,
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
        if span.character_id is not None and span.character_id.startswith(ROLE_PREFIX)
    }
    synthesised = tuple(
        CharacterProfile(
            id=role,
            display_name=role.removeprefix(ROLE_PREFIX).split("@")[0].replace("-", " "),
            gender=ROLE_GENDERS.get(role.removeprefix(ROLE_PREFIX).split("@")[0]),
            spoken_characters=volume.get(role, 0),
            chapter_ids=tuple(chapters.get(role, ())),
            # A role is minted with one surface form -- its display name --
            # never a roster entry with variant names to fold together.
            aliases=(role.removeprefix(ROLE_PREFIX).split("@")[0].replace("-", " "),),
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


def _model_roster(
    inspection: BookInspection,
    extracted: Mapping[str, tuple[TextSpan, ...]],
    roster_model: str,
    client: Client | None,
    cancel: CancellationToken | None,
) -> tuple[tuple[CharacterProfile, ...], str | None]:
    """Infer the roster by asking a model about each chapter in turn.

    One call per chapter that contains dialogue, merged afterwards. The
    alternative offline pass is `spacy_roster.infer_roster`, which reads the
    whole book at once because it can.
    """
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

    return characters, narrator_id


def _chapter_roster(
    characters: Sequence[CharacterProfile],
    chapter_id: str,
    narrator_id: str | None,
) -> tuple[CharacterProfile, ...]:
    """Narrow the roster to the characters this chapter can plausibly contain.

    The prompt used to carry the whole book's roster into every chapter, and
    had to: the model roster is built one chapter at a time and cannot say
    where anyone appears until attribution has already run, so `chapter_ids`
    is empty at this point on that path. A roster with no placement is passed
    through untouched, or filtering it would empty every chapter.

    A spaCy roster does know. It read the whole book to build the roster, and
    recorded the chapters each name was seen in, so a chapter can be offered
    the twenty characters it contains rather than the hundred the book does.
    The narrator is kept regardless of where they were seen: they are marked
    in the prompt block and so have to be in it.
    """
    if not any(character.chapter_ids for character in characters):
        return tuple(characters)
    keep = {
        character.id for character in characters if chapter_id in character.chapter_ids
    }
    if narrator_id is not None:
        keep.add(narrator_id)
    narrowed = tuple(character for character in characters if character.id in keep)
    # A chapter whose speakers are all named by role rather than by name
    # matches nobody. Passing nothing would skip the model call and narrate
    # every line of it; passing the book lets attribution mint those roles.
    return narrowed or tuple(characters)


def _with_current_roster(
    record: AttributionRecord,
    inspection: BookInspection,
    roster_model: str,
) -> AttributionRecord:
    """Re-derive a stored record's genders without re-attributing its quotes.

    The spans are the expensive half and they are untouched by this: the
    attribution prompt carries only character ids and display names, never
    gender, so nothing about how a character is gendered can change which
    quote was assigned to whom. Keying the record on the roster's own version
    instead would be correct and ruinous -- it would re-buy every model call in
    the book to change one field per character.

    The offline roster costs 8 seconds on a 380k-character book and 18 on
    Dune, so it is simply re-run. Only the gender is transplanted, and only
    onto ids the record already holds: the cached spans refer to those ids,
    and adopting a fresh roster wholesale could leave a span pointing at a
    character no longer present.

    A model-derived roster is returned untouched, because refreshing it would
    cost exactly what this exists to avoid.
    """
    pipeline = spacy_roster.pipeline_for(roster_model)
    if pipeline is None:
        return record
    extracted = {
        chapter.id: extract_spans(chapter.id, chapter.text)
        for chapter in inspection.chapters
    }
    fresh, _ = spacy_roster.infer_roster(
        inspection.chapters, extracted, pipeline=pipeline
    )
    genders = {character.id: character.gender for character in fresh}
    characters = tuple(
        type(character)(
            id=character.id,
            display_name=character.display_name,
            gender=genders.get(character.id, character.gender),
            spoken_characters=character.spoken_characters,
            chapter_ids=character.chapter_ids,
            aliases=character.aliases,
        )
        for character in record.characters
    )
    refreshed = replace(
        record,
        characters=dialogue_tags.apply(
            characters,
            dialogue_tags.tag_genders(inspection.chapters, record.spans),
        ),
    )
    before = {character.id: character.gender for character in record.characters}
    after = {character.id: character.gender for character in refreshed.characters}
    if after != before:
        # Persist, or the store keeps asserting what the old inference said
        # while the render uses this. `list_castings`, a series' pins, and
        # anyone reading the store directly would all see a gender that no
        # render actually used. write_attribution replaces the row wholesale,
        # so this is an upsert rather than a duplicate.
        #
        # Guarded on an actual change because that rewrite also deletes and
        # reinserts every quote span -- thirteen thousand rows for Dune -- and
        # a steady-state render has nothing to correct.
        store.write_attribution(refreshed)
        log_event(
            _LOGGER,
            "attribution_roster_refreshed",
            context={
                "boundary": "characters",
                "attribution_id": refreshed.attribution_id,
                "regendered": sum(
                    1 for key, value in after.items() if before.get(key) != value
                ),
            },
        )
    return refreshed


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

    Called only from the shell. Cancellation is checked as chapter calls
    complete, because a long book is a long sequence of model calls, and the
    only other check happens once before rendering starts.

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
        return _with_current_roster(cached, inspection, roster_model)

    # Extracted once and reused: both passes below need the same partition,
    # and scanning a 600k-character book twice for it is pure waste.
    extracted = {
        chapter.id: extract_spans(chapter.id, chapter.text)
        for chapter in inspection.chapters
    }

    spacy_pipeline = spacy_roster.pipeline_for(roster_model)
    if spacy_pipeline is not None:
        # One offline pass over the whole book, replacing the per-chapter model
        # roster entirely. Nothing below this branch reaches a model, and the
        # attribution pass that follows is untouched: it still receives a
        # roster of the same shape and answers against it the same way.
        characters, narrator_id = spacy_roster.infer_roster(
            inspection.chapters, extracted, pipeline=spacy_pipeline
        )
    else:
        characters, narrator_id = _model_roster(
            inspection, extracted, roster_model, client, cancel
        )

    # Chapters are independent calls, so they run concurrently and are
    # re-ordered below: the record's spans must stay in chapter order.
    attributed: dict[str, tuple[tuple[SpeakerSpan, ...], AttributionCoverage]] = {}
    workers = min(_ATTRIBUTION_CONCURRENCY, len(inspection.chapters))
    if workers > 0:
        rosters = {
            chapter.id: _chapter_roster(characters, chapter.id, narrator_id)
            for chapter in inspection.chapters
        }
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    attribute_chapter,
                    chapter,
                    rosters[chapter.id],
                    model_id,
                    client=client,
                    spans=extracted[chapter.id],
                    narrator_id=narrator_id,
                ): chapter
                for chapter in inspection.chapters
            }
            for future in as_completed(futures):
                if cancel is not None:
                    cancel.raise_if_cancelled()
                chapter = futures[future]
                attributed[chapter.id] = future.result()
                _log_coverage(chapter.id, attributed[chapter.id][1])

    spans: list[SpeakerSpan] = []
    for chapter in inspection.chapters:
        spans.extend(attributed[chapter.id][0])

    record = AttributionRecord(
        attribution_id=key,
        book_id=source_hash,
        model_id=model_id,
        prompt_version=PROMPT_VERSION,
        params=PARAMS,
        # Attribution has just decided who speaks each quote, so `"..." she
        # said` now genders a character we can name. Applied here rather than
        # in the roster because the roster runs before any speaker is known.
        characters=dialogue_tags.apply(
            _measured(characters, spans),
            dialogue_tags.tag_genders(inspection.chapters, spans),
        ),
        spans=tuple(spans),
    )
    store.write_attribution(record)
    return record
