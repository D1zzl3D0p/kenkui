"""Attribution turns dialogue spans into speaker spans, or gives up cleanly."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace

import pytest

import kenkui as kk
from kenkui._characters import discover_characters, resolve_attribution, store
from kenkui._characters.attribution import attribute_chapter
from kenkui._characters.infer import normalise_roster, slugify
from kenkui._characters.models import CharacterRoster
from kenkui._characters.quotes import extract_spans
from kenkui._domain.casting import CharacterProfile
from kenkui.cancellation import CancellationToken
from kenkui.errors import CancelledError

# The coverage fixture below carries exactly this many quoted runs.
_FIXTURE_QUOTES = 3
_REVIEWED_VERSIONS = 2

NARRATION_A = "The inspector waited by the door. "
QUOTED = '"You are late again,"'
TAG = " he said."
LINE = QUOTED + TAG
NARRATION_B = " Nobody answered him."
TEXT = NARRATION_A + LINE + NARRATION_B
BOOK = "a" * 64


class ScriptedClient:
    """Answers roster prompts and attribution prompts from fixed replies."""

    def __init__(self, speaker: str = "javert", *, broken: bool = False) -> None:
        """Answer with *speaker*, or unusable text when *broken*."""
        self.speaker = speaker
        self.broken = broken
        self.calls: list[str] = []

    def complete(self, model: str, prompt: str) -> str:
        """Return a scripted reply for whichever prompt this is."""
        assert model
        self.calls.append(prompt)
        if self.broken:
            return "the model is having a day"
        if "List the speaking characters" in prompt:
            return json.dumps(
                {
                    "characters": [
                        {"id": "javert", "name": "Javert", "gender": "masculine"}
                    ]
                }
            )
        return json.dumps({"attributions": [{"quote_id": 0, "speaker": self.speaker}]})


def _inspection(text: str = TEXT) -> kk.BookInspection:
    chapter = kk.ChapterInspection("ch1", 0, "One", len(text), text)
    metadata = kk.BookMetadata("T", "A", cover_available=False)
    return kk.BookInspection(metadata, (chapter,))


def test_discovery_can_stop_before_attribution() -> None:
    """Discovering a roster performs no quote attribution or attribution writes."""
    client = ScriptedClient()
    roster = discover_characters(_inspection(), "fake/model", client=client)
    assert tuple(character.id for character in roster.characters) == ("javert",)
    assert len(client.calls) == 1
    assert "List the speaking characters" in client.calls[0]
    assert store.list_castings() == ()


def test_supplied_roster_skips_discovery_and_separates_cached_edits() -> None:
    """A corrected roster cannot hit attribution cached for the original input."""
    automatic = resolve_attribution(
        _inspection(), BOOK, "fake/model", client=ScriptedClient()
    )
    client = ScriptedClient()
    roster = CharacterRoster(
        (CharacterProfile("javert", "Inspector Javert", "feminine", 0, ()),)
    )
    reviewed = resolve_attribution(
        _inspection(), BOOK, "fake/model", client=client, roster=roster, reviewed=True
    )
    assert len(client.calls) == 1
    assert "List the speaking characters" not in client.calls[0]
    assert "Inspector Javert" in client.calls[0]
    # Explicit review takes precedence over the contradictory "he said" tag.
    assert reviewed.characters[0].gender == "feminine"
    assert reviewed.attribution_id != automatic.attribution_id
    assert (
        resolve_attribution(
            _inspection(),
            BOOK,
            "fake/model",
            client=client,
            roster=roster,
            reviewed=True,
        )
        == reviewed
    )
    assert len(client.calls) == 1
    changed = replace(
        roster, characters=(replace(roster.characters[0], display_name="Javert"),)
    )
    revised = resolve_attribution(
        _inspection(), BOOK, "fake/model", client=client, roster=changed, reviewed=True
    )
    assert revised.attribution_id != reviewed.attribution_id
    assert len(client.calls) == _REVIEWED_VERSIONS


@pytest.mark.parametrize(
    ("gender", "expected"), [(None, "masculine"), ("feminine", "feminine")]
)
def test_review_preserves_known_genders_and_keeps_unspecified_values_inferable(
    gender: str | None, expected: str
) -> None:
    """Review supersedes confident evidence only where the caller supplied a value."""
    roster = CharacterRoster((CharacterProfile("javert", "Javert", gender, 0, ()),))
    inspection = kk.BookInspection(
        _inspection().metadata,
        tuple(
            kk.ChapterInspection(f"ch{index}", index, "One", len(TEXT), TEXT)
            for index in range(_FIXTURE_QUOTES)
        ),
    )
    record = resolve_attribution(
        inspection,
        BOOK,
        "fake/model",
        client=ScriptedClient(),
        roster=roster,
        reviewed=True,
    )
    assert record.characters[0].gender == expected


class _AliasFoldClient:
    """Names one woman "Corwi" in chapter one and "Lizbyet Corwi" in chapter two.

    merge_rosters folds these into a single character keyed on
    "lizbyet-corwi" before any attribution call happens, so both chapters'
    attribution answers name the folded id -- exactly what a real model
    would do, since that is the only id the roster block hands it by then.
    """

    def __init__(self) -> None:
        """Track calls so the fixture reads like the ones above it."""
        self.calls: list[str] = []
        self._rosters_seen = 0

    def complete(self, model: str, prompt: str) -> str:
        """Return the next scripted roster or attribution answer."""
        assert model
        self.calls.append(prompt)
        if "List the speaking characters" in prompt:
            self._rosters_seen += 1
            character = (
                {"id": "corwi", "name": "Corwi", "gender": "feminine"}
                if self._rosters_seen == 1
                else {
                    "id": "lizbyet-corwi",
                    "name": "Lizbyet Corwi",
                    "gender": "feminine",
                }
            )
            return json.dumps({"characters": [character]})
        return json.dumps(
            {"attributions": [{"quote_id": 0, "speaker": "lizbyet-corwi"}]}
        )


def test_slugify_produces_stable_ids() -> None:
    """Ids key everything downstream, so they must not drift."""
    assert slugify("Elizabeth Bennet") == "elizabeth-bennet"
    assert slugify("Éponine") == "eponine"
    assert slugify("  ") == ""


def test_roster_rejects_pronouns() -> None:
    """A pronoun id would merge every unrelated speaker into one voice."""
    roster = normalise_roster(
        [
            {"id": "he", "name": "He"},
            {"id": "javert", "name": "Javert", "gender": "masculine"},
        ]
    )
    assert [c.id for c in roster] == ["javert"]


def test_roster_is_sorted_so_two_runs_agree() -> None:
    """The plan fingerprint requires a stable roster order."""
    forward = normalise_roster([{"id": "b", "name": "B"}, {"id": "a", "name": "A"}])
    reverse = normalise_roster([{"id": "a", "name": "A"}, {"id": "b", "name": "B"}])
    assert forward == reverse


def test_roster_ignores_junk_entries() -> None:
    """A model returning nonsense must not produce a nonsense character."""
    assert normalise_roster([1, "x", {}, {"name": "  "}]) == ()


def test_dialogue_is_attributed_and_narration_is_not() -> None:
    """Narration is never sent to a model and never gets a speaker."""
    record = resolve_attribution(
        _inspection(), BOOK, "fake/model", client=ScriptedClient()
    )
    assert [s.character_id for s in record.spans] == [None, "javert", None]


def test_spans_still_partition_the_chapter() -> None:
    """Segmentation reconstructs chapter text from these spans."""
    record = resolve_attribution(
        _inspection(), BOOK, "fake/model", client=ScriptedClient()
    )
    assert "".join(TEXT[s.start : s.end] for s in record.spans) == TEXT


def test_an_unknown_speaker_stays_unattributed() -> None:
    """Narrating a line nobody could place beats voicing it as the wrong one."""
    record = resolve_attribution(
        _inspection(), BOOK, "fake/model", client=ScriptedClient("unknown")
    )
    assert all(s.character_id is None for s in record.spans)


def test_a_speaker_outside_the_roster_becomes_a_role() -> None:
    """A name the roster missed is a speaker, not a non-answer.

    Rejecting it made the roster a hard ceiling on attribution: a character
    the model named correctly but the roster never listed could not be
    placed at any price, and unknown is read in the narrator's voice. The
    cost of the open vocabulary is that a hallucinated name also becomes a
    voice, which is the better failure of the two.
    """
    record = resolve_attribution(
        _inspection(), BOOK, "fake/model", client=ScriptedClient("someone-else")
    )
    placed = [s.character_id for s in record.spans if s.character_id is not None]
    assert placed
    assert all(cid.startswith("role:someone-else@") for cid in placed)


def test_a_pronoun_answer_is_rejected() -> None:
    """A pronoun would merge unrelated speakers into one voice."""
    record = resolve_attribution(
        _inspection(), BOOK, "fake/model", client=ScriptedClient("he")
    )
    assert all(s.character_id is None for s in record.spans)


def test_a_broken_model_narrates_rather_than_stalling() -> None:
    """The book still reads, in one voice, instead of failing the render."""
    record = resolve_attribution(
        _inspection(), BOOK, "fake/model", client=ScriptedClient(broken=True)
    )
    assert "".join(TEXT[s.start : s.end] for s in record.spans) == TEXT
    assert all(s.character_id is None for s in record.spans)


def test_characters_carry_measured_volume_and_chapters() -> None:
    """Casting weights by volume and forbids same-chapter sharing."""
    record = resolve_attribution(
        _inspection(), BOOK, "fake/model", client=ScriptedClient()
    )
    javert = next(c for c in record.characters if c.id == "javert")
    # The quoted run only: a dialogue tag is narration, so it must not count
    # toward the speaker's volume or a chatty tag would inflate their weight.
    assert javert.spoken_characters == len(QUOTED)
    assert javert.chapter_ids == ("ch1",)


def test_a_character_who_never_speaks_is_dropped() -> None:
    """The voice pool is too shallow to spend one on a silent character."""
    record = resolve_attribution(
        _inspection(), BOOK, "fake/model", client=ScriptedClient("unknown")
    )
    assert record.characters == ()


def test_a_second_call_hits_the_store_and_calls_no_model() -> None:
    """Re-rendering a book must not pay for attribution twice."""
    client = ScriptedClient()
    resolve_attribution(_inspection(), BOOK, "fake/model", client=client)
    before = len(client.calls)
    resolve_attribution(_inspection(), BOOK, "fake/model", client=client)
    assert len(client.calls) == before


def test_the_stored_record_round_trips() -> None:
    """What comes back must equal what went in, spans and all."""
    written = resolve_attribution(
        _inspection(), BOOK, "fake/model", client=ScriptedClient()
    )
    assert store.read_attribution(written.attribution_id) == written


def test_cancellation_is_honoured_before_any_model_call() -> None:
    """A long book is a long sequence of calls; Ctrl-C must not wait them out."""
    token = CancellationToken()
    token.cancel()
    client = ScriptedClient()
    with pytest.raises(CancelledError):
        resolve_attribution(
            _inspection(), BOOK, "fake/model", client=client, cancel=token
        )
    assert client.calls == []


def test_book_text_with_braces_does_not_break_the_prompt() -> None:
    """A novel containing {} would otherwise raise inside str.format."""
    text = 'He wrote {x} on the board. "Solve it," she said.'
    record = resolve_attribution(
        _inspection(text), BOOK, "fake/model", client=ScriptedClient()
    )
    assert "".join(text[s.start : s.end] for s in record.spans) == text


def test_no_roster_call_for_a_chapter_without_speech() -> None:
    """Front matter and description are common; asking about them is spend."""
    quiet = kk.ChapterInspection("ch0", 0, "Front", 24, "No speech whatsoever here.")
    speaking = kk.ChapterInspection("ch1", 1, "One", len(TEXT), TEXT)
    metadata = kk.BookMetadata("T", "A", cover_available=False)
    client = ScriptedClient()
    resolve_attribution(
        kk.BookInspection(metadata, (quiet, speaking)),
        BOOK,
        "fake/model",
        client=client,
    )
    rosters = [p for p in client.calls if "List the speaking characters" in p]
    assert len(rosters) == 1


def test_coverage_separates_unknown_from_dropped() -> None:
    """Two different failures that produced one indistinguishable value.

    A model that answers "unknown" has considered the quote and declined; a
    model that never returns the id has dropped it, which is a defect. Both
    became None, so a truncated or malformed response was indistinguishable
    from ordinary model caution and could only be found by reading the text.
    """
    inspection = _inspection(
        'Chapter One\n\n"One," he said. "Two," she said. "Three," they said.'
    )
    chapter = inspection.chapters[0]
    spans = extract_spans(chapter.id, chapter.text)
    dialogue = [span for span in spans if span.is_dialogue]
    assert len(dialogue) == _FIXTURE_QUOTES

    # ScriptedClient answers quote_id 0 only, so every later quote is dropped.
    _, coverage = attribute_chapter(
        chapter,
        (CharacterProfile("dhatt", "Dhatt", None, 0, ()),),
        "fake/model",
        client=ScriptedClient("unknown"),
        spans=spans,
    )
    assert coverage.quotes == len(dialogue)
    assert coverage.unknown == 1
    assert coverage.dropped == len(dialogue) - 1
    assert coverage.answered == 0


def test_folded_aliases_survive_measurement_and_the_store() -> None:
    """The gap this closes: aliases dying between merge_rosters and the store.

    merge_rosters computes aliases correctly on its own, but resolve_attribution
    rebuilds every profile through _measured on the way to storage. A test
    that only calls merge_rosters cannot see _measured silently dropping the
    field back to its default -- this one exercises the whole path a book
    actually takes, roster inference through the stored record.
    """
    chapter_one = kk.ChapterInspection("ch1", 0, "One", len(TEXT), TEXT)
    chapter_two = kk.ChapterInspection("ch2", 1, "Two", len(TEXT), TEXT)
    metadata = kk.BookMetadata("T", "A", cover_available=False)
    inspection = kk.BookInspection(metadata, (chapter_one, chapter_two))

    record = resolve_attribution(
        inspection, BOOK, "fake/model", client=_AliasFoldClient()
    )
    corwi = next(c for c in record.characters if c.id == "lizbyet-corwi")
    assert set(corwi.aliases) == {"Corwi", "Lizbyet Corwi"}

    stored = store.read_attribution(record.attribution_id)
    assert stored is not None
    restored = next(c for c in stored.characters if c.id == "lizbyet-corwi")
    assert set(restored.aliases) == {"Corwi", "Lizbyet Corwi"}


class _ConcurrentClient:
    """Answers every roster and attribution prompt while tracking overlap."""

    def __init__(self) -> None:
        """Start with no observed concurrency."""
        self._lock = threading.Lock()
        self._in_flight = 0
        self.max_concurrent = 0

    def complete(self, model: str, prompt: str) -> str:
        """Hold each call briefly so overlapping calls are observable."""
        assert model
        if "List the speaking characters" in prompt:
            return json.dumps(
                {
                    "characters": [
                        {"id": "javert", "name": "Javert", "gender": "masculine"}
                    ]
                }
            )
        with self._lock:
            self._in_flight += 1
            self.max_concurrent = max(self.max_concurrent, self._in_flight)
        time.sleep(0.15)
        with self._lock:
            self._in_flight -= 1
        return json.dumps({"attributions": [{"quote_id": 0, "speaker": "javert"}]})


def _multi_chapter_inspection(count: int) -> kk.BookInspection:
    chapters = tuple(
        kk.ChapterInspection(f"ch{i}", i, f"One {i}", len(LINE), LINE)
        for i in range(count)
    )
    metadata = kk.BookMetadata("T", "A", cover_available=False)
    return kk.BookInspection(metadata, chapters)


def test_chapters_are_attributed_concurrently() -> None:
    """Chapters are independent model calls; they must not queue single-file."""
    client = _ConcurrentClient()
    resolve_attribution(_multi_chapter_inspection(4), BOOK, "fake/model", client=client)
    assert client.max_concurrent > 1


def test_attribution_prompts_carry_no_recent_speakers() -> None:
    """Continuity chaining is gone; no prompt may ask for recent speakers."""
    client = ScriptedClient()
    resolve_attribution(_inspection(), BOOK, "fake/model", client=client)
    assert client.calls
    assert all("Recently speaking" not in prompt for prompt in client.calls)


@pytest.mark.parametrize("workers", [1, 3])
def test_cancellation_stops_unstarted_chapters(
    monkeypatch: pytest.MonkeyPatch, workers: int
) -> None:
    """Cancellation during active calls must not buy all forty chapter calls."""
    monkeypatch.setattr("kenkui._characters._ATTRIBUTION_CONCURRENCY", workers)
    token = CancellationToken()
    barrier = threading.Barrier(workers)

    class CancellingClient(ScriptedClient):
        def complete(self, model: str, prompt: str) -> str:
            """Wait for the active batch, then cancel before workers refill."""
            response = super().complete(model, prompt)
            barrier.wait(timeout=5)
            token.cancel()
            return response

    client = CancellingClient()
    roster = CharacterRoster((CharacterProfile("javert", "Javert", None, 0, ()),))
    with pytest.raises(CancelledError):
        resolve_attribution(
            _multi_chapter_inspection(40),
            BOOK,
            "fake/model",
            client=client,
            roster=roster,
            cancel=token,
        )
    assert len(client.calls) == workers
    assert store.list_castings() == ()


def test_progress_reports_chapters_on_calling_thread_and_cache_hits() -> None:
    """Both model stages and reused attribution remain visible to a caller."""
    owner = threading.get_ident()
    events: list[tuple[str, int, int, str | None]] = []

    def progress(stage: str, completed: int, total: int, chapter: str | None) -> None:
        assert threading.get_ident() == owner
        events.append((stage, completed, total, chapter))

    inspection = _multi_chapter_inspection(3)
    client = ScriptedClient()
    resolve_attribution(
        inspection, BOOK, "fake/model", client=client, on_progress=progress
    )
    for stage in ("characters", "attribution"):
        stage_events = [event for event in events if event[0] == stage]
        assert [event[1] for event in stage_events] == list(range(4))
        assert {event[2] for event in stage_events} == {len(inspection.chapters)}
        assert {event[3] for event in stage_events[1:]} == {"ch0", "ch1", "ch2"}
    events.clear()
    calls = len(client.calls)
    resolve_attribution(
        inspection, BOOK, "fake/model", client=client, on_progress=progress
    )
    assert events == [("attribution", 0, 3, None), ("attribution", 3, 3, None)]
    assert len(client.calls) == calls


@pytest.mark.parametrize("stage", ["characters", "attribution"])
def test_initial_progress_callback_can_cancel_without_model_calls(stage: str) -> None:
    """Stage-start callbacks can stop before the first provider invocation."""
    token = CancellationToken()
    client = ScriptedClient()

    def progress(current: str, completed: int, total: int, chapter: str | None) -> None:
        del completed, total, chapter
        if current == stage:
            token.cancel()

    if stage == "characters":
        with pytest.raises(CancelledError):
            discover_characters(
                _inspection(),
                "fake/model",
                client=client,
                cancel=token,
                on_progress=progress,
            )
    else:
        with pytest.raises(CancelledError):
            resolve_attribution(
                _inspection(),
                BOOK,
                "fake/model",
                client=client,
                roster=CharacterRoster(()),
                cancel=token,
                on_progress=progress,
            )
    assert client.calls == []
