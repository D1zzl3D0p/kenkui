"""casting.sqlite3 holds attribution and cast assignments.

Not the fail-open PCM cache. Reads fail soft to a miss, since anything lost is
recoverable by recomputing; writes fail loud, because silently losing a cast
means a re-render silently re-casts the book.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest

import kenkui as kk
from kenkui._characters import resolve_cast, store
from kenkui._domain.casting import CastingRequest, CharacterProfile
from kenkui._domain.planning import SpeakerSpan
from kenkui._tts import production

if TYPE_CHECKING:
    from pathlib import Path

    from kenkui.voices.types import PerceivedGender

BOOK = "a" * 64
MODEL = "fake/model"
PROMPT_VERSION = "characters-v1"
PARAMS = {"temperature": 0.0}
_TWO_CASTS = 2


@pytest.fixture
def attribution() -> store.AttributionRecord:
    """One book with two characters and a partitioned chapter."""
    key = store.attribution_key(BOOK, MODEL, PROMPT_VERSION, PARAMS)
    return store.AttributionRecord(
        attribution_id=key,
        book_id=BOOK,
        model_id=MODEL,
        prompt_version=PROMPT_VERSION,
        params=PARAMS,
        characters=(
            CharacterProfile("elizabeth", "Elizabeth", "feminine", 400, ("ch1",)),
            CharacterProfile("darcy", "Darcy", "masculine", 300, ("ch1", "ch2")),
        ),
        spans=(
            SpeakerSpan("ch1", 0, 10, None),
            SpeakerSpan("ch1", 10, 20, "elizabeth"),
            SpeakerSpan("ch1", 20, 30, None),
        ),
    )


def _cast(attribution_id: str, method: str) -> store.CastRecord:
    return store.CastRecord(
        cast_id=store.cast_key(attribution_id, method, {}, "eponine", "eponine"),
        attribution_id=attribution_id,
        method=method,
        narrator_voice_id="eponine",
        unknown_voice_id="eponine",
        assignments=(("elizabeth", "anna", True), ("darcy", "charles", False)),
    )


def test_store_lives_beside_the_manifest(isolated_cache_root: Path) -> None:
    """A self-describing name, unlike the opaque PCM cache next to it."""
    assert store.default_store_path() == isolated_cache_root / "casting.sqlite3"


def test_attribution_round_trips(attribution: store.AttributionRecord) -> None:
    """Characters, chapter links, and spans all survive a write and read."""
    store.write_attribution(attribution)
    assert store.read_attribution(attribution.attribution_id) == attribution


def test_missing_attribution_reads_as_none() -> None:
    """An absent key is a miss, not an error."""
    assert store.read_attribution("f" * 64) is None


def test_two_casts_share_one_attribution(
    attribution: store.AttributionRecord,
) -> None:
    """Cast keys hang below attribution keys, so exploring casts is free."""
    store.write_attribution(attribution)
    store.write_cast(_cast(attribution.attribution_id, "gendered"))
    store.write_cast(_cast(attribution.attribution_id, "random"))
    assert len(store.list_castings()) == _TWO_CASTS
    assert store.read_attribution(attribution.attribution_id) is not None


def test_cast_round_trips_with_pins(attribution: store.AttributionRecord) -> None:
    """A pinned choice must survive, so a re-solve never overwrites it."""
    store.write_attribution(attribution)
    written = _cast(attribution.attribution_id, "gendered")
    store.write_cast(written)
    loaded = store.read_cast(written.cast_id)
    assert loaded is not None
    assert {c: (v, p) for c, v, p in loaded.assignments} == {
        "elizabeth": ("anna", True),
        "darcy": ("charles", False),
    }


def test_assignments_come_back_in_a_stable_order(
    attribution: store.AttributionRecord,
) -> None:
    """Write order must not change the record, matching cast_key's behaviour."""
    store.write_attribution(attribution)
    written = _cast(attribution.attribution_id, "gendered")
    store.write_cast(written)
    forward = store.read_cast(written.cast_id)
    reversed_write = store.CastRecord(
        cast_id=written.cast_id,
        attribution_id=written.attribution_id,
        method=written.method,
        narrator_voice_id=written.narrator_voice_id,
        unknown_voice_id=written.unknown_voice_id,
        assignments=tuple(reversed(written.assignments)),
    )
    store.write_cast(reversed_write)
    assert store.read_cast(written.cast_id) == forward


def test_remove_casting_keeps_the_attribution(
    attribution: store.AttributionRecord,
) -> None:
    """Rebuilding a cast is free; re-deriving attribution costs a model pass."""
    store.write_attribution(attribution)
    cast = _cast(attribution.attribution_id, "gendered")
    store.write_cast(cast)
    store.remove_casting(cast.cast_id)
    assert store.list_castings() == ()
    assert store.read_attribution(attribution.attribution_id) is not None


def test_remove_attribution_cascades_to_its_casts(
    attribution: store.AttributionRecord,
) -> None:
    """Dropping the roster must not strand casts that reference it."""
    store.write_attribution(attribution)
    store.write_cast(_cast(attribution.attribution_id, "gendered"))
    store.remove_attribution(attribution.attribution_id)
    assert store.read_attribution(attribution.attribution_id) is None
    assert store.list_castings() == ()


def test_attribution_key_covers_everything_that_determines_content() -> None:
    """A stale key would silently reuse output from a different prompt."""
    base = store.attribution_key(BOOK, MODEL, PROMPT_VERSION, PARAMS)
    assert base != store.attribution_key("b" * 64, MODEL, PROMPT_VERSION, PARAMS)
    assert base != store.attribution_key(BOOK, "other/model", PROMPT_VERSION, PARAMS)
    assert base != store.attribution_key(BOOK, MODEL, "characters-v2", PARAMS)
    other_params = {"temperature": 1}
    assert base != store.attribution_key(BOOK, MODEL, PROMPT_VERSION, other_params)


def test_the_identity_model_enters_the_attribution_key() -> None:
    """A roster-changing identity choice cannot reuse another attribution."""
    params = {"temperature": 0.0}
    base = store.attribution_key("book", "m", "v", params)
    assert store.attribution_key("book", "m", "v", params, identity_model_id="") == base
    assert (
        store.attribution_key(
            "book",
            "m",
            "v",
            params,
            identity_model_id="glm",
            identity_reasoning="high",
        )
        != base
    )


def test_cast_key_is_order_independent_for_pins() -> None:
    """Two callers writing the same cast differently must land on one row."""
    first = store.cast_key("x", "gendered", {"a": "1", "b": "2"}, "n", "u")
    second = store.cast_key("x", "gendered", {"b": "2", "a": "1"}, "n", "u")
    assert first == second


def test_corrupt_database_reads_as_a_miss(isolated_cache_root: Path) -> None:
    """Everything here is recoverable by recomputing, so a read fails soft."""
    (isolated_cache_root / "casting.sqlite3").write_bytes(b"not a database")
    assert store.read_attribution("a" * 64) is None
    assert store.list_castings() == ()


def test_corrupt_database_fails_loud_on_write(
    isolated_cache_root: Path, attribution: store.AttributionRecord
) -> None:
    """Silently losing a cast would silently re-cast the book on re-render."""
    (isolated_cache_root / "casting.sqlite3").write_bytes(b"not a database")
    with pytest.raises(OSError, match="could not write attribution"):
        store.write_attribution(attribution)


def test_writing_twice_replaces_rather_than_duplicates(
    attribution: store.AttributionRecord,
) -> None:
    """Re-resolving the same key must not accumulate rows."""
    store.write_attribution(attribution)
    store.write_attribution(attribution)
    assert store.read_attribution(attribution.attribution_id) == attribution


def test_the_store_path_follows_a_redirected_cache_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Resolution must happen per call, not once at import.

    The whole suite's isolation rests on this. Binding default_cache_root at
    import time silently sends every test's writes to the developer's real
    cache, which is a leak nothing else here would notice.
    """
    monkeypatch.setattr(production, "default_cache_root", lambda: tmp_path)
    assert store.default_store_path() == tmp_path / store.STORE_NAME


def _voice(voice_id: str, traits: PerceivedGender) -> kk.Voice:
    """Return a loaded voice the solver can draw from."""
    return kk.Voice(
        id=voice_id,
        name=voice_id.title(),
        enabled=True,
        provenance="Project-owned recording by Test Speaker",
        license_id="CC0-1.0",
        commercial_use_allowed=True,
        language="en-US",
        content_fingerprint="9" * 64,
        compatible_model_revisions=("pocket-tts/model@0123456789abcdef",),
        state="loaded",
        perceived_gender=traits,
    )


def test_resolving_a_cast_stores_it_beneath_its_attribution(
    attribution: store.AttributionRecord,
) -> None:
    """Solving without storing leaves the public cast verbs answering for nothing.

    list_castings and remove_casting are exported, so a solved cast that is
    never written makes the first always empty and the second always a no-op.
    """
    store.write_attribution(attribution)
    outcome = resolve_cast(
        attribution,
        CastingRequest(
            characters=attribution.characters,
            pool=(_voice("anna", "feminine"), _voice("charles", "masculine")),
            explicit={"elizabeth": "anna"},
            narrator_voice_id="eponine",
            unknown_voice_id="eponine",
            method="gendered",
        ),
    )
    stored = store.list_castings()
    assert len(stored) == 1
    assert dict(outcome.assignments) == {
        character: voice for character, voice, _ in stored[0].assignments
    }


def test_a_stored_cast_records_which_choices_the_caller_pinned(
    attribution: store.AttributionRecord,
) -> None:
    """An explicit entry must come back pinned, so a re-solve cannot move it."""
    store.write_attribution(attribution)
    resolve_cast(
        attribution,
        CastingRequest(
            characters=attribution.characters,
            pool=(_voice("anna", "feminine"), _voice("charles", "masculine")),
            explicit={"elizabeth": "anna"},
            narrator_voice_id="eponine",
            unknown_voice_id="eponine",
            method="gendered",
        ),
    )
    pinned = {c: p for c, _, p in store.list_castings()[0].assignments}
    assert pinned["elizabeth"] is True
    assert pinned["darcy"] is False


def test_a_store_without_the_column_is_migrated(
    tmp_path: Path, attribution: store.AttributionRecord
) -> None:
    """An operator's existing attributions survive the upgrade."""
    path = tmp_path / "old.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE books(book_id TEXT PRIMARY KEY);
        CREATE TABLE characters(
            attribution_id TEXT NOT NULL,
            character_id TEXT NOT NULL,
            display_name TEXT NOT NULL,
            gender TEXT,
            spoken_characters INTEGER NOT NULL,
            ordinal INTEGER NOT NULL,
            PRIMARY KEY (attribution_id, character_id));
        """
    )
    connection.commit()
    connection.close()

    store.write_attribution(attribution, path)
    assert store.read_attribution(attribution.attribution_id, path) is not None


def test_character_aliases_round_trip(tmp_path: Path) -> None:
    """A stored character keeps every name it was known by."""
    record = store.AttributionRecord(
        attribution_id=store.attribution_key(BOOK, MODEL, PROMPT_VERSION, PARAMS),
        book_id=BOOK,
        model_id=MODEL,
        prompt_version=PROMPT_VERSION,
        params=PARAMS,
        characters=(
            CharacterProfile(
                id="corwi",
                display_name="Lizbyet Corwi",
                gender="feminine",
                spoken_characters=10,
                chapter_ids=("ch1",),
                aliases=("Corwi", "Lizbyet Corwi"),
            ),
        ),
        spans=(),
    )
    store.write_attribution(record, tmp_path / "s.sqlite3")
    read = store.read_attribution(record.attribution_id, tmp_path / "s.sqlite3")
    assert read is not None
    assert read.characters[0].aliases == ("Corwi", "Lizbyet Corwi")
