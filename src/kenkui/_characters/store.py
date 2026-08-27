"""Durable store for attribution and cast assignments.

Lives at ``<cache_root>/casting.sqlite3``, beside ``manifest.json`` and the
PCM cache, and is named for what it holds: an operator can point at it, inspect
it, and delete it, which an opaque ``cache.sqlite3`` mixing casting decisions
with half a gigabyte of audio does not allow.

This is **not** the fail-open PCM cache. Reads fail soft to a miss, because
everything here is recoverable by recomputing. Writes fail loud, because
silently losing a cast means a later re-render silently re-casts the book.

Cast rows hang below attribution rows, so exploring many castings of one book
costs exactly one model pass.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing, contextmanager, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kenkui._domain.casting import CharacterProfile
from kenkui._domain.planning import SpeakerSpan
from kenkui._tts import production

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

STORE_NAME = "casting.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS books(
    book_id TEXT PRIMARY KEY);

CREATE TABLE IF NOT EXISTS attributions(
    attribution_id TEXT PRIMARY KEY,
    book_id TEXT NOT NULL REFERENCES books(book_id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    params_json TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS characters(
    attribution_id TEXT NOT NULL
        REFERENCES attributions(attribution_id) ON DELETE CASCADE,
    character_id TEXT NOT NULL,
    display_name TEXT NOT NULL,
    gender TEXT,
    spoken_characters INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    aliases_json TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (attribution_id, character_id));

CREATE TABLE IF NOT EXISTS character_chapters(
    attribution_id TEXT NOT NULL
        REFERENCES attributions(attribution_id) ON DELETE CASCADE,
    character_id TEXT NOT NULL,
    chapter_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    PRIMARY KEY (attribution_id, character_id, chapter_id));

CREATE TABLE IF NOT EXISTS quote_spans(
    attribution_id TEXT NOT NULL
        REFERENCES attributions(attribution_id) ON DELETE CASCADE,
    chapter_id TEXT NOT NULL,
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    character_id TEXT,
    PRIMARY KEY (attribution_id, chapter_id, start));

CREATE TABLE IF NOT EXISTS casts(
    cast_id TEXT PRIMARY KEY,
    attribution_id TEXT NOT NULL
        REFERENCES attributions(attribution_id) ON DELETE CASCADE,
    method TEXT NOT NULL,
    narrator_voice_id TEXT NOT NULL,
    unknown_voice_id TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS cast_assignments(
    cast_id TEXT NOT NULL REFERENCES casts(cast_id) ON DELETE CASCADE,
    character_id TEXT NOT NULL,
    voice_id TEXT NOT NULL,
    pinned INTEGER NOT NULL,
    PRIMARY KEY (cast_id, character_id));
"""


@dataclass(frozen=True, slots=True)
class AttributionRecord:
    """One book's roster and speaker spans, as derived by one model."""

    attribution_id: str
    book_id: str
    model_id: str
    prompt_version: str
    params: Mapping[str, Any]
    characters: tuple[CharacterProfile, ...]
    spans: tuple[SpeakerSpan, ...]


@dataclass(frozen=True, slots=True)
class CastRecord:
    """One resolved cast. ``assignments`` is (character, voice, pinned)."""

    cast_id: str
    attribution_id: str
    method: str
    narrator_voice_id: str
    unknown_voice_id: str
    assignments: tuple[tuple[str, str, bool], ...]


def default_store_path() -> Path:
    """Return the managed store location, beside the voice manifest.

    Resolved through the module rather than a bound name, so the attribute
    lookup happens per call. Tests redirect the cache root that way, and
    binding the function at import time would send every test's writes to the
    developer's real cache.
    """
    return production.default_cache_root() / STORE_NAME


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def attribution_key(  # noqa: PLR0913, PLR0917 - every input that determines content.
    book_id: str,
    model_id: str,
    prompt_version: str,
    params: Mapping[str, Any],
    chapters: tuple[str, ...] = (),
    schemas: tuple[str, ...] = (),
    *,
    roster_model_id: str = "",
) -> str:
    """Key attribution by everything that determines its content.

    The prompt version is included deliberately: without it, editing a prompt
    would silently reuse output produced by the previous one.

    ``chapters`` and ``schemas`` matter for the same reason. Spans are offsets
    into one chapter's normalized text, so a record derived over a selection
    reused for the whole book leaves every other chapter unattributed, and a
    record derived under an earlier parser or normalizer carries offsets that
    no longer land where they did. Both are absent by default so a caller that
    supplies neither keys exactly as before.
    """
    material: dict[str, Any] = {
        "book_id": book_id,
        "model_id": model_id,
        "params": dict(params),
        "prompt_version": prompt_version,
    }
    if chapters:
        material["chapters"] = list(chapters)
    if schemas:
        material["schemas"] = list(schemas)
    if roster_model_id and roster_model_id != model_id:
        # Absent when one model does both, so the common case keys exactly as
        # before. A separately named roster model is different material: the
        # characters it invents decide who every later attribution can name.
        material["roster_model_id"] = roster_model_id
    return hashlib.sha256(_canonical(material).encode()).hexdigest()


def cast_key(
    attribution_id: str,
    method: str,
    explicit: Mapping[str, str],
    narrator_voice_id: str,
    unknown_voice_id: str,
) -> str:
    """Key a cast below its attribution, so cast variants share one model pass."""
    return hashlib.sha256(
        _canonical(
            {
                "attribution_id": attribution_id,
                "explicit": dict(sorted(explicit.items())),
                "method": method,
                "narrator": narrator_voice_id,
                "unknown": unknown_voice_id,
            }
        ).encode()
    ).hexdigest()


def _migrate(connection: sqlite3.Connection) -> None:
    """Add columns that post-date a store an operator already has.

    CREATE TABLE IF NOT EXISTS does nothing to an existing table, so a new
    column reaches a fresh database and no other. Adding it here keeps a
    store written by an earlier version readable and writable rather than
    making the operator discard their attributions.
    """
    existing = {
        row["name"] for row in connection.execute("PRAGMA table_info(characters)")
    }
    if "aliases_json" not in existing:
        connection.execute(
            "ALTER TABLE characters "
            "ADD COLUMN aliases_json TEXT NOT NULL DEFAULT '[]'"
        )


@contextmanager
def _connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open the store, creating it and its private directory if absent."""
    location = path or default_store_path()
    location.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with closing(sqlite3.connect(location)) as connection:
        connection.row_factory = sqlite3.Row
        # Required for the delete cascades below; SQLite leaves it off.
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(_SCHEMA)
        _migrate(connection)
        # Private like the voice manifest; a filesystem without chmod is not
        # a reason to refuse the store.
        with suppress(OSError):
            location.chmod(0o600)
        yield connection


@contextmanager
def _reading(path: Path | None = None) -> Iterator[sqlite3.Connection | None]:
    """Open for reading, yielding None when the store is unusable.

    A miss and a corrupt file are the same outcome here: recompute.
    """
    try:
        with _connect(path) as connection:
            yield connection
    except (sqlite3.Error, OSError):
        yield None


def write_attribution(record: AttributionRecord, path: Path | None = None) -> None:
    """Persist one attribution, replacing any earlier row with the same key."""
    try:
        with _connect(path) as connection, connection:
            connection.execute(
                "INSERT OR IGNORE INTO books(book_id) VALUES(?)", (record.book_id,)
            )
            connection.execute(
                "DELETE FROM attributions WHERE attribution_id=?",
                (record.attribution_id,),
            )
            connection.execute(
                "INSERT INTO attributions(attribution_id,book_id,model_id,"
                "prompt_version,params_json) VALUES(?,?,?,?,?)",
                (
                    record.attribution_id,
                    record.book_id,
                    record.model_id,
                    record.prompt_version,
                    _canonical(dict(record.params)),
                ),
            )
            for ordinal, character in enumerate(record.characters):
                connection.execute(
                    "INSERT INTO characters(attribution_id,character_id,"
                    "display_name,gender,spoken_characters,ordinal,aliases_json) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (
                        record.attribution_id,
                        character.id,
                        character.display_name,
                        character.gender,
                        character.spoken_characters,
                        ordinal,
                        json.dumps(list(character.aliases)),
                    ),
                )
                for index, chapter_id in enumerate(character.chapter_ids):
                    connection.execute(
                        "INSERT INTO character_chapters(attribution_id,"
                        "character_id,chapter_id,ordinal) VALUES(?,?,?,?)",
                        (record.attribution_id, character.id, chapter_id, index),
                    )
            for span in record.spans:
                connection.execute(
                    "INSERT INTO quote_spans(attribution_id,chapter_id,start,end,"
                    "character_id) VALUES(?,?,?,?,?)",
                    (
                        record.attribution_id,
                        span.chapter_id,
                        span.start,
                        span.end,
                        span.character_id,
                    ),
                )
    except sqlite3.Error as error:
        message = f"could not write attribution to {path or default_store_path()}"
        raise OSError(message) from error


def read_attribution(
    attribution_id: str, path: Path | None = None
) -> AttributionRecord | None:
    """Return one stored attribution, or None on a miss or unusable store."""
    with _reading(path) as connection:
        if connection is None:
            return None
        try:
            row = connection.execute(
                "SELECT * FROM attributions WHERE attribution_id=?",
                (attribution_id,),
            ).fetchone()
            if row is None:
                return None
            chapters: dict[str, list[str]] = {}
            for link in connection.execute(
                "SELECT character_id,chapter_id FROM character_chapters "
                "WHERE attribution_id=? ORDER BY character_id,ordinal",
                (attribution_id,),
            ):
                chapters.setdefault(link["character_id"], []).append(
                    link["chapter_id"]
                )
            characters = tuple(
                CharacterProfile(
                    item["character_id"],
                    item["display_name"],
                    item["gender"],
                    item["spoken_characters"],
                    tuple(chapters.get(item["character_id"], ())),
                    tuple(json.loads(item["aliases_json"])),
                )
                for item in connection.execute(
                    "SELECT * FROM characters WHERE attribution_id=? ORDER BY ordinal",
                    (attribution_id,),
                )
            )
            spans = tuple(
                SpeakerSpan(
                    item["chapter_id"],
                    item["start"],
                    item["end"],
                    item["character_id"],
                )
                for item in connection.execute(
                    "SELECT * FROM quote_spans WHERE attribution_id=? "
                    "ORDER BY chapter_id,start",
                    (attribution_id,),
                )
            )
        except sqlite3.Error:
            return None
        return AttributionRecord(
            attribution_id=row["attribution_id"],
            book_id=row["book_id"],
            model_id=row["model_id"],
            prompt_version=row["prompt_version"],
            params=json.loads(row["params_json"]),
            characters=characters,
            spans=spans,
        )


def write_cast(record: CastRecord, path: Path | None = None) -> None:
    """Persist one cast, replacing any earlier row with the same key."""
    try:
        with _connect(path) as connection, connection:
            connection.execute("DELETE FROM casts WHERE cast_id=?", (record.cast_id,))
            connection.execute(
                "INSERT INTO casts(cast_id,attribution_id,method,narrator_voice_id,"
                "unknown_voice_id) VALUES(?,?,?,?,?)",
                (
                    record.cast_id,
                    record.attribution_id,
                    record.method,
                    record.narrator_voice_id,
                    record.unknown_voice_id,
                ),
            )
            for character_id, voice_id, pinned in record.assignments:
                connection.execute(
                    "INSERT INTO cast_assignments(cast_id,character_id,voice_id,"
                    "pinned) VALUES(?,?,?,?)",
                    (record.cast_id, character_id, voice_id, int(pinned)),
                )
    except sqlite3.Error as error:
        message = f"could not write cast to {path or default_store_path()}"
        raise OSError(message) from error


def _cast_from_row(connection: sqlite3.Connection, row: sqlite3.Row) -> CastRecord:
    assignments = tuple(
        (item["character_id"], item["voice_id"], bool(item["pinned"]))
        for item in connection.execute(
            "SELECT * FROM cast_assignments WHERE cast_id=? ORDER BY character_id",
            (row["cast_id"],),
        )
    )
    return CastRecord(
        cast_id=row["cast_id"],
        attribution_id=row["attribution_id"],
        method=row["method"],
        narrator_voice_id=row["narrator_voice_id"],
        unknown_voice_id=row["unknown_voice_id"],
        assignments=assignments,
    )


def read_cast(cast_id: str, path: Path | None = None) -> CastRecord | None:
    """Return one stored cast, or None on a miss or unusable store."""
    with _reading(path) as connection:
        if connection is None:
            return None
        try:
            row = connection.execute(
                "SELECT * FROM casts WHERE cast_id=?", (cast_id,)
            ).fetchone()
            if row is None:
                return None
            return _cast_from_row(connection, row)
        except sqlite3.Error:
            return None


def list_castings(path: Path | None = None) -> tuple[CastRecord, ...]:
    """Return every stored cast. Filtering composes over this."""
    with _reading(path) as connection:
        if connection is None:
            return ()
        try:
            rows = connection.execute("SELECT * FROM casts ORDER BY cast_id").fetchall()
            return tuple(_cast_from_row(connection, row) for row in rows)
        except sqlite3.Error:
            return ()


def remove_casting(cast_id: str, path: Path | None = None) -> None:
    """Discard one cast. The pure solver rebuilds it for free."""
    try:
        with _connect(path) as connection, connection:
            connection.execute("DELETE FROM casts WHERE cast_id=?", (cast_id,))
    except sqlite3.Error as error:
        message = f"could not remove cast from {path or default_store_path()}"
        raise OSError(message) from error


def remove_attribution(attribution_id: str, path: Path | None = None) -> None:
    """Discard one attribution and every cast beneath it.

    Distinct from remove_casting because the costs differ by orders of
    magnitude: rebuilding a cast is free, re-deriving attribution is a fresh
    model pass. One verb doing both would hide that.
    """
    try:
        with _connect(path) as connection, connection:
            connection.execute(
                "DELETE FROM attributions WHERE attribution_id=?", (attribution_id,)
            )
    except sqlite3.Error as error:
        message = f"could not remove attribution from {path or default_store_path()}"
        raise OSError(message) from error
