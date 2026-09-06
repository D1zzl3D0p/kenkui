# Series Continuity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A character who appears in several volumes of a series keeps one voice across all of them.

**Architecture:** No new casting policy. The solver already picks the least-used voice of the right gender, so a series only has to give it a wider view: characters the series already knows arrive as `explicit` pins, and the voice-usage counter starts from what the series has already spent. A new pair of store tables holds the series' canonical characters and every surface form they have been seen under, and identity resolution across volumes reuses `_characters/identity.py` unchanged.

**Tech Stack:** Python 3.11-3.13, SQLite via stdlib `sqlite3`, pytest, ruff, mypy. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-27-series-continuity-design.md`

## Global Constraints

- Python `>=3.11,<3.14`. No new runtime dependencies.
- Run `.venv/bin/pytest tests/ -q --no-cov` from the repo root. Lint with `.venv/bin/ruff check src tests`, typecheck with `.venv/bin/mypy src`. All three must pass before every commit.
- `_characters/identity.py` keeps its stated bias: under-merging (one person, two voices) is preferred to over-merging (two people, one voice). No task may invert this.
- Series records are **model-independent**. They key on `(series_id, canonical_id)`, never on an `attribution_id`. Re-attributing a volume with a different model must not re-cast a series.
- A pipeline that never calls `.series()` must produce byte-identical segment identities to today. Verified safe by construction: segment identity is built from an explicit field list (`_domain/planning.py:645`), not from the operations tuple, so a `Series` operation cannot leak into it. The only way a series changes a segment is by changing a character's `voice_id`, which is correct and is exactly what a pin is for. A first volume, having nothing to pin and nothing spent, solves identically to a non-series render.
- Store writes follow the existing pattern in `_characters/store.py`: `_connect` for writes (raises `OSError` on failure), `_reading` for reads (returns `None` and degrades to empty on an unusable store).
- New `ErrorCode` members use the existing lowercase-string convention (`errors.py:20-33`) and need an entry in the message table (`errors.py:89-108`).

---

### Task 1: Carry every surface form on a character profile

The spec's prerequisite. `merge_rosters` folds several display names into one character and keeps only the head's, so volume 1 records `Kaladin Stormblessed` and loses `Kaladin` — the name volume 3 will use. Without this the alias table cannot be populated and series matching has one string per character to work with.

**Files:**
- Modify: `src/kenkui/_domain/casting.py` (`CharacterProfile`)
- Modify: `src/kenkui/_characters/infer.py` (`merge_rosters`)
- Modify: `src/kenkui/_characters/store.py` (persist and read the field)
- Test: `tests/test_identity.py`, `tests/test_casting_store.py`

**Interfaces:**
- Produces: `CharacterProfile.aliases: tuple[str, ...] = ()`, sorted and deduplicated, holding every display name folded into this character (including its own).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_identity.py`. `_profile` is the existing helper at `tests/test_identity.py:87`.

```python
def test_merged_names_are_kept_as_aliases() -> None:
    """Folding Lizbyet Corwi into Corwi must not lose the other surface form.

    Volume 1 records the fullest name it saw and volume 3 uses a shorter one.
    Keeping only the head's display name throws away the string that would
    have matched them.
    """
    merged = merge_rosters(
        (
            (_profile("corwi", "Corwi"),),
            (_profile("lizbyet-corwi", "Corwi"),),
            (_profile("lizbyet-corwi", "Lizbyet Corwi"),),
        )
    )
    assert len(merged) == 1
    assert set(merged[0].aliases) == {"Corwi", "Lizbyet Corwi"}


def test_a_lone_character_aliases_to_its_own_name() -> None:
    """Every character has at least one surface form: the one it was given."""
    merged = merge_rosters(((_profile("dhatt", "Dhatt"),),))
    assert merged[0].aliases == ("Dhatt",)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_identity.py -k aliases -v --no-cov`
Expected: FAIL with `AttributeError: 'CharacterProfile' object has no attribute 'aliases'`

- [ ] **Step 3: Add the field**

In `src/kenkui/_domain/casting.py`, add to `CharacterProfile` after `chapter_ids`:

```python
    # Every surface form this character was seen under, sorted. The display
    # name alone cannot find them again: a later volume says "Kaladin" where
    # this one recorded "Kaladin Stormblessed", and merge_rosters keeps only
    # the head's name. Defaulted so every existing construction still works.
    aliases: tuple[str, ...] = ()
```

- [ ] **Step 4: Accumulate the folded names in merge_rosters**

In `src/kenkui/_characters/infer.py`, inside `merge_rosters`, collect names per head id while merging and emit them on the profile. Add before the merge loop:

```python
    aliases: dict[str, set[str]] = {}
```

Inside the loop over `by_id.items()`, after `head_id` is computed, record both
the folded character's name and the head's own:

```python
        seen_names = aliases.setdefault(head_id, set())
        seen_names.add(character.display_name)
        seen_names.add(head.display_name)
```

and pass them when building the merged profile:

```python
aliases = (tuple(sorted(aliases[head_id])),)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_identity.py -v --no-cov`
Expected: PASS

- [ ] **Step 6: Persist the field**

Add the column to the `characters` table in `_SCHEMA`, for databases created
from now on:

```sql
    aliases_json TEXT NOT NULL DEFAULT '[]',
```

**That alone is not enough.** `_SCHEMA` runs under `CREATE TABLE IF NOT
EXISTS`, which silently does nothing to a table that already exists, so an
operator's live store keeps a `characters` table without the column and the
next write fails with `table characters has no column named aliases_json`.
Add a guarded migration that runs after the schema, wherever `_SCHEMA` is
executed:

```python
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
            "ALTER TABLE characters ADD COLUMN aliases_json TEXT NOT NULL DEFAULT '[]'"
        )
```

Write the field in `write_attribution` alongside the other character columns,
as `json.dumps(list(character.aliases))`, and read it back where characters
are loaded, as `tuple(json.loads(row["aliases_json"]))`.

- [ ] **Step 6b: Test the migration against a pre-existing store**

```python
def test_a_store_without_the_column_is_migrated(tmp_path: Path) -> None:
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

    store.write_attribution(_record(), path)
    assert store.read_attribution(_record().attribution_id, path) is not None
```

Build `_record()` the way the surrounding tests in
`tests/test_casting_store.py` build an `AttributionRecord`. The point of the
test is that writing to a store created before the column existed succeeds.

- [ ] **Step 7: Test the round trip**

Append to `tests/test_casting_store.py`:

```python
def test_character_aliases_round_trip(tmp_path: Path) -> None:
    """A stored character keeps every name it was known by."""
    record = _record(
        characters=(
            CharacterProfile(
                id="corwi",
                display_name="Lizbyet Corwi",
                gender="feminine",
                spoken_characters=10,
                chapter_ids=("ch1",),
                aliases=("Corwi", "Lizbyet Corwi"),
            ),
        )
    )
    store.write_attribution(record, tmp_path / "s.sqlite3")
    read = store.read_attribution(record.attribution_id, tmp_path / "s.sqlite3")
    assert read is not None
    assert read.characters[0].aliases == ("Corwi", "Lizbyet Corwi")
```

Build `_record` from whatever record helper `tests/test_casting_store.py`
already uses; if it takes no `characters` argument, construct an
`AttributionRecord` inline with the same fields the surrounding tests use.

- [ ] **Step 8: Run the full suite, lint, typecheck**

Run: `.venv/bin/pytest tests/ -q --no-cov && .venv/bin/ruff check src tests && .venv/bin/mypy src`
Expected: all pass

- [ ] **Step 9: Commit**

```bash
git add src/kenkui/_domain/casting.py src/kenkui/_characters/ tests/
git commit -m "feat: keep every folded name as a character alias"
```

---

### Task 2: Series tables and their records

**Files:**
- Modify: `src/kenkui/_characters/store.py`
- Test: `tests/test_series_store.py` (create)

**Interfaces:**
- Consumes: `_connect`, `_reading`, `_SCHEMA` from `store.py`.
- Produces:
  - `SeriesCharacter` frozen dataclass: `canonical_id: str`, `display_name: str`, `gender: str | None`, `voice_id: str`, `spoken_characters: int`, `aliases: tuple[str, ...]`
  - `SeriesRecord` frozen dataclass: `series_id: str`, `narrator_voice_id: str`, `characters: tuple[SeriesCharacter, ...]`
  - `read_series(series_id: str, path: Path | None = None) -> SeriesRecord | None`
  - `write_series(record: SeriesRecord, path: Path | None = None) -> None`
  - `list_series(path: Path | None = None) -> tuple[SeriesRecord, ...]`
  - `remove_series(series_id: str, path: Path | None = None) -> None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_series_store.py`:

```python
"""A series outlives any one book, and any one model that read it."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kenkui._characters import store

if TYPE_CHECKING:
    from pathlib import Path


def _character(
    canonical_id: str, voice_id: str, spoken: int = 100
) -> store.SeriesCharacter:
    return store.SeriesCharacter(
        canonical_id=canonical_id,
        display_name=canonical_id.title(),
        gender="masculine",
        voice_id=voice_id,
        spoken_characters=spoken,
        aliases=(canonical_id.title(),),
    )


def test_a_series_round_trips(tmp_path: Path) -> None:
    """Everything written is everything read back."""
    path = tmp_path / "s.sqlite3"
    record = store.SeriesRecord(
        series_id="stormlight",
        narrator_voice_id="eponine",
        characters=(_character("kaladin", "alf"), _character("shallan", "aoife")),
    )
    store.write_series(record, path)
    assert store.read_series("stormlight", path) == record


def test_characters_are_ordered_by_speech(tmp_path: Path) -> None:
    """The listing reads as the cast in prominence order."""
    path = tmp_path / "s.sqlite3"
    store.write_series(
        store.SeriesRecord(
            "wot",
            "eponine",
            (_character("nynaeve", "aoife", 10), _character("rand", "alf", 900)),
        ),
        path,
    )
    read = store.read_series("wot", path)
    assert read is not None
    assert [c.canonical_id for c in read.characters] == ["rand", "nynaeve"]


def test_writing_again_replaces_the_series(tmp_path: Path) -> None:
    """A volume updates the series; it does not accumulate duplicates."""
    path = tmp_path / "s.sqlite3"
    store.write_series(
        store.SeriesRecord("s", "eponine", (_character("a", "alf"),)), path
    )
    store.write_series(
        store.SeriesRecord(
            "s", "eponine", (_character("a", "alf"), _character("b", "aoife"))
        ),
        path,
    )
    read = store.read_series("s", path)
    assert read is not None
    assert len(read.characters) == 2


def test_a_missing_series_reads_as_none(tmp_path: Path) -> None:
    """An absent series is a miss, not an error."""
    assert store.read_series("absent", tmp_path / "s.sqlite3") is None


def test_listing_and_removal(tmp_path: Path) -> None:
    """The pair that makes a series something a caller can name and discard."""
    path = tmp_path / "s.sqlite3"
    store.write_series(
        store.SeriesRecord("a", "eponine", (_character("x", "alf"),)), path
    )
    store.write_series(
        store.SeriesRecord("b", "eponine", (_character("y", "aoife"),)), path
    )
    assert [r.series_id for r in store.list_series(path)] == ["a", "b"]
    store.remove_series("a", path)
    assert [r.series_id for r in store.list_series(path)] == ["b"]


def test_an_unusable_store_lists_empty(tmp_path: Path) -> None:
    """Reads degrade rather than raise, matching read_attribution."""
    assert store.list_series(tmp_path / "never-created.sqlite3") == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_series_store.py -q --no-cov`
Expected: FAIL with `AttributeError: module 'kenkui._characters.store' has no attribute 'SeriesCharacter'`

- [ ] **Step 3: Add the tables**

Append to `_SCHEMA` in `src/kenkui/_characters/store.py`:

```sql
CREATE TABLE IF NOT EXISTS series(
    series_id TEXT PRIMARY KEY,
    narrator_voice_id TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS series_characters(
    series_id TEXT NOT NULL REFERENCES series(series_id) ON DELETE CASCADE,
    canonical_id TEXT NOT NULL,
    display_name TEXT NOT NULL,
    gender TEXT,
    voice_id TEXT NOT NULL,
    spoken_characters INTEGER NOT NULL,
    PRIMARY KEY (series_id, canonical_id));

CREATE TABLE IF NOT EXISTS series_aliases(
    series_id TEXT NOT NULL REFERENCES series(series_id) ON DELETE CASCADE,
    alias TEXT NOT NULL,
    canonical_id TEXT NOT NULL,
    PRIMARY KEY (series_id, alias));
```

- [ ] **Step 4: Add the records**

```python
@dataclass(frozen=True, slots=True)
class SeriesCharacter:
    """One person across a series, and the voice they keep."""

    canonical_id: str
    display_name: str
    gender: str | None
    voice_id: str
    spoken_characters: int
    aliases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SeriesRecord:
    """A series' cast, ordered by accumulated speech.

    Keyed on the series and the character, never on an attribution: a volume
    re-read by a different model must not re-cast the series.
    """

    series_id: str
    narrator_voice_id: str
    characters: tuple[SeriesCharacter, ...]
```

- [ ] **Step 5: Add the four store functions**

```python
def write_series(record: SeriesRecord, path: Path | None = None) -> None:
    """Persist one series, replacing whatever it held before.

    A volume updates the series wholesale rather than appending, so a
    re-render cannot leave a character behind under a stale voice.
    """
    try:
        with _connect(path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO series(series_id,narrator_voice_id) "
                "VALUES(?,?)",
                (record.series_id, record.narrator_voice_id),
            )
            connection.execute(
                "DELETE FROM series_characters WHERE series_id=?",
                (record.series_id,),
            )
            connection.execute(
                "DELETE FROM series_aliases WHERE series_id=?", (record.series_id,)
            )
            for character in record.characters:
                connection.execute(
                    "INSERT INTO series_characters(series_id,canonical_id,"
                    "display_name,gender,voice_id,spoken_characters) "
                    "VALUES(?,?,?,?,?,?)",
                    (
                        record.series_id,
                        character.canonical_id,
                        character.display_name,
                        character.gender,
                        character.voice_id,
                        character.spoken_characters,
                    ),
                )
                for alias in character.aliases:
                    connection.execute(
                        "INSERT OR REPLACE INTO series_aliases(series_id,alias,"
                        "canonical_id) VALUES(?,?,?)",
                        (record.series_id, alias, character.canonical_id),
                    )
    except sqlite3.Error as error:
        message = f"could not write series to {path or default_store_path()}"
        raise OSError(message) from error


def _series_from_row(connection: sqlite3.Connection, row: sqlite3.Row) -> SeriesRecord:
    aliases: dict[str, list[str]] = {}
    for item in connection.execute(
        "SELECT alias,canonical_id FROM series_aliases WHERE series_id=? "
        "ORDER BY alias",
        (row["series_id"],),
    ):
        aliases.setdefault(item["canonical_id"], []).append(item["alias"])
    characters = tuple(
        SeriesCharacter(
            canonical_id=item["canonical_id"],
            display_name=item["display_name"],
            gender=item["gender"],
            voice_id=item["voice_id"],
            spoken_characters=item["spoken_characters"],
            aliases=tuple(aliases.get(item["canonical_id"], ())),
        )
        for item in connection.execute(
            "SELECT * FROM series_characters WHERE series_id=? "
            "ORDER BY spoken_characters DESC, canonical_id",
            (row["series_id"],),
        )
    )
    return SeriesRecord(
        series_id=row["series_id"],
        narrator_voice_id=row["narrator_voice_id"],
        characters=characters,
    )


def read_series(series_id: str, path: Path | None = None) -> SeriesRecord | None:
    """Return one stored series, or None on a miss or unusable store."""
    with _reading(path) as connection:
        if connection is None:
            return None
        try:
            row = connection.execute(
                "SELECT * FROM series WHERE series_id=?", (series_id,)
            ).fetchone()
            return None if row is None else _series_from_row(connection, row)
        except sqlite3.Error:
            return None


def list_series(path: Path | None = None) -> tuple[SeriesRecord, ...]:
    """Return every stored series. Filtering composes over this."""
    with _reading(path) as connection:
        if connection is None:
            return ()
        try:
            rows = connection.execute(
                "SELECT * FROM series ORDER BY series_id"
            ).fetchall()
            return tuple(_series_from_row(connection, row) for row in rows)
        except sqlite3.Error:
            return ()


def remove_series(series_id: str, path: Path | None = None) -> None:
    """Drop a series and its pins. The next volume is cast fresh."""
    with _connect(path) as connection:
        connection.execute("DELETE FROM series WHERE series_id=?", (series_id,))
```

The `series_aliases` cascade needs `PRAGMA foreign_keys = ON`, which
`_connect` already sets (`store.py:199`).

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_series_store.py -q --no-cov`
Expected: PASS

- [ ] **Step 7: Full suite, lint, typecheck, commit**

```bash
.venv/bin/pytest tests/ -q --no-cov && .venv/bin/ruff check src tests && .venv/bin/mypy src
git add src/kenkui/_characters/store.py tests/test_series_store.py
git commit -m "feat: store a series' cast above any one book"
```

---

### Task 3: Match a volume's roster against its series

The identity half. Pure functions over names and records — no I/O, no model — so it can be tested exhaustively and cheaply.

**Files:**
- Create: `src/kenkui/_characters/series.py`
- Test: `tests/test_series_identity.py` (create)

**Interfaces:**
- Consumes: `store.SeriesRecord`, `store.SeriesCharacter` (Task 2); `CharacterProfile.aliases` (Task 1); `identity.same_person`, `identity.detect_titles` (existing).
- Produces:
  - `match_roster(record: SeriesRecord | None, characters: Sequence[CharacterProfile]) -> dict[str, str]` — book character id to series canonical id, for characters the series already knows.
  - `merged_series(record, characters, assignments, narrator_voice_id, series_id) -> SeriesRecord` — the series as it stands after this volume, ready to write.

- [ ] **Step 1: Write the failing test**

Create `tests/test_series_identity.py`:

```python
"""One person across volumes, decided from the names alone."""

from __future__ import annotations

from kenkui._characters import store
from kenkui._characters.series import match_roster, merged_series
from kenkui._domain.casting import CharacterProfile


def _known(
    canonical_id: str, display: str, aliases: tuple[str, ...]
) -> store.SeriesCharacter:
    return store.SeriesCharacter(
        canonical_id=canonical_id,
        display_name=display,
        gender="masculine",
        voice_id="alf",
        spoken_characters=500,
        aliases=aliases,
    )


def _profile(character_id: str, display: str, *aliases: str) -> CharacterProfile:
    return CharacterProfile(
        id=character_id,
        display_name=display,
        gender=None,
        spoken_characters=100,
        chapter_ids=("ch1",),
        aliases=tuple(sorted({display, *aliases})),
    )


def test_an_exact_alias_matches() -> None:
    """Volume 3 says "Kaladin"; volume 1 recorded it as an alias."""
    record = store.SeriesRecord(
        "stormlight",
        "eponine",
        (
            _known(
                "kaladin-stormblessed",
                "Kaladin Stormblessed",
                ("Kaladin", "Kaladin Stormblessed"),
            ),
        ),
    )
    matched = match_roster(record, (_profile("kaladin", "Kaladin"),))
    assert matched == {"kaladin": "kaladin-stormblessed"}


def test_a_nested_name_matches_without_an_exact_alias() -> None:
    """Identity resolution, not just string equality.

    The series recorded only the full name; this volume uses the short one
    and it was never stored as an alias, so an exact hit cannot find it.
    """
    record = store.SeriesRecord(
        "stormlight",
        "eponine",
        (_known("dalinar-kholin", "Dalinar Kholin", ("Dalinar Kholin",)),),
    )
    matched = match_roster(record, (_profile("dalinar", "Dalinar"),))
    assert matched == {"dalinar": "dalinar-kholin"}


def test_two_people_sharing_a_surname_do_not_match() -> None:
    """Over-merging gives two people one voice, which is the worse failure."""
    record = store.SeriesRecord(
        "s",
        "eponine",
        (_known("charles-hayter", "Charles Hayter", ("Charles Hayter",)),),
    )
    assert (
        match_roster(record, (_profile("charles-musgrove", "Charles Musgrove"),)) == {}
    )


def test_an_ambiguous_short_form_matches_nobody() -> None:
    """Two hosts could claim it, so it names neither."""
    record = store.SeriesRecord(
        "s",
        "eponine",
        (
            _known("charles-hayter", "Charles Hayter", ("Charles Hayter",)),
            _known("charles-musgrove", "Charles Musgrove", ("Charles Musgrove",)),
        ),
    )
    assert match_roster(record, (_profile("charles", "Charles"),)) == {}


def test_no_series_yet_matches_nothing() -> None:
    """The first volume has nothing to match against."""
    assert match_roster(None, (_profile("kaladin", "Kaladin"),)) == {}


def test_merging_accumulates_speech_and_aliases() -> None:
    """A returning character's totals grow; their voice does not change."""
    record = store.SeriesRecord(
        "s",
        "eponine",
        (
            _known(
                "kaladin-stormblessed",
                "Kaladin Stormblessed",
                ("Kaladin Stormblessed",),
            ),
        ),
    )
    updated = merged_series(
        record,
        (_profile("kaladin", "Kaladin"),),
        {"kaladin": "alf"},
        "eponine",
        "s",
    )
    kaladin = next(
        c for c in updated.characters if c.canonical_id == "kaladin-stormblessed"
    )
    assert kaladin.voice_id == "alf"
    assert kaladin.spoken_characters == 600
    assert "Kaladin" in kaladin.aliases


def test_merging_adds_a_newcomer() -> None:
    """A character the series has not met joins it with the voice just solved."""
    updated = merged_series(
        None,
        (_profile("shallan", "Shallan Davar"),),
        {"shallan": "aoife"},
        "eponine",
        "s",
    )
    assert updated.series_id == "s"
    assert updated.narrator_voice_id == "eponine"
    assert [c.canonical_id for c in updated.characters] == ["shallan"]
    assert updated.characters[0].voice_id == "aoife"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_series_identity.py -q --no-cov`
Expected: FAIL with `ModuleNotFoundError: No module named 'kenkui._characters.series'`

- [ ] **Step 3: Write the module**

Create `src/kenkui/_characters/series.py`:

```python
"""Deciding when a character in this volume is one the series already knows.

Pure functions over names and records: no model, no I/O. The rules are the
ones `identity` already applies within a book, which matters more here --
a series has more names competing for the same short forms than any single
volume does, so over-merging is likelier and costs more.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kenkui._characters.identity import detect_titles, same_person
from kenkui._characters.store import SeriesCharacter, SeriesRecord

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from kenkui._domain.casting import CharacterProfile


def match_roster(
    record: SeriesRecord | None, characters: Sequence[CharacterProfile]
) -> dict[str, str]:
    """Map this volume's character ids onto the series' canonical ids.

    An exact alias decides on its own: the series recorded that surface form
    against exactly one person. Otherwise every name this character is known
    by is compared to every name the series knows, and a match counts only
    when it lands on exactly one series character -- two candidates means the
    name belongs to neither, which is `resolve_short_forms`' rule and the
    reason a series does not quietly merge its two Charleses.
    """
    if record is None:
        return {}
    by_alias: dict[str, set[str]] = {}
    for known in record.characters:
        for alias in (*known.aliases, known.display_name):
            by_alias.setdefault(alias, set()).add(known.canonical_id)
    titles = detect_titles(sorted(by_alias))

    matched: dict[str, str] = {}
    for character in characters:
        names = (*character.aliases, character.display_name)
        hosts = {
            canonical
            for name in names
            for alias, owners in by_alias.items()
            for canonical in owners
            if alias == name or same_person(alias, name, titles)
        }
        if len(hosts) == 1:
            matched[character.id] = next(iter(hosts))
    return matched


def merged_series(  # noqa: PLR0913 - one call site, all inputs explicit.
    record: SeriesRecord | None,
    characters: Sequence[CharacterProfile],
    assignments: Mapping[str, str],
    narrator_voice_id: str,
    series_id: str,
) -> SeriesRecord:
    """Return the series as it stands after this volume.

    A returning character keeps the voice the series gave them and gains this
    volume's speech and surface forms. A newcomer joins with whatever voice
    the solver just chose.
    """
    matched = match_roster(record, characters)
    known = {c.canonical_id: c for c in (record.characters if record else ())}
    for character in characters:
        voice_id = assignments.get(character.id)
        if voice_id is None:
            continue
        canonical = matched.get(character.id, character.id)
        existing = known.get(canonical)
        aliases = {*character.aliases, character.display_name}
        if existing is None:
            known[canonical] = SeriesCharacter(
                canonical_id=canonical,
                display_name=character.display_name,
                gender=character.gender,
                voice_id=voice_id,
                spoken_characters=character.spoken_characters,
                aliases=tuple(sorted(aliases)),
            )
            continue
        known[canonical] = SeriesCharacter(
            canonical_id=canonical,
            display_name=existing.display_name,
            # The series keeps the first gender it was sure of: a later
            # volume answering None must not un-gender a cast character.
            gender=existing.gender if existing.gender is not None else character.gender,
            voice_id=existing.voice_id,
            spoken_characters=existing.spoken_characters + character.spoken_characters,
            aliases=tuple(sorted({*existing.aliases, *aliases})),
        )
    return SeriesRecord(
        series_id=series_id,
        narrator_voice_id=(
            record.narrator_voice_id if record is not None else narrator_voice_id
        ),
        characters=tuple(
            sorted(known.values(), key=lambda c: (-c.spoken_characters, c.canonical_id))
        ),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_series_identity.py -q --no-cov`
Expected: PASS

- [ ] **Step 5: Full suite, lint, typecheck, commit**

```bash
.venv/bin/pytest tests/ -q --no-cov && .venv/bin/ruff check src tests && .venv/bin/mypy src
git add src/kenkui/_characters/series.py tests/test_series_identity.py
git commit -m "feat: resolve a volume's roster against its series"
```

---

### Task 4: Seed the solver's voice-usage counter

`solve` starts `load` at zero for every voice, so volume 3 restarts the spread instead of continuing volume 1's. This is the whole of the "least-used voice" behaviour working series-wide.

**Files:**
- Modify: `src/kenkui/_domain/casting.py` (`CastingRequest`, `solve`)
- Test: `tests/test_casting_solver.py`

**Interfaces:**
- Produces: `CastingRequest.prior_load: Mapping[str, int]` defaulting to an empty dict; `solve` initialises `load` from it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_casting_solver.py`. `_voice`, `_character` and `_request` are the existing helpers at lines 22, 36 and 61.

```python
def test_prior_load_steers_the_next_volume() -> None:
    """A voice the series has already spent is not the least-used one.

    Without this the second volume restarts the count and hands its first
    character the same voice the first volume did.
    """
    characters = (_character("newcomer", "feminine", 100, ("ch1",)),)
    fresh = solve(_request(characters))
    spent = solve(
        _request(characters, prior_load={fresh.assignments["newcomer"]: 10_000})
    )
    assert spent.assignments["newcomer"] != fresh.assignments["newcomer"]


def test_no_prior_load_is_todays_behaviour() -> None:
    """A pipeline that never mentions a series must cast exactly as before."""
    characters = (_character("solo", "feminine", 100, ("ch1",)),)
    assert (
        solve(_request(characters)).assignments
        == solve(_request(characters, prior_load={})).assignments
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_casting_solver.py -k prior_load -v --no-cov`
Expected: FAIL with `TypeError: CastingRequest.__init__() got an unexpected keyword argument 'prior_load'`

- [ ] **Step 3: Add the field and use it**

In `CastingRequest`, after `method`:

```python
    # Voice usage carried in from outside this book, in spoken characters.
    # A series continues its spread across volumes rather than restarting
    # it; empty is exactly today's behaviour.
    prior_load: Mapping[str, int] = MappingProxyType({})
```

Import `MappingProxyType` from `types` — a plain `{}` default is mutable and
shared across instances.

In `solve`, replace the `load` initialisation:

```python
load: dict[str, int] = {voice.id: request.prior_load.get(voice.id, 0) for voice in pool}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_casting_solver.py -q --no-cov`
Expected: PASS

- [ ] **Step 5: Full suite, lint, typecheck, commit**

```bash
.venv/bin/pytest tests/ -q --no-cov && .venv/bin/ruff check src tests && .venv/bin/mypy src
git add src/kenkui/_domain/casting.py tests/test_casting_solver.py
git commit -m "feat: let a cast continue an earlier volume's voice spread"
```

---

### Task 5: The `.series()` pipeline method

**Files:**
- Modify: `src/kenkui/_domain/operations.py` (new `Series` operation, union member)
- Modify: `src/kenkui/pipeline.py` (the method)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `Series(series_id: str, book: int | None = None, allow_recast: bool = False, allow_narrator_change: bool = False)`; `Pipeline.series(...) -> Pipeline`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:

```python
def test_series_records_intent_without_reading_anything() -> None:
    """Membership is declared, never derived: no EPUB carries it."""
    pipeline = kk.epub("book.epub").series("stormlight", book=3)
    recorded = pipeline.operations[-1]
    assert recorded.series_id == "stormlight"
    assert recorded.book == 3
    assert recorded.allow_recast is False
    assert recorded.allow_narrator_change is False


def test_an_empty_series_id_is_refused() -> None:
    """A series with no name cannot be looked up again."""
    with pytest.raises(kk.ValidationError) as error:
        kk.epub("book.epub").series("   ")
    assert error.value.code == kk.ErrorCode.INVALID_SERIES


def test_a_negative_book_number_is_refused() -> None:
    with pytest.raises(kk.ValidationError) as error:
        kk.epub("book.epub").series("stormlight", book=0)
    assert error.value.code == kk.ErrorCode.INVALID_SERIES


def test_two_series_declarations_are_refused() -> None:
    """A book belongs to one series."""
    with pytest.raises(kk.ValidationError) as error:
        kk.epub("book.epub").series("a").series("b")
    assert error.value.code == kk.ErrorCode.DUPLICATE_OPERATION
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_pipeline.py -k series -v --no-cov`
Expected: FAIL with `AttributeError: 'Pipeline' object has no attribute 'series'`

- [ ] **Step 3: Add the error code**

In `src/kenkui/errors.py`, beside the other casting codes:

```python
    INVALID_SERIES = "invalid_series"
```

and in the message table:

```python
    ErrorCode.INVALID_SERIES: "A series needs a name and a positive book number.",
```

- [ ] **Step 4: Add the operation**

In `src/kenkui/_domain/operations.py`, beside the other intent records, and add
`| Series` to the `Operation` union:

```python
@dataclass(frozen=True, slots=True)
class Series:
    """Which series this book belongs to, and how strictly to honour it.

    Declared, never derived: no EPUB in practice carries series metadata,
    so there is nothing to read it from.
    """

    series_id: str
    book: int | None = None
    allow_recast: bool = False
    allow_narrator_change: bool = False
```

- [ ] **Step 5: Add the method**

In `src/kenkui/pipeline.py`, beside the other intent methods:

```python
    def series(
        self,
        series_id: str,
        *,
        book: int | None = None,
        allow_recast: bool = False,
        allow_narrator_change: bool = False,
    ) -> Pipeline:
        """Return a branch tying this book to a series.

        A character the series already cast keeps their voice, and voices
        continue spreading across volumes rather than restarting. ``book`` is
        recorded for ordering only: continuity is decided by identity, not by
        volume number.

        Two ways a series can be contradicted fail before any model call:
        a pinned voice missing from the pool, and a narrator differing from
        the one the series recorded. ``allow_recast`` re-solves the affected
        characters and updates their pins; ``allow_narrator_change`` adopts
        the new narrator from this volume on. Both log when they take effect.
        """
        name = series_id.strip()
        if not name or (book is not None and book < 1):
            raise ValidationError(ErrorCode.INVALID_SERIES)
        return self._append(
            Series(
                series_id=name,
                book=book,
                allow_recast=allow_recast,
                allow_narrator_change=allow_narrator_change,
            ),
            before_tts=True,
        )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_pipeline.py -k series -v --no-cov`
Expected: PASS. `test_two_series_declarations_are_refused` passes via the
existing `append_unique` behaviour in `_append`.

- [ ] **Step 7: Update the export test and commit**

`tests/test_pipeline.py::test_public_exports_are_intentional` pins the public
name set; no new public name is added by this task, so it should still pass.
Run the full suite to confirm.

```bash
.venv/bin/pytest tests/ -q --no-cov && .venv/bin/ruff check src tests && .venv/bin/mypy src
git add src/kenkui/_domain/operations.py src/kenkui/pipeline.py src/kenkui/errors.py tests/test_pipeline.py
git commit -m "feat: declare a book's series membership on the pipeline"
```

---

### Task 6: Refuse a render that would contradict the series

Both checks are knowable without a model call, so they belong in `validate()` beside the existing intent rules rather than at render time.

**Files:**
- Modify: `src/kenkui/errors.py` (two codes)
- Modify: `src/kenkui/validation.py` (the rules)
- Modify: `src/kenkui/pipeline.py` (`validate` calls them with the pool and the series)
- Test: `tests/test_series_validation.py` (create)

**Interfaces:**
- Consumes: `store.read_series` (Task 2), `Series` (Task 5).
- Produces: `ErrorCode.SERIES_VOICE_MISSING`, `ErrorCode.SERIES_NARRATOR_CHANGED`; `series_intent_errors(operations, record, pool_ids) -> tuple[ErrorCode, ...]` in `validation.py`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_series_validation.py`:

```python
"""A series is contradicted before a model call, not after paying for one."""

from __future__ import annotations

from kenkui._characters import store
from kenkui._domain.operations import AssignVoices, Series
from kenkui.errors import ErrorCode
from kenkui.validation import series_intent_errors


def _record() -> store.SeriesRecord:
    return store.SeriesRecord(
        series_id="s",
        narrator_voice_id="eponine",
        characters=(
            store.SeriesCharacter(
                "kaladin", "Kaladin", "masculine", "alf", 100, ("Kaladin",)
            ),
        ),
    )


def _operations(**overrides: object) -> tuple[object, ...]:
    series: dict[str, object] = {"series_id": "s"}
    series.update(overrides)
    return (
        AssignVoices(
            narrator_voice_id=overrides.pop("narrator", "eponine"),  # type: ignore[arg-type]
            unknown_voice_id="eponine",
            cast=(),
            method="gendered",
        ),
        Series(**series),  # type: ignore[arg-type]
    )


def test_a_pinned_voice_missing_from_the_pool_fails() -> None:
    """The operator unloaded a voice between volumes."""
    errors = series_intent_errors(_operations(), _record(), frozenset({"aoife"}))
    assert ErrorCode.SERIES_VOICE_MISSING in errors


def test_allow_recast_accepts_the_loss() -> None:
    errors = series_intent_errors(
        _operations(allow_recast=True), _record(), frozenset({"aoife"})
    )
    assert ErrorCode.SERIES_VOICE_MISSING not in errors


def test_a_changed_narrator_fails() -> None:
    """A series narrator changing is almost always a mistake."""
    errors = series_intent_errors(
        _operations(narrator="marius"), _record(), frozenset({"alf", "marius"})
    )
    assert ErrorCode.SERIES_NARRATOR_CHANGED in errors


def test_allow_narrator_change_accepts_it() -> None:
    errors = series_intent_errors(
        _operations(narrator="marius", allow_narrator_change=True),
        _record(),
        frozenset({"alf", "marius"}),
    )
    assert ErrorCode.SERIES_NARRATOR_CHANGED not in errors


def test_a_first_volume_has_nothing_to_contradict() -> None:
    """No stored series means no constraint."""
    assert series_intent_errors(_operations(), None, frozenset({"alf"})) == ()


def test_a_pipeline_without_a_series_is_unaffected() -> None:
    """Every existing pipeline keeps validating exactly as it did."""
    only_voices = (
        AssignVoices(
            narrator_voice_id="eponine",
            unknown_voice_id="eponine",
            cast=(),
            method="gendered",
        ),
    )
    assert series_intent_errors(only_voices, _record(), frozenset()) == ()
```

`AssignVoices`' fields are `narrator_voice_id`, `unknown_voice_id`, `cast`
and `method` (`src/kenkui/_domain/operations.py:77`), which is what the
helper above builds.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_series_validation.py -q --no-cov`
Expected: FAIL with `ImportError: cannot import name 'series_intent_errors'`

- [ ] **Step 3: Add the error codes**

In `src/kenkui/errors.py`:

```python
    SERIES_VOICE_MISSING = "series_voice_missing"
    SERIES_NARRATOR_CHANGED = "series_narrator_changed"
```

```python
    ErrorCode.SERIES_VOICE_MISSING: (
        "A voice this series already cast is not in the pool."
    ),
    ErrorCode.SERIES_NARRATOR_CHANGED: (
        "This series was recorded with a different narrator voice."
    ),
```

- [ ] **Step 4: Write the rule**

In `src/kenkui/validation.py`:

```python
def series_intent_errors(
    operations: Sequence[Operation],
    record: SeriesRecord | None,
    pool_ids: frozenset[str],
) -> tuple[ErrorCode, ...]:
    """Return the ways this render would contradict its series.

    Checked here rather than at render time because both are knowable from
    the store and the operations alone: failing after a book has been
    attributed spends a model pass to learn something free.
    """
    series = next((item for item in operations if isinstance(item, Series)), None)
    if series is None or record is None:
        return ()
    errors: list[ErrorCode] = []
    if not series.allow_recast and any(
        character.voice_id not in pool_ids for character in record.characters
    ):
        errors.append(ErrorCode.SERIES_VOICE_MISSING)
    narrator = next(
        (
            item.narrator_voice_id
            for item in operations
            if isinstance(item, AssignVoices)
        ),
        None,
    )
    if (
        not series.allow_narrator_change
        and narrator is not None
        and narrator != record.narrator_voice_id
    ):
        errors.append(ErrorCode.SERIES_NARRATOR_CHANGED)
    return tuple(errors)
```

Import `Series` and `AssignVoices` from `._domain.operations` and
`SeriesRecord` under `TYPE_CHECKING`.

- [ ] **Step 5: Call it from validate()**

In `Pipeline.validate`, after the existing `render_intent_errors` extension:

```python
series = next((item for item in self.operations if isinstance(item, Series)), None)
if series is not None:
    from ._characters import store  # noqa: PLC0415 - store is only

    # touched by a pipeline that declared a series.
    from .voices.provision import list_voices  # noqa: PLC0415

    issues.extend(
        _issue(code)
        for code in series_intent_errors(
            self.operations,
            store.read_series(series.series_id),
            frozenset(voice.id for voice in list_voices() if voice.state == "loaded"),
        )
    )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_series_validation.py -q --no-cov`
Expected: PASS

- [ ] **Step 7: Full suite, lint, typecheck, commit**

```bash
.venv/bin/pytest tests/ -q --no-cov && .venv/bin/ruff check src tests && .venv/bin/mypy src
git add src/kenkui/errors.py src/kenkui/validation.py src/kenkui/pipeline.py tests/test_series_validation.py
git commit -m "feat: refuse a render that would contradict its series"
```

---

### Task 7: Wire the series into resolution

**Files:**
- Modify: `src/kenkui/pipeline.py` (`_resolve_all`)
- Test: `tests/test_series_resolution.py` (create)

**Interfaces:**
- Consumes: `match_roster`, `merged_series` (Task 3); `CastingRequest.prior_load` (Task 4); `store.read_series`, `store.write_series` (Task 2).

- [ ] **Step 1: Write the failing test**

Create `tests/test_series_resolution.py`:

```python
"""Volume two sounds like volume one."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import kenkui as kk
from kenkui._characters import store
from tests.helpers import make_epub, xhtml

if TYPE_CHECKING:
    from pathlib import Path


class _Roster:
    """Names one character, and attributes every quote to them."""

    def __init__(self, character_id: str) -> None:
        self.character_id = character_id

    def complete(self, model: str, prompt: str) -> str:
        assert model
        if "List the speaking characters" in prompt:
            return json.dumps(
                {
                    "characters": [
                        {
                            "id": self.character_id,
                            "name": self.character_id.title(),
                            "gender": "feminine",
                        }
                    ]
                }
            )
        return json.dumps(
            {"attributions": [{"quote_id": 0, "speaker": self.character_id}]}
        )


def _render_volume(tmp_path: Path, character_id: str, book: int) -> dict[str, str]:
    """Resolve one volume of a series and return its cast assignments."""
    path = make_epub(
        tmp_path / f"volume-{book}.epub",
        chapters={"one": xhtml(f'<h1>One</h1><p>"Hello," said {character_id}.</p>')},
        spine=("one",),
    )
    resolved = (
        kk.epub(path)
        .series("s", book=book)
        .infer_characters("fake/model")
        .attribute_quotes("fake/model")
        .assign_voices(narrator="eponine")
        .resolve()
    )
    return dict(resolved._resolved.cast_assignments)


def test_a_returning_character_keeps_their_voice(tmp_path: Path) -> None:
    """The whole point: one person, one voice, across volumes."""
    first = _render_volume(tmp_path, "javert", book=1)
    second = _render_volume(tmp_path, "javert", book=2)
    assert first["javert"] == second["javert"]


def test_a_newcomer_does_not_take_a_spent_voice(tmp_path: Path) -> None:
    """Volume two keeps spreading rather than restarting the count."""
    first = _render_volume(tmp_path, "javert", book=1)
    second = _render_volume(tmp_path, "cosette", book=2)
    assert second["cosette"] != first["javert"]


def test_a_book_outside_a_series_touches_no_series_state(tmp_path: Path) -> None:
    """Every existing pipeline keeps behaving exactly as it did."""
    path = make_epub(
        tmp_path / "solo.epub",
        chapters={"one": xhtml('<h1>One</h1><p>"Hello," said javert.</p>')},
        spine=("one",),
    )
    kk.epub(path).assign_voice("eponine").resolve()
    assert store.list_series() == ()
```

The model calls have to be intercepted: `_render_volume` as written would
reach a provider. Inject `_Roster(character_id)` the way
`tests/test_attribution.py` injects `ScriptedClient` — read that file's
`resolve_attribution(..., client=...)` calls and thread the same `client`
argument through, adding a `client` parameter to `_render_volume` if the
pipeline does not accept one directly. `make_epub` and `xhtml` are the
existing helpers `tests/test_write_preflight.py:32` uses; import them from
wherever that module imports them.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_series_resolution.py -q --no-cov`
Expected: FAIL — voices differ between volumes, because nothing reads the series.

- [ ] **Step 3: Wire it into `_resolve_all`**

In `src/kenkui/pipeline.py`, in `_resolve_all` after `record` is obtained and
before `resolve_cast` is called:

```python
series = next((item for item in pipeline.operations if isinstance(item, Series)), None)
stored = store.read_series(series.series_id) if series is not None else None
explicit = dict(casting.cast)
prior_load: dict[str, int] = {}
if stored is not None:
    by_canonical = {c.canonical_id: c for c in stored.characters}
    for book_id, canonical in match_roster(stored, record.characters).items():
        known = by_canonical[canonical]
        # A pin the pool cannot honour only reaches here under
        # allow_recast; validate() refuses it otherwise. Dropping it
        # lets the solver choose afresh, which is what was asked for.
        if known.voice_id in {voice.id for voice in pool}:
            explicit[book_id] = known.voice_id
    for known in stored.characters:
        prior_load[known.voice_id] = (
            prior_load.get(known.voice_id, 0) + known.spoken_characters
        )
```

Pass `explicit=explicit` and `prior_load=prior_load` into the `CastingRequest`,
and after `resolve_cast` returns, persist the series:

```python
    if series is not None:
        store.write_series(
            merged_series(
                stored,
                record.characters,
                outcome.assignments,
                casting.narrator_voice_id,
                series.series_id,
            )
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_series_resolution.py -q --no-cov`
Expected: PASS

- [ ] **Step 5: Log the forced overrides**

When `series.allow_recast` dropped at least one pin, or
`series.allow_narrator_change` is adopting a different narrator, `log_event`
at `logging.WARNING` with `boundary="series"`, following `_log_collisions`.
A forced render must still say what it did.

- [ ] **Step 6: Full suite, lint, typecheck, commit**

```bash
.venv/bin/pytest tests/ -q --no-cov && .venv/bin/ruff check src tests && .venv/bin/mypy src
git add src/kenkui/pipeline.py tests/test_series_resolution.py
git commit -m "feat: cast a volume against the series it belongs to"
```

---

### Task 8: Public listing and removal, and the docs

**Files:**
- Modify: `src/kenkui/__init__.py` (exports)
- Modify: `tests/test_pipeline.py` (`test_public_exports_are_intentional`)
- Modify: `docs/usage.md`
- Test: `tests/test_series_store.py`

**Interfaces:**
- Consumes: `store.list_series`, `store.remove_series` (Task 2).
- Produces: public `kk.list_series()`, `kk.remove_series(series_id)`, `kk.SeriesRecord`, `kk.SeriesCharacter`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_series_store.py`:

```python
def test_the_public_pair_is_exported() -> None:
    """Mirrors list_castings / remove_casting: see it, or start over."""
    import kenkui as kk

    assert callable(kk.list_series)
    assert callable(kk.remove_series)
    assert kk.SeriesRecord is store.SeriesRecord
    assert kk.SeriesCharacter is store.SeriesCharacter
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_series_store.py -k exported -v --no-cov`
Expected: FAIL with `AttributeError: module 'kenkui' has no attribute 'list_series'`

- [ ] **Step 3: Export them**

In `src/kenkui/__init__.py`, extend the existing `._characters.store` import
to bring in `SeriesCharacter`, `SeriesRecord`, `list_series` and
`remove_series`, and add all four names to `__all__` in sorted position.

- [ ] **Step 4: Update the export contract test**

`tests/test_pipeline.py::test_public_exports_are_intentional` pins the exact
public name set. Add `"SeriesCharacter"`, `"SeriesRecord"`, `"list_series"`
and `"remove_series"` to it in sorted position.

- [ ] **Step 5: Document it**

Add a `## Series` section to `docs/usage.md` after the casting material:

````markdown
## Series

A character who appears in several volumes should sound like one person in all
of them. Declare which series a book belongs to and Kenkui keeps their voice:

```python
kk.epub("oathbringer.epub").series("stormlight", book=3)
```

Membership is declared, never derived. EPUBs do not carry series metadata in
practice — Calibre keeps it in its own database — so the name is yours to
choose, and `book` is recorded for ordering only. Continuity is decided by
who the characters are, not by volume number.

A character the series already cast keeps their voice. A newcomer is cast from
the least-used voices of their gender, counting what earlier volumes already
spent, so voices keep spreading across the series rather than restarting each
book.

Two things fail before any model call runs:

| code | meaning | override |
|---|---|---|
| `series_voice_missing` | a voice this series already cast is no longer loaded | `allow_recast=True` |
| `series_narrator_changed` | the narrator differs from the one recorded | `allow_narrator_change=True` |

Each override logs a warning when it takes effect, so a forced render still
says what it did.

`list_series()` shows every series and its cast in prominence order.
`remove_series(series_id)` drops its pins, and the next volume is cast fresh.
````

- [ ] **Step 6: Full suite, lint, typecheck, commit**

```bash
.venv/bin/pytest tests/ -q --no-cov && .venv/bin/ruff check src tests && .venv/bin/mypy src
git add src/kenkui/__init__.py tests/ docs/usage.md
git commit -m "feat: expose a series' cast for listing and removal"
```
