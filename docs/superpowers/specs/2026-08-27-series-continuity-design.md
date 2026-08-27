# Series continuity

Status: draft for review
Date: 2026-08-27

## Problem

A character who appears in five volumes of a series should sound like one
person across all five. Today every book is cast independently: `book_id` is
the widest scope the store knows (`_characters/store.py:37`), so nothing
carries a voice from one volume to the next. Rendering *The Way of Kings* and
then *Oathbringer* gives Kaladin two unrelated voices.

Two facts constrain the design.

**Series membership cannot be derived.** Of 40 EPUBs sampled from the user's
Calibre library, **none** carry series metadata. Calibre keeps series in its
own database and does not write `calibre:series` or `belongs-to-collection`
into the OPF; Oathbringer's metadata has `calibre:title_sort` and
`calibre:timestamp` and nothing else. Membership must be declared by the
caller.

**Character ids are not stable across books, for the same reason they are not
stable across chapters.** The roster prompt asks for an id "stable across the
whole book" and models do not reliably give one — that is what produced
`corwi` and `lizbyet-corwi` as two people in a single book, fixed in
`f95ac14`. Across volumes the problem is strictly worse: book 1 says
`kaladin-stormblessed`, book 3 says `kaladin`, and no shared prompt context
connects them.

## What already works

The solver needs no new policy for a series. `_domain/casting.py:170`:

```python
chosen = min(
    free or admissible,
    key=lambda voice: (load.get(voice.id, 0), voice.id),
)
```

`admissible` is already the gender-matched subset of the pool, and `load`
counts spoken characters per voice, so the solver picks the least-used voice
of the right gender and only repeats one when the pool is exhausted. That is
exactly the behaviour a series wants; it just needs to see the series rather
than one book.

Two changes make the existing solver do all the work:

- characters the series already knows are passed as `explicit`, which the
  solver already honours as pins (`casting.py:145`);
- `load` is initialised from the series' accumulated usage instead of zero, so
  volume 3 keeps spreading across voices rather than restarting the count.

No pinning policy, no threshold, no "who deserves a series voice" rule.
Recurring characters keep their voice because they are pinned; newcomers are
cast by the same least-used rule that already governs a single book. Voice
reuse when the pool runs out behaves series-wide exactly as it does within a
book, and `Collision` reporting is unchanged.

## Design

### A. Declaring membership

```python
kk.epub("oathbringer.epub").series("stormlight", book=3)
```

A new pipeline method recording a `Series(series_id, book)` operation.
`series_id` is a caller-chosen string; there is nothing to derive it from.
`book` is optional, used for ordering in `list_series()` output and for
nothing else — continuity is resolved by identity, not by volume number.

### B. Store

Two tables above `book_id`, mirroring the existing shapes:

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

`series_aliases` is what lets volume 3's `kaladin` find volume 1's
`kaladin-stormblessed`. `spoken_characters` accumulates across volumes and is
what seeds the solver's `load`.

Series records are **model-independent by design**. They key on the series and
the character, never on an attribution. Re-attributing volume 1 with a
different model must not re-cast a series.

### C. Resolving a volume against its series

1. Attribute the book exactly as today.
2. For each character in the merged roster, look for a series match: an exact
   alias hit first, then `identity.same_person` against the series' known
   display names, then `identity.resolve_short_forms` for bare names.
3. Matched characters contribute `book_character_id -> voice_id` to
   `explicit`. The key is the id **this book's** roster used, not the series
   canonical id: `solve` validates every explicit key against the book's own
   roster (`casting.py:141`), and a canonical id from volume 1 need not
   appear in volume 3's. The canonical id is the store's key; the book's id
   is the solver's.
4. `load` is seeded from `series_characters.spoken_characters`.
5. `solve` runs unchanged.
6. Newcomers are written back with their assigned voice, and every surface
   form seen for a character is written to `series_aliases`.

**This design has one prerequisite.** Step 6 needs the surface forms a
character was known by, and `CharacterProfile` does not carry them:
`merge_rosters` folds several display names into one character and keeps only
the head's. Volume 1 therefore records `Kaladin Stormblessed` and loses
`Kaladin`, which is the very name volume 3 will use.

`CharacterProfile.aliases` was designed for this in the character-identification
spec and closed unimplemented, because nothing then needed it. This does. It
must land first: without it, series matching falls back to a single display
name per character and the alias table cannot do its job.

The identity rules are the ones already in `_characters/identity.py`,
including the bias the module is built around: under-merging gives one person
two voices, over-merging gives two people one voice, and the second is worse.
That bias matters more across volumes than within one, because a series has
more names competing for the same short forms.

**Accepted cost.** A character whose name in volume 5 cannot be matched to
volume 1 is treated as a newcomer and gets a second voice. This is the same
failure the module already accepts within a book, and the same one accepted
for minted roles: a wrong-but-distinct voice beats a wrong merge.

### D. Failing before spending anything

Both ways a series can be violated are knowable without a model call, because
the series pins and the narrator voice are both in the store and the pipeline
operations. Both are therefore reported by `validate()`, before attribution
runs, alongside the existing intent checks.

**A pinned voice that is no longer in the pool** — the operator unloaded or
removed it between volumes — fails with `ErrorCode.SERIES_VOICE_MISSING`,
naming the characters and the lost voice.

**A narrator voice that differs from the one the series recorded** fails with
`ErrorCode.SERIES_NARRATOR_CHANGED`.

Each has an explicit override:

```python
.series("stormlight", book=3, allow_recast=True)
.series("stormlight", book=3, allow_narrator_change=True)
```

`allow_recast` re-solves the affected characters from the current pool and
updates their pins. `allow_narrator_change` records the new narrator as the
series narrator from this volume on. Both log a warning when they take effect,
so a forced render still says what it did.

Two named flags rather than one `force`, because they are different decisions
with different consequences, and a caller who accepts a lost voice has not
thereby agreed to change the narrator. **Open for review:** a single `force`
is smaller surface if that distinction is not worth two parameters.

### E. Public API

Mirroring the existing `list_castings` / `remove_casting` pair:

```python
def list_series() -> tuple[SeriesRecord, ...]: ...
def remove_series(series_id: str) -> None: ...
```

`SeriesRecord` carries `series_id`, `narrator_voice_id`, and
`characters: tuple[tuple[str, str, str | None, str], ...]` — canonical id,
display name, gender, voice id — ordered by accumulated speech, so the
listing reads as the cast in prominence order.

Removing a series drops its pins; the next render of any volume casts it
fresh. No per-character override in this iteration.

## Testing

- Identity resolution across volumes: volume 1's `Kaladin Stormblessed` and
  volume 3's `Kaladin` resolve to one canonical id and one voice; two
  different characters sharing a surname across volumes do not merge.
- A recurring character keeps their voice when volume 2 is cast; a newcomer
  gets the least-used voice of their gender, counting volume 1's usage.
- `validate()` reports `SERIES_VOICE_MISSING` and `SERIES_NARRATOR_CHANGED`
  before any model call, and both overrides suppress the failure and log.
- A series render with no prior volumes behaves exactly as a non-series render
  does today, including identical segment identities, so adopting `.series()`
  on a book already rendered does not re-synthesize it.
- `list_series` / `remove_series` round-trip.

## Out of scope

Series metadata on the M4B output. Per-character series overrides. An eager
"resolve the whole series at once" mode: this design is incremental, and a
reconcile pass over completed series is a separate feature if the incremental
result proves insufficient.
