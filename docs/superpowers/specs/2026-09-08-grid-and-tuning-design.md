# Addressable Grid and Per-Book Tuning

Date: 2026-09-08
Status: design, awaiting review

## Purpose

Kenkui can render a book but cannot correct one. Today a book is tuned by
five global pause durations, a per-series pronunciation dict living in the
driver script, and whole-book attribution that is either right or wrong.
There is no way to say "this line is Jessica, not the narrator", "drop the
pause here", or "in this chapter, *lead* rhymes with *feed*".

This design adds that capability, and the substrate it needs: a durable way
to address any position in a book, and a per-book document that records
corrections against those addresses.

The eventual consumer is an annotation GUI — something closer to music
transcription software than a text editor, where the book is displayed as
addressable rows and corrections are made by pointing. That GUI is **not**
in scope here. What is in scope is the addressing scheme and the annotation
layer it writes to, designed so the GUI is a client of an existing format
rather than the thing that invents one.

## Goals

- Address any position in a book durably, at any resolution.
- Correct speaker attribution for an individual line or a pattern of lines.
- Add, retune, and remove silence at an individual position.
- Scope pronunciation to a region rather than the whole book.
- Persist all of the above in a per-book file that travels with the EPUB,
  diffs in git, and is hand-editable.
- Make the iterate → listen → correct loop cheap enough to run repeatedly on
  one book.
- Unify the four partitioning schemes the codebase currently maintains
  separately.

## Non-goals

- The annotation GUI itself.
- Coreference and gender-assignment quality. That is a separate project
  (see *Relationship to other work*).
- Arbitrary character-range annotation. Addressing is grid-quantized.
- Changing which TTS engine is used or how voices are cast.

## Architecture

Three layers, strictly ordered:

```
Layer 0   the grid          derived from chapter text alone; no settings, no model
Layer 1   tuning            per-book corrections anchored to Layer 0
Layer 2   compilation       planning merges 0 and 1 into segments and silence
```

The load-bearing rule: **nothing anchors to anything but Layer 0.** A manual
split is itself a Layer 1 annotation against Layer 0, not a modification of
the grid other annotations see. This keeps annotations composable and lets
Layer 0 improve without orphaning them.

---

## 1. The addressable grid (Layer 0)

### Why a new partition

The codebase already maintains three exact partitions, each satisfying the
same invariant — concatenating the pieces reproduces the chapter text
exactly:

| partition | location | why it cannot serve as the grid |
|---|---|---|
| dialogue / narration | `_characters/quotes.py:120` | cannot name unquoted text; all narration between quotes is one span |
| structural pieces | `_domain/structure.py:120` | shape depends on pause settings via `break_tiers` (`structure.py:64`) |
| TTS chunks | `_domain/planning.py:652` | shape depends on chunk budget and engine limits |

The second row is the trap. `break_tiers` derives boundaries from which pause
durations are non-zero, so anchoring to structural pieces would mean turning
on `line_ms` silently re-shapes the grid and moves every anchor after it.

The grid must therefore be a fourth partition whose definition depends on
**nothing but the canonical chapter text**. No tunables, no model, no
settings. That is the operative meaning of "bulletproof".

### The hierarchy

```
chapter → paragraph → line → sentence → phrase → word → character
```

- **chapter** — `ChapterInspection` (`inspection.py:24`), exists
- **paragraph** — `structure._blocks` (`structure.py:94`), splits on `\n{2,}`, exists
- **line** — `structure._lines` (`structure.py:106`), splits on `\n`, exists
- **sentence** — new; `[.!?…]` with an abbreviation guard
- **phrase** — new; `[,;:]` — clause-level, and the level TTS chunking targets
- **word / character** — derived on demand, never materialized

Five of seven already exist as implementations; they are not currently
organized as a hierarchy.

**Materialization stops at phrase.** A novel is ~150k words; building objects
below phrase per book is waste. The *address space* extends to the character,
but word and character coordinates resolve lazily inside a phrase when
something asks. Only pronunciation and the pathological chunk fallback ask.

### Quotes are not a level

Dialogue crosscuts the hierarchy in both directions: one quote may span
several sentences (`"Get out. Now. I mean it."`), and one sentence may
contain a quote plus its tag (`"Yes," she said.`).

Resolution: **quote edges force a boundary at whatever level they land in,
and `is_dialogue` becomes an attribute of a unit rather than a rank in the
tree.** The second example resolves for free — the comma inside the quote is
already a phrase delimiter, so it splits into `"Yes,"` / `she said.` with the
quote occupying exactly one phrase.

### Split rules

In priority order, all computed from canonical chapter text alone:

1. **Block boundary** (`\n{2,}`) — always splits. Paragraph level.
2. **Single newline** — always splits. Line level. Carries verse, epigraphs,
   and Herbert's Irulan headnotes.
3. **Quote-run edges** — every span from `extract_spans` opens and closes a
   unit. This rule does most of the real work: it makes every quoted
   utterance individually addressable by construction.
4. **Sentence-terminal punctuation** — `[.!?…]+["')\]]*\s+`, guarded against
   a short abbreviation list and single-initial capitals.
   `identity.PREFIX_TITLES` (`_characters/identity.py:24`) is reusable as-is.
5. **Clause punctuation** — `[,;:]["')\]]*\s+`. Phrase level.

### Bias toward splitting

The grid **addresses**; it does not **segment**. A grid boundary becomes a
TTS segment boundary only if an annotation attaches to it or chunking needs
it. This makes the two failure directions asymmetric:

- **Over-splitting** (`Dr.` | `Yueh smiled.`) — two adjacent units, both
  narrator, nothing attaches, no boundary forced. Zero effect on audio. Cost
  is one extra row in a future UI.
- **Under-splitting** (`"Yes," she said. "No," he answered.` as one unit) —
  two speakers trapped in one addressable unit. The correction cannot be
  expressed at all.

Under-splitting destroys expressiveness; over-splitting is cosmetic.
**The splitter is therefore biased to split.** This is why the abbreviation
guard in rule 4 is allowed to be short and imperfect.

### Exactness is the safety property

Boundary *placement* is a quality metric to tune. Partition *exactness* is a
safety property to assert:

```
"".join(units) == chapter.text
```

deterministic, and dependent only on the text. This matches the invariant
asserted at `_characters/quotes.py:7-10`, `planning.py:617`, and the final
check in `_chunk_span`. Property-test it across the Calibre library.

An English sentence splitter cannot be made linguistically correct on
arbitrary prose, and chasing that is a tar pit. Only exactness needs to be
bulletproof.

---

## 2. Addressing

### Labeled paths

A path is a labeled map, not a positional array:

```json
{"chapter": "xhtml/ch08", "paragraph": 3, "line": 1, "sentence": 2}
```

Labeled rather than positional so that truncated addresses are
self-describing and so a level can be added later without rewriting existing
sidecars.

**The chapter component is a stable spine ID, not an index.**
`select_chapters` (`pipeline.py:108`) already uses stable IDs precisely so
they survive source changes. Chapter therefore accepts an ID, a list of IDs,
or `*` — but not a range, since IDs are unordered. Chapter ranges remain with
`select_chapter_range` (`pipeline.py:119`), which owns spine ordering.

### Prefix addressing

A truncated path addresses the whole subtree. `{"chapter": "xhtml/ch08",
"paragraph": 3}` is all of paragraph 3. `{"chapter": "xhtml/ch08"}` is the
chapter. The empty path is the book.

**Coarser anchors are more stable anchors.** An annotation on a paragraph is
immune to every change below the paragraph level — tightening the
abbreviation guard, recovering an italic run, or fixing a quote-parsing bug
renumbers sentences inside that paragraph without touching it. The guidance
is therefore: *use the coarsest path that expresses your intent.*

### This eliminates the exact-tiling invariant

`_spans_for` (`planning.py:607`) currently requires `SpeakerSpan`s to tile
the chapter exactly, reconstructs the text, and raises `EMPTY_SPEECH` at
`planning.py:617` when it does not match. That check exists because a span is
a raw offset range with no other way to prove coverage.

With subtree addressing, coverage holds by construction: **the empty path
defaults to the narrator, and everything else is an override.** No partition
to validate, no reconstruction, no `EMPTY_SPEECH` on this path. Attribution
output shrinks from every span in the chapter to only the spans that are not
narrator.

### Set-valued components

Any component may be a set, which turns a path into a pattern matching a set
of nodes:

| form | meaning |
|---|---|
| `3` | concrete |
| `"*"` | all |
| `[2, 4, 7]` | explicit list |
| `"2..4"` | range |
| `-1` | last |

This is the whole vocabulary. No predicates, no text matching, no every-Nth —
that is where a pattern language becomes a programming language. If a rule
cannot be expressed, write the paths out.

It enables things like `{"chapter": "*", "paragraph": 1}` → Irulan, which
covers Dune's chapter-opening epigraphs in one rule.

### Content hashes and match counts

An exact path carries the **content hash** of what it addressed. Path gives
locality and legibility; hash gives verification. When they disagree, the
system surfaces a conflict rather than silently applying an annotation to the
wrong sentence — matching how `identity.py` drops ambiguous names rather than
guessing, and how roster checkpoints refuse with `source_changed`.

A pattern cannot be content-hashed. Patterns therefore record their **match
count at authoring time**, and a change in that count on load raises a
warning. This is the pattern-world analogue of the hash, and it catches the
case where a book gains a front-matter chapter and `paragraph: 1` silently
re-aims.

### Gaps are one-sided and normalized

Silence anchors to the gap **after** a path. One side only, so a gap has
exactly one address and two annotations cannot fight over it from opposite
directions.

"After" rather than "before" because `planning.py:591-593` already attaches
silence to the end of a segment; matching existing semantics avoids a
translation layer. The cost is that book-opening silence has no anchor; that
is a render-level lead-in setting, not an annotation.

**Normalization is the other half.** Subtree addressing reintroduces
collisions by another door: `after {paragraph: 3}` and `after {paragraph: 3,
sentence: -1}` denote the same physical gap. Every anchor therefore
normalizes by resolving its path to its last leaf; that leaf's trailing gap
is the key. Two annotations meaning the same gap then genuinely collide and
go through precedence, rather than both applying with one silently winning.

---

## 3. The tuning layer (Layer 1)

### Three operations, each an ordered rule tuple

`append_unique` (`_domain/operations.py:136`) enforces one operation per
type. Tuning therefore accumulates into a single operation per kind rather
than appending one operation per rule:

- `Attributions` — ordered tuple of (pattern, character_id)
- `Silences` — ordered tuple of (pattern, duration_ms)
- `Pronunciations` — ordered tuple of (pattern, word→replacement mapping)
- `GridEdits` — ordered tuple of splits and merges

This is the shape `pauses()` already uses — five keyword durations in one
call rather than five calls — and it keeps the type-uniqueness invariant
intact while making the rule list a first-class ordered collection, which is
what the precedence tiebreaker needs.

Calling `.attribute()` twice **extends** the tuple. It does not raise.

Because these operations replace rather than append, they cannot route
through `_append()`; they need a replace-or-create helper that also preserves
`_resolved` (see *Where the merge happens*).

`Annotations` — the sidecar loader carrying the path, digest, and per-kind
load counts — classifies as tuning, though it holds no rules of its own.

### Grid edits

The escape hatch for a badly-placed boundary. A split is "unit X, break at
inner offset N"; a merge is "units X through Y are one". Both anchor to
Layer 0, never to the effective grid, so they compose and a Layer 0
improvement does not orphan them.

Given the eager-split bias, **merge is expected to be the common operation**
and split the rare one. If that expectation holds, splits may prove
unnecessary entirely — see *Open questions*.

### Silence semantics

Derived tier reasons take the **maximum** among themselves, as
`structure.py:1-7` already specifies: "Its duration is the maximum of them,
never the sum." A manual silence then **replaces** that result outright.

There is exactly one mode, `set`. `set: 0` removes a pause. There is no
additive mode — hand-placed silences are absolute and will not scale when
global tiers are retuned, which is accepted in exchange for one rule instead
of two.

### Pronunciation is literal, not regex

Entries remain literal strings with the existing semantics in
`_domain/spoken/lexicon.py`: word-boundary lookarounds (`_LB`/`_RB`),
case- and diacritic-insensitive matching (`_fold`), capitalization-shape
transfer onto the replacement (`_shaped`), and single-shot matching that
cannot chain or loop.

Regex was considered and rejected. It would have bought morphological
variants in one rule (`Harkonnens`) and nothing else that the current
implementation lacks, at the cost of making `_shaped()` meaningless and
introducing catastrophic backtracking as a failure mode on a code path that
runs over every segment and takes part in segment identity — with no timeout
available in Python's `re`.

What the lexicon gains here is **scope**. Anchoring a pronunciation to a
region makes homographs expressible (*lead*, *read*, *bow*), which is
impossible today, and moves the lexicon out of the driver script and into
the book's own file.

---

## 4. Precedence

Two mechanisms, applied in order.

### Provenance first

```
0. narrator default          the empty path
1. machine attribution       from casting.sqlite3
2. human tuning              Layer 1 rules
```

A hand-written rule always beats machine attribution, regardless of
specificity. Without this, machine attribution — which emits fully concrete
paths — would beat every hand-written pattern under the specificity rule
below, inverting the entire model.

### Specificity within the human layer

**Strict subset wins, order-free.** `{chapter: "ch08", paragraph: 3}` matches
a strict subset of `{chapter: "ch08", paragraph: "*"}`, so it wins no matter
which order they appear in. Subset is computed exactly, per component.

**Overlap without subset falls back to declaration order**, later wins, and
raises a warning. Example:

```
A = {chapter: "ch08", paragraph: "*"}   → paul
B = {chapter: "*",    paragraph: 1}     → irulan
```

Both match `ch08 ¶1`. Neither match set contains the other. Any
component-counting or left-to-right scheme would pick one for no principled
reason — this incomparability is the actual source of CSS specificity's
unpredictability. Declaration order is deterministic and authorable: it is
fixed by moving a line.

Declaration order comes free from the rule tuple's ordering.

### Consequences

A fully concrete path is a subset of everything matching it, so exact
overrides win automatically. Storing them separately from patterns in the
sidecar is a storage-ergonomics decision (a GUI appending thousands of clicks
should not churn hand-written rules), not a precedence rule.

---

## 5. The three tiers and the API

### The tiers

| tier | changes when | lives in | applied by |
|---|---|---|---|
| **identity** | every book | the library / driver table | `.metadata()`, `.series()` |
| **tuning** | every book, as it is dialed in | the sidecar | `where=` methods |
| **style** | when taste changes | a reusable style function | `.pipe(style)` |

Different lifetimes, different storage, different application mechanism.

**The rule that makes the tier legible:**

> If it takes `where`, it is about this book. If it does not, it is about
> your style.

Mechanically true, visible in autocomplete, and it extends automatically to
any method added later. It is also the rule that decides serialization, so
one concept does three jobs.

Known wart: `.metadata()` and `.series()` take no `where` but are per-book.
They are the identity tier — their values come from the library table, they
are not dialed in, and they do not serialize.

### Tuning method shape

Uniform `verb(what, where=...)` — payload positional, `where` a keyword
defaulting to the whole book:

```python
.attribute("irulan", where={"chapter": "*", "paragraph": 1})
.silence(900, where={"chapter": "xhtml/ch08", "paragraph": 3})
.pronounce({"Atreides": "Ah-tray-deez"})                       # whole book
.pronounce({"lead": "leed"}, where={"chapter": "xhtml/ch12"})
```

`where` accepts a pattern or a tuple of patterns. `pronounce`'s payload is a
mapping because entries are written in bulk; the shape is uniform, the
payload type follows the verb.

### `pronounce()` must split

`SpokenForm` (`pipeline.py:190`) currently carries `numbers`, `features`,
`builtin_lexicon`, and `lexicon` in one operation. The first three are style
— how numbers and vocal gestures are read — and the last is this book's
proper nouns. One operation straddling both tiers breaks the `where` rule at
the exact method that introduces it.

- `.pronounce(words, where=...)` → tuning
- `.spoken_form(numbers="standard", roman=False)` → style

The style method's name matches the internal operation type. It is flagged as
an open naming question below.

### Style as a function

No new API is needed; `pipe()` (`pipeline.py:85`) already is it:

```python
def house_style(book: kk.Pipeline) -> kk.Pipeline:
    return (
        book.infer_characters("spacy")
        .attribute_quotes(MODEL)
        .assign_voices(narrator="ivy", unknown="michael")
        .pauses(chapter_ms=1200, paragraph_ms=400)
        .spoken_form(numbers="conservative")
    )


book = kk.book(epub).pipe(house_style).annotations()
```

This makes the axis visible in the code's shape and stops `example.py`'s loop
from repeating four constants per book.

### Duplicate calls

`DUPLICATE_OPERATION` stops firing for re-calls. Behaviour follows the tier:

- **tuning accumulates** — `.attribute()` twice extends the rule tuple
- **style replaces** — `.pauses()` twice means the second wins
- **identity replaces** — same

The error survives only where two operations are genuinely incompatible
(`select_chapters` vs `select_chapter_range`).

This is a deliberate change to current public behaviour: `.pauses()` called
twice currently raises. The tension worth noting is that `pronounce()`'s
docstring argues for raising on caller mistakes rather than silently
rendering something unasked-for. Replacement is not silent ignoring — the
most recent call wins, as in any settings object — and `book.style` makes the
effective value inspectable.

### Introspection

`book.identity`, `book.tuning`, and `book.style` are **properties**: tuple
scans over `operations` with no parse and no I/O, consistent with
`metadata_intent` being a property while `inspect()` is a method.

```
>>> book.style
Style — 5 operations
  characters     spacy
  attribution    openrouter/deepseek/deepseek-v4-flash
  voices         narrator=ivy  unknown=michael  method=gendered
  pauses         chapter=1200ms  paragraph=400ms
  spoken form    numbers=conservative  builtin=on

>>> book.tuning
Tuning — Dune - Frank Herbert.kenkui.json — 15 rules, 3 unsaved

  attribute   9 rules
    irulan      chapter=*  ¶1
    jessica     xhtml/ch08  ¶3  s2
    paul        xhtml/ch08  ¶7                    unsaved
    … 6 more

  silence     2 rules
    900ms       xhtml/ch08  ¶3
    0ms         xhtml/ch12  ¶14                   unsaved

  pronounce   4 rules
    10 words    whole book
    1 word      xhtml/ch12                        unsaved
```

Requirements:

- **Declaration order, never sorted.** Order is the precedence tiebreaker; a
  sorted display would misrepresent the render.
- **Degenerate levels elided at display.** `¶3 s2`, not `¶3 line 1 s2`. The
  stored path keeps the line level; only the display drops single-child
  levels.
- **Truncated but iterable.** A GUI-driven book has thousands of exact
  overrides; the summary stays readable while iteration yields everything.
- **`unsaved` derived from load counts.** The `Annotations` operation records
  how many rules it loaded per kind; rules past that index were added in code.
- **Implemented as `__repr__`,** not `__str__` — the point is typing
  `book.tuning` in a REPL.
- **No match counts.** The property is cheap, so it cannot know a pattern
  matches zero units. Drift, hash mismatch, and incomparable overlap are
  reported by `validate()`, which already does work.

The partition must be **total**: every `Operation` lands in exactly one tier,
including `SynthesizeSpeech`, which files under style. A test asserts
exhaustiveness over `Operation` subclasses so a new unclassified operation
fails CI rather than vanishing from all three views. This follows
`test_dependency_contract.py` and `test_import_boundaries.py`.

---

## 6. The sidecar

`Dune - Frank Herbert.kenkui.json`, beside the EPUB — matching the
`cover.jpg` convention already used at `spikes/examples/example.py:94`.

Loaded by `.annotations(path=None)`, written by `.write_annotations(path=None)`.
Loading twice raises `DUPLICATE_OPERATION`, matching how double selection is
guarded.

### Why not the existing store

`casting.sqlite3` lives in `~/Library/Caches/kenkui/v1/`
(`_characters/store.py:60`, resolved against `_tts/production.py:73`). The two kinds of data have opposite provenance:

|  | derived | human |
|---|---|---|
| what | grid, machine attribution, casts, audio | attribution, silence, pronunciation, splits |
| if lost | spend model calls, rebuild | **gone forever** |

Putting irreplaceable hand-made corrections in a rebuildable cache is a
category error, one `remove_attribution` or cache-clear away from destroying
the work.

The sidecar also travels with the book, diffs in git, is hand-editable when
the GUI is wrong or absent, and is already the future GUI's document format —
no export step.

### What serializes

**Every operation that carries a path**, which is exactly `book.tuning`. One
partition function serves both introspection and serialization, so there is
no second implementation to drift.

Style and identity stay in code: they are choices about rendering and
bibliography, not facts about this book's text.

Corrections that are not saved apply to that render only, which makes
experimenting free.

### Series lexicons

Pass the shared dict in code — `DUNE_LEXICON` handed to all four Dune books.
The sidecar's job begins with corrections made while dialing in one book. If
saved, shared entries land in that book's file too and dedupe on reload.

Duplication across the four sidecars is the intended default: each book
travels alone and may drift. An import-reference mechanism inside the sidecar
is deferred until the duplication actually hurts.

### The content digest

`.annotations()` records the sidecar's **path and content digest**, not just
the path. Without the digest the plan fingerprint would not change when the
file is edited, and a stale render would be served from cache. This mirrors
`resolve()`'s use of `source_digest` (`pipeline.py:383`).

Consequence: after `write_annotations()` the pipeline must be re-annotated to
pick up the new digest. This is deliberate — the invalidation is visible
rather than hidden.

---

## 7. The folds

All three existing partitions fold into the grid.

### quotes → grid

`extract_spans` is already a pure function of text and is already rule 3 of
the splitter. Grid construction consumes it; units carry `is_dialogue`;
attribution reads the grid rather than calling `extract_spans` directly. No
downside.

### structure → grid

`split_structural` currently conflates two things: **where the boundaries
are** (a pure function of text and headings) and **which ones carry silence**
(dependent on pause settings). The grid owns the first; pause settings select
from the second.

### chunking → grid

`_chunk_span` (`planning.py:652`) becomes "pack whole grid units up to
budget, descending a level when the pieces do not fit."

The `_BREAK_TIERS` ladder (`planning.py:82`) **is** the hierarchy, written
once as a flat regex ladder and re-derived at every call site. Note that its
sentence tier currently jams sentence-terminal and clause punctuation into a
single alternation, so the chunker cannot presently distinguish a sentence
break from a clause break. The hierarchy separates them, which is what allows
"prefer sentence, accept phrase, never word."

`MAX_SEPARATOR_FREE_CHARACTERS = 200` (`planning.py:79`) guards runs
containing none of `.!?,;:` — but a phrase is by definition delimited by
exactly those characters, so an over-budget phrase *is* that pathological
run. Per the existing calibration note, that is 31–126 runs per whole book.
Descending below phrase therefore goes from routine to rare, and the tier
ladder deletes.

**This fixes a known defect.** `MIN_BREAK_FILL = 0.7` (`planning.py:110`)
measures that only 64–71% of breaks currently land on a line or clause
boundary — roughly a third of segment boundaries are mid-clause today, which
`_CLEAN_BREAK_TIERS`' comment describes as "an audible break mid-clause".
Grid-driven packing pushes that to ~100% by construction, and `MIN_BREAK_FILL`
largely dissolves since greedy unit packing fills naturally.

**Spoken-form ordering must be preserved.** `_append_chapter`
(`planning.py:537`) applies `to_spoken` and *then* chunks, because spoken form
expands text ("£5" → "five pounds"). Packing therefore applies spoken form per
unit and packs by *spoken* length, while grid identity stays in canonical
coordinates. This preserves the property that boundaries are decided before
spoken form, so no offset map is ever needed.

### Cache invalidation

Folding changes how boundaries are computed, so **every segment ID changes and
the entire audio cache invalidates.** The library re-renders.

This is accepted. The cost is GPU hours and disk, not API spend: attribution
records survive in `casting.sqlite3` and `_with_current_roster` re-derives
genders without re-buying model calls. Check free space on the Data volume
before kicking off a full re-render.

`CHUNKING_SCHEMA_VERSION` and `STRUCTURAL_CHUNKING_SCHEMA_VERSION`
(`planning.py:52-53`) are the designed mechanism for exactly this. A unified
`grid-v1` replaces both and invalidates cleanly.

The `break_tiers` docstring (`structure.py:65-71`) is visibly scarred by a
past instance of this problem — it excludes `chapter_ms` purely to avoid
"changing every segment identity and invalidating every cache entry while
changing no segment text at all." That exclusion becomes unnecessary once
boundaries no longer depend on pause settings.

---

## 8. The read model and the dial-in loop

### `script()`

```python
script = book.script()
for row in script.at({"chapter": "xhtml/ch08"}):
    print(row.path, row.character, row.provenance, row.silence_after_ms, row.text)
```

`inspect().casting.spans` gives chapter-wide offset spans, which are
unreadable as a review artifact. `script()` returns one row per grid unit:
path, text, character, voice, silence-after, and **provenance** — `default` /
`machine` / `rule[3]` / `override`.

Provenance is the point. It makes the precedence model debuggable instead of
mystifying, and it is exactly the row a future GUI renders.

`script()` is a **method**, not a property, because it requires a parse — this
codebase reserves methods for things that do work. The returned `Script`
behaves as a Mapping (`script[path]`, `script.at(pattern)`, iteration over the
whole book) but materializes a chapter's rows on first access rather than
building all ~50k units of Dune to answer one question.

It works before `resolve()` too, showing grid and rules with `unresolved`
provenance, so patterns can be checked without paying for model calls.

### `select()` and `preview()`

`select(*patterns)` generalizes chapter selection to any level, so a probe
can be a single paragraph. `select_chapters` becomes the chapter-level
ergonomic case; `select_chapter_range` stays as-is because it needs spine
ordering.

`preview(path)` writes a lightweight audio file for the selection. Writing an
M4B for a twenty-second probe would pay for chaptering, metadata, and
encoding that the loop does not need.

### The loop

```python
book = kk.book(epub).pipe(house_style).annotations().resolve()

for row in book.script().at({"chapter": "xhtml/ch08"}):
    print(row.path, row.character, row.provenance, row.text[:60])

book = (
    book.attribute("irulan", where={"chapter": "*", "paragraph": 1})
    .attribute(
        "jessica", where={"chapter": "xhtml/ch08", "paragraph": 3, "sentence": 2}
    )
    .silence(900, where={"chapter": "xhtml/ch08", "paragraph": 3})
)

book.select({"chapter": "xhtml/ch08", "paragraph": 3}).preview("probe.wav")
book.write_annotations()
book.tts().write(epub.with_suffix(".m4b"), overwrite=True)
```

### The cost model

- speaker change on a unit → forces span edges → re-synthesizes ~1–3 segments
- silence at a gap that is **already** a segment boundary → **zero
  re-synthesis**; silence is excluded from segment identity (`planning.py:288`)
- silence at a gap that is not yet a boundary → forces one → ≤2 segments
- grid split/merge with nothing attached → **zero**; no audio changes

**Probe renders should warm the real cache.** Grid boundaries are a pure
function of chapter text, so a sub-chapter selection ought to produce
byte-identical segments to the full render, meaning every probe populates
cache entries the final render reuses. This is the property that makes the
loop worth building, and it is an assertion requiring a test, not a fact. The
one known exception is the selection's trailing gap, which has no following
unit from which to derive silence.

### Where the merge happens

`_append` (`pipeline.py:593`) preserves `_resolved` for an allowlist of
operations — `SynthesizeSpeech`, `MetadataIntent`, `SpokenForm`, `Pauses` —
whose docstring reads: "consume resolved attribution without changing it."

**Tuning operations join that allowlist.** They layer over machine
attribution rather than changing it, so a correction preserves `_resolved`
entirely: no re-resolution, no store read, no model calls. Just a re-plan and
the two or three segments whose identity actually changed.

This holds **only if the layer merge happens in planning, not in
resolution.** Today `resolved.spans` is the final span set handed to
`execute_sequential` (`pipeline.py:588`). Resolution must keep producing only
the machine layer, and planning must merge machine spans with tuning
operations to produce effective spans.

This is the one structural requirement the fast loop imposes. Getting it
backwards silently costs a full resolve on every correction.

---

## 9. Errors and warnings

Three findings are **warnings**, not errors — they should be visible without
blocking a render:

- anchor content-hash mismatch
- pattern match-count drift
- incomparable rule overlap

`ValidationIssue` (`api.py:77`) carries only `code` and `message`, and
`ValidationResult.is_valid` means "no issues at all" (`api.py:91-93`).
Routing warnings through it as-is would make every drifted anchor fatal.

So `ValidationIssue` gains a severity, and `is_valid` comes to mean "no
errors" rather than "no issues". This touches a public type that `write_m4b`
uses to decide whether to raise (`pipeline.py:549`), so it needs care —
the render path must continue to refuse on errors exactly as it does now.

Errors remain errors: an unresolvable path, an unknown character ID, a
malformed sidecar.

---

## 10. Testing

**Grid exactness (property).** `"".join(units) == chapter.text` across the
Calibre library. The eval README records 376 books, 32 of which do not parse
today; the parseable remainder is the corpus.

**Grid determinism.** Same text in, same grid out, independent of pause
settings, chunk budget, selection, and model.

**Fold equivalence (differential).** For a representative sample, assert the
new chunker's output is byte-identical to the old for settings where it
should be, and characterize where it deliberately differs. This is what makes
the cache-invalidation claim honest rather than hopeful.

**Break quality (metric, not assertion).** Measure the fraction of segment
boundaries landing on a line or clause boundary before and after. The current
baseline is 64–71% per `MIN_BREAK_FILL`'s comment; the target is ~100%.

**Probe/full-render segment identity.** Render a selection and the whole
chapter; assert segment IDs match except at the selection's trailing edge.

**Precedence.** Subset ordering is order-free; incomparable overlap warns and
falls to declaration order; provenance beats specificity.

**Round-trip.** `write_annotations()` → `annotations()` reproduces
`book.tuning` exactly.

**Tier partition exhaustiveness.** Every `Operation` subclass is classified.

---

## 11. Rollout

1. Grid construction and the exactness property test, with no consumers.
   Nothing changes for existing renders.
2. Addressing, patterns, precedence, and the sidecar format — pure, testable
   without touching planning.
3. `script()` and `select()`. Read-only; still no render change.
4. The tuning operations and the planning merge. First behaviour change,
   gated on the `_resolved` preservation test.
5. The three folds and `grid-v1`. The cache-invalidating step, gated on the
   differential test.
6. `preview()`, introspection, and the `pronounce`/`spoken_form` split.

Steps 1–3 are additive and safe to land independently. Step 5 is the only one
that forces a library re-render, and it is deliberately last so the loop is
usable before the expensive migration.

---

## 12. Open questions

- **`.spoken_form()` naming.** It matches the internal operation type but is
  clumsier than the single-verb names around it (`pauses`, `pronounce`,
  `metadata`, `series`).
- **`.silence()` vs `.pause()`.** `.silence()` is chosen because `.pauses()`
  already exists for global tiers and the two sit on opposite sides of the
  tier boundary. One character of difference between two different concepts
  would be a trap.
- **Abbreviation guard contents.** Starts from `identity.PREFIX_TITLES`; how
  far to extend it is a quality question to tune against the corpus, not a
  correctness one.
- **Whether `GridEdits` is needed at all.** The eager-split bias is designed
  to make under-splitting impossible, which is the only failure that costs
  expressiveness. If it succeeds, manual splits never fire and merges are
  cosmetic. Worth measuring on Dune before building the operation.
- **Sidecar import references** for shared series lexicons. Deferred.
- **Attribute-based patterns** (`emphasis: true`, `is_dialogue: true`).
  Deferred; positional patterns ship first. Once italics survive the parser,
  an emphasis-matching rule would be a better Irulan rule than a positional
  one, and a much better handle on Herbert's interior monologue.

---

## Relationship to other work

This project was split from a second one: **coreference and gender
assignment quality** — folding Paul / Paul Atreides / Muad'Dib / Usul into
one identity, and fixing misgendered minor characters such as the Baron's
lieutenant after Piter. That work is measurement-first and needs gold labels
for identity and gender, which do not exist yet.

The sidecar is that gold-label store. Every correction made while dialing in
a book is a labeled error in a diffable file, produced as a by-product of
work that is worth doing anyway.

## Parser prerequisite

`_epub/parser.py:196` flattens inline markup: `<em>` and `<i>` emit no
boundary and no marker, so italics vanish into plain text. For Dune this is
expensive — Herbert's interior monologue is italicized and unquoted, it is a
large share of the book, and it is exactly the text that should be in Paul's
voice rather than the narrator's. Today it is invisible to the pipeline and
unaddressable by any annotation.

Recovering emphasis at the parser is cheap and unlocks a whole class of
lines. It is folded into this project as a prerequisite for the grid carrying
an `emphasis` attribute, even though pattern-matching on that attribute is
deferred.
