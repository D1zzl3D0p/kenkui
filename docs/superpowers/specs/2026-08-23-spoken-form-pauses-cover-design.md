# Spoken Form, Structural Pauses, and Custom Cover — Design

Date: 2026-08-23
Status: Approved design, ready for implementation planning

## 1. Purpose

Add four capabilities to the Kenkui core pipeline:

1. numeric normalization for speech (`100,000` → `one hundred thousand`);
2. configurable silence around line breaks, paragraph breaks, headings, and
   chapter boundaries;
3. correction of commonly mispronounced words via a built-in lexicon plus
   per-book caller overrides;
4. embedding a caller-supplied cover image instead of the source cover.

Three of the four share one spine — a new pure text stage plus a structural
split ahead of chunking. The cover is independent and can land at any time.

## 2. Governing constraint

Every feature here changes what the engine says or how long the output runs.
The core specification makes three of those consequences load-bearing:

- core §9/§20: `speech_characters` and `normalized_speech_characters` describe the
  normalized source, and the latter must match the server's credit policy;
- core §15: planning is pure and deterministic;
- core §21: cache loss may cost performance but must never change correctness.

The design therefore keeps one **canonical** normalized text as the authority
for billing, inspection, chapter identity, and attribution offsets, and derives
a separate **spoken** text that only the synthesis engine ever sees.

## 3. Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | New `spoken-form-v1` stage; canonical text unchanged | Billing stays honest to the source book; attribution spans keep their offsets |
| D2 | Pauses are generated silence at spill time, not prosody hints | Exact millisecond control; silence never reaches a worker or the cache, so retuning costs no re-synthesis |
| D3 | Four boundary kinds across five knobs, each independently variable | Caller sets any tier to 0 to disable it, including its chunking cost |
| D4 | Built-in lexicon plus `.pronounce()` caller overrides | Shared defaults for English, escape hatch for invented names |
| D5 | Number handling is tiered (`conservative`/`standard`/`aggressive`) | Ambiguous forms require explicit opt-in; anything a tier declines falls through to the lexicon |
| D6 | **The whole stage is off unless requested** | Existing pipelines stay byte-identical and every cache entry stays valid |
| D7 | Silence is modelled as *gaps between segments*, not per-segment fields | Makes pause stacking impossible by construction |
| D8 | The inter-chapter gap belongs to the **preceding** chapter's marker | Skipping to a chapter lands on speech; no dead air at any chapter start |
| D9 | Cover content hash enters the fingerprint; the path does not | core §14: output paths must not alter semantics |

## 4. Stage architecture

```
EPUB
  └─ visible text extraction              epub-visible-text-v1   (unchanged)
       └─ normalize_text()                nfc-space-newline-v1   (unchanged)
            │                             ← CANONICAL TEXT
            ├─ inspect().speech_characters
            ├─ attribution span offsets
            ├─ OutputChapter.speech_characters
            └─ ExecutionStats.normalized_speech_characters   ← the bill
            ↓
          speaker spans                   (unchanged)
            ↓
          structural split                epub-structure-v1     ← NEW
            ↓
          spoken form                     spoken-form-v1        ← NEW
            ↓
          chunker                         tts-chunks-v2 | v3
            ↓
          SpeechSegment.text              → engine, cache key
                      .character_count    → ExecutionStats.synthesized_characters
```

### 4.1 Insertion point

`_compile_segments` (`src/kenkui/_domain/planning.py:362`) currently reads:

```python
text = chapter.text[span.start : span.end]
for chunk_index, chunk in enumerate(_chunk_span(chapter, text)):
```

The new stages sit between those two statements. Nothing upstream changes.

### 4.2 Why the structural split precedes spoken form

Spoken form changes string lengths, so heading offsets recorded against
canonical text would need an offset map to survive it. Splitting *first*, in
canonical coordinates, then applying spoken form to each resulting piece,
removes the need for any offset map. When no pause tier is enabled the split
yields exactly one piece and the whole arrangement collapses to a no-op.

### 4.3 Exactness invariants

Three levels, each exact, each independently testable:

```
chunks   join to  to_spoken(piece)
pieces   join to  span text
spans    join to  canonical chapter text
```

The existing single invariant (`"".join(chunks) == text`) is replaced by these
three. `_chunk_span`'s current whole-chapter validation
(`chapter.speech_characters == len(chapter.text)`) still applies against
canonical text and moves to the split boundary.

## 5. Billing

`total_speech_characters` changes from *sum of segment character counts* to
*sum of canonical chapter lengths*:

```python
total = sum(len(chapter.text) for chapter in inspection.chapters)
```

These are identical today, so the change is a no-op when spoken form is off,
and automatically correct when it is on. No per-segment apportionment of
canonical characters is required.

`ExecutionStats` needs no new field. `normalized_speech_characters` and
`synthesized_characters` (`_execution/coordinator.py:331-334`) already exist and
are currently always equal; text expansion is what makes them diverge, which is
what the core specification anticipated.

## 6. Spoken form

### 6.1 Matching model

One left-to-right pass over a combined ranked matcher. At each position the
first matching rule wins, its replacement is emitted, and **the output is never
re-scanned**, so rules cannot loop or cascade. Rank order:

1. caller lexicon entries, longest phrase first;
2. built-in lexicon entries, longest phrase first;
3. number rules for the active tier.

### 6.2 Lexicon rules

- An entry is a literal source phrase mapped to a literal plain-English
  respelling. Not IPA — Pocket-TTS accepts no phoneme input.
- Matching is whole-word on Unicode word boundaries, case-insensitive,
  multi-word phrases permitted.
- The replacement inherits the source's capitalization shape: `ALL CAPS`,
  `Titlecase`, otherwise lowercase.
- Built-in entries additionally match diacritic-insensitively, so
  `coup de grace` is caught alongside `coup de grâce`. Caller entries match
  exactly as written.
- Caller entries override built-ins on overlap.
- Built-in data lives in a versioned data file. Adding an entry is a version
  bump and therefore a visible, cache-invalidating change rather than silent
  output drift.
- Caller input is validated and bounded — non-empty keys and values, a cap on
  entry count and total length — because on the server side this is untrusted
  input.

### 6.3 Number tiers

Anything a tier declines is left verbatim for `.pronounce()` to handle.

| Form | conservative | standard | aggressive |
|---|---|---|---|
| grouped integers, decimals, negatives, ordinals | ✅ | ✅ | ✅ |
| percent (`40%`) | ✅ | ✅ | ✅ |
| currency with leading symbol (`$1.50`) | ✅ | ✅ | ✅ |
| units directly following a number (`5 km`) | ✅ | ✅ | ✅ |
| years read as pairs (`1984` → nineteen eighty-four) | — | ✅ | ✅ |
| clock times (`3:45`), numeric ranges (`1914–1918`) | — | ✅ | ✅ |
| roman numerals after Chapter/Part/Book/Act/Scene/Volume/Section/Appendix, or after a regnal name | — | ✅ | ✅ |
| bare roman numerals, `No. 5`, fractions | — | — | ✅ |
| `St.`, `Dr.`, `Mrs.`, phone numbers | never | never | never |

Notes:

- Bare roman numerals in the aggressive tier require an explicit stoplist. `I`
  is a pronoun; `MIX`, `DID`, `CIVIC`, `MILD`, `DIM`, `LID` are words.
- Year detection applies only to bare 4-digit tokens in 1000–2099 that are not
  comma-grouped and not adjacent to a unit, currency symbol, or percent sign.
- `Dr.` and `Mrs.` are deliberately never expanded; Pocket-TTS handles them.
  `St.` is never expanded because Saint/Street is genuinely ambiguous.

### 6.4 Language

English only. Number words use en-US conventions (no "one hundred *and* one").
A **non-English narrator voice disables the stage entirely** rather than
mangling the text; the narrator's `language` is already resolved in the plan.

## 7. Structure and pauses

### 7.1 Parser change

`ChapterInspection` gains heading offsets into canonical chapter text:

```python
headings: tuple[tuple[int, int], ...] = ()
```

The extracted **text is unchanged** — heading text is already emitted into the
body by `_emit_element` (`src/kenkui/_epub/parser.py:181`). Only the record of
where headings are is new, so `epub-visible-text-v1` remains valid for text and
the structure record carries its own `epub-structure-v1` version.

### 7.2 The gap model

Between any two adjacent segments in plan order there is exactly one **gap**. A
gap may have several *reasons*; its duration is the **maximum** of them, never
the sum.

```
chapter N, last prose ][  gap  ][ "Chapter Twelve" ][  gap  ][ chapter N+1 prose
                          ▲                            ▲
              max(chapter_ms,                    heading_after_ms
                  heading_before_ms)
```

```
... prose ][  gap  ][ "A Section Heading" ][  gap  ][ prose ...
               ▲                              ▲
      max(paragraph_ms,                 heading_after_ms
          heading_before_ms)
```

A chapter boundary and a chapter title's "before" pause are **the same gap**, so
stacking is impossible by construction rather than by a rule an implementer has
to remember.

Gap 0 — before the book's first segment — is always zero. Dead air at the head
of a file is a defect, not a pause.

### 7.3 Attribution (D8)

Every gap folds into the **preceding** segment's trailing pad. Consequences:

- the inter-chapter gap is counted in chapter N's duration;
- chapter N+1's marker begins exactly on its first spoken word;
- skipping forward lands on speech; a scrubber shows no dead air at any
  chapter start;
- no leading-pad mechanism is required anywhere, since gap 0 is zero.

### 7.4 Chunking v3

`tts-chunks-v3` is **v2, restricted**: pre-split each span at the forced
structural boundaries, then run the *unmodified* v2 chunker inside each piece.

- Concatenation-exactness is inherited rather than re-proved.
- The tuned constants `MIN_BREAK_FILL` and `MAX_SEPARATOR_FREE_CHARACTERS` are
  not forked.
- The forced-break set is derived from **which pause durations are non-zero**:

  | tier | forced boundary | engages v3? |
  |---|---|---|
  | `chapter_ms` | chapter edges — already segment boundaries | **no** |
  | `heading_before_ms` or `heading_after_ms` | recorded heading offsets | yes |
  | `paragraph_ms` | `\n\n` runs | yes |
  | `line_ms` | single `\n` | yes |

  `chapter_ms` deliberately does not engage v3. `_compile_segments` already
  iterates chapter by chapter, so a chapter edge is inherently a segment
  boundary. A pipeline whose only pause is `.pauses(chapter_ms=1500)` therefore
  stays on `tts-chunks-v2` with **every cache entry still valid** — it changes
  the gap table and the plan fingerprint, and no segment text at all.

- The break set enters segment identity; the *durations* do not. Retuning
  600ms → 400ms is free. Turning a tier on or off re-chunks. Leaving `line_ms`
  at 0 costs nothing in fragmentation, which matters because verse-dense text
  would otherwise reduce to one segment per line.

Enabling paragraph breaks roughly doubles segment count on a 594k-character
book (~987 → ~1800) in exchange for no chunk ever straddling a paragraph.

### 7.5 Rendering silence

The coordinator pads `SegmentAudio` **after** `_validate_audio` has run on the
raw worker output, and writes the corresponding zero frames when spilling:

```
worker / cache → _validate_audio(raw)     ← all existing safety invariants intact
                      ↓
               segment_audio(item) + trailing pad frames
                      ↓
               _spill_chapter writes segment PCM, then zero frames
```

`SegmentAudio.byte_count` is a derived property of `frame_count`
(`src/kenkui/_tts/protocols.py:53-56`), so padding one field simultaneously
corrects:

- `_validate_pcm_parts`, which sums `byte_count` per chapter and requires the
  part file size to match exactly;
- `chapter_frame_boundaries_ms`, which derives markers from frame counts;
- the reported total duration.

**No changes are required in `_audio/production.py` or `_audio/m4b.py`.**

Padding must be counted against `MAX_CHAPTER_PCM_BYTES` and
`MAX_TOTAL_PCM_BYTES`, which is correct — silence occupies real bytes.

Silence never reaches a worker and is never cached, so pause configuration
stays out of segment identity.

## 8. Cover

`Pipeline.metadata(cover=...)` widens from `Literal["source"] | None` to
`Literal["source"] | Path | None`. `CoverIntent` gains `FILE`.

Resolution happens in the shell at plan time, exactly as voice and attribution
resolution already do — the pure planner receives a finished value:

- open with `O_NOFOLLOW`; require a regular file with a single hard link;
- reject anything larger than 8 MiB;
- sniff magic bytes and accept JPEG and PNG only, so FFmpeg is never handed
  arbitrary input;
- SHA-256 of the content enters the plan fingerprint. The path does not (core §14).

`OutputMetadata` gains `cover_content_hash: str | None`; `source_cover_available`
is retained unchanged.

Validated bytes are copied into the workspace `.cover` file by the same
mechanism `materialize_source_cover` uses today
(`src/kenkui/_audio/cover.py`), so `production.py` needs no change beyond
selecting the byte source. `preflight(expect_cover=True)` applies when a file
cover is set.

An invalid or unreadable caller-supplied cover **fails the render loudly** with
a new stable `EncodingError(COVER_INVALID)`. There is no silent fallback to the
source cover — the caller asked for a specific image.

## 9. Identity and versioning

Added to `SchemaVersions`: `structure`, `spoken_form`. `CHUNKING_SCHEMA_VERSION`
becomes `v2` or `v3` depending on whether any forced break tier is active.

Segment identity gains the following fields **only when the corresponding
feature is enabled**, preserving D6:

| Field | Present when |
|---|---|
| `spoken_form_schema` | spoken form requested |
| `numbers_tier` | spoken form requested |
| `lexicon_identity` (digest of built-in version + sorted caller pairs) | spoken form requested |
| `structure_schema` | any forced break tier active |
| `break_tiers` (sorted frozenset) | any forced break tier active |

This mirrors the existing precedent at `planning.py:494-498`, where
`speaker_id`/`voice_id` are added only for attributed speech so single-voice
identities stay byte-identical.

Plan fingerprint additionally gains pause durations and the cover content hash.
`trailing_silence_ms` is **excluded** from segment identity.

## 10. Public API

```python
Pipeline.pronounce(lexicon=None, *, numbers="conservative", builtin=True)
Pipeline.pauses(*, chapter_ms=0, heading_before_ms=0,
                heading_after_ms=0, paragraph_ms=0, line_ms=0)
Pipeline.metadata(cover=Path(...))          # widened
```

Two new operation records — `SpokenForm`, `Pauses` — appended through the
existing `append_unique` path with `before_tts=True`, so duplicate-operation
and ordering validation come for free.

Neither method is implied by any other. A pipeline that calls neither renders
byte-identically to today (D6).

## 11. Errors

Three new stable codes:

| Code | Class | Raised when |
|---|---|---|
| `COVER_INVALID` | `EncodingError` | a caller-supplied cover fails validation (§8) |
| `INVALID_PRONUNCIATION` | `ValidationError` | a lexicon entry is empty, unbounded, or exceeds the caller cap |
| `INVALID_PAUSE` | `ValidationError` | a pause duration is negative or exceeds its bound |

Everything else reuses existing codes.

## 12. Package structure

New:

```
src/kenkui/_domain/spoken/
├── __init__.py          to_spoken() — the combined single-pass matcher
├── numbers.py           tier rules and number-to-words
├── lexicon.py           matching, capitalization shape, validation
└── data/
    └── lexicon-v1.json  versioned built-in entries
src/kenkui/_domain/structure.py   forced-break derivation and the gap model
```

Modified: `_domain/operations.py`, `_domain/planning.py`, `_epub/parser.py`,
`inspection.py`, `pipeline.py`, `_execution/coordinator.py`, `_audio/cover.py`,
`errors.py`.

`_domain/text.py` is **not** modified. `normalize_text()` and
`nfc-space-newline-v1` are the canonical authority and stay frozen; this is the
mechanism that keeps billing honest, so leaving the file untouched is a
deliberate check on the design rather than an omission.

Splitting spoken form into its own package keeps `text.py` and `planning.py`
from becoming grab-bags; `planning.py` is already 624 lines.

## 13. Testing

Per core specification §26, pure logic is tested separately from effects.

**Highest value:**

- **Opt-out guard** — a golden test asserting that a pipeline with neither
  `.pronounce()` nor `.pauses()` produces byte-identical segment IDs *and* an
  identical plan fingerprint to the current implementation. This makes D6
  enforceable rather than aspirational.
- **Three-level exactness property test** — over arbitrary generated text:
  chunks join to `to_spoken(piece)`, pieces join to the span, spans join to the
  canonical chapter text.

**Unit (pure):**

- number tier golden tables, one per tier, including every "never" case;
- roman numeral stoplist, especially `I` and word-forming numerals;
- lexicon capitalization shape, diacritic-insensitive built-in matching,
  longest-phrase-first, caller-over-built-in, and no-re-scan;
- gap model: max-not-sum where a chapter boundary meets a heading-before pause;
- gap 0 is always zero;
- `tts-chunks-v3` reduces exactly to v2 when no break tier is active;
- forced-break derivation from non-zero durations;
- non-English narrator disables the stage;
- caller lexicon bound and validation rejections.

**Integration (effects):**

- silence padding versus part-size validation and chapter marker arithmetic,
  through the fake assembler;
- the inter-chapter gap lands in chapter N's duration, not N+1's (D8);
- cover accept/reject matrix — symlink, non-regular, multi-link, oversize,
  wrong magic bytes, valid JPEG, valid PNG;
- `normalized_speech_characters` holds steady while `synthesized_characters`
  rises when spoken form is enabled;
- cache hit/miss equivalence with spoken form enabled.

## 14. Sequencing

1. **Spoken form** — establishes the conditional identity-versioning pattern.
2. **Structure and pauses** — reuses that pattern; depends on the structural
   split introduced alongside spoken form.
3. **Cover** — independent; may land at any point.

## 15. Out of scope

- Non-English number words or lexicons.
- IPA or phoneme-level pronunciation control.
- SSML.
- Per-chapter or per-character pause overrides.
- Caller-supplied cover for formats other than JPEG and PNG.
- Exposing `to_spoken()` as a public API.
