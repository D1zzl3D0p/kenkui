# Changelog

All notable changes to Kenkui are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [10.2.0] - 2026-09-29

### Changed

- New synthesis pipelines announce authored chapter titles by default. Use
  `chapter_titles(enabled=False)` to opt out, or configure title pauses and
  spoken-only overrides. Matching opening headings are reused; source text
  and character offsets remain intact.

- EPUB table-of-contents targets now define logical chapters, joining internal
  file splits and preserving unlisted content in reading order. Fragment targets
  can delimit multiple chapters in one file. Without usable navigation, use
  opening headings, meaningful titles, or sequential `Untitled section N` labels.
  Internal epigraph credits no longer name whole sections. Re-inspect sources and
  refresh chapter selections after upgrading: regrouped chapters have new IDs.


### Added

- Initial native PDF source support via `pdf()` or `book()`, with explicit
  `pdf_processing(mode="native")`, immutable `prepare()` checkpoints and
  `pdf_report()` evidence. Install the optional `pdf` extra. Preparation runs
  automatically before PDF resolution/rendering; inspection requires preparation.
- Independently callable header/footer removal and paragraph reconstruction in
  `kenkui.pdf_processing`, versioned recipes, edit logs, source-change guards and
  preparation-aware plan/attribution identities. Empty recipes disable cleanup.
  Persistent caches are not yet available.
- Extend the native PDF recipe with source-attested line-break word repair and
  conservative same-page footnote/reference removal. Unknown spellings remain
  unchanged and are reported; word repair preserves all letters.
- Recognize corroborated top-of-page folios and numbered running titles, including
  facing-page layouts. Years and numbers inside prose are retained.
- Add optional `pdf-layout` extraction through an isolated, bounded CPU worker,
  with provisioned local Docling/RapidOCR assets, selective OCR, native-text
  completeness checks, progress stages and strict result validation.
- Add independently callable numbered note-section and visual-material removal,
  verified footnotes within layout paragraphs, and repeated OCR furniture removal.
- Validate PDF recipe prerequisites before extraction, reject invalid resource
  limit types, and honor cancellation before cleanup callbacks run.

### Fixed

- Incomplete quote-attribution responses are retried and rejected before caching;
  old response and attribution caches are revalidated for full quote coverage.
- Cached model responses now honor cancellation before reading and after validation.
- Failed model calls report bounded diagnostics without exposing book text.

## [10.1.1] - 2026-09-20

### Fixed

- A scene ornament is no longer read aloud. A book that divides its scenes
  with `* * *`, `#`, or a similar glyph run on a line of its own was handing
  that run to the engine, which said it however it saw fit. Such a line is now
  recognized from the canonical text and speaks as nothing, and the block after
  one opens a scene, so `pauses(scene_ms=...)` covers these books too without
  the publisher having labelled anything. Suppression does not depend on
  `pronounce()`: a pipeline that never asked for a spoken form is exactly the
  one affected.

  Canonical text is untouched, so billing, chapter identity, offsets and
  sidecar anchors are unchanged. The segments that held an ornament do change
  text, so those -- and only those -- re-synthesize once. A chapter containing
  nothing but an ornament is a separator page rather than a scene break and is
  left exactly as it was.

## [10.1.0] - 2026-09-20

### Added

- `pauses(scene_ms=...)` gives mid-chapter scene breaks their own duration.
  A scene break is detected at parse time from an `<hr/>` or a block the
  publisher labelled as one (`class`/`epub:type`, including an ornament hidden
  from assistive technology); normalization erases the distinction later, so it
  cannot be recovered afterwards. The tier is off unless set, so no existing
  render changes. `ChapterInspection.scene_ranges` reports the detected blocks
  and `ScriptRow.is_scene_start` marks the row that opens a scene, while the
  silence it implies falls on the row before it.

## [10.0.0] - 2026-09-20

Kenkui 10 is a ground-up rewrite. It shares no code with the 2.x series, and
none of the 2.x API carries over. The version is `10` in binary: the second
generation.

### Added

- An immutable, typed `Pipeline` API: `book()`, fluent intent methods,
  `validate()`, `inspect()`, `resolve()`, and `write()`, plus `magic_run()` for
  one-call rendering.
- Local speech synthesis with Pocket-TTS 2.1.0, in spawned worker processes,
  with a private per-book audio cache and atomic M4B publication through
  FFmpeg.
- Explicit voice provisioning with `load_voice()`, `unload_voice()`,
  `add_voice()`, `remove_voice()`, and `list_voices()`, over a built-in catalog
  of 26 upstream voices and 95 precompiled
  [kenkui-voices](https://huggingface.co/datasets/D1zzl3D0p/kenkui-voices).
- Multi-voice casting: character discovery with spaCy or a LiteLLM model, an
  optional identity pass that merges aliases, LiteLLM quote attribution
  (including unnamed speakers), and a deterministic casting solver with
  `gendered` and `random` methods.
- Character review with `resolve(until="characters")` and `with_characters()`.
- Series continuity: `.series()` keeps characters' voices across volumes.
- Speech shaping: `pronounce()` with lexicons, number-reading tiers, and vocal
  gesture collapsing; `pauses()`; and cover art control.
- The dial-in loop: `script()`, `attribute()`, `silence()`, scoped
  `pronounce()`, `select()`, `preview()`, and sidecar corrections with
  `annotations()` and `write_annotations()`.
- Progress events, cooperative cancellation, and stable `ErrorCode` values on
  every failure.
- Runnable examples in `examples/`.

### Fixed

- Quote attribution now returns gender directly in a separate per-speaker table.
  Valid, consistent evidence survives storage and offline roster refreshes;
  conflicting evidence is flagged and reviewed genders retain precedence.
  A controlled four-book passage evaluation found no attribution regression
  when this field was added to the role-aware prompt; see the evaluation report
  for sample limits and provider-routing anomalies.
- Character attribution now requests distinct, gender-qualified identities for
  unnamed people sharing a role, using pronouns and actions in context. Casting
  recognizes those qualifiers; sparse unopposed dialogue tags fill unknown
  genders, inverted tags and common adverbs are recognized, and ambiguous tag
  evidence is logged. The prompt version advances to avoid reusing old merges.
- Chapter names now prefer EPUB table-of-contents labels over epigraph headings
  and internal document titles. Calibre split continuations inherit the label
  with a part number; filename-only titles use the numbered fallback.

[Unreleased]: https://github.com/D1zzl3D0p/kenkui/compare/v10.2.0...HEAD
[10.2.0]: https://github.com/D1zzl3D0p/kenkui/compare/v10.1.1...v10.2.0
[10.1.1]: https://github.com/D1zzl3D0p/kenkui/compare/v10.1.0...v10.1.1
[10.1.0]: https://github.com/D1zzl3D0p/kenkui/compare/v10.0.0...v10.1.0
[10.0.0]: https://github.com/D1zzl3D0p/kenkui/releases/tag/v10.0.0
