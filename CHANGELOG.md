# Changelog

All notable changes to Kenkui are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/D1zzl3D0p/kenkui/compare/v10.0.0...HEAD
[10.0.0]: https://github.com/D1zzl3D0p/kenkui/releases/tag/v10.0.0
