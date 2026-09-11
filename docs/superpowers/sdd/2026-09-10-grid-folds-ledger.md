# Grid Folds SDD Ledger

## Identity

- Plan: `docs/superpowers/plans/2026-09-10-grid-folds.md`
- Authoritative design: `docs/superpowers/specs/2026-09-10-grid-folds-design.md`
- Worktree: `/Users/dizzler/Projects/Repos/kenkui-v2/kenkui/.worktrees/grid-folds`
- Branch: `feat/grid-folds`
- Target branch: `fix/chunking-prosody-and-gender`
- Verified starting tip: `d72c1e8e3789ab17ee1687ed11779a8c1effd523`

This ledger belongs only to the plan above. Do not reuse the prior
`.superpowers/sdd/2026-09-08-book-tuning-dial-in-loop` workspace.

## Working protocol

- Append task outcomes, review findings, deferred minors, and verification
  evidence here before moving to the next task.
- Record an adjudication that changes or clarifies behavior as a line beginning
  with `Ruling:` so the final handoff can enumerate it mechanically.
- The approved design is authoritative over the implementation plan if wording
  differs.
- Keep implementer and reviewer scopes sequential because the migration's file
  interfaces overlap heavily.

## Pre-flight — 2026-09-10

### Worktree and base

- `rtk git status --short --branch` -> `## feat/grid-folds`; no tracked or
  untracked changes at entry.
- `rtk git worktree list --porcelain` -> the isolated worktree and the target
  checkout both pointed at `d72c1e8`.
- `rtk git rev-parse HEAD fix/chunking-prosody-and-gender` -> both resolved to
  `d72c1e8e3789ab17ee1687ed11779a8c1effd523`.
- `rtk git log -1 --oneline --decorate HEAD` ->
  `d72c1e8 (HEAD -> feat/grid-folds, fix/chunking-prosody-and-gender) docs: plan grid folds migration`.

### Full gate baseline

The commands from the plan were run separately so every baseline result was
retained even when an earlier step failed.

- `rtk uv run ruff format --check .` -> pass; `188 files already formatted`.
- `rtk uv run ruff check .` -> pass; `All checks passed!`.
- `rtk uv run mypy` -> fail; 1 error across 152 checked source files:
  `src/kenkui/_characters/spacy_roster.py:482: error: Cannot find implementation or library stub for module named "spacy" [import-not-found]`.
  The exact planned command installs the default/dev environment, while spaCy
  is an optional project extra and is absent from the clean worktree venv.
- `rtk uv run pytest` -> 1,567 collected; 7 deselected, 1 skipped at collection,
  1,560 selected; final result `1515 passed, 46 skipped, 7 deselected, 1 warning
  in 94.45s`. Coverage measured 7,806 statements / 641 missed and 2,414
  branches / 270 partial, for 89.85% total. Pytest printed
  `FAIL Required test coverage of 90% not reached`, although the wrapper
  process reported exit status 0.

Baseline gate status: **not green** because strict mypy cannot import the
optional spaCy dependency and measured coverage is below the required 90%.
Neither issue was modified during pre-flight.

### Existing opt-in grid corpus property

- Initial planned form:
  `rtk env KENKUI_RUN_CORPUS=1 uv run pytest tests/test_grid_exactness.py -ra`
  could not initialize `/Users/dizzler/.cache/uv` under the sandbox. Escalated
  access was declined, so no cache permissions or contents were changed.
- Equivalent already-created-venv run:
  `rtk env KENKUI_RUN_CORPUS=1 .venv/bin/pytest tests/test_grid_exactness.py -vv -ra --no-cov`
  -> `39 passed, 1 skipped in 16.86s` across the test's capped first 40 local
  EPUBs.
- Sole parse failure: `Dark One - Brandon Sanderson`; skipped as pre-existing
  with `SourceError: A spine chapter has no visible text.`
- All 39 parseable books satisfied exact chapter tiling. The current property
  checks exactness only; determinism and an uncapped all-EPUB pass remain Task
  10 work under the new design.

### Dune planning-only smoke

- Source: `/Users/dizzler/Projects/Calibre Library/Frank Herbert/Dune (466)/Dune - Frank Herbert.epub`.
- The smoke used `compile_execution_plan` directly and never entered the
  coordinator/render path. It made no provider/model calls and synthesized no
  audio.
- Scope: chapter index 5 (`ch-v1-2cc5ecca00df5ea4a766ea67`, title `1`),
  paragraphs 1-4, canonical range `[0, 774)`.
- Inputs reproduce the checked-in example's Dune lexicon and house style, the
  existing stored `characters-v5` attribution, the most-pinned existing cast,
  and the local loaded voice manifest. The casting database was opened through
  SQLite's immutable URI mode.
- Result: 6 segments; 774 canonical characters; 797 spoken characters;
  canonical SHA-256
  `236264d0c4ca443ab525b5a0890486351ef970c4d432dd7c0e83ceb5919776c5`;
  spoken SHA-256
  `d397a4bdd1b20ab5e11f8443030bb8cb1a459b5e16035e3c25ef3503d59be98a`;
  semantic fingerprint
  `0d53c7d50ed77bb381aabd302c9429e8467ce2d411273132bc0fe5881622220d`.
- Compact exact artifact, including canonical/spoken boundaries, ordered
  speakers, voices, and effective trailing silences:
  `docs/superpowers/artifacts/2026-09-10-grid-folds-dune-baseline.json`.

### Task/file interface scan

The scan covered `_domain`, `_characters`, resolution/pipeline boundaries, and
the grid/attribution/planning/tuning/selection tests named by the plan.

- `src/kenkui/_domain/grid.py` directly imports both
  `_characters.identity.PREFIX_TITLES` and `_characters.quotes.extract_spans`.
  Moving both pure inputs together in Task 2 is the required cycle break;
  moving only the scanner leaves the dependency violation intact.
- `_characters.quotes` is imported by `_characters.__init__`,
  `_characters.attribution`, `_characters.spacy_roster`, and tests including
  `test_attribution.py` and `test_tuning_merge.py`. Task 2 must migrate every
  import before deleting the private module; Task 4 then replaces production
  rescans with grid dialogue ranges.
- `planning.py` is a shared hotspot for Tasks 5, 7, 8, and 9. It currently owns
  `_structural_pieces`, direct `split_structural` use, grid-derived tuning
  helpers, `_chunk_span`, both break-tier ladders, and both chunking schema
  constants. Those tasks must remain ordered; parallel edits would collide and
  blur behavior ownership.
- `grid.py` is shared by Tasks 2 and 3 and may need focused query additions in
  Tasks 4 and 5. Freeze the Task 1 oracle before the Task 2 move, then land the
  range index before consumers switch.
- The new `grid_packing.py` is created in Task 6, integrated in Task 7, and
  cleaned in Task 9. Task 7 must not grow a second implementation in
  `planning.py`.
- Existing tests overlap multiple stages: `test_attribution.py` and
  `test_tuning_merge.py` import the legacy scanner, while planning, pause,
  identity, and selection tests all exercise `planning.py`. Task-scoped test
  changes need review against the Task 1 oracle rather than broad rewrites.
- The corpus test is capped at 40 EPUBs and records parser failures only as
  pytest skips. Task 10's all-parseable and determinism evidence needs a
  separate uncapped measurement or a deliberate extension of that opt-in test.

No behavioral rulings or deferred review findings exist at pre-flight.

## Task log

### Task 1 — Freeze the legacy oracle and measurements — 2026-09-10

- Added `tests/legacy_grid_folds_oracle.py`, a test-only observation boundary
  over the current quote scanner, structural splitter, `_chunk_span`, and pure
  planning compiler. It is unreachable from production imports and is
  intentionally temporary until Task 9 removes the migration oracle.
- Added exact, meaningful fixtures for straight quotes, smart quotes, nested
  quotes, title and initial abbreviation guards, headings, repeated blank-block
  separators, individual lines, emphasis flags, dialogue edges,
  separator-free prose, indivisible long tokens, and spoken-form expansion.
- The planning fixture records concatenated spoken text, per-segment text,
  speaker/voice order, canonical source ranges, spoken ranges, and effective
  trailing silence. Its spoken form expands `IV` to `Four` and `21%` to
  `twenty-one percent` while the canonical ranges remain fixed.
- Added the measurement artifact
  `docs/superpowers/artifacts/2026-09-10-grid-folds-break-quality-baseline.json`.
  It classifies each legacy internal boundary as a paragraph/line/sentence/
  dialogue/phrase grid edge or as a punctuation-or-hyphen/whitespace/hard-token
  cut within one leaf. The artifact explicitly defines its values as baseline
  evidence, not an acceptance percentage for the replacement packer.
- Representative baseline: 3 internal boundaries total; 1 sentence grid edge;
  2 within-leaf emergency cuts (1 whitespace at canonical offset 999 and 1
  hard-token cut at offset 1000). This deliberately small fixture set
  characterizes each class; corpus-wide measurement remains Task 10.
- Focused oracle run:
  `rtk .venv/bin/pytest tests/test_grid_folds_legacy_oracle.py -q --no-cov`
  -> `11 passed in 0.02s`.
- Targeted regression run:
  `rtk .venv/bin/pytest tests/test_grid_folds_legacy_oracle.py tests/test_quote_extraction.py tests/test_structure.py tests/test_grid.py tests/test_planning.py tests/test_planning_multi_voice.py tests/test_identity_stability.py tests/test_tuning_merge.py tests/test_select_preview.py -q --no-cov`
  -> `221 passed in 5.07s`.
- Changed-file formatting and lint:
  `rtk .venv/bin/ruff format --check tests/legacy_grid_folds_oracle.py tests/test_grid_folds_legacy_oracle.py`
  -> `2 files already formatted`; and the matching `ruff check` ->
  `All checks passed!`.
- Changed-file strict typing:
  `rtk .venv/bin/mypy tests/legacy_grid_folds_oracle.py tests/test_grid_folds_legacy_oracle.py`
  -> `Success: no issues found in 2 source files`.

No behavioral ruling was required and no Task 1 finding was deferred. The
pre-flight full-gate spaCy and aggregate-coverage failures remain unchanged.

#### Task 1 review gate

- Independent task-scoped review of `4bc681a` approved the task with no
  Critical, Important, or Minor findings.
- Review confirmed the oracle is excluded from production packaging/imports,
  expectations freeze legacy behavior rather than the future packer, and the
  break-quality artifact separates ordinary grid edges from characterized
  within-leaf emergency cuts.

### Task 2 — Break the domain/characters cycle — 2026-09-10

- Moved the pure quote scanner and `TextSpan` unchanged from
  `_characters/quotes.py` to `_domain/quotes.py`; SHA-256 of the old and new
  source content is identically
  `89d0d695adb5246e554c6391e743f770c40c7dbe297ba91fe156f9c2991c0e5f`.
- Updated every production and test consumer to import the scanner from
  `_domain.quotes`, then deleted `_characters/quotes.py`. No compatibility
  module or re-export remains at that path.
- Moved canonical `PREFIX_TITLES` ownership to `_domain/titles.py`. Both grid
  sentence splitting and character identity now consume the domain constant,
  preserving the existing name-merging behavior while reversing the dependency
  in the required direction.
- Extended `tests/test_import_boundaries.py` with a recursive AST boundary that
  rejects absolute or relative imports of `_characters` anywhere under
  `_domain`. The existing import scanner was strengthened to resolve relative
  imports so the boundary cannot be bypassed by spelling alone.
- Strengthened the Task 1 grid fixture to freeze every paragraph/line/sentence/
  phrase coordinate, canonical start/end offset, dialogue flag, emphasis flag,
  text slice, adjacency, complete coverage, and exact source reconstruction.
- Focused/affected regression run:
  `rtk .venv/bin/pytest tests/test_import_boundaries.py tests/test_grid_folds_legacy_oracle.py tests/test_quote_extraction.py tests/test_grid.py tests/test_grid_exactness.py tests/test_identity.py tests/test_attribution.py tests/test_narration.py tests/test_spacy_roster.py tests/test_tuning_merge.py -q --no-cov`
  -> `173 passed, 41 skipped in 4.17s`; the opt-in corpus cases account for 40
  skips and the optional spaCy dependency for the other skip.
- Full formatting and lint checks:
  `rtk .venv/bin/ruff format --check .` -> `192 files already formatted`; and
  `rtk .venv/bin/ruff check .` -> `All checks passed!`.
- Full strict typing still reports only the pre-flight optional-dependency
  failure: `src/kenkui/_characters/spacy_roster.py:482` cannot import `spacy`
  across 155 checked files. The affected-file check excluding that already
  known module passed: `Success: no issues found in 14 source files`.

No behavioral ruling was required and no Task 2 finding was deferred. The
pre-flight full-gate spaCy and aggregate-coverage failures remain unchanged.

#### Task 2 review gate

- Independent task-scoped review of `6664170` approved the task with no
  Critical, Important, or Minor findings.
- Review confirmed byte-identical scanner relocation, complete consumer
  migration, no compatibility shim, robust absolute/relative import-boundary
  enforcement, and full legacy fixture equality for coordinates, flags, and
  exact tiling.

### Task 3 — Add the derived structural range index — 2026-09-10

- Added immutable `StructuralIndex`, an ordered `Mapping[Path, LeafRange]` that
  derives chapter, paragraph, line, sentence, and phrase prefix ranges from an
  ordered leaf sequence. Every value is a half-open range of leaf indices; the
  index stores neither canonical text nor canonical offsets and retains no
  copy of the input leaf tuple.
- Added `GapReason`, a compact `IntFlag` bit set with phrase, sentence, line,
  paragraph, and chapter closure reasons. One immutable value after each leaf
  records all coincident structural closures, including the complete set at
  the final chapter leaf.
- Index construction validates a non-empty stable chapter ID, positive
  one-based hierarchy coordinates, a `(1, 1, 1, 1)` first path, non-empty
  monotonic contiguous canonical ranges beginning at zero, unique leaf paths,
  gapless sibling increments, and child-coordinate resets. These transition
  rules prove prefix contiguity and nesting before index construction.
- Unit coverage proves exact ranges at every prefix depth, parent/child
  nesting, sibling exclusion, multi-reason gaps, malformed-grid rejection,
  empty-grid behavior, deterministic equality/hash behavior, and independence
  from canonical offsets, dialogue flags, and emphasis flags.
- Extended the opt-in real-library exactness property to build the index twice
  for every chapter, assert deterministic equality and one gap value per leaf,
  and prove that every leaf belongs to all five of its indexed prefixes while
  retaining exact canonical reconstruction.
- Focused and affected regression run:
  `rtk .venv/bin/pytest tests/test_grid_index.py tests/test_grid.py tests/test_paths.py tests/test_select_preview.py tests/test_sidecar.py tests/test_planning.py tests/test_import_boundaries.py -q --no-cov`
  -> `187 passed in 5.55s`.
- Real-library property:
  `rtk env KENKUI_RUN_CORPUS=1 .venv/bin/pytest tests/test_grid_exactness.py -q -ra --no-cov`
  -> `39 passed, 1 skipped in 31.75s`; the sole skip remains the pre-existing
  `Dark One - Brandon Sanderson` empty-visible-text parse failure.
- Full formatting and lint checks:
  `rtk .venv/bin/ruff format --check .` -> `193 files already formatted`; and
  `rtk .venv/bin/ruff check .` -> `All checks passed!`.
- Strict typing of the changed production/test files passed across 3 source
  files. Full mypy still reports only the pre-flight optional-dependency
  failure at `src/kenkui/_characters/spacy_roster.py:482` for absent `spacy`,
  now across 156 checked source files.

No behavioral ruling was required and no Task 3 finding was deferred. The
pre-flight aggregate-coverage and optional-spaCy failures remain unchanged.

#### Task 3 review gate

- Independent task-scoped review found one Important defect: when paragraph
  coordinates changed while child sentence coordinates reset to one, the gap
  omitted `GapReason.SENTENCE` even though the nested sentence range closed.
- Resolved by treating a paragraph change as closing the nested sentence range
  and adding the `One.\n\nTwo.` regression fixture.
- No other Critical, Important, or Minor findings were reported.
- Scoped re-review of fix commit `34a6240` approved the resolution with no new
  findings; focused verification reported 15 passed and 40 opt-in skips.

### Task 4 — Fold quote attribution into the grid — 2026-09-10

- Added immutable `DialogueRange` values and `dialogue_ranges`, a grid query
  that coalesces contiguous dialogue-marked leaves into canonical half-open
  ranges without retaining text or independently scanning quote punctuation.
- Replaced every production `_characters` use of `extract_spans` with ranges
  derived from `build_grid`. Model roster discovery, spaCy roster inference,
  first-person narration detection, and chapter attribution now consume only
  grid-provided dialogue ranges; `extract_spans` remains a production detail
  only of grid construction while the Task 1 oracle still exists.
- Attribution reconstructs the same complete narration/dialogue `SpeakerSpan`
  tiling around those dialogue ranges, preserving stored span offsets,
  coverage accounting, quote ids, prompt text, and unknown-speaker behavior.
- Centralized per-book range materialization in `_dialogue_by_chapter`.
  Resolution builds each chapter grid once and reuses its ranges for roster
  discovery and concurrent attribution. A 600,000-character regression spies
  on both possible build sites and proves exactly one grid build for the
  chapter while retaining exact output tiling.
- Representative differential coverage compares straight, smart, and nested
  quote inputs against `legacy_quote_partition`; a focused grid test proves
  sentence/phrase leaves inside one quotation coalesce to one attribution
  range.
- Real-library differential property:
  `rtk env KENKUI_RUN_CORPUS=1 .venv/bin/pytest tests/test_grid_exactness.py -q -ra --no-cov`
  -> `39 passed, 1 skipped in 33.02s`. Every grid-derived dialogue range
  matched the Task 1 legacy oracle for every chapter in all 39 parseable EPUBs;
  the sole skip remains the pre-existing `Dark One - Brandon Sanderson`
  empty-visible-text parse failure.
- Full character/attribution-focused regression run:
  `rtk .venv/bin/pytest tests/test_attribution.py tests/test_casting_solver.py tests/test_casting_store.py tests/test_character_llm.py tests/test_dialogue_tags.py tests/test_identity.py tests/test_identity_stability.py tests/test_narration.py tests/test_series_identity.py tests/test_series_resolution.py tests/test_series_store.py tests/test_series_validation.py tests/test_spacy_roster.py tests/test_tuning_merge.py -q --no-cov`
  -> `274 passed, 1 skipped in 5.72s`; the skip is the absent optional spaCy
  dependency.
- Quote/grid/import/oracle regression run:
  `rtk .venv/bin/pytest tests/test_quote_extraction.py tests/test_import_boundaries.py tests/test_grid.py tests/test_grid_folds_legacy_oracle.py -q --no-cov`
  -> `63 passed in 0.23s`.
- Full formatting and lint checks passed: `193 files already formatted` and
  `All checks passed!`.
- Strict typing across all 11 changed production and test modules passed when
  disabling only the pre-flight `import-not-found` diagnostic for optional
  spaCy. The unmodified full strict check still reports exactly the documented
  `src/kenkui/_characters/spacy_roster.py:482` missing-spaCy error across 156
  source files.

No behavioral ruling was required and no Task 4 finding was deferred. The
pre-flight aggregate-coverage and optional-spaCy failures remain unchanged.

#### Task 4 review gate

- Independent task-scoped review found one Important defect:
  `dialogue_ranges` merged distinct adjacent source quotations such as
  `"a""b"` and `“a”“b”`, erasing their shared mandatory quote edge and making
  two speakers inseparable.
- Resolved by recording a one-based `dialogue_run` identity on every
  dialogue-marked grid leaf. It is immutable metadata derived during the one
  quote scan already performed by grid construction; the flat `Unit` tuple
  remains the sole text partition. `dialogue_ranges` now coalesces structural
  leaves only when their source quotation identity matches.
- Added straight- and smart-quote oracle differentials proving both adjacent
  quotation ranges exactly match Task 1's scanner output, plus attribution
  tests proving both quote ids reach the model input and can receive different
  speakers.
- Focused grid/oracle/attribution run -> `55 passed in 3.75s`; structural-index,
  import-boundary, and scanner regression run -> `58 passed in 0.23s`.
- Full character/attribution-focused rerun -> `276 passed, 1 skipped in 5.60s`;
  the skip remains the absent optional spaCy dependency.
- Real-library differential rerun -> `39 passed, 1 skipped in 32.59s`, with the
  same pre-existing `Dark One - Brandon Sanderson` parse failure.
- Changed-file strict mypy passed across 3 source files, and full Ruff passed.

No behavioral ruling was required and no review finding remains deferred.

- Scoped re-review of fix commit `f3795ba` approved Task 4 with no new
  findings. It confirmed adjacent quote-run separation, within-quote leaf
  coalescing, immutable leaf-owned metadata, and corpus oracle equality.

### Task 5 — Fold structural gaps into the grid — 2026-09-10

- Replaced pause-dependent structural text pieces with immutable canonical
  `BlockRange` and `LineRange` discovery. `_domain.structure` no longer accepts
  `Pauses`, chooses enabled tiers, carries durations, or owns text pieces.
- Added parser-derived `is_heading` metadata to grid leaves and extended
  `GapReason` with heading-before and heading-after. `StructuralIndex.gaps` now
  records line, paragraph, chapter, and coincident heading closures without
  consulting pause settings.
- Planning builds one grid and structural index per chapter, reuses that grid
  for attribution tuning, scoped pronunciation, and manual-gap resolution,
  and consumes canonical grid gaps rather than calling a structural scanner.
  A static import-boundary test rejects planning imports of block/line
  discovery.
- Pause policy now translates grid reasons after legacy packing. Coincident
  paragraph and heading reasons retain max-not-sum semantics. While the legacy
  chunker remains until Tasks 6–9, an enabled derived gap is treated as a
  semantic mandatory cut without using its numeric duration to choose the
  boundary; changing an enabled value therefore leaves canonical boundaries
  unchanged.
- Manual silence still replaces derived silence rather than adding to it.
  Added coverage for explicit zero at both paragraph and inter-chapter gaps;
  the latter now correctly suppresses `chapter_ms` instead of being raised
  again by the inter-chapter maximum.
- Moved the deleted legacy structure implementation into the Task 1 test-only
  oracle. Differential fixtures prove pure grid reasons translate to the same
  effective values on the same canonical heading/line/paragraph gaps, and
  adjacent coincident heading reasons choose the maximum duration.
- Final focused/affected regression run:
  `rtk .venv/bin/pytest tests/test_structure.py tests/test_grid.py tests/test_grid_index.py tests/test_grid_folds_legacy_oracle.py tests/test_import_boundaries.py tests/test_planning.py tests/test_planning_multi_voice.py tests/test_identity_stability.py tests/test_tuning_merge.py tests/test_select_preview.py tests/test_script.py tests/test_sidecar.py -q --no-cov`
  -> `299 passed in 5.76s`.
- Full project regression before the final explicit-zero assertion:
  `rtk .venv/bin/pytest -q --no-cov` -> `1551 passed, 46 skipped, 7 deselected,
  1 warning in 78.36s`; the affected suite above passed after that assertion.
- Full Ruff format/check passed across 193 files. Strict mypy passed across all
  10 changed production/test modules. Full mypy still reports only the
  pre-flight missing optional `spacy` import at
  `src/kenkui/_characters/spacy_roster.py:482` across 156 source files.

No behavioral ruling was required and no Task 5 finding is deferred. The
pre-flight aggregate-coverage and optional-spaCy failures remain unchanged.

#### Task 5 review gate

- Independent task-scoped review of `e291349` approved Task 5 with no
  Critical, Important, or Minor findings.
- Review confirmed pause-independent structure discovery, complete coincident
  gap reasons, no planning-side rescan, post-grid pause translation, explicit
  zero replacement, and legacy effective-gap equality across 336 structural
  and pause combinations.

### Task 6 — Implement hierarchical grid packing — 2026-09-10

- Added pure `_domain/grid_packing.py` with immutable typed `PackingInput`,
  `SpokenRegion`, `SpokenMapping`, and `PackedRange` values. The module imports
  only grid/index and path types; it has no pipeline, attribution, synthesis,
  planning, structure-discovery, or pause-policy dependency.
- Packing projects canonical grid ranges into concatenated transformed text,
  attempts paragraph, line, sentence, then phrase ranges, descends only when a
  candidate exceeds the configured character budget, and greedily combines
  adjacent fitting pieces inside each mandatory canonical interval.
- An isolated over-budget phrase fallback chooses the final punctuation or
  hyphen edge inside the budget, then whitespace, then a hard token cut. Each
  within-leaf output records its `FallbackCut` category; ordinary output
  boundaries remain grid or mandatory edges.
- Spoken mappings validate unchanged runs exactly and let expanded/contracted
  text drive fit decisions while output retains stable canonical envelopes and
  exact spoken subranges. Emergency pieces inside one expanded replacement may
  share its canonical envelope and remain distinct by their spoken ranges.
- Result validation enforces non-empty output, exact contiguous spoken
  reconstruction, the hard bound, ordered in-grid canonical envelopes, and
  preservation of every mandatory boundary.
- Added focused cases for hierarchy descent, greedy recombination, mandatory
  cuts, all three fallback categories, spoken expansion, multiple independently
  transformed regions, expanded-token emergency cuts, ordinary-boundary
  provenance, invalid inputs, and the module dependency boundary.
- Added a deterministic randomized property over 100 generated prose inputs,
  varied budgets, and sampled mandatory grid edges; every run reconstructs
  spoken text exactly, stays bounded, preserves cuts, and compares equal on a
  second invocation.
- Focused packer/grid/index/import/oracle run -> `62 passed in 0.33s`.
- Full Ruff format/check passed across 195 files. Strict mypy passed for both
  new files.
- The delegated implementer exhausted its agent quota after drafting the
  module, so the primary agent completed validation hardening, tests, and the
  task commit while retaining the required task-scoped review gate.

No behavioral ruling was required and no Task 6 finding is deferred. The
pre-flight aggregate-coverage and optional-spaCy failures remain unchanged.

#### Task 6 review gate

- Independent task-scoped review found two Important defects. A valid leading
  zero-spoken replacement mapped the first emergency piece past its canonical
  start, and the punctuation-first fallback omitted typographic ellipsis even
  though the grid recognizes it as sentence punctuation.
- Resolved zero-width mapping ambiguity by projecting its shared spoken edge
  to the canonical start for a lower envelope and the canonical end for an
  upper envelope. Added a regression for `A -> ""` followed by an expanded
  `B`, proving exact bounded reconstruction without dropped canonical coverage.
- Added typographic ellipsis to the punctuation/hyphen fallback class and a
  separator-free regression proving it cuts after `…` rather than reporting a
  hard-token cut.
- Post-fix focused packer/grid/index/import/oracle run -> `64 passed`; Ruff and
  strict mypy passed for both changed files.
- No other Critical, Important, or Minor findings were reported; scoped
  re-review remains required before Task 7.
- Scoped re-review of fix commit `4c866a2` approved both resolutions with no
  remaining findings, including extra adjacent/interior zero-width mapping
  probes.

### Task 7 — Integrate the packer with planning — 2026-09-11

- Replaced the production `_chunk_span` call with `pack_grid`. The legacy
  definition remains temporarily reachable only from the Task 1 test oracle
  and is guarded by a static AST test proving planning never calls it.
- Planning now builds one grid and structural index per chapter, reusing the
  same leaves for effective attribution, scoped pronunciation, manual gaps,
  packing, and selected-plan silence placement. A selected-plan spy test proves
  exactly one planning-side grid build per materialized chapter.
- Mandatory canonical cuts include effective speaker/voice span edges, scoped
  lexicon region edges, explicit silence edges including zero, and enabled
  derived-silence edges. Spoken form is applied independently inside those
  intervals, retaining global canonical-to-spoken replacement mappings.
- Packed results construct speech segments and trailing gaps directly. Empty
  or whitespace-only pieces carry forward without becoming synthesis inputs;
  emergency pieces retain their explicit `FallbackCut` provenance in the
  temporary source observation record.
- Restored the legacy stable `EMPTY_SPEECH` validation for empty chapters and
  inconsistent `speech_characters` before grid packing.
- Updated separator-free planning expectations to the approved design: the
  general 200-character guard no longer drives production boundaries; only an
  over-budget phrase invokes the isolated fallback, and all output remains
  under the 1,000-character ceiling.
- Added a differential whose two 640-character paragraphs deliberately move
  from legacy `[994, 288]` to hierarchical `[642, 640]` chunks while preserving
  the exact spoken stream. Added plan-origin tests proving ordinary boundaries
  are grid edges and a hard-token cut is explicitly characterized.
- Existing speaker/voice, scoped pronunciation, explicit/manual silence,
  tuning-resolution checkpoint, and selection suites pass. Full/selection
  tests retain every wholly-contained segment identity and permit at most the
  two intersected edge chunks plus trailing selection-gap treatment to differ.
- Ruling: selection edges are enforced as mandatory post-pack clips over the
  stable full-grid packing. Feeding a selection start into a fresh greedy pack
  would shift downstream boundaries and violate the authoritative requirement
  that wholly-contained full/preview segments share identities; clipping only
  intersected edge segments satisfies both exact selection and cache reuse.
- Focused planning/multi-voice/tuning/selection/identity/script/packer run ->
  `186 passed in 6.19s`; packer validation coverage subsequently increased the
  packer-focused result to 19 passed.
- Full regression without coverage -> `1575 passed, 46 skipped, 7 deselected,
  1 warning in 78.66s` before the final added validation case.
- Full Ruff format/check passed across 195 files. Changed-file strict mypy
  passed.
- Full gate with coverage before the final added validation case -> `1575
  passed, 46 skipped, 7 deselected, 1 warning`; aggregate coverage was 89.57%,
  still below the 90% requirement. Full mypy still reports only the pre-flight
  optional-spaCy import error. Task 9 deletion and the final gate must close
  both branch-wide acceptance gaps.
- The delegated implementer exhausted its quota after beginning the integration;
  the primary agent completed and verified this task.

#### Task 7 review repair

- Independent review of `8fa25e7` did not approve the first integration pass.
  It reported three Important findings and no Critical or Minor findings:
  whitespace-only packed ranges could be reassembled past the 1,000-character
  ceiling; selection and billing independently rebuilt the same chapter grid;
  and the planning oracle invoked the current compiler on both sides instead
  of remaining independent.
- Removed the unbounded post-pack carry. Whitespace-only source spans are
  assigned to adjacent effective speech before mandatory speaker cuts are
  formed; bounded whitespace output is then attached to available capacity on
  the following and preceding synthesizable ranges without exceeding the hard
  ceiling. The 1,200-space regression reconstructs the complete text in two
  bounded ranges with contiguous canonical origins and a characterized
  whitespace fallback.
- Ruling: a canonical span containing only whitespace has no synthesizable
  speaker. Preserve phase-1 behavior by assigning it to the following
  effective speech span before packing (or the preceding span at chapter end),
  rather than emitting an engine-invalid whitespace segment. If a run cannot
  fit wholly with the following speech, fill available capacity on the
  preceding speech and retain a `WHITESPACE` fallback marker. Only a middle run
  too large for both adjacent bounded speech segments can remain omitted,
  matching phase 1's removal of whitespace-only chunks.
- Added an optional prebuilt-grid input to `selected_ranges`; planning passes
  its cached grid to both selection clipping and billing. The spy now patches
  both the planning and selection module symbols and compiles from one already
  materialized inspection, proving one boundary scan per source chapter.
- Replaced the dynamic legacy-plan observer with literal immutable data
  captured at `d72c1e8`. The current compiler is observed separately and
  compared field-for-field for transformed spoken text, segment text, speaker
  and voice order, canonical/spoken ranges, and effective silences.
- Post-repair focused planning/selection/oracle/packer run -> `120 passed`;
  changed-file Ruff format/check and strict mypy passed. Full regression
  without coverage -> `1577 passed, 46 skipped, 7 deselected, 1 warning in
  77.33s`.
- The first re-review resolved the independent differential and one-grid
  findings, but reported two Important follow-ups: a manual zero gap at the end
  of an inaudible span could remove its whitespace and churn the following
  identity, and selected planning rebuilt `StructuralIndex` in
  `grid_silences`.
- Manual and derived gaps inside an unspeakable attribution span now settle at
  its preceding effective boundary before mandatory cuts are constructed.
  Exact regression coverage proves adding an explicit zero gap to the space in
  `"A." "B."` leaves both segments, IDs, and `(0, 0)` silences unchanged.
- Planning now caches `StructuralIndex` beside each grid and threads it into
  selected silence calculation. The strengthened spy patches both constructors
  and proves exactly one grid plus one index build per materialized chapter.
- Second repair focused run -> `121 passed`; changed-file Ruff and strict mypy
  passed.
- The broad regression exposed two pre-existing Script parity cases not present
  in the narrower run: without attribution spans, a whitespace grid leaf still
  owns manual gap settlement. Unspeakable normalization now uses the union of
  whitespace-only speaker spans and whitespace-only grid leaves; all three
  300/zero/absent gap cases pass again.
- The second re-review then found one remaining Important case: redistributing
  a 1,200-space run extended the preceding range beyond the normalized gap, so
  end-keyed gap attachment lost a non-zero manual silence. Gap ownership is now
  determined after redistribution from the last emitted range whose canonical
  start precedes the effective gap. The exact long-whitespace regression proves
  full reconstruction, 208/1,000 bounded segment lengths, stable segment IDs,
  contiguous origins, `WHITESPACE` provenance, and `(900, 0)` effective
  silence with a manual gap.
- Third repair focused planning/multi-voice/tuning/selection/identity/script/
  packer/oracle run -> `204 passed`; changed-file Ruff and strict mypy passed.
- Full regression after the third repair -> `1578 passed, 46 skipped, 7
  deselected, 1 warning in 78.36s`.

#### Task 7 final review gate

- Independent third-repair re-review approved Task 7 with no Critical,
  Important, or Minor findings.
- Reviewer probes confirmed exact 1,208-character reconstruction in bounded
  208/1,000-character segments, stable manual/no-manual segment identities,
  manual `(900, 0)` and derived `line_ms=700` silence preservation, identical
  selected/full silence, contiguous origins, and `FallbackCut.WHITESPACE`.
- Review also confirmed Script retains whitespace rows while their gaps settle
  onto preceding effective speech, and selected planning constructs exactly one
  `StructuralIndex` per chapter.

No semantic difference from phase 1 was accepted beyond the approved packing
boundary changes and the whitespace-only ruling above. No Task 7 review finding
is deferred.

### Task 8 — Unify identities under grid-v1 — 2026-09-11

- Replaced `CHUNKING_SCHEMA_VERSION = "tts-chunks-v4"` and
  `STRUCTURAL_CHUNKING_SCHEMA_VERSION = "tts-chunks-v5"` with the single
  explicit `GRID_CHUNKING_SCHEMA_VERSION = "grid-v1"` segment-identity input.
  Every segment, including clipped selection edges, now reaches the same
  constructor and receives that input exactly once.
- Removed pause-tier and structure-schema fields from segment identities and
  deleted their identity-only plumbing through chapter compilation and
  selection clipping. Plan-level structure schema and effective silence remain
  in the semantic fingerprint because they still describe rendered audio.
- Retained every synthesis-relevant segment input: canonical chapter and
  ordinal/chunk position, content hash and normalization version, clipped-edge
  selection coordinates, attributed speaker and voice, and active spoken-form
  configuration.
- Migrated the plain-plan golden IDs and fingerprint to grid-v1. A payload spy
  proves plain, paused, and spoken segments each contain grid-v1 once and contain
  neither retired pause-tier field; independently reproduced v4 and v5 payloads
  prove both old ID namespaces are disjoint from every corresponding new ID.
- A cache regression stores a genuine v4-keyed segment and proves a grid-v1
  lookup misses it without deleting or corrupting the legacy database row or
  PCM. The legacy entry remains independently readable.
- Strengthened preview coverage to compare identities explicitly: every wholly
  contained interior segment is identical between full and selected plans, and
  a whole-chapter preview shares the exact full-plan IDs later observed as warm
  cache hits.
- Existing attribution-store reuse and tuning-sidecar round-trip/checkpoint
  tests remain green; neither persistence format nor schema was changed.
- Focused planning/multi-voice/tuning/identity/selection/preview/cache/sidecar/
  attribution run -> `257 passed in 43.73s`. Changed-file Ruff format/check and
  strict mypy passed.
- No cache files were deleted and no Task 9 legacy-code cleanup was performed.
  The pre-flight aggregate-coverage and optional-spaCy full-gate failures remain
  unchanged for the final task.

No new semantic ruling was required and no Task 8 finding is deferred.

#### Task 8 review gate

- Independent review approved `4ccfb65` with no Critical, Important, or Minor
  findings.
- Review confirmed every production segment path either constructs one identity
  containing exactly one `grid-v1` input or reuses an already-built segment;
  only obsolete tier fields were removed and all synthesis-relevant fields
  remain.
- Reviewer reran the 257-test focused suite and verified v4/v5 disjointness,
  undeleted readable legacy cache data, attribution/sidecar compatibility, and
  equal full/preview identities for wholly contained segments. Ruff and strict
  mypy passed.

### Task 9 — Delete legacy partitioning and tighten modules — 2026-09-11

- Deleted the retired `_chunk_span`, `_break_offset`,
  `_separator_free_end`, `_BREAK_TIERS`, `_CLEAN_BREAK_TIERS`,
  `MIN_BREAK_FILL`, `MAX_SEPARATOR_FREE_CHARACTERS`, and `POCKET_SEPARATORS`
  implementation and constants. Also removed the unused `_structural_gaps`,
  `_pause_pieces`, and `_fragments` transitional helpers.
- Removed the obsolete structure schema from `SchemaVersions`, plan
  fingerprint serialization, and `_domain/structure.py`; grid-v1 segment IDs
  plus effective segment/silence content now carry the applicable semantics.
- Deleted `tests/legacy_grid_folds_oracle.py`. Compact migration evidence
  remains as literal d72c1e8 plan values and fixed quote/gap fixtures in the
  migration test, the identity tests' literal v4/v5 payload reconstruction,
  and the checked-in break-quality baseline artifact. No test helper executes a
  retired algorithm.
- Strengthened the import-boundary test from “no call” to absence of every
  legacy function/constant definition, and retained AST enforcement that
  `_domain` never imports `_characters` and planning never imports structural
  discovery scanners. Direct searches also found no quote/block/line scanner in
  planning or attribution and no `_characters` import under `_domain`.
- Replaced remaining tests that referenced retired constants/helpers with
  direct grid-packer assertions and literal pre-migration boundary measurements.
  Corpus dialogue comparison now uses the canonical domain scanner directly.
- Updated `docs/architecture.md` for the one-grid ownership model,
  transformation-before-fit, hierarchy/fallback behavior, mandatory semantic
  cuts, whitespace settlement, grid-v1 invalidation, retained legacy cache
  data, and explicit cache-pruning/free-space choice.
- Added a narrow mypy override for the lazily imported optional `spacy` package,
  matching the existing pocket-tts missing-stubs policy without installing the
  optional runtime. Full strict mypy now passes: `Success: no issues found in
  157 source files`.
- Added focused branch coverage for adjacent/trailing unspeakable spans,
  leading/consecutive/trailing whitespace redistribution, invalid span lookup,
  and empty gap settlement. Targeted cleanup suite -> `140 passed, 40 skipped`;
  focused defensive additions -> `20 passed`; full Ruff format/check passed
  across 194 files.
- Full project gate -> `1580 passed, 46 skipped, 7 deselected, 1 warning in
  84.63s`; branch coverage reached 90.11%, clearing the configured 90% floor.
- The delegated implementer exhausted its quota before producing changes; the
  primary agent completed and verified the task.

#### Task 9 review gate

- Independent review found no Critical or Important findings and approved the
  runtime deletion, layering, compact fixtures, architecture, and narrow spaCy
  mypy override. Its focused suite reported `91 passed, 40 skipped`; changed-
  file Ruff and full strict mypy passed.
- Two Minor cleanup findings were fixed before Task 10: `_gap_enabled` and
  `_compile_segments` no longer describe the retired legacy/frozen chunker,
  and the AST deletion guard now includes `_structural_gaps`, `_pause_pieces`,
  `_fragments`, `CHUNKING_SCHEMA_VERSION`,
  `STRUCTURAL_CHUNKING_SCHEMA_VERSION`, and `STRUCTURE_SCHEMA_VERSION`.
- Post-fix focused check -> `62 passed`; changed-file Ruff format/check passed.

No runtime rollback flag, dormant legacy path, or deferred Task 9 finding
remains.
