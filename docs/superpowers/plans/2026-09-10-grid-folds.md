# Grid Folds — Implementation Plan

> **Execution:** Use subagent-driven development. Dispatch one implementer at a
> time, then a task-scoped reviewer. Preserve progress in this plan's SDD
> ledger. Do not skip review because the migration is internally coupled.

**Goal:** Make the addressable grid the sole text partition and replace legacy
quote rescanning, structural rescanning, and regex-tier chunking with one
hierarchical grid packer.

**Spec:** `docs/superpowers/specs/2026-09-10-grid-folds-design.md`

**Base:** `fix/chunking-prosody-and-gender` at or after `d1c956b`.

**Working directory:** `/Users/dizzler/Projects/Repos/kenkui-v2/kenkui`

## Global constraints

- Work in an isolated worktree; do not disturb the user's main checkout.
- Never push, publish, delete caches, or start a library-wide render.
- Never use `git stash`; the stash stack is shared across worktrees.
- Never `git add -A`; stage named files.
- Python 3.12 is the verified environment. The package currently claims 3.11
  support but does not import there; fixing that is out of scope.
- No new runtime dependencies.
- Keep strict mypy and project Ruff rules clean.
- Maintain at least 90% coverage.
- Run targeted tests before each task commit. Run the full gate at integration
  milestones and completion:
  `uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pytest`
- Grid construction depends only on canonical chapter text and parser-provided
  structure/emphasis—not pauses, model, voice, tuning, selection, or budget.
- Resolution continues to produce the machine layer. Human tuning merges only
  in planning; corrections must not trigger re-resolution.
- Preserve current public APIs and sidecar readability.
- The legacy chunker is temporary test-oracle code and must be deleted before
  the final review.

## Pre-flight

- [ ] Create/verify an isolated worktree and branch from the current target
  branch tip.
- [ ] Run the full gate and record exact baseline counts.
- [ ] Run the existing opt-in grid corpus property and record parse failures.
- [ ] Run a Dune planning smoke baseline and save a compact artifact containing
  canonical/spoken text hashes, segment boundaries, speakers, voices, and
  effective silences. Do not synthesize the whole book.
- [ ] Create the SDD ledger and record this plan path as its identity.
- [ ] Scan task/file interfaces for conflicts before Task 1.

## Task 1: Freeze the legacy oracle and measurements

**Purpose:** Capture what must remain equal before changing ownership or
deleting algorithms.

**Files:** tests around grid, quotes, structure, planning, selection; a test-only
oracle/fixture module if needed; evaluation documentation if the repository has
an established location.

- [ ] Extract or wrap the current quote, structure, and `_chunk_span` behavior
  behind test-only helpers without changing production behavior.
- [ ] Add representative cases for straight/smart/nested quotes, titles and
  initials, headings, blank blocks, lines, emphasis, dialogue edges,
  separator-free prose, long tokens, and spoken-form expansion.
- [ ] Capture legacy planning observations: concatenated spoken text,
  speaker/voice sequence, canonical source ranges, effective silence, and
  segment boundaries.
- [ ] Define a break-quality report that distinguishes structural/clause grid
  edges from emergency cuts. It is a metric artifact, not a brittle assertion
  against the old percentage.
- [ ] Run targeted tests and commit.

**Review gate:** Fixtures must assert meaningful values and must not encode the
new algorithm as expected output. The oracle must remain unreachable from
production code.

## Task 2: Break the domain/characters cycle

**Purpose:** Establish the dependency direction required by the folds without
changing grid output.

**Files:** `_domain` scanner/constants modules, `_domain/grid.py`,
`_characters/identity.py`, `_characters/quotes.py`, their tests and imports.

- [ ] Move the pure quote scanner into `_domain`.
- [ ] Move the prefix-title data used by sentence splitting into `_domain`.
- [ ] Update imports and delete the private `_characters.quotes` module once no
  consumers remain.
- [ ] Add an import-boundary test proving `_domain` never imports
  `_characters`.
- [ ] Prove paths, offsets, dialogue flags, emphasis flags, and exact tiling are
  unchanged from Task 1 fixtures.
- [ ] Run targeted tests and the import smoke test; commit.

**Review gate:** No compatibility re-export may recreate the cycle. No behavior
change is allowed in this task.

## Task 3: Add the derived structural range index

**Purpose:** Provide tree-like traversal while retaining one flat canonical
representation.

**Files:** `_domain/grid.py` or a focused grid-index module, paths utilities,
grid tests.

- [ ] Define an immutable index mapping every chapter/paragraph/line/sentence/
  phrase prefix to one contiguous half-open leaf-index range.
- [ ] Define compact gap metadata that can carry multiple structural reasons.
- [ ] Derive both exclusively from ordered `Unit` values and pure structure
  boundaries.
- [ ] Validate monotonicity, contiguity, nesting, one-based coordinates, stable
  chapter IDs, and deterministic equality.
- [ ] Add unit tests plus the real-library exactness property; commit.

**Review gate:** The index must not store duplicate text or independently owned
offsets. The leaf tuple is the sole source of truth.

## Task 4: Fold quote attribution into the grid

**Purpose:** Stop `_characters` from reparsing chapter text.

**Files:** attribution modules and tests, grid query helpers as needed.

- [ ] Expose dialogue ranges from dialogue-marked contiguous grid leaves.
- [ ] Replace all current `extract_spans` consumer sites with grid-derived
  ranges.
- [ ] Preserve deliberate per-chapter caching or replace it with cached grid
  construction; never scan a 600k-character chapter twice.
- [ ] Differentially prove attribution inputs match the Task 1 oracle for
  representative and corpus text.
- [ ] Run all character/attribution tests and commit.

**Review gate:** No quote scanning remains under `_characters`; grid building is
not repeated per attribution stage.

## Task 5: Fold structural gaps into the grid

**Purpose:** Separate pure boundary discovery from pause policy.

**Files:** `_domain/structure.py`, `_domain/grid.py`, `_domain/planning.py`,
pause/grid/planning tests.

- [ ] Make block/line discovery return pure canonical edges or ranges.
- [ ] Attach line/paragraph/chapter reasons to grid gaps independently of
  `Pauses`.
- [ ] Change pause planning to translate reasons into durations after packing.
- [ ] Preserve maximum-of-derived reasons and explicit-silence replacement,
  including explicit zero.
- [ ] Delete planning-side structure rescans.
- [ ] Differentially prove effective silence remains attached to the same
  canonical gaps; run targeted tests and commit.

**Review gate:** Changing pause values must not change grid equality or packed
canonical boundaries before semantic mandatory cuts are applied.

## Task 6: Implement hierarchical grid packing

**Purpose:** Add the replacement algorithm without yet switching production.

**Files:** create `_domain/grid_packing.py`; create focused packer tests.

- [ ] Define typed packer inputs for ordered leaves/ranges, per-region spoken
  text and canonical mappings, mandatory cuts, and character budget.
- [ ] Traverse the range index from paragraph through line, sentence, and
  phrase, descending only when a candidate does not fit.
- [ ] Greedily combine adjacent fitting pieces without crossing a mandatory
  cut.
- [ ] Implement the isolated over-budget-leaf fallback: punctuation/hyphen,
  whitespace, then hard token cut.
- [ ] Return packed canonical/spoken ranges sufficient for planning to build
  stable segment identities.
- [ ] Assert exact spoken reconstruction, non-empty output, hard bounds,
  determinism, and mandatory-boundary preservation.
- [ ] Add property tests and commit.

**Review gate:** The module must not import pipeline, attribution, synthesis, or
pause policy. Ordinary output boundaries must be grid edges; fallback cuts must
be explicitly identified.

## Task 7: Integrate the packer with planning

**Purpose:** Switch production planning to the grid packer while preserving all
phase-1 semantics.

**Files:** `_domain/planning.py`, `_domain/grid_packing.py`, planning/tuning/
selection tests.

- [ ] Convert effective speaker spans, voice changes, scoped pronunciation
  regions, explicit silence, and selection edges into mandatory cuts.
- [ ] Apply spoken form per region before fit decisions while retaining
  canonical offset maps.
- [ ] Replace `_chunk_span` calls with the new packer.
- [ ] Construct segments and trailing gaps from packed results.
- [ ] Preserve resolution checkpoint identity and avoid store/model calls after
  tuning-only edits.
- [ ] Prove selected/full plans share all wholly contained segment identities;
  only clipped edges/trailing selection gap may differ.
- [ ] Compare new and legacy oracle observations for spoken content,
  speaker/voice ordering, canonical coverage, and effective silence.
- [ ] Run the full gate and commit.

**Review gate:** `_resolution.py` must not consume tuning operations. Planning
must not rescan text boundaries. Any semantic differential requires a written
ruling against the spec before proceeding.

## Task 8: Unify identities under grid-v1

**Purpose:** Make the deliberate cache transition explicit and deterministic.

**Files:** planning identity code and cache/preview tests.

- [ ] Replace both legacy chunking schema constants with one `grid-v1` input.
- [ ] Ensure every new speech segment identity includes it exactly once.
- [ ] Remove obsolete pause-tier identity inputs now that pause settings do not
  alter textual packing boundaries, while retaining every semantic identity
  input that changes audio.
- [ ] Prove unchanged full/preview interior segments have equal identities.
- [ ] Prove old IDs differ cleanly and attribution/sidecar stores remain
  readable.
- [ ] Do not delete old cache files; run targeted tests and commit.

**Review gate:** Identity tests must detect both stale-cache reuse and needless
identity churn.

## Task 9: Delete legacy partitioning and tighten modules

**Purpose:** Finish the replacement rather than shipping two architectures.

**Files:** planning, structure, obsolete test-oracle helpers, documentation.

- [ ] Delete `_chunk_span`, regex break ladders, `MIN_BREAK_FILL`, obsolete
  schema constants, and unused rescanning helpers.
- [ ] Delete the temporary oracle implementation after retaining compact
  differential results/fixtures sufficient to explain the migration.
- [ ] Confirm `_domain` has no `_characters` import and planning/attribution
  contain no raw quote or structure scans.
- [ ] Document `grid-v1`, cache invalidation, emergency leaf behavior, and the
  explicit cache-pruning choice.
- [ ] Run Ruff, mypy, full pytest, and coverage; commit.

**Review gate:** Search-based deletion claims must be backed by import tests and
behavior tests. No runtime rollback flag or dormant legacy path may remain.

## Task 10: Corpus, Dune, and quality verification

**Purpose:** Validate the migration against real books before final review.

- [ ] Run grid exactness/determinism over every parseable local EPUB and record
  counts plus pre-existing failures.
- [ ] Run the break-quality metric over the same representative corpus. Report
  normal grid-edge boundaries and emergency cuts separately.
- [ ] Compare a Dune planning smoke against the pre-flight artifact: canonical
  and spoken hashes, speaker/voice order, tuning application, effective
  silence, selection plans, and tier summaries.
- [ ] Check free space on the Data volume and report it; do not launch a full
  render or delete cache entries.
- [ ] Run the full project gate and record exact results in the ledger.
- [ ] Commit only documentation/fixture updates produced by verification.

**Review gate:** Corpus parse failures must be classified as pre-existing or
regressions. Emergency cuts must be characterized, not hidden in an aggregate.

## Task 11: Final whole-branch review and one fix wave

- [ ] Generate a merge-base-to-HEAD review package.
- [ ] Dispatch the final reviewer on the most capable available model.
- [ ] Point it explicitly at all ledger rulings, deferred minors, identity
  inputs, the planning/resolution boundary, and temporary-oracle deletion.
- [ ] If Critical or Important findings exist, dispatch one implementer with
  the complete list, then one scoped re-review of the fix wave.
- [ ] Adjudicate residuals in the ledger; do not silently discard findings.
- [ ] Re-run the full gate and corpus property on final HEAD.
- [ ] Use the branch-finishing workflow and let the user choose integration.

## Required completion evidence

The final handoff must include:

- commit range and branch/worktree paths;
- full gate counts and coverage;
- corpus exactness/determinism counts;
- old/new break-quality measurements and emergency-cut count;
- Dune smoke comparison;
- cache invalidation and free-space disclosure;
- confirmation that legacy runtime/test-oracle code is deleted;
- every `Ruling:` ledger entry and all deferred findings;
- any behavior deliberately different from phase 1.

## New-session handoff

Start the next session with:

> Execute `docs/superpowers/plans/2026-09-10-grid-folds.md` in
> `/Users/dizzler/Projects/Repos/kenkui-v2/kenkui` using subagent-driven
> development: fresh implementer per task, task-scoped review between tasks,
> ledger-backed recovery, and a final whole-branch review. Read
> `docs/superpowers/specs/2026-09-10-grid-folds-design.md` first; it is the
> authority. Begin with pre-flight and do not reuse the old dial-in-loop SDD
> workspace.

The next session must verify the actual target branch tip and dirty-tree state
rather than assuming this document's base SHA is still current.
