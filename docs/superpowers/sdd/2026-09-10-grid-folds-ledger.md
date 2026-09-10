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
