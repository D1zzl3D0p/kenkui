# Task 12 implementation report

## Scope and result

Implemented `Pipeline.script()` and exported frozen `Script` / `ScriptRow` types.
The read model snapshots `inspect()` without resolving models, builds each
chapter's grid only on first row access, iterates rows in source order, filters
with existing pattern/sibling semantics, and supports exact or unambiguous
partial path lookup. Rows retain canonical text and flags, effective character,
speaker provenance and original winning rule index, and effective following
silence. Missing or ambiguous lookup paths raise `KeyError`.

`materialized` returns first-access chapter order. `warnings` returns immutable
`ValidationIssue` snapshots in operation/rule declaration order. Neither property
parses or builds a grid. Drift checks cover attributions, silences, and
pronunciations, detect missing anchors, and compare authoring-time match counts
only after the relevant scope has been materialized. Warnings never relocate a
rule or change whether it applies.

Added dedicated `ANCHOR_DIGEST_MISMATCH` and `PATTERN_MATCH_COUNT_DRIFT` warning
codes. Both findings have severity `warning`; inexpensive `validate()` is
unchanged.

## Decisions and requirement corrections

- Followed the task's explicit laziness prose over its contradictory
  `stale.script().warnings` example, as confirmed by the parent agent. The test
  now constructs a script, consumes the relevant chapter with `at()`, then reads
  warnings. Cost: the caller must explicitly request that chapter's rows before
  grid-dependent diagnostics exist; warning-property access stays cheap.
- Used real `CH08_ID` / `CH09_ID` fixture chapter identifiers instead of the
  brief's literal `ch08`, which is not an actual parsed ID in this repository.
- A chapter-selected pipeline only checks exact rules inside its selection.
  It defers broad/cross-chapter counts and rules outside that selection, per the
  parent's ruling. No extra source inspection or grid work is triggered merely
  to assess whole-book authoring counts. Even an explicit selection that happens
  to contain every source chapter uses this conservative deferral.
- Reused the planner's `_machine_lookup`, `_gaps_over`, and `_structural_pieces`,
  and the sidecar's `_anchor`, to preserve Task 9 subtree hashing and Task 11
  effective speaker/silence semantics. Silence includes derived heading, line,
  paragraph and chapter pauses, manual replacement (including zero), and the
  existing final-book-zero rule. No planning or resolution behavior changed.
- `Script` is mapping-like, not a `Mapping` subclass: its required iteration
  yields `ScriptRow` values, while a true mapping would iterate keys. The
  supported surface is `script[path]`, `script.at(pattern)`, row iteration,
  `materialized`, and `warnings`.
- Updated `tests/test_pipeline.py` because it owns the exact public export
  allowlist, despite the brief pointing to `tests/test_package.py`; added the
  requested package-level export checks there as well.

## Verification

All commands ran in
`/Users/dizzler/Projects/Repos/kenkui-v2/kenkui/.worktrees/book-tuning-dial-in-loop`.

1. Red phase:
   `rtk uv run pytest tests/test_script.py -v --no-cov`
   — **17 failed**, as expected: missing `Pipeline.script` / missing script
   module. `--no-cov` avoids applying the repository's 90% whole-suite threshold
   to a focused test module; the full gate retains normal coverage enforcement.
2. Initial implementation:
   `rtk uv run pytest tests/test_script.py tests/test_package.py -v --no-cov`
   — **19 passed** in 0.11s.
3. Static checks initially found three overlong lines, one magic test constant,
   an unparameterized tuple annotation, and two mypy property-narrowing artifacts
   around cache snapshots. These were corrected before final verification.
4. Expanded focused checks:
   `rtk uv run ruff format src/kenkui/script.py tests/test_script.py && rtk uv run ruff check . && rtk uv run mypy && rtk uv run pytest tests/test_script.py tests/test_package.py -v --no-cov`
   — formatter clean after formatting; Ruff passed; mypy passed for **146 source
   files**; **26 tests passed** in 0.38s.
5. Full required gate:
   `rtk uv run ruff format --check . && rtk uv run ruff check . && rtk uv run mypy && rtk uv run pytest`
   — **PASS**: 180 files already formatted; Ruff passed; mypy passed for
   **146 source files**; **1,444 passed, 45 skipped, 7 deselected, 1 warning**
   in **86.18s**. Overall branch-aware coverage **91.46%**, exceeding the required
   90%; `src/kenkui/script.py` coverage **99%** (all statements covered; only the
   empty-grid silence branch is unvisited).
6. `rtk git diff --check` — passed.

Focused tests cover canonical partition/order, cached row identity, lazy chapter
access and iterator creation, immutable rows/paths/warnings, missing and ambiguous
lookups, sparse/set/last selectors, default/machine/rule/unresolved provenance,
specificity and declaration-order rule indices, agreement with planning,
pre-resolution model-call exclusion, source emphasis, derived and manual silences,
all annotation families, missing/stale and unchanged subtree hashes, exact and
cross-chapter count scope, warning deduplication/order, sidecar round trips, and
chapter/range selection behavior.

## Self-review and concerns

- Reviewed the public surface, cache behavior, source and row ordering,
  declaration-order provenance, warning immutability, no-model path, and reuse of
  sidecar/planning semantics. No unrelated edits or rendering changes are needed.
- The read model depends on private pure planning/sidecar helpers intentionally;
  future internal refactors must preserve these consumers or move the common
  helpers together. This avoids maintaining separate subtly different algorithms.
- A selected script is intentionally not a complete whole-source drift audit.
  To assess a broad saved match count, materialize an unselected script's full
  rule scope. Reading `warnings` alone does not perform an audit.
- Grid construction is lazy; source parsing itself follows the existing
  `Pipeline.inspect()` behavior. Resolved pipelines use their retained inspection
  snapshot. This task does not introduce a chapter-streaming EPUB parser.
- No whole-Calibre corpus or real model/audio run was requested or needed for
  this read-only consumer. Existing opt-in suite exclusions remain in effect.

## Commit

Commit message: `feat(script): add the per-unit read model with provenance`.
Only Task 12 implementation, tests, and this report are staged. The report is
force-added because `.superpowers/` is ignored, consistent with existing tracked
task reports. The final commit hash is returned to the parent agent.
