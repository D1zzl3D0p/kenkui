# Task 9 report: Versioned annotation sidecars

## Implementation

- Added `src/kenkui/_domain/sidecar.py` with `SIDECAR_VERSION = 1`, pure `serialize` / `deserialize` codecs, the `<stem>.kenkui.json` path convention, authoring snapshots, and atomic UTF-8 publication.
- Added `Pipeline.annotations(path=None)` and `Pipeline.write_annotations(path=None)`. Loading reads the sidecar once, hashes those exact bytes with SHA-256, appends its rules to existing immutable per-kind tuples, reindexes declarations consecutively, and adds one unique `Annotations` record with the path, digest, and per-kind loaded counts. Duplicate loading is rejected before reading the second path.
- Existing resolution and roster checkpoints survive loading through the established tuning branch behavior. Existing pipelines and their already-loaded content digests never change when saving or editing a file.
- Serialization uses `tier_of` to select tuning and excludes `Annotations`; style and identity are absent. Array order determines precedence, and JSON object insertion order preserves authored operation-family order. Selector sets serialize in sorted canonical order; pronunciation values reload as immutable sorted tuples.
- Validation rejects non-object payloads, missing/unsupported/non-integer versions (including booleans), unknown fields, malformed sections/rules/patterns, invalid attribution/silence/lexicon values, invalid digest/count metadata, and simultaneous digest/count fields. Malformed JSON, invalid UTF-8, and inaccessible files use the new sanitized `INVALID_SIDECAR` error.
- Added trailing defaulted `Rule.matched: int | None = None`, approved by the coordinating agent, so authored match counts survive deserialization for subsequent drift checks. Existing positional construction remains compatible.
- Writes use a temporary file in the destination directory, flush and fsync it, then atomically replace the destination. Failed publication preserves the prior sidecar and cleans the temporary file. Explicit destinations that alias the source EPUB are refused.

## Anchor and compatibility rulings

The coordinating agent approved the following interpretation of the brief/spec:

1. Saving is explicit and may inspect the source EPUB. Annotation scope is the entire source book, independent of a render-time chapter selection. Grid construction is limited to chapters touched by rules, and repeated patterns reuse one computed anchor.
2. A selection with an exact chapter, exclusively exact selectors, and exactly one contiguous addressed subtree receives the digest of its complete canonical text span. `unit_digest` hashes a copy of its first grid unit extended through its last unit, preserving the established `sha256:<16 lowercase hex>` format. A display-elided line remains eligible when all matching units belong to the same addressed subtree.
3. Explicit wildcard/set/range/last selectors, whole-book selections, and sparse patterns spanning distinct subtrees receive matched leaf-unit counts. Distinct subtrees receive counts even when their spans happen to be adjacent. Sibling counts derived from the grid resolve `-1` at each numeric level.
4. A missing target has no text to hash and saves `matched: 0`; saving does not invent a digest or reject a rule whose stable chapter/path no longer exists. Loading hand-authored rules without either optional anchor field remains supported.
5. The brief's `repr(book.tuning)` test cannot run at this point because tuning introspection belongs to Task 14. The round-trip regression compares operation/rule semantics directly, removing newly authored digest/count metadata from the comparison. It also verifies that saving did not attach metadata to the original immutable rules.

## Test evidence

All commands ran through RTK in the assigned worktree.

- Red: `rtk uv run pytest tests/test_sidecar.py -q --no-cov` failed **11 tests** at the expected missing `annotations` / `write_annotations` methods.
- Initial green: the same command passed **11 tests in 0.24s**.
- Expanded sidecar suite: the same command passed **55 tests in 0.08s**.
- Related regression group: `rtk uv run pytest tests/test_sidecar.py tests/test_tuning_operations.py tests/test_patterns.py tests/test_precedence.py tests/test_grid.py tests/test_pipeline.py tests/test_import_boundaries.py -q --no-cov` passed **171 tests in 0.42s** before the final adjacent-subtree regression was added.
- `rtk uv run mypy`: **success, no issues in 142 source files**.
- `rtk uv run ruff check src/kenkui/_domain/sidecar.py src/kenkui/_domain/tuning.py src/kenkui/pipeline.py src/kenkui/errors.py tests/test_sidecar.py`: **all checks passed**.
- `rtk uv run ruff format --check` for those same five files: **5 files already formatted**.
- `rtk git diff --check`: passed.
- Full suite: `rtk uv run pytest -q` passed **1397 tests, 45 skipped, 7 deselected, 1 warning in 85.52s**. **Total coverage: 91.26%**, satisfying the configured 90% gate. The warning is the existing deliberate duplicate-ZIP-member fixture in `tests/test_epub.py`.

Focused tests use `--no-cov` because the configured coverage threshold applies to the whole package; the full run retains the 90% coverage gate.

## Self-review

- Verified round trips preserve operation-family order, per-kind declaration order, patterns, values, and immutable lexical entries. Arrays remain authoritative even when input Rule indices are out of order.
- Verified existing inline declarations precede loaded declarations, subsequent inline calls append after them, load counts describe only the file, and returned checkpoints reference the same resolved/roster objects.
- Verified file digests identify exactly the parsed bytes and change for formatting-only edits; saving a previously loaded pipeline refreshes the file's anchors without changing its in-memory snapshot.
- Verified exact paragraph and sparse single-sentence hashes cover the complete intended span. Self-review caught and fixed the adjacent-distinct-subtree case and added a regression for it.
- Verified wildcard/set/span/last/whole-book counts, missing matches, selected versus unselected render chapters, empty sidecars, absent source books for load/empty-save, malformed schemas, invalid values, invalid UTF-8, missing destinations, and atomic replace failure.
- Verified source overwrite protection and temporary-file cleanup. No user files outside the assigned task were changed; no stashes or subagents were used.

## Concerns and handoff

- Actual drift diagnostics and renderer/plan integration remain later tasks. `Rule.digest`, `Rule.matched`, and `Annotations.digest` are now available; the existing execution planner still needs to consume tuning/annotation identity as planned. Task 9 verifies changed annotation operations, matching the brief's fingerprint test.
- Later drift code must use the same full-subtree hash and leaf-count interpretation. `authoring_snapshot` centralizes the current calculations for reuse; care is needed to compare old metadata before replacing it with current metadata.
- Loading an absent sidecar is a sanitized `INVALID_SIDECAR` error; version-only sidecars are valid. Optional anchor fields support hand authoring. Saving recalculates current anchors, including replacing a missing exact anchor with count zero.
- Existing destination directories are required. Atomic replacement prevents partial files; this task does not introduce file locking or concurrent-editor conflict detection.

## Review fix round 1: Inline precedence and shared-rule deduplication

Review identified two important errors in the initial loader: it put loaded rules after existing inline declarations, and it duplicated shared rules each time code was reapplied before loading a saved sidecar. The ordering described in the initial implementation/self-review above is superseded by this fix.

`annotations()` now retains loaded rules as the prefix of each operation family, appends nonduplicate existing inline rules in their original order, and reindexes the entire merged sequence contiguously. Equal-specificity inline rules therefore win regardless of whether code declared them before or after the load call.

Deduplication compares only the exact `(Pattern, value)` pair across the loaded/inline boundary, ignoring `Rule.index`, `Rule.digest`, and `Rule.matched`. A duplicate keeps the loaded copy and its authoring metadata. `Annotations.loaded` consequently remains the exact prefix length, and unsaved code additions occupy only the suffix. The coordinating agent explicitly confirmed that repetitions within either source must remain intact to preserve that source's declaration semantics; this fix does not globally deduplicate or reorder them. Different patterns and different payloads remain distinct.

Validation:

- Red: `rtk uv run pytest tests/test_sidecar.py -k 'loading_extends or equal_specificity or repeated_save_load or deduplication' -q --no-cov` reproduced both findings: **4 failed, 54 deselected in 0.29s**.
- Green: `rtk uv run pytest tests/test_sidecar.py tests/test_tuning_operations.py tests/test_precedence.py tests/test_pipeline.py -q --no-cov` passed **141 tests in 0.26s**.
- `rtk uv run mypy`: **success, no issues in 142 source files**.
- `rtk uv run ruff check src/kenkui/pipeline.py tests/test_sidecar.py`: **all checks passed**.
- `rtk uv run ruff format --check src/kenkui/pipeline.py tests/test_sidecar.py`: **2 files already formatted**.
- `rtk git diff --check`: passed.
- Full suite: `rtk uv run pytest -q` passed **1400 tests, 45 skipped, 7 deselected, 1 warning in 83.42s**; **91.21% total coverage**, satisfying the 90% gate. The warning remains the deliberate duplicate-ZIP-member fixture.

Self-review verified equal-specificity precedence through `resolve_rules`, contiguous indices, unchanged pipeline checkpoints, immutable input branches, distinct pattern/value retention, intentional repetitions within both sources, and an exact loaded/unsaved boundary. The repeated save/load regression exercises all three tuning families across three cycles and asserts byte-for-byte file stability. Its shared inline rules have different indices and lack the stored anchor metadata, confirming neither affects duplicate detection. Only the loader, sidecar regressions, and this report changed; no subagents or stashes were used. There are no new downstream concerns beyond the original planned drift/renderer integration.
