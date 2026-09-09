# Task 8 report: Tuning operations, tiers, and immutable pipeline methods

## Implementation

- Added frozen `Attributions`, `Silences`, and `Pronunciations` records carrying ordered `tuple[Rule, ...]` values. Added frozen `Annotations(path, digest, loaded)`, defensively copying its loaded mapping into a read-only `MappingProxyType`.
- Extended the existing `Operation` union without replacing the repository's union-based operation model.
- Added `replace_or_append`, which replaces a matching exact operation type at its original position and appends missing types after validating synthesis order.
- Added `Pipeline.attribute` and `Pipeline.silence`, plus `WhereArg` and an internal `_add_rule` helper. Raw sparse coordinate mappings are parsed with Task 6's parser; already parsed `Pattern` values pass through; `None` selects the whole book. Each declaration gets `index=len(existing_rules)` and appends at the tuple's end, preserving Task 7's tuple-order precedence.
- Extended `Pipeline.pronounce` with `where`. Explicit lexicons become ordered `Pronunciations` rules. Values are immutable sorted entry tuples from the established lexicon validator. Omitting a lexicon adds no empty tuning declaration; explicitly passing an empty mapping records an empty lexicon rule. Global number presets, built-in pronunciation, and feature overrides replace `SpokenForm` settings.
- Added one explicit tier mapping for all 14 operation families. Metadata, series, and chapter selection are identity; attribution, silence, pronunciation, and annotation corrections are tuning; the six production/model intent families are style. Unknown types return `None`.
- Centralized checkpoint handling in `_branch`. Tuning operations preserve `_resolved` and `_roster`; changes to model/cast intent invalidate `_resolved` as before. Changing selection or character discovery also invalidates `_roster`.

## Plan-compatibility decisions

The coordinating agent explicitly confirmed these rulings during implementation:

1. Every style/identity re-call replaces, including `infer_characters`, `attribute_quotes`, and `tts`. The latter methods were omitted from the brief's switch list, but the user-approved general rule applies to them. Selection methods remain on `append_unique`, including duplicate and mixed-selection rejection.
2. Existing metadata-after-TTS behavior is preserved. `replace_or_append` has a keyword-only `before_tts=True` default; metadata uses `False`. Existing settings can be replaced after TTS, while adding other new pre-synthesis stages still fails with `INVALID_OPERATION_ORDER`.
3. `InferCharacters` and `AttributeQuotes` are style operations.
4. `Pronunciations` owns Pipeline-provided lexicons. `SpokenForm.lexicon` remains available to legacy direct domain callers, but `Pipeline.pronounce` leaves it empty. Rendering integration belongs to later work in this plan.

## Test evidence

All commands ran from the assigned worktree through RTK.

- Red: `rtk uv run pytest tests/test_tuning_operations.py tests/test_tiers.py -v --no-cov` failed collection before implementation with missing `Annotations` and missing `kenkui._domain.tiers` (exit 2).
- Green initial focused run: the same command passed 31 tests in 0.08s.
- Final focused run, including extra invalidation regressions: the same command passed **35 tests in 0.33s**.
- Pipeline/resolution regression subset: `rtk uv run pytest tests/test_tuning_operations.py tests/test_tiers.py tests/test_pipeline.py tests/test_resolution_inspection.py -v --no-cov` passed **90 tests in 3.35s** before the four final invalidation cases were added.
- Initial full run: `rtk uv run pytest -v` produced **5 failed, 1327 passed, 45 skipped, 7 deselected, 1 warning in 83.58s**. Failures were the four existing pipeline assertions for old duplicate/lexicon storage behavior and the lexicon-file-to-pipeline assertion. All were updated to assert the new contract; selection duplicate checks remain enforced.
- Final full run: `rtk uv run pytest -v` passed with **1336 passed, 45 skipped, 7 deselected, 1 warning in 82.26s**. **Total coverage: 91.07%**, satisfying the configured 90% gate.
- `rtk uv run mypy`: **success, no issues in 140 source files**.
- `rtk uv run ruff check src/kenkui/_domain/operations.py src/kenkui/_domain/tiers.py src/kenkui/pipeline.py tests/test_tuning_operations.py tests/test_tiers.py tests/test_pipeline.py tests/test_spoken_lexicon_files.py`: **all checks passed**.
- `rtk uv run ruff format --check` for those same seven files: **7 files already formatted**.
- `rtk git diff --check`: passed.
- `rtk uv run ruff check .`: **15 existing violations, exclusively in unchanged `tests/test_paths.py`** (missing module/function docstrings and test magic-number warnings). These are outside Task 8 and were left untouched.

The focused commands use `--no-cov` because a small subset cannot satisfy the repository-wide 90% threshold; the full run retains the configured coverage requirement.

## Self-review

- Confirmed one tuning operation per family across repeated calls, preservation of declaration order and rule indices, immutable branching, and defensive snapshots of mappings.
- Exercised whole-book defaults, sparse mappings, existing Patterns, zero and maximum silence durations, and rejection of negative, excessive, boolean, non-integer, and absent durations.
- Verified lexicon accumulation separately from last-call global speech settings.
- Verified retained machine checkpoints by calling `resolve()` on tuned branches and asserting the same resolved snapshot and no new branch from resolution.
- Verified replacement of model/cast/series intent drops the old resolution snapshot without mutating the original.
- Exercised style and identity replacements, TTS idempotent replacement, original operation position, metadata append after TTS, and refusal of a new silence operation after TTS.
- Kept explicit regression coverage for both repeated selection methods and both orders of mixed selection methods.
- The tier census iterates operation-module classes and tests union membership, matching the repository's `Operation` type alias; each known family also has a direct tier assertion.

## Concerns and handoff

- This is the operation/API layer only. New tuning records still require planned downstream inspection, serialization, gap normalization, and rendering consumption. In particular, Pipeline-provided lexicons are now in `Pronunciations`; old planning code reads only `SpokenForm.lexicon`. Downstream integration must consume the new rules before the overall plan is complete.
- Annotation loading methods are not part of this task; only the immutable operation record and tier/checkpoint classification were added.
- No subagents were spawned, no files were stashed, and only Task 8 implementation, regression tests, and this report are included in the commit.
