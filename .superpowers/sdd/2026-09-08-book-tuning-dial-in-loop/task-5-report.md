# Task 5 report: Paths and subtree addressing

## Decisions

- Added `LEVELS` in the required chapter-to-phrase order.
- Implemented `Path` as a frozen, slotted dataclass with all five components
  defaulting to `None`; unspecified components represent an unconstrained
  level.
- `parse_path` accepts sparse mappings, permits explicit `None` values, and
  rejects unknown keys, non-string chapter values, booleans/non-positive or
  non-integer indexed values with `ValidationError(ErrorCode.INVALID_PATH)`.
- `path_of` copies the five 1-based coordinates and chapter ID from `Unit`.
- `contains` performs component-wise wildcard matching, including reflexive
  containment and the whole-book path.
- `render_path` uses the requested labels and two-space separators, elides
  stored line `1`, and renders an empty display as `whole book`.
- Added the stable `INVALID_PATH` error code and default message.

## Test evidence

- `uv run pytest tests/test_paths.py -v`: test execution passed, but the
  repository coverage gate failed because the isolated file produced 24%
  coverage against the configured 90% threshold.
- `uv run pytest tests/test_paths.py tests/test_grid.py --no-cov -q`: **17
  passed**.
- `uv run pytest --no-cov -q`: **passed** (full suite; RTK emitted no failure
  output).
- `uvx ruff check src/kenkui/_domain/paths.py src/kenkui/errors.py`: **All
  checks passed**.

## Self-review

- The stored line coordinate is retained even though display rendering omits
  line `1`, preserving immutable addressing semantics for later matching.
- Validation deliberately rejects `bool` despite Python's `bool` subclassing
  `int`, so boolean input cannot masquerade as a coordinate.
- `contains` does not impose hierarchy-contiguity rules; it follows the brief's
  exact per-level wildcard definition and therefore supports holes.
- No existing files outside the Task 5 scope were changed.

## Concerns

- The ordinary isolated pytest command cannot satisfy this repository's global
  coverage threshold; use `--no-cov` for focused evidence or run the full suite
  with coverage when coverage accounting is required.

## Review fix round 1

- Corrected the earlier hole test: a path may not skip paragraph before a
  sentence/line/phrase. Line remains optionally omitted because the required
  full-path example deliberately omits it; phrase still requires sentence.
- Added `Path.__post_init__` validation so direct construction enforces valid
  chapter/type/positive coordinate values and the same hierarchy invariants as
  parsing. `contains` consequently operates only on valid prefix paths.
- Evidence: `uv run pytest tests/test_paths.py tests/test_grid.py --no-cov -q`
  => **25 passed**; `uv run pytest -q` => **passed**; Ruff => **All checks
  passed**; `uv run mypy src/kenkui/_domain/paths.py` => **Success**.
