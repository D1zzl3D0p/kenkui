# Task 6 report: Patterns and subset ordering

## Design choices

- Implemented all five selectors as frozen, slotted value objects. `OneOf`
  defensively freezes its values, and `OneOf`/`Span` validate positive integer
  coordinates even when constructed directly.
- Implemented `Pattern` as a frozen `Mapping[str, Selector]`, stored as a
  canonical level-ordered tuple. Omitted levels remain omitted in the mapping
  and are interpreted as `Any()` by matching and subset comparison, preserving
  sparse paths rather than imposing a contiguous hierarchy.
- `parse_pattern` accepts exact values, `*`, positive integer lists, inclusive
  `lo..hi` ranges, and `-1` for `Last()`. Chapter IDs support only exact strings
  or `*`; chapter lists, ranges, and `Last()` raise
  `ValidationError(ErrorCode.INVALID_PATTERN)`.
- `matches` resolves `Last()` from the sibling count keyed by the concrete
  parent coordinate tuple while all other selectors are context-free.
- Selector ordering returns `True` for containment, `False` when the other
  selector is strictly narrower, and `None` for incomparable or unresolved
  relationships. Pattern ordering combines per-level direction, returning
  `None` when constraints cross in opposite directions.

## Exact test evidence

- Red phase: `uv run pytest tests/test_patterns.py -v` failed during collection
  with `ImportError: cannot import name 'parse_pattern'`.
- `uv run pytest tests/test_patterns.py -v --no-cov`: **22 passed**.
- `uv run pytest tests/test_patterns.py tests/test_paths.py --no-cov -q`:
  **39 passed**.
- `uv run pytest --no-cov`: **1288 passed, 45 skipped, 7 deselected**.
- `uv run ruff check src/kenkui/_domain/paths.py src/kenkui/errors.py
  tests/test_patterns.py`: **All checks passed**.
- `uv run mypy src/kenkui/_domain/paths.py tests/test_patterns.py`:
  **Success: no issues found**.
- `git diff --check`: **passed**.

## Self-review

- Boolean values are rejected explicitly despite being integer subclasses.
- Empty lists, reversed ranges, non-positive coordinates, unknown levels, and
  non-index strings outside the chapter level are rejected with the new stable
  error code.
- Equal patterns are subsets of each other; a concrete selector is narrower
  than a wildcard; overlapping finite sets and unresolved `Last()` comparisons
  are incomparable.
- `Pattern.is_whole_book()` recognizes both an empty pattern and patterns made
  solely of explicit wildcards.
- Existing `Path` validation and sparse coordinate behavior are unchanged.

## Concerns

- The brief's isolated pytest command triggers the repository-wide 90% coverage
  gate and therefore reports a coverage failure despite passing every pattern
  test. Focused evidence uses `--no-cov`; the complete suite was also run.
- A missing sibling-count entry makes `Last().covers(...)` return `False`, as
  matching cannot resolve the final coordinate without that context.
