# AI Notes

These notes record known mismatches between the desired library-core shape and
the current codebase. Use `AGENTS.md` as the working contract and
`docs/ARCHITECTURE.md` for the intended architecture.

## Remaining Structural Mismatches

- `src/kenkui/nlp/__init__.py` still contains cache helpers, roster metadata,
  segment assembly, and legacy orchestration entrypoints. Split it before making
  deeper NLP behavior changes.
- `src/kenkui/parsing.py` still combines output naming, annotated-cache loading,
  speaker auto-assignment, ETA tracking, stitching, cover embedding, and
  `AudioBuilder`.
- `src/kenkui/services/voice_service.py` still combines catalog lookup,
  exclusion config, preview synthesis, cast assignment, and review formatting.
- Some reader/rendering paths still use direct `print()` calls. Core modules
  should move toward logging or callbacks.
- Provider extension hooks exist for local/remote-style modes, but remote
  execution remains external-client work.

## Repository Noise

Generated/environment artifacts such as `build/`, `dist/`, `.venv/`,
`.pytest_cache/`, `.ruff_cache/`, and package metadata are not architecture.

## Cleanup Guidance

- Preserve `kenkui.models` and `kenkui.api` public imports while moving internals.
- Prefer focused modules under existing ownership boundaries over new generic
  utility modules.
- Add no-interactive-I/O tests when removing prompts or terminal output from
  core code.
- Keep remote/provider abstractions concrete and minimal.

