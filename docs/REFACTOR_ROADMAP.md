# Refactor Roadmap

This roadmap tracks the library-core cleanup only.

## Done In The First Pass

- Split `kenkui.models` into a package while preserving the historical
  `kenkui.models` import path.
- Moved root package implementation wrappers into `kenkui.api`.
- Reduced `kenkui.__init__` to a public export facade.
- Added typed exception categories in `kenkui.errors`.
- Removed interactive HuggingFace setup prompts from the core library path.
- Made Ruff a passing repository gate with scoped ignores for unrelated legacy
  style choices.

## Next Work

- Split `nlp.__init__` into focused cache, roster cache, spaCy loading, segment
  assembly, and compatibility-export modules.
- Make `services.nlp_service` plus `nlp.pipeline` the only supported NLP
  orchestration path, then deprecate legacy orchestration entrypoints.
- Split rendering internals behind `kenkui.rendering` modules while keeping
  `kenkui.parsing.AudioBuilder` as a compatibility import during transition.
- Split voice behavior into catalog, exclusion config, cast assignment, review
  formatting, and preview synthesis modules.
- Replace remaining direct `print()` calls in core rendering/readers with
  logging or callbacks.
- Expand no-interactive-I/O tests beyond HuggingFace auth.

## Non-Goals

- No in-repo CLI.
- No in-repo FastAPI server.
- No remote execution implementation.
- No cloud auth, billing, benchmarking, scheduler, or farm orchestration.
- No behavior-changing cache migrations without explicit tests.

## Quality Gates

- `python -m ruff check .`
- `python -m pytest -q`
- Unit tests must avoid real network calls, real model downloads, and heavy audio
  synthesis unless marked as integration or slow.

