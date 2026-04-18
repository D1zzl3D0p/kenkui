# AI Notes

These notes are intentionally separate from `AGENTS.md`. The guide describes how the repo should be approached. This file records places where the current codebase does not fully match that shape yet.

## Structural mismatches

- Business rules are partly duplicated between the CLI and service layers.
  `src/kenkui/cli/add.py` contains nontrivial voice-assignment and gender-pool logic that overlaps with `src/kenkui/services/voice_service.py`.

- Scene-break detection is duplicated.
  Similar `_SCENE_BREAK_RE` and `_is_scene_break()` logic exists in `src/kenkui/workers.py` and `src/kenkui/nlp/__init__.py`.

- Gender-pronoun normalization is duplicated.
  `src/kenkui/cli/add.py:_gender_pool()` and `src/kenkui/services/voice_service.py:gender_from_pronoun()` solve the same problem in different layers.

- The service package has no explicit shared contract.
  `src/kenkui/services/__init__.py` is empty, which is fine technically, but it means the service layer is discoverable only by scanning files rather than by reading an exported surface.

## Repository-shape mismatches

- The working tree currently has local modifications in:
  `src/kenkui/models.py`
  `src/kenkui/server/worker.py`
  `src/kenkui/workers.py`
  Treat current behavior in those files as in-flight until those edits are settled.

- The repository root contains generated or environment directories that are not part of the logical architecture:
  `build/`
  `dist/`
  `.venv/`
  `.mypy_cache/`
  `.ruff_cache/`
  `.pytest_cache/`
  `.kenkui_cache/`

- The `docs/` tree currently contains macOS metadata files:
  `docs/.DS_Store`
  `docs/superpowers/.DS_Store`
  These are not part of the documentation model and should be treated as noise.

## Guidance for future cleanup

- Move reusable voice-assignment rules into `services/voice_service.py` and let CLI code focus on prompts, review screens, and user flow.
- Extract duplicated scene-break and pronoun helpers into a shared module only if both call sites are meant to stay active long term.
- Consider adding a short `docs/` index if more AI- or human-facing documentation accumulates.
