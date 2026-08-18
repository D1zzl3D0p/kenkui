# Task 3 report: local single-voice vertical slice

## Scope and changed files

- Added `tests/integration/test_single_voice_pipeline.py`.
  - Creates a fixed two-chapter EPUB fixture.
  - Uses the existing private fake synthesis specification with the real local
    FFmpeg M4B assembler; it does not activate unavailable Pocket-TTS assets.
  - Exercises `Pipeline.write(..., workers=1)`, independently probes ordered
    chapter titles with host `ffprobe`, fully decodes with host `ffmpeg`, checks
    positive normalized speech statistics, and observes the terminal
    `Completed` event.
- No production source changes were made. The existing coordinator, spawned
  process pool, assembler, and private cache already satisfied the requested
  failure, ordering, atomic-publication, cache-equivalence, and non-inflated
  statistics boundaries. Existing focused coverage is in
  `tests/test_process_execution.py`, `tests/test_ffmpeg_audio.py`,
  `tests/test_cache.py`, and `tests/test_native_ffmpeg.py`.

## Red-test proof

The new test was added before any production edit and first executed with:

```sh
KENKUI_RUN_NATIVE=1 uv run pytest --no-cov -m native tests/integration/test_single_voice_pipeline.py
```

Its first collection was red because an integration subdirectory cannot import
root-level `test_epub` helpers (`ModuleNotFoundError: test_epub`). After making
the fixture self-contained, the test was red again because its probe helper was
missing the `cast` import (`NameError: cast`). Both were test-harness defects,
not production behavior; they were corrected before the first behavioral run.
No production behavior was red, and no orchestration correction was warranted.
The corrected first behavioral run passed: `1 passed in 0.62s`.

## Final focused test evidence

### Deterministic focused suite

```sh
KENKUI_RUN_NATIVE=1 uv run pytest --no-cov tests/integration/test_single_voice_pipeline.py tests/test_process_execution.py tests/test_ffmpeg_audio.py tests/test_cache.py tests/test_native_ffmpeg.py
```

Result: `58 passed, 4 deselected in 51.43s`.

This exercises deterministic spawned serial/parallel plan-order behavior,
worker failure and cancellation cleanup, mocked FFmpeg failure/publication
boundaries, private cache cold/warm equivalence and cancellation behavior, and
statistics assertions. The four native tests were deselected by the repository
configuration's default `-m=not native` expression despite
`KENKUI_RUN_NATIVE=1`.

### Environment-gated host FFmpeg suite

```sh
KENKUI_RUN_NATIVE=1 uv run pytest --no-cov -m native tests/integration/test_single_voice_pipeline.py tests/test_native_ffmpeg.py
```

Result: `4 passed in 2.58s`.

This directly exercised host `ffmpeg`/`ffprobe`: the new integration fixture
published a decodable M4B with chapters `("One", "Two")`; existing native
coverage additionally checked serial/parallel outputs and cold/warm private
cache decode equivalence.

## Commit

Implementation commit: `708fa2f67a0bae01df0a5e56386c01bfad6677e9`

## Self-review

- The acceptance path remains private: the test supplies bindings only through
  the established private test seam and introduces no public cache path, server
  identifier, or scheduling control.
- It retains spawned synthesis, plan-order rendering, atomic publication, and
  bounded host command execution; it does not change their implementation.
- The FFprobe assertion is independent of the assembler's returned metadata,
  and the FFmpeg decode verifies the emitted container can be consumed.
- Existing focused tests cover the requested pre-commit callback/cancellation,
  serial-vs-parallel, and warm-cache boundaries without duplicating them in the
  integration test.

## Concerns

- Task 1 remains blocked by the absent approved Pocket-TTS model and voice
  assets. Consequently, the environment-gated vertical evidence uses the
  deterministic fake synthesizer plus real local FFmpeg, not real Pocket-TTS
  inference or an authorized production prompt.
- The brief's exact command does not select native tests under the current
  pytest default marker expression. The explicit `-m native` supplemental run
  above is required to obtain the stated FFmpeg probe/decode evidence.
