# Troubleshooting and stable errors

Catch `KenkuiError` subclasses and branch on `error.code`, not message text.
`ErrorCode` values are machine-readable and messages are sanitized; paths,
provider exceptions, and subprocess output are intentionally not public.

```python
import kenkui as kk

try:
    kk.epub("book.epub").assign_voice("voice").tts().inspect()
except kk.KenkuiError as error:
    print(error.code.value, str(error))
```

## Python or locked environment

Use CPython 3.11, 3.12, or 3.13. Other versions are outside the declared range.
For contributors, verify rather than update the committed lock:

```console
uv lock --check
uv sync --frozen --all-groups
```

Run repository commands through `uv run`, or install the built wheel into the
active environment.

## Source and EPUB failures

- `unsupported_format`: use an `.epub` path.
- `source_not_found` / `source_not_readable`: check the path, regular-file state,
  and current-user read permission.
- `malformed_epub`: ZIP/package/XHTML structure is invalid or unsupported.
- `unsafe_archive_path`: a member path is unsafe; do not extract or rewrite an
  untrusted archive to bypass this check.
- `archive_limit`: a fixed ZIP/XML/spine/text safety budget was exceeded.
- `empty_chapter`, `chapter_not_found`, `reversed_chapter_range`: inspect chapter
  IDs and visible text before selecting.

See [security boundaries](security.md) for exact limits.

## Intent, output, events, and cancellation

- `voice_required` and `tts_required`: call `assign_voice(...).tts()` in order.
- `duplicate_operation` / `invalid_operation_order`: create a new immutable branch
  with each operation once and place selections/normalization/voice before TTS.
- `invalid_workers`: use `"auto"` or a positive integer (not `True`/`False`).
- `invalid_output`: use an `.m4b` path whose parent directory already exists.
- `output_exists`: choose another path or explicitly pass `overwrite=True`.
- `callback_failed`: an event callback raised before publication commit; fix it
  or remove it. No candidate was published. Publication `StageCompleted` and
  terminal `Completed` happen only after commit and are best-effort, so exceptions
  from those callbacks are generically logged and do not turn success into failure.
- `cancelled`: the supplied token was cancelled. Cancellation is expected to
  leave no published candidate.

## FFmpeg capability or artifact failure

Install FFmpeg 5+ with both commands and AAC support on `PATH` (see
[installation](installation.md)). Stable distinctions are:

- `ffmpeg_not_found`, `ffprobe_not_found` — executable discovery;
- `ffmpeg_unsupported`, `ffprobe_unsupported` — capability preflight;
- `encoding_failed`, `cover_failed` — candidate construction;
- `probe_failed`, `decode_failed`, `invalid_artifact` — independent validation;
- `publication_failed` — validated candidate could not be atomically published.

A failure never publishes the candidate. To reproduce the explicit native tier:

```console
ffmpeg -version
ffprobe -version
KENKUI_RUN_NATIVE=1 uv run pytest --no-cov -m native tests/test_native_ffmpeg.py
```

## Renderer, Pocket, voice, and synthesis

- `renderer_unavailable`: no manifest at `KENKUI_POCKET_MANIFEST` or the
  managed default. Run `kk.load_voice("<id>")` once. Inspection and validation
  work without it.
- `voice_not_provisioned`: the voice is registered but not loaded. The message
  names the call that fixes it: `kk.load_voice("<id>")`.
- `voice_unknown`: the ID is in neither the built-in catalog nor the manifest.
  `kk.list_voices()` shows everything available.
- `engine_not_cloning_capable`: a `wav` voice needs the gated
  `kyutai/pocket-tts` weights to compile. Accept the terms and authenticate, or
  supply a pre-compiled `.safetensors` instead.
- `voice_variety_invalid`: unrecognised variety or state, or an `add_voice`
  ID that collides with a built-in catalog name.
- `pocket_package_missing`: `pocket-tts` is a required dependency; reinstall.
- `pocket_version_unsupported`: exactly 2.1.0 is required.
- `pocket_model_invalid` / `pocket_voice_invalid`: local manifest/tree/config or
  prompt/rights declarations failed preflight; do not fall back to a downloader.
- `voice_provenance_required`, `voice_disabled`, `voice_incompatible`,
  `voice_unresolved`: correct the explicit voice registry material.
- Pocket model/voice load and inference codes, `synthesis_failed`, and
  `invalid_audio`: provider/worker output failed a sanitized execution boundary.

No approved model/voice ships with this repository. Real Pocket tests must remain
skipped until the separate gate described on the
[Pocket adapter page](pocket-tts-adapter.md) is explicitly approved.
