# Immutable API and execution

Only names exported by `kenkui.__all__` are public. Modules whose names start
with `_` are implementation details and may change without compatibility notice.

## Construct and branch

`book(path)` dispatches a supported suffix and currently accepts `.epub` only;
`epub(path)` records EPUB intent without opening the path. `Pipeline` and its
operations are frozen values. Every fluent method returns a new branch.

```python
import kenkui as kk

source = kk.book("novel.epub")
all_chapters = source.assign_voice("narrator").tts()
selection = (
    source.select_chapters("chapter-a", "chapter-c")
    .normalize_text()
    .assign_voice("narrator")
    .tts()
    .metadata(title="Novel", author="Writer", cover="source")
)
assert source.operations == ()
```

Use either `select_chapters(*ids)` or the inclusive
`select_chapter_range(start_id, end_id)`, before `tts()`. Chapter IDs come from
`inspect()` and are stable functions of canonical EPUB member/fragment identity
and occurrence. Duplicate operations and invalid ordering fail immediately.

## Validate and inspect

`validate()` checks path readability and required voice/TTS intent cheaply; it
does not parse the source or prove that a voice can be resolved. It returns a
frozen `ValidationResult` containing zero or more `ValidationIssue(code,
message)` values.

`inspect()` securely parses the EPUB and applies chapter selection. It returns a
frozen `BookInspection`: source metadata and an ordered tuple of
`ChapterInspection(id, index, title, speech_characters, text)`. Inspection does
not import a synthesis provider or invoke FFmpeg.

```python
validation = selection.validate()
if not validation.is_valid:
    for issue in validation.issues:
        print(issue.code.value, issue.message)
else:
    inspection = selection.inspect()
    print(inspection.metadata.title)
    for chapter in inspection.chapters:
        print(chapter.id, chapter.speech_characters)
```

## Write, events, result, and overwrite

`write()` and `write_m4b()` are equivalent. The parent directory must already
exist and the suffix must be `.m4b`. Existing regular output raises
`output_exists` unless `overwrite=True`. Publication uses an atomic final replace;
validation, rendering, callback, cancellation, or encoding failure cannot publish
the work-in-progress candidate.

```python
from kenkui import CancellationToken, StageProgress

token = CancellationToken()


def progress(event: object) -> None:
    if isinstance(event, StageProgress):
        print(event.stage, event.completed, event.total, event.chapter_id)


result = selection.write_m4b(
    "novel.m4b",
    on_event=progress,
    cancel=token,
    workers="auto",
    overwrite=False,
)
print(result.output)
print(result.stats)
```

The frozen `Result` contains the published `Path` and `ExecutionStats`:
normalized and synthesized character counts, synthesized segment count, rendered
chapter count, and duration in milliseconds. Warm-cache results retain semantic
work statistics; they do not pretend cached speech ceased to exist.

Events are frozen and sequence-numbered. Successful order is `Started`, stage
start/progress/completion events, then `Completed`; recoverable conditions may
emit `Warning`. Progress is emitted in plan order even when workers finish out of
order. Event callbacks execute in the coordinator; if one raises before commit,
execution fails as `callback_failed`, workers/workspace are cleaned, and nothing
is published. This includes every pre-commit event, including publication
`StageStarted` and `StageProgress`, each of which is followed by a cancellation
check. The coordinator then creates and validates a final private snapshot,
checks cancellation once more, and runs no callback between that check and the
atomic commit. Only after a successful commit are publication `StageCompleted`
and terminal `Completed` emitted. Both are best-effort; either callback may fail
without revoking the published output. A publish failure emits neither.

## Cancellation and workers

`CancellationToken.cancel()` is thread-safe and idempotent. Pass the token to a
write and cancel it from another thread or an event callback. Cancellation is
cooperative at bounded orchestration points, terminates/then kills children with
bounded waits when necessary, raises `CancelledError` (`cancelled`), and never
publishes an artifact.

`workers` must be `"auto"` or a positive, non-boolean integer. Auto considers
available CPUs and chapter count. Both automatic and explicit requests are
bounded by chapter count and the conservative hard cap of two. Rendering uses
only Python **spawned** worker processes, including where `fork` is available.
Each worker constructs one engine and processes its bounded static segment batch
serially; workers do not share model state, and the parent never constructs an
engine.

## Current production gate

Ordinary writes fail closed with `renderer_unavailable` unless
`KENKUI_POCKET_MANIFEST` explicitly names an approved owner-controlled local
manifest. When supplied, the assigned voice ID is resolved from that strict
manifest and the public pipeline uses Pocket, FFmpeg, and the private OS cache.
This activation mechanism does not approve any model or voice: operators must
complete the real-inference and rights gates described under Models and voices.
