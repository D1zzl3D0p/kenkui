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

## Provisioning voices

Rendering reads a manifest; it never downloads. Provisioning is explicit, and
`load_voice` is the only thing that makes a voice renderable:

```python
import kenkui as kk

kk.load_voice("eponine")

result = (
    kk.book("book.epub")
    .normalize_text()
    .assign_voice("eponine")
    .tts()
    .write("book.m4b")
)
```

`load_voice` is idempotent: a voice whose asset is present and hash-verified
performs no network access. The first call for a language downloads that
language's engine, roughly 225 MB, plus a roughly 6.5 MB embedding.

Writing with a voice that was registered but never loaded raises
`voice_not_provisioned`, before any worker process is spawned, and the message
names the call that fixes it.

### The five verbs

```python
add_voice(path, *, voice_id, name, language, provenance,
          license_id, commercial_use_allowed, voice_rights) -> Voice
load_voice(voice_id)   -> Voice   # registered -> loaded
unload_voice(voice_id) -> Voice   # loaded -> registered, prunes the engine
remove_voice(voice_id) -> None    # deletes the entry entirely
list_voices()          -> tuple[Voice, ...]
```

A voice is `registered`, `loaded`, or — reported by `list_voices` only —
`missing`, when the manifest says loaded but the file is gone. Built-in voices
are implicitly registered by the catalog, so `load_voice` works on them
directly. `unload_voice` keeps your rights metadata; `remove_voice` discards it.

There are no bulk provisioning verbs. Provisioning several voices at once
composes over `list_voices()`:

```python
for voice in kk.list_voices():
    if voice.variety == "built-in" and voice.language == "english":
        kk.load_voice(voice.id)

engines = {v.engine for v in kk.list_voices() if v.state == "loaded"}
disk = sum(e.size_bytes for e in engines)
```

Engines are derived state with no verbs of their own: they are provisioned when
a voice needs one and pruned when the last loaded voice referencing one goes
away. `Voice.engine` exposes the engine a loaded voice uses.

## Casting characters

One voice is the degenerate cast. These two are the same pipeline:

```python
kk.epub("book.epub").assign_voice("eponine")
kk.epub("book.epub").assign_voices(narrator="eponine")
```

A full cast adds character inference and dialogue attribution, both of which
name the model to use:

```python
result = (
    kk.epub("book.epub")
    .infer_characters(model="anthropic/claude-sonnet-5")
    .attribute_quotes(model="anthropic/claude-sonnet-5")
    .assign_voices(narrator="eponine", method="gendered")
    .tts()
    .write("book.m4b")
)
```

Order does not matter. Only `tts()` must come last.

### Roles

`narrator` speaks everything that is not attributed dialogue. `unknown` speaks
dialogue nobody could be placed for, and defaults to the narrator's voice, so
an unplaced line sounds like narration rather than like a third character. Set
it separately to make the distinction audible:

```python
.assign_voices(narrator="eponine", unknown="paul")
```

Both roles are excluded from the pool characters are cast from. The narrator
speaks in every chapter, so sharing its voice with a character would collide
everywhere.

### Methods

A method decides which voices a character is eligible for. The shared solver
then does the assigning.

| Method | Eligible voices |
| --- | --- |
| `gendered` | voices whose `perceived_gender` matches the character's |
| `random` | the whole pool |

Neither is random in the sense of varying between runs. Same book, same
method, same pool always gives the same cast: reproducibility is required, and
the solver's ordering supplies the variation instead.

A character whose gender was never inferred falls back to the whole pool. A
voice whose gender was never *sourced* never joins a gendered pool, because a
missing trait is an admission of ignorance rather than a wildcard.

### Pinning a choice

```python
.assign_voices(narrator="eponine", cast={"javert": "charles"})
```

Pinned entries are constraints on the solver, not suggestions.

### How voices are chosen

Characters who speak in the same chapter never share a voice. Among the voices
still free, the solver takes the least-used one, weighted by how much each
character actually speaks — so a lead does not land on the voice a walk-on
already holds. Voices spread before they repeat, and repeat only once the pool
is under pressure.

When the pool cannot satisfy that, the solver minimises the clash and logs
`cast_collision`. It is not raised and not reported to the caller: a collision
means the pool ran short, which is fixed by provisioning more voices.

### Stored work

Attribution is expensive; casting is free. Both are stored in
`casting.sqlite3` beside the voice manifest, keyed so that exploring ten
castings of one book costs one model pass.

```python
kk.list_castings()
kk.remove_casting(casting_id)      # free to rebuild
kk.remove_attribution(attribution_id)  # cascades; costs a fresh model pass
```

The two removal verbs are separate because their costs differ by orders of
magnitude.

### Registering your own voice

`add_voice` takes a local `.wav` or `.safetensors` and requires every rights
field explicitly — Kenkui infers none of them:

```python
kk.add_voice(
    "narrator.wav",
    voice_id="narrator",
    name="House Narrator",
    language="english",
    provenance="recorded 2026-08-19 with documented consent",
    license_id="proprietary",
    commercial_use_allowed=True,
    voice_rights="owned outright",
)
kk.load_voice("narrator")
```

A `.wav` is compiled into an embedding at `load_voice` time, which needs the
gated cloning-capable weights. A `.safetensors` is registered as-is and needs
no model at all. See Models and voices for the gating details.

### Manifest location

Kenkui manages a manifest at `~/Library/Caches/kenkui/v1/manifest.json` (macOS)
or `${XDG_CACHE_HOME:-~/.cache}/kenkui/v1/manifest.json` (Linux).
`KENKUI_POCKET_MANIFEST` overrides that path for operator-controlled
deployments, with unchanged strict-validation semantics. With neither present,
`write()` fails closed with `renderer_unavailable`.
