# Immutable API and execution

Only names exported by `kenkui.__all__` are public. Modules whose names start
with `_` are implementation details and may change without compatibility notice.
The [public API reference](api.md) lists every exported name and the current
method signatures; this guide explains how to compose them.

## Construct and branch

`book(path)` dispatches a supported suffix and currently accepts `.epub` only;
`epub(path)` records EPUB intent without opening the path. `Pipeline` and its
operations are frozen values. Intent-building methods return new branches.

Standalone scripts must call `resolve()`, provisioning, and rendering from an
`if __name__ == "__main__":` guard or a function invoked by that guard. Kenkui
uses spawned processes on every supported platform; workers import the script
again. Keep expensive effects out of module-level initialization. Pure pipeline
construction can remain at module scope.

```python
import kenkui as kk

source = kk.book("novel.epub")
all_chapters = source.assign_voice("narrator").tts()
selection = (
    source.select_chapters("chapter-a", "chapter-c")
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

## Compose ordinary functions

Use `pipe(function, *args, **kwargs)` to keep reusable operations in ordinary
functions. The pipeline becomes the function's first argument; its return
value becomes the result of the call. A function returning a `Pipeline` keeps
the chain going, while a function returning a report ends it with that report.
The function runs immediately, so any I/O it performs happens at that point.

```python
import kenkui as kk


def speech_style(pipeline: kk.Pipeline, *, paragraph_ms: int) -> kk.Pipeline:
    return pipeline.pronounce().pauses(paragraph_ms=paragraph_ms)


pipeline = (
    kk.epub("book.epub")
    .assign_voice("eponine")
    .pipe(speech_style, paragraph_ms=250)
    .tts()
)
```

The same function can be called directly with an unconfigured pipeline as its
first argument. No subclass, registration, or modification to Kenkui's
`Pipeline` class is required.

## One-call rendering

`magic_run(book_path, *, narrator, multi=False, model="openrouter/deepseek/deepseek-v4-flash")`
provides the small, defaulted rendering surface. It derives an `.m4b` output
beside the EPUB and returns the normal `Result`.

```python
import kenkui as kk

if __name__ == "__main__":
    single = kk.magic_run("novel.epub", narrator="eponine")
    # To choose character casting instead, use multi=True with a provisioned pool.
```

`multi=True` adds character inference, quote attribution, and automatic casting.
The default model is the LiteLLM identifier `openrouter/deepseek/deepseek-v4-flash`; pass
`model=` to select another configured provider/model. The helper exposes no
selection, output, overwrite, callback, worker, or cast controls; use the
fluent API for those cases.

## Validate and inspect

`validate()` checks path readability and required voice/TTS intent cheaply; it
does not parse the source or prove that a voice can be resolved. It returns a
frozen `ValidationResult` containing zero or more `ValidationIssue(code,
message)` values.

Before resolution, `inspect()` securely parses the EPUB and applies chapter
selection. It returns a frozen `BookInspection`: source metadata and an ordered
tuple of `ChapterInspection(id, index, title, speech_characters, text)` values.
After resolution, it returns the saved snapshot with its casting information
(see [Resolve before write](#resolve-before-write)). Inspection never calls a
model or invokes FFmpeg.

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
A successful publication clears that book's private audio cache, so a finished
book stops holding its rendered PCM on disk. Pass ``keep_audio_cache=True`` to
keep the segments and re-render the book without paying for synthesis again
(for example while retuning pauses or voices).

```python
from kenkui import CancellationToken, CastResolved, StageProgress

token = CancellationToken()


def progress(event: object) -> None:
    if isinstance(event, CastResolved):
        print(dict(event.assignments))
    elif isinstance(event, StageProgress):
        print(event.stage, event.completed, event.total, event.chapter_id)


if __name__ == "__main__":
    result = selection.write_m4b(
        "novel.m4b",
        on_event=progress,
        cancel=token,
        workers="auto",
        overwrite=False,
        keep_audio_cache=False,
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
emit `Warning`. Synthesis progress is emitted in plan order even when workers
finish out of order; quote attribution reports chapters as they finish.
Event callbacks execute on the calling thread; if one raises before commit,
execution fails as `callback_failed`, workers/workspace are cleaned, and nothing
is published. This includes every pre-commit event, including publication
`StageStarted` and `StageProgress`, each of which is followed by a cancellation
check. The coordinator then creates and validates a final private snapshot,
checks cancellation once more, and runs no callback between that check and the
atomic commit. Only after a successful commit are publication `StageCompleted`
and terminal `Completed` emitted. Both are best-effort; either callback may fail
without revoking the published output. A publish failure emits neither.

For a multi-voice pipeline, `CastResolved` arrives after attribution and casting
but before any renderer worker starts. Its `assignments` contain
`(character_id, voice_id)` pairs, so a callback can log or inspect the cast and
cancel before synthesis begins. Single-voice runs emit the same event with no
assignments.

Model work also reports progress, before rendering starts: `characters` covers
roster discovery and `attribution` covers quote assignment. Each stage emits
`StageStarted`, an initial `StageProgress` at zero, progress as chapters finish,
and `StageCompleted`. The spaCy roster reports start and completion for the whole
selection. Reused attribution reports completion without repeating model calls.
`write()` includes these stages in the same sequence as planning and rendering.

Use the same callback for a standalone checkpoint:

```python
checkpoint = selection.resolve(on_event=progress, cancel=token)
review = selection.resolve(until="characters", on_event=progress, cancel=token)
```

Each successful `resolve()` emits `Started` and `Completed`. Reusing an unchanged
checkpoint emits only those two events. An observer exception fails resolution
as `callback_failed`; completed cache or series writes are not rolled back.

## Cancellation and workers

`CancellationToken.cancel()` is thread-safe and idempotent. Pass the token to
`resolve(cancel=token)` or `write(..., cancel=token)` and cancel it from another
thread or an event callback. Resolution checks cancellation between model calls
and before committing series changes. It cancels queued work and prevents further
provider retries, including during retry backoff; already running provider calls
must return before cancellation finishes.
Completed attribution and cast cache entries may remain available for a retry.
Cancellation is
cooperative at bounded orchestration points, terminates/then kills children with
bounded waits when necessary, raises `CancelledError` (`cancelled`), and never
publishes an artifact.

`workers` must be `"auto"` or a positive, non-boolean integer. Auto considers
available CPUs, reserving two when possible, and chapter count. Both automatic
and explicit requests are bounded by chapter count and the hard cap of sixteen.
The caller must be allowed to create child processes; Python multiprocessing
daemon processes cannot invoke this renderer. Rendering uses
only Python **spawned** worker processes, including where `fork` is available.
Each worker constructs one engine and processes its bounded static segment batch
serially; workers do not share model state, and the parent never constructs an
engine.

## Provisioning voices

Rendering reads a manifest; it never downloads. Provisioning is explicit, and
`load_voice` is the only thing that makes a voice renderable:

```python
import kenkui as kk

if __name__ == "__main__":
    kk.load_voice("eponine")
    result = kk.book("book.epub").assign_voice("eponine").tts().write("book.m4b")
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

A full cast adds character inference and dialogue attribution. Each names its
model independently. Discovery can also use the optional local
[spaCy pipeline](installation.md#optional-offline-character-discovery-with-spacy);
quote attribution uses a configured LiteLLM provider:

```python
import os

if __name__ == "__main__":
    model = os.environ["KENKUI_ANALYSIS_MODEL"]
    result = (
        kk.epub("book.epub")
        .infer_characters(model=model)
        .attribute_quotes(model=model)
        .assign_voices(narrator="eponine", method="gendered")
        .tts()
        .write("book.m4b")
    )
```

### Review characters before attribution

`resolve(until="characters")` runs character discovery and returns a pipeline
checkpoint. It needs an `infer_characters()` operation, but no narrator voice,
quote-attribution operation, or synthesis operation. `inspect().roster` exposes
an immutable `CharacterRoster`; `inspect().casting` remains `None`.

Use ordinary Python to edit the roster, then pass it to `with_characters()`:

```python
import os
from dataclasses import replace

import kenkui as kk

if __name__ == "__main__":
    model = os.environ["KENKUI_ANALYSIS_MODEL"]
    discovered = (
        kk.epub("book.epub").infer_characters(model).resolve(until="characters")
    )
    roster = discovered.inspect().roster
    assert roster is not None
    print(roster.characters)

    corrected = replace(
        roster,
        characters=tuple(
            replace(character, display_name="Alice Example", aliases=("Alice", "Al"))
            if character.id == "alice"
            else character
            for character in roster.characters
        ),
    )
    reviewed = discovered.with_characters(corrected)
    cast = reviewed.attribute_quotes(model).assign_voices(narrator="eponine").resolve()
    print(cast.inspect().casting)
    cast.tts().write("book.m4b")
```

`with_characters()` performs no I/O and leaves the discovered checkpoint
unchanged. The replacement roster can rename, add, or remove characters and
change aliases, gender, or the first-person narrator's character ID
(`narrator_id`, independently of the narrator voice). IDs must be unique
lowercase slugs; reserved attribution markers and pronouns cannot be ordinary
character IDs. A narrator ID must refer to a roster member, and chapter IDs
must come from the inspected selection. Invalid edits raise `invalid_roster`.

Speech counts are measured after attribution. Known reviewed genders take
precedence over inferred dialogue evidence; `None` leaves gender open to later
inference. An empty `chapter_ids` tuple leaves a character available throughout
the selection. An empty roster narrates all speech without attribution calls.
For a nonempty roster, attribution can still discover additional chapter-scoped
speakers, as described under [Unnamed speakers](#unnamed-speakers).

Continuing to attribution reuses the roster and does not repeat discovery.
The supplied roster enters the attribution cache key, so corrections cannot
reuse assignments derived from a different roster. Repeating character
resolution on unchanged input reuses its checkpoint. Adding attribution,
casting, series, or rendering intent preserves it; changing chapter selection
discards it.

These checkpoints live in memory. If source bytes change, continuing with the
old roster raises `source_changed`. Call `resolve(until="characters")` to
discover the new source, then review it again; earlier edits remain available
on the original checkpoint and are not silently applied to different text.

### Resolve before write

`write()` resolves voices, model attribution, and casting when it needs to.
Call `resolve()` first to create an inspectable checkpoint before rendering.
It returns another `Pipeline`, so method chaining continues normally:

```python
import os

import kenkui as kk

if __name__ == "__main__":
    model = os.environ["KENKUI_ANALYSIS_MODEL"]
    resolved = (
        kk.epub("book.epub")
        .infer_characters(model)
        .attribute_quotes(model)
        .assign_voices(narrator="eponine")
        .resolve()
    )
    inspection = resolved.inspect()
    casting = inspection.casting
    assert casting is not None
    print(dict(casting.assignments))
    print(casting.characters)
    print(casting.collisions)

    result = resolved.tts().write("book.m4b")
```

Before resolution, `inspect().casting` is `None`. Afterwards it contains a
frozen `CastingInspection`: narrator and unknown voice IDs, the character
roster, assignment pairs, attributed speaker spans, and same-chapter voice
sharing. Span offsets refer to the chapter text in that same inspection.
A single-voice pipeline has a casting inspection too, with an empty character
roster and assignments when no attribution was requested. Inspection performs
no model calls and, for a resolved pipeline, returns the saved snapshot.

For unchanged source bytes, another `resolve()` reuses the checkpoint, as does
`write()`. Adding synthesis, metadata, pronunciation, or pauses preserves it;
changing selection, character analysis, casting, or series intent invalidates
it. Configure selection, pronunciation, pauses, and casting before `tts()`;
metadata may also be applied after it.

A checkpoint records the source bytes it analyzed. If the EPUB changes,
`inspect()` still shows that checkpoint and rendering raises `SourceError`
with code `source_changed` before synthesis. Explicitly call `resolve()` again
to analyze the changed book and obtain a new checkpoint. If you stopped at
character discovery, refresh with `resolve(until="characters")` and review the
roster again first. This avoids silently
rendering text with a cast or attribution you reviewed for different text.

### Narrator and unknown voices

`narrator` speaks everything that is not attributed dialogue. `unknown` speaks
dialogue nobody could be placed for, and defaults to the narrator's voice, so
an unplaced line sounds like narration rather than like a third character. Set
it separately to make the distinction audible:

```python
.assign_voices(narrator="eponine", unknown="paul")
```

Character casting prefers voices distinct from both configured voices. If none
remain, it shares the available voices with narration. With one loaded voice,
the entire book is spoken by that voice, including all attributed dialogue.

### Unnamed speakers

`attribute_quotes()` can also place a speaker the text identifies without
naming, such as a guard, innkeeper, or first man. This happens automatically:
use the normal inference, attribution, and casting pipeline above; do not add
or pin a role identifier yourself. Each identified role is scoped to its
chapter. Casting tries to give speakers in one scene different voices, but
shares them when the available pool requires it. The same role word in another
chapter is treated as another speaker. When the text
does not identify the speaker, the dialogue remains `unknown` and uses the
fallback described above.

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

A character whose gender was never inferred uses the whole character pool.
For a known gender, sourced matching voices are preferred. If none match,
casting falls back to the available pool; a missing trait is not treated as
evidence of a match.

### Pinning a choice

```python
.assign_voices(narrator="eponine", cast={"javert": "charles"})
```

Pinned entries are constraints on the solver, not suggestions.

### How voices are chosen

Casting avoids sharing voices within a chapter when the pool permits it. Among the voices
still free, the solver takes the least-used one, weighted by how much each
character actually speaks — so a lead does not land on the voice a walk-on
already holds. Voices spread before they repeat, and repeat only once the pool
is under pressure.

Voice sharing is expected when the cast exceeds the available pool: a book
with 120 characters and six voices still gets a complete cast. The solver
balances reuse and records same-chapter sharing as `cast_collision` log entries
without failing the conversion. Additional voices can improve distinctness.

### Stored work

Attribution is expensive; casting is free. Both are stored in
`casting.sqlite3` beside the voice manifest, keyed so that exploring ten
castings of one book costs one model pass.

```python
kk.list_castings()
kk.remove_casting(casting_id)  # free to rebuild
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

## Series

A character who appears in several volumes should sound like one person in
all of them. Declare which series a book belongs to and Kenkui keeps their
voice:

```python
kk.epub("oathbringer.epub").series("stormlight", book=3)
```

Membership is declared, never derived. EPUBs do not carry series metadata in
practice, so the name is yours to choose. `book` is recorded with the render's
intent and nothing reads it: `list_series()` orders by series name, and
continuity never consults it. Pass it if it documents your own call sites;
it changes nothing. Continuity is decided by who the characters are:
identity resolution matches this volume's roster against every name the
series has seen, the same way it matches names within one book.

A character the series already cast keeps their voice. A newcomer is cast
from the least-used voices of their gender, counting what earlier volumes
already spent, so voices keep spreading across the series rather than
restarting each book. `narrator` and `unknown` behave as they do without a
series; only cast characters accumulate.

### Fixing a wrong assignment

Pass `cast={...}` the same way you would without a series. An explicit
assignment for this render wins over the series' stored pin, and the series
then adopts it: this book and every one rendered after it use the voice you
just named. This is the supported way to correct one character's voice
without discarding the whole series.

```python
kk.epub("book.epub").series("stormlight", book=4).assign_voices(
    narrator="eponine", cast={"kaladin": "charles"}
)
```

### What refuses a render

Two ways a series can be contradicted are checked in `validate()`, before any
model call:

| code | meaning | override |
|---|---|---|
| `series_voice_missing` | a voice this series already cast is not currently loaded | `allow_recast=True` |
| `series_narrator_changed` | this render's narrator differs from the one the series recorded | `allow_narrator_change=True` |

Every `write()` and `write_m4b()` calls `validate()`, including writes from an
already resolved checkpoint. Resolving first does **not** bypass these checks.
Set the corresponding flag on `.series(...)` to authorize the change before
resolving and rendering:

- `allow_recast=True` re-solves the affected character and **persists** the
  replacement voice, so the series converges on it from this volume on.
- `allow_narrator_change=True` **persists** the new narrator as the series'
  narrator from this volume on.

Standalone `resolve()` does not run full render validation: it can produce an
inspectable cast without `.tts()`. If a continuity conflict is not authorized,
resolution preserves the affected stored voice or narrator, and a subsequent
write still refuses it. Use `validate()` to inspect those issues cheaply.

Every case where this volume's cast disagrees with what the series has
stored — a dropped pin, an explicit `cast=` override, or a changed narrator —
logs a `WARNING` with `boundary="series"`, whether or not a flag was passed.
An authorized override still says what it did.

```python
kk.epub("book.epub").series(
    "stormlight", book=5, allow_recast=True, allow_narrator_change=True
)
```

### Listing and removal

```python
kk.list_series()
kk.remove_series(series_id)
```

`list_series()` returns every stored `SeriesRecord`, each holding its
characters in prominence order — most spoken first. `remove_series(series_id)`
drops a series' pins entirely; the next volume declaring that series id is
cast fresh, with no memory of the ones before it.

## Shaping speech

Nothing below is applied unless you ask for it. A pipeline that calls neither
`pronounce()` nor `pauses()` renders exactly as it did before these existed,
down to the segment identities, so no cached audio is invalidated by upgrading.

```python
from pathlib import Path

import kenkui as kk

pipeline = (
    kk.epub("book.epub")
    .pronounce({"Cthulhu": "kuh-THOO-loo"}, numbers="standard")
    .pauses(chapter_ms=1500, heading_after_ms=600, paragraph_ms=250)
    .assign_voice("alba")
    .tts()
    .metadata(cover=Path("cover.jpg"))
)
```

### Pronunciation and numbers

`pronounce()` changes what the engine says, never what you are billed for.
`ExecutionStats.normalized_speech_characters` keeps counting the source text
while `synthesized_characters` follows the expansion.

Normalization is not part of this and is not optional. NFC, line endings,
Unicode spaces, and whitespace runs are settled when the source is parsed, and
character counts refer to that normalized string. `pronounce()` is the only
stage that rewrites text for speech, and it runs last, per segment.

A small built-in lexicon applies by default once you call `pronounce()`; pass
`builtin=False` to disable it. Your own entries always win over it, match whole
words case-insensitively, and take the source's capitalization shape, so one
entry covers `cello`, `Cello`, and `CELLO`.

A lexicon is also tuning: `where=` scopes entries to part of the book the
same way [`attribute()` and `silence()`](#tuning-a-book-and-the-dial-in-loop)
do. A whole-book table stays in effect everywhere; a scoped call overrides
only the keys it names, inside its own region, which is what lets one entry
read *lead* as "leed" in one chapter and "led" in another:

```python
pipeline = (
    kk.epub("book.epub")
    .pronounce({"lead": "leed"})  # whole book, unless overridden below
    .pronounce({"lead": "led"}, where={"chapter": "xhtml/ch12", "paragraph": 3})
)
```

Repeated calls to `pronounce()` compose rather than replace: `numbers`,
`builtin`, and any feature keyword each keep the value a previous call gave
them until a later call names that same setting again, so adding a
correction never silently resets a house style already chosen.

Keep a larger table in a file rather than a literal, and read it with
`read_lexicon()`. It accepts a plain object of word to replacement, or the
shape the shipped table uses, and validates exactly as an inline dict does.
`builtin_lexicon()` returns a copy of what Kenkui ships, so you can extend it
rather than replace it.

```python
mine = kk.builtin_lexicon() | kk.read_lexicon("pronunciations.json")
pipeline = kk.epub("book.epub").pronounce(mine, builtin=False)
```

`numbers` selects how much guessing you accept. Anything a tier declines is
left verbatim for your own entries to handle.

| tier | adds |
|---|---|
| `off` | nothing |
| `conservative` (default) | grouped integers, decimals, negatives, ordinals, percent, currency, units after a number |
| `standard` | years as pairs, clock times, numeric ranges, Roman numerals after Chapter/Part/Act or a regnal name |
| `aggressive` | bare Roman numerals, `No. 5`, fractions |

A tier is a preset over individually switchable features, not a package. Pass
any of `currency`, `percent`, `ordinals`, `units`, `decimals`, `integers`,
`years`, `clock`, `roman`, `fractions`, `numbered` as a keyword to override
it: `False` declines a form the tier supplies, `True` asks for one it does
not, without accepting the rest of the tier that carries it.

```python
pipeline.pronounce(numbers="standard", roman=False)
```

Features compose rather than nest, so declining a specific form leaves a
general one free to match inside it: `currency=False` alone reads `£5` as
`£five`, because the integer rule still applies. Decline `integers` too to
leave the digits alone.

An override changes segment identity, so a book already rendered without one
re-synthesizes. Passing none leaves identities byte-identical.

`St.`, `Dr.`, and `Mrs.` are never expanded at any tier. English only: a
non-English narrator voice disables the stage rather than mangling the text.

### Pauses

Each boundary carries its own duration in milliseconds, and zero disables that
tier completely — including the extra segmenting it would otherwise cause, so
leaving `line_ms` at zero costs nothing.

Durations are free to retune: changing 250 ms to 600 ms reuses every cached
segment, because only turning a tier on or off changes where a segment ends.
`chapter_ms` never re-segments at all.

The gap between two chapters belongs to the chapter that precedes it, so
skipping forward lands on speech rather than silence, and a book never ends on
dead air. Where several reasons meet — a chapter ending immediately before a
chapter title — the longest one wins rather than all of them adding up.

### Cover art

`metadata(cover=...)` accepts `"source"` (the default), `None`, or a path to a
JPEG or PNG. A supplied image that cannot be read or is not one of those two
formats fails the render rather than quietly falling back to the book's own
art. The plan records the image's content, not its location, so moving the file
does not change the output.

## Tuning a book and the dial-in loop

A pipeline's intent lives in three tiers, each mirrored by a read-only
property:

* `identity` — the source and its selection: what makes this render *this*
  book.
* `style` — reusable taste that travels across books: number tiers, the
  built-in lexicon switch, feature overrides, pause durations.
* `tuning` — corrections anchored to one book's text: attribution, silence,
  and pronunciation rules declared with `where=`.

```python
pipeline.identity  # Source, selection
pipeline.style  # SpokenForm settings, Pauses
pipeline.tuning  # Attributions, Silences, Pronunciations rules, grouped
```

All three are cheap tuple scans over declared operations — no parse, no I/O.
`repr` renders a truncated summary; iterate a property for every declared
operation or rule.

### Addressing a position

A rule's `where=` names a path through the grid Kenkui splits every chapter
into: `chapter`, `paragraph`, `line`, `sentence`, `phrase`. Each level below
`chapter` accepts a positive index, the wildcard `"*"`, `-1` for the last
child, a list of indices, or an inclusive `"lo..hi"` string range. Omitting a
level matches every value at it; omitting `chapter` reaches every chapter.
Passing a tuple of patterns adds one rule per pattern, in declaration order.

```python
book.attribute("irulan", where={"chapter": "*", "paragraph": 1})
book.attribute(
    "jessica", where={"chapter": "xhtml/ch08", "paragraph": 3, "sentence": 2}
)
book.silence(900, where={"chapter": "xhtml/ch08", "paragraph": 3})
```

`attribute()` and `silence()` accumulate rules the same way a scoped
`pronounce()` lexicon does: a strict subset outranks the broader pattern it
sits inside; a later declaration breaks a tie between equal or incomparable
patterns. `validate()` warns, rather than fails, when two rules overlap
without one containing the other — the warning names both rule indices so
you know which line to move.

### Reading the result: `script()`

`script()` returns one row per grid unit, without resolving voices or calling
a model:

```python
for row in book.script().at({"chapter": "xhtml/ch08"}):
    print(row.path, row.character, row.provenance, row.silence_after_ms, row.text)
```

`row.provenance` is `"default"`, `"machine"`, `"rule"`, or — before
`resolve()` has run — `"unresolved"`, naming exactly which layer decided that
row; `row.rule_index` points at the winning declaration when it is a rule.
`Script` behaves like a mapping (`script[path]`, `script.at(pattern)`,
iteration over the whole book) and materializes a chapter's rows only on
first access, so checking a pattern against a long book does not build every
unit in it.

### Previewing a correction

`select(*patterns)` generalizes chapter selection to a union of grid
subtrees at any level, so a probe can be a single paragraph. `preview(path)`
renders that selection to a `.wav` file, skipping the metadata and
chaptering a full `write()` would pay for:

```python
book.select({"chapter": "xhtml/ch08", "paragraph": 3}).preview("probe.wav")
```

### Saving corrections: the sidecar

`write_annotations(path=None)` saves every attribution, silence, and
pronunciation rule to a JSON file beside the EPUB — an `.epub` suffix becomes
`.kenkui.json` — anchored with a content digest so a rule keeps its meaning
even if a later edit shifts the surrounding text. `annotations(path=None)`
loads that file back as the pipeline's tuning baseline; rules declared inline
afterward are appended, never replacing what was saved.

The full loop:

```python
book = kk.book("dune.epub").annotations()  # load prior corrections, if any

for row in book.script().at({"chapter": "xhtml/ch08"}):
    print(row.path, row.character, row.text[:60])

book = book.attribute(
    "jessica", where={"chapter": "xhtml/ch08", "paragraph": 3, "sentence": 2}
).silence(900, where={"chapter": "xhtml/ch08", "paragraph": 3})

book.select({"chapter": "xhtml/ch08", "paragraph": 3}).preview("probe.wav")
book.write_annotations()
book.tts().write("dune.m4b", overwrite=True)
```

A rule scoped elsewhere never moves a segment identity, so retuning one line
does not re-render a library.
