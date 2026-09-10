# Public API reference

Import these names from `kenkui`. This page covers `kenkui.__all__`; signatures and
fields below are rendered from the source at documentation build time. Underscore
modules and operation implementation classes are private, even when a type annotation
mentions them. Prefer the constructors and fluent methods to creating
`Pipeline.operations` directly. `__version__` is the installed package's version
string.

See the [usage guide](usage.md) for complete workflows, operation ordering, checkpoint
reuse, publication semantics, and examples.

## Constructors and convenience rendering

`book()` and `epub()` record a path without opening it. `magic_run()` performs
resolution and rendering, returning the same result as `Pipeline.write()`.

::: kenkui.book

::: kenkui.epub

::: kenkui.magic_run

## Pipeline and source

Intent methods return new immutable branches. `pipe()` immediately calls an ordinary
function; `validate()` reads inexpensive local state; `inspect()` parses the selected
source or returns a saved checkpoint. `resolve()` may load local models, make provider
calls, and save casting/series records. `write()` and `write_m4b()` validate and
publish an M4B using spawned synthesis processes. Call effectful workflows from a
guarded script entry point.

`attribute()`, `silence()`, and the scoped form of `pronounce()` declare per-book
tuning rules addressed with `where=`; `annotations()` and `write_annotations()` load
and save them as a JSON sidecar. `select()` and `preview()` narrow a render to part of
a book for a quick probe. `identity`, `tuning`, and `style` summarize a pipeline's
declared intent by tier. See [tuning a book and the dial-in
loop](usage.md#tuning-a-book-and-the-dial-in-loop) for the full workflow.

::: kenkui.Source

::: kenkui.Pipeline

::: kenkui.MetadataIntent

## The tuning read model

`Pipeline.script()` returns a `Script`: one `ScriptRow` per grid unit, carrying the
effective speaker, its provenance, and the silence that follows it, without
resolving voices or calling a model.

::: kenkui.Script

::: kenkui.ScriptRow

## Inspection and character review

`BookInspection.roster` is populated at a character-review checkpoint;
`BookInspection.casting` is populated after full resolution. They are absent before
the corresponding work is done. Span offsets refer to the normalized chapter text in
the same inspection. Edit frozen records using `dataclasses.replace()` and pass a
reviewed roster to `with_characters()`.

::: kenkui.BookInspection

::: kenkui.BookMetadata

::: kenkui.ChapterInspection

::: kenkui.CharacterRoster

::: kenkui.CharacterProfile

::: kenkui.CastingInspection

::: kenkui.SpeakerSpan

::: kenkui.Collision

## Voices and provisioning

`list_voices()` reads catalog/local state. Registration, loading, unloading, and
removal modify local voice state; loading may download assets or compile a supplied
WAV. Rendering never provisions voices implicitly. See
[provisioning](usage.md#provisioning-voices) for state transitions and [models and
voices](models-and-voices.md) for asset requirements.

::: kenkui.Voice

::: kenkui.Engine

::: kenkui.list_voices

::: kenkui.add_voice

::: kenkui.load_voice

::: kenkui.unload_voice

::: kenkui.remove_voice

## Pronunciation lexicons

`builtin_lexicon()` returns a fresh dictionary. `read_lexicon()` reads and validates a
local JSON file. Pass either result to `Pipeline.pronounce()`; reading a lexicon does
not change a pipeline by itself.

::: kenkui.builtin_lexicon

::: kenkui.read_lexicon

## Stored casts and series

These functions use the managed casting database by default. Their optional `path`
argument selects a database file, not a voice manifest. `list_castings()` returns
immutable records with `cast_id`, `attribution_id`, `method`, `narrator_voice_id`,
`unknown_voice_id`, and `assignments` fields. Each assignment is a `(character_id,
voice_id, pinned)` tuple. The record's implementation class is private; use its fields
and the public removal verbs. Removing attribution cascades to its stored casts and
requires new attribution work later. Removing a series forgets its continuity pins.

::: kenkui.list_castings

::: kenkui.remove_casting

::: kenkui.remove_attribution

::: kenkui.list_series

::: kenkui.remove_series

::: kenkui.SeriesRecord

::: kenkui.SeriesCharacter

## Validation and execution results

`validate()` reports issues as values; execution methods raise the relevant exception
if validation fails. A successful `Result.output` is the published path. Its
statistics describe semantic work even when synthesis uses cached audio; they do not
measure provider charges.

::: kenkui.ValidationIssue

::: kenkui.ValidationResult

::: kenkui.Result

::: kenkui.ExecutionStats

## Events and cancellation

Callbacks receive immutable `ExecutionEvent` values. Sequence numbers describe
delivery order. See [events](usage.md#write-events-result-and-overwrite) for the
pre-publication callback failure boundary and best-effort terminal events.
Cancellation is cooperative; a running model call cannot be forcibly recalled.

::: kenkui.ExecutionEvent

::: kenkui.Started

::: kenkui.StageStarted

::: kenkui.StageProgress

::: kenkui.StageCompleted

::: kenkui.CastResolved

::: kenkui.Warning

::: kenkui.Completed

::: kenkui.CancellationToken

## Errors

Catch `KenkuiError` or a narrower subclass and branch on `error.code`, an `ErrorCode`
member. Human-readable messages are not a stable matching interface. See
[troubleshooting](troubleshooting.md) for recovery actions.

::: kenkui.ErrorCode

::: kenkui.KenkuiError

::: kenkui.SourceError

::: kenkui.ValidationError

::: kenkui.VoiceError

::: kenkui.ModelError

::: kenkui.RenderError

::: kenkui.EncodingError

::: kenkui.CancelledError

