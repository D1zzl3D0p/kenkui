# Architecture

Kenkui uses a `src` layout and a functional-core/imperative-shell design.
Public source/pipeline/result/event values are immutable. EPUB parsing,
normalization, selection, and semantic planning are kept separate from
filesystem, process, provider, cache, FFmpeg, callback, and publication effects.
Internal plans and binding factories are deliberately not public APIs.

## Deterministic source and planning core

EPUB spine order is authoritative. Visible text extraction excludes active and
hidden content, inserts semantic block boundaries, then applies versioned
`nfc-space-newline-v1` normalization: CR/CRLF become LF, designated Unicode
spaces become ordinary spaces, Unicode is NFC-normalized, horizontal runs are
collapsed per line, surrounding space is removed, and runs above two newlines
are reduced to two. Case and punctuation are preserved. Character counts refer
to this exact normalized string.

The pure planner consumes an immutable inspection, exact source-bytes SHA-256,
resolved voice metadata, model revision, and ordered intent. It builds one flat
canonical grid from paragraph, line, sentence, phrase, quote, and emphasis
boundaries, then derives immutable structural ranges and gap reasons from those
leaves. Planning, attribution, selection, and tuning consume that shared grid;
they do not rescan quotes, blocks, or lines.

The `grid-v1` packer transforms spoken-form regions before measuring them and
packs the largest fitting paragraph, line, sentence, then phrase ranges without
crossing speaker, scoped-pronunciation, explicit-silence, or enabled-gap cuts.
Every synthesis segment contains at most 1,000 spoken characters. Only one
over-budget phrase enters the emergency splitter, which prefers punctuation or
hyphens, then whitespace, and finally a hard token cut. Emergency provenance is
retained for tests and quality metrics. Whitespace-only ranges are never sent to
the engine; bounded redistribution preserves adjacent speech, silence, and
canonical coverage wherever synthesizable output can represent them.
The planner emits schema versions, content hashes, stable segment
identities, resolved metadata, and a canonical semantic fingerprint. Canonical
JSON key ordering and bounded UTF-8 hashing make equivalent semantic inputs
produce the same plan regardless of output path, callback, worker count, cache
location, or run ID. The source file and resolved voice/model inputs—not any
cached row—remain the authority for every run.

## Coordinator and spawned isolation

The coordinator is the only owner of planning, ordering, event callbacks,
assembly, and atomic publication. It does not construct or call a synthesis
engine. Rendering uses a bounded pool of children created from Python's `spawn`
multiprocessing context, even where `fork` exists. Each worker receives one
frozen pickle-safe engine specification and a bounded static task batch,
constructs one engine, and reuses it serially for that batch. The parent never
constructs an engine.

Segments are worker/cache units, but public progress and output metadata remain
semantic chapter units. Each validated segment's PCM is appended to its chapter's
private part file in the run workspace as it arrives, so a run holds one segment
of samples rather than a chapter or a book, and a chapter costs the same memory
however long it runs. A part is fsynced and offered to assembly and checkpointing
only once its chapter is complete; an abandoned chapter's part leaves with its
failed run. Nothing caps a chapter's length: the budgets in `kenkui.limits` bound
one worker's result and one run's total, which is a bound on untrusted output
rather than on content. A chapter whose character count estimates past
`LONG_CHAPTER_HOURS` is reported as a `Warning` during planning and rendered
regardless, because only the person who chose the chapters can say whether
fourteen hours of endnotes is a mistake. Only metadata --
identities, frame counts, durations -- travels on to assembly, which concatenates
the parts in one linear pass and aggregates exact frames into one M4B marker per
selected chapter. Parts are untrusted like any other worker output: each must be
a regular file whose size matches its chapter's metadata exactly.

`workers="auto"` reserves two CPUs for the rest of the system and is bounded by
chapter count and a hard cap of sixteen; the ceiling is memory, because each
worker copies a private model snapshot and holds its own model instance. The
scheduler bounds combined live and completed-but-not-emitted work, validates
per-segment and whole-run PCM budgets, accepts completion out of order, and emits
results/events strictly in plan order.

Worker audio never crosses a multiprocessing pipe. A worker writes one
versioned, bounded header and raw PCM to a private result path using sibling-temp
plus atomic rename, then waits for the parent to remove that file as an
acknowledgement before advancing. The parent validates containment, type,
no-follow/link/identity state, size, primitive-only metadata, task identity, and
audio invariants before bounded reads and immediate removal, while the reusable
worker may remain alive. Final results are withheld until workers have exited and
cleaned their engines. Startup, timeout, malformed result, provider, callback,
and cancellation paths use bounded terminate grace, kill escalation, and final
join; no unbounded join is allowed.

## Private cache semantics and ownership

Segment PCM caching is a private execution optimization, never a public pipeline
parameter or source of semantic truth. Only internal `ExecutionBindings` can own
a cache directory. Public callers cannot choose its location, issue SQL, select
a schema, or trust it instead of parsing/planning current input. No cache object
or SQLite connection is retained or sent to a child.

The owner-only root contains `cache.sqlite3` and content-addressed PCM sidecars
under `payloads/`. Root, payload directory, database, locks, temporary files, and
sidecars are checked without following links. Unsafe owner, type, mode, identity,
or hardlink state disables/bypasses caching without touching an external target.
Payload operations are descriptor-relative and private modes are enforced only
through validated descriptors.

Keys contain synthesis semantics only: cache/PCM contract versions, segment
identity and text digest, voice content and rights identities, model/engine PCM
configuration, sample rate, and channels. Worker count, output/cache paths,
callbacks, and run IDs are excluded. Unknown schemas are never treated as
compatible. Hits are fully hashed/validated before use and merged with misses in
plan order. Payloads are bounded, hashed while written, fsynced, atomically
published, then transactionally referenced.

`grid-v1` deliberately replaced both `tts-chunks-v4` and `tts-chunks-v5` in
segment identity, so legacy PCM entries are clean misses. The migration neither
deletes nor rewrites those rows or payloads. Cache pruning remains an explicit
operator/publication choice; check free space before a large library rerender.

Caching is offline and **fail-open for rendering**: missing/corrupt/truncated or
hash-mismatched payloads, malformed metadata, unknown schema, lock timeout,
unwritable storage, or SQLite failure become misses. Such failures cannot turn an
otherwise correct uncached render into failure. This does not mean unsafe data is
accepted; it means the cache is bypassed.

Retention is per-book and explicit. A successful publication clears the
published book's rows, runs, book, and unreferenced payloads, so a finished
book stops holding its rendered audio on disk; ``write(..., keep_audio_cache=
True)`` opts out. Run rows keep a finite bound of 1,024. Each maintenance call
scans at most 64 orphan names. Payload reclamation takes the digest lock and
rechecks references under ``BEGIN IMMEDIATE`` immediately before unlinking.
Cleanup removes only validated old regular temporaries and unreferenced regular,
single-link sidecars; poisoned/unsafe nodes are left untouched. Book/voice/run
metadata is pruned in bounded batches.

## FFmpeg shell and publication

The native shell discovers and capability-checks `ffmpeg`/`ffprobe`, invokes them
with argv (not a shell), bounds diagnostics, builds AAC MP4/M4B metadata/chapters
and optional source cover, probes semantic output, and performs a full decode.
Only a validated candidate is atomically published. Public errors are stable and
sanitized; local paths and subprocess/provider diagnostics stay private.

Every callback exception before publication commit becomes `callback_failed`,
terminates active work through coordinator unwinding, removes the workspace, and
publishes nothing. Publication `StageStarted` and `StageProgress` are fatal
pre-commit callbacks followed by cancellation checks. The final private snapshot
is then validated, cancellation is checked, and no callback runs between that
point and atomic commit. A successful commit is followed by best-effort
publication `StageCompleted` and terminal `Completed`; observer errors are only
generic-log records and cannot revoke durable success. Publish failure emits no
publication completion. Logs contain only generic stage/cache categories and
terminal stable codes: cache keys/ordinals, source/output paths, text,
voice/model paths, and provider diagnostics are excluded.

## Attribution as a resolved input

Character and series records live in `_characters.models`, independently of
SQLite. `_characters.series` handles identity matching and merging;
`_characters.continuity` derives series pins, prior voice usage, and diagnostics
from supplied records. These functions perform no I/O and return frozen values.
The imperative functions in `_resolution` own reading records, resolving
resources, reporting decisions, and persisting the resulting series through
`_characters.store`. The fluent `Pipeline` delegates to that implementation;
its methods record intent and provide convenient entry points into the work.

Character discovery is separately materializable through
`resolve(until="characters")`. It retains a `CharacterRoster` and the exact
source snapshot without binding voices or attributing quotes. Pure validation
in `_characters.review` canonicalizes replacements supplied through
`with_characters()`. Adding attribution or casting preserves that roster;
changing the source selection invalidates it.

Attribution accepts a supplied roster and skips discovery. Its cache identity
includes the supplied roster and review status, keeping edited input separate
from automatically derived records. Reviewed known genders survive later
dialogue inference; unspecified genders remain inferable. Roster aliases are
included in prompts for supplied rosters. The automatic path retains its
existing prompts and cache identity.

Character inference and dialogue attribution call a language model, which the
pure planner must not do. They are resolved in the shell and handed to the
planner as finished values, exactly as voice metadata already is: the planner
receives a roster, speaker spans, and a cast, and reaches neither a model nor
a store.

`resolve()` and `write()` share one resolution implementation. Resolution
hashes and parses the same private source snapshot using `_source`, which also
supplies rendering's bounded snapshot function. The returned pipeline retains
frozen source and casting inspection values alongside private rendering
bindings. `inspect()` exposes the values; bindings stay internal.

Synthesis, metadata, pronunciation, and pauses preserve a resolved checkpoint
because attribution uses canonical text. Other intent changes discard it.
Before rendering, the coordinator compares its private source snapshot with
the resolved source hash and rejects mismatches as `source_changed`. An
explicit `resolve()` refreshes a cast checkpoint whose source bytes have changed.
When a roster checkpoint is present, refresh discovery with
`resolve(until="characters")` and review the new input first.
This keeps both the reviewable checkpoint and the render tied to exact input.

All model traffic happens in the parent process, beside the network voice
provisioning already performs. The render path is unchanged: spawned workers
install a socket-denying audit hook and set the offline environment variables,
so no credential and no request can reach them.

A book narrated in the first person names its narrator during roster
inference, and attribution marks them in the roster it sends. Their id is an
ordinary character id, so their spoken lines and their narration differ only in
which voice casting gives them. Character casting prefers voices distinct from
narration, but shares the available voices when no distinct voice exists. A
single loaded voice can therefore narrate every part of the book.

Attribution also handles a speaker the text identifies by role but never names,
such as a guard, innkeeper, or first man. The model returns a short role word;
resolution scopes that identity to the chapter and turns it into an ordinary
character profile for casting. Two role speakers in one scene therefore receive
different voices, while a guard in a later chapter may reuse one. A role that
cannot be identified remains unattributed rather than becoming a guessed
character.

Attribution prompts preserve explicit gender in unnamed role identifiers and
ask for distinct identifiers when several people share a role. The pure
`role_gender` parser accepts compact qualified roles without inferring gender
from occupations. Post-attribution dialogue tags fill unknown genders from
unopposed evidence; overriding a known inference still requires three votes
and a two-to-one margin. Ambiguous evidence is logged for operators. Prompt
changes advance `PROMPT_VERSION`, preventing cached merged identities from
surviving a new resolution. No public schema or render-worker behavior changes.

The model also returns a `speaker_genders` map alongside the quote assignments.
Only supported values for attributed speakers survive decoding. The
`gender_evidence` module merges this evidence conservatively across aliases and
chapters, retaining only unanimous values, then applies it before dialogue-tag
checks and reviewed overrides. `AttributionRecord.gender_evidence` persists in
an additively migrated SQLite JSON column. A later spaCy roster refresh reapplies
this evidence rather than replacing a contextual correction with proximity
votes. Old records migrate with empty evidence; the updated prompt version
causes fresh resolution to obtain the new model response.

Roster identity is deliberately conservative. Clear aliases fold into one
character, but ambiguous short names and conflicting honorifics do not. A
duplicate voice is locally audible; assigning two distinct people one voice is
a more damaging error.

## Grid-folded segmentation

Attribution produces speaker spans in canonical grid coordinates. Effective
speaker changes become mandatory packer cuts; whitespace-only attribution spans
are assigned to adjacent effective speech because they cannot produce valid
engine input. Structural gaps remain pure grid reasons until pause policy turns
them into durations after packing.

A chapter with no attributed dialogue is one narration span. Speaker and voice
enter segment identity for attributed speech, while every segment contains the
single `grid-v1` chunking input. Pause tier names and structure schema do not
enter segment identity; silence never reaches synthesis, and a duration-only
retune reuses unchanged PCM.

## Canonical text and the spoken form

Normalized chapter text stays the single authority for billing, inspection,
chapter identity, and attribution offsets. When a caller asks for it, a
separate versioned `spoken-form-v1` stage derives the string the engine
actually speaks, and nothing else consumes that string. This is what lets
`normalized_speech_characters` keep describing the book the caller supplied
while `synthesized_characters` follows the expansion.

Segment compilation establishes canonical semantic regions, speaks each region
independently, retains exact canonical-to-spoken replacement maps, and then
packs by spoken length. The mappings keep selection and source provenance in
canonical coordinates even when a number or pronunciation expands or contracts
text. The spoken-form schema, number tier, and lexicon identity enter segment
identity whenever that stage is active. The same discipline governs the plan
fingerprint: an optional key is absent, never null, when its feature is not in
play.

## Silence as a gap between segments

Pauses are generated silence, not prosody hints. Between any two adjacent
segments there is exactly one gap, and a gap may have several reasons; its
duration is the maximum of them, never the sum, so a chapter boundary meeting a
chapter title's leading pause cannot compound. Modelling gaps rather than
per-segment durations makes that impossible by construction.

Each gap folds into the preceding segment's trailing pad, so the inter-chapter
gap is counted in the earlier chapter and skipping forward lands on speech. The
coordinator pads `SegmentAudio` only after the raw worker output has been
validated; because `byte_count` is derived from `frame_count`, that single
adjustment keeps the assembler's part-size check, the chapter markers, and the
reported duration in agreement without changing the FFmpeg shell at all.
Silence never reaches a worker or the cache, so pause durations stay out of
segment identity and retuning them costs no re-synthesis.
