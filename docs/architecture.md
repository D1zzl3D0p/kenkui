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
resolved voice metadata, model revision, and ordered intent. Under frozen
`tts-chunks-v2`, it splits every selected chapter into non-empty segments of at
most 1000 characters, ranking break points by how natural the resulting pause
sounds -- line break, then whitespace beside punctuation, then any whitespace,
then the dash family -- and taking the best one that still fills most of the
window, with a hard boundary for long tokens. It additionally caps runs holding
none of `.!?,;:`, which the engine cannot subdivide, so no run reaches synthesis
far enough over the engine's own chunk budget to generate past its limit. Concatenating a chapter's segments exactly
reconstructs its normalized text. It emits schema versions, content hashes, stable segment
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
semantic chapter units. As each chapter completes, its ordered segment PCM is
spilled to a private part file in the run workspace and leaves memory, so a run
holds at most one chapter of samples rather than a whole book. Only metadata --
identities, frame counts, durations -- travels on to assembly, which concatenates
the parts in one linear pass and aggregates exact frames into one M4B marker per
selected chapter. Parts are untrusted like any other worker output: each must be
a regular file whose size matches its chapter's metadata exactly.

`workers="auto"` reserves two CPUs for the rest of the system and is bounded by
chapter count and a hard cap of sixteen; the ceiling is memory, because each
worker copies a private model snapshot and holds its own model instance. The
scheduler bounds combined live and completed-but-not-emitted work, validates
per-segment, per-chapter, and whole-run PCM budgets, accepts completion out of
order, and emits results/events strictly in plan order.

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

Caching is offline and **fail-open for rendering**: missing/corrupt/truncated or
hash-mismatched payloads, malformed metadata, unknown schema, lock timeout,
unwritable storage, or SQLite failure become misses. Such failures cannot turn an
otherwise correct uncached render into failure. This does not mean unsafe data is
accepted; it means the cache is bypassed.

Finite quotas are 4,096 segment rows, 512 MiB of distinct payload bytes, and
1,024 run rows. Each maintenance call evicts/scans bounded batches (32 database
rows and at most 64 orphan names). Payload reclamation takes the digest lock and
rechecks references under `BEGIN IMMEDIATE` immediately before unlinking.
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

Character inference and dialogue attribution call a language model, which the
pure planner must not do. They are resolved in the shell and handed to the
planner as finished values, exactly as voice metadata already is: the planner
receives a roster, speaker spans, and a cast, and reaches neither a model nor
a store.

`resolve()` and `write()` are two entry points into one resolution
implementation, not two paths. A pipeline carrying resolved values renders
identically to one that resolves during `write()`; appending any operation
discards them, because changing intent invalidates them and re-resolving
against a populated store is a lookup.

All model traffic happens in the parent process, beside the network voice
provisioning already performs. The render path is unchanged: spawned workers
install a socket-denying audit hook and set the offline environment variables,
so no credential and no request can reach them.

## Span-then-chunk segmentation

Attribution produces speaker spans that partition a chapter. The frozen
`tts-chunks-v2` chunker then runs inside each span, so concatenating every
chunk still reproduces the chapter exactly.

A chapter with no attributed dialogue is one span, which is byte-for-byte what
the chunker saw before attribution existed. Speaker and voice enter a segment's
identity only for attributed speech, so single-voice segment identities and
plan fingerprints are unchanged and no previously cached segment is
invalidated.

## Canonical text and the spoken form

Normalized chapter text stays the single authority for billing, inspection,
chapter identity, and attribution offsets. When a caller asks for it, a
separate versioned `spoken-form-v1` stage derives the string the engine
actually speaks, and nothing else consumes that string. This is what lets
`normalized_speech_characters` keep describing the book the caller supplied
while `synthesized_characters` follows the expansion.

Segment compilation runs split, then speak, then chunk. Splitting first, in
canonical coordinates, means the spoken stage cannot move a boundary that has
already been decided, which removes the need to map offsets through a
length-changing transformation. Exactness therefore holds at three levels:
chunks join to the spoken form of their piece, pieces join to their span, and
spans join to the canonical chapter text.

`tts-chunks-v3` is `tts-chunks-v2` restricted rather than replaced: the
structural split feeds the unmodified v2 chunker one piece at a time, so
concatenation-exactness is inherited and the tuned break constants are not
forked. Every field the new stages contribute — the spoken-form schema, the
number tier, the lexicon identity, the structure schema, the break tiers —
enters a segment's identity only when that feature is active, so a pipeline
requesting neither pronunciation nor pauses produces byte-identical identities
and invalidates no cached segment. The same discipline governs the plan
fingerprint: a key is absent, never null, when its feature is not in play.

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
