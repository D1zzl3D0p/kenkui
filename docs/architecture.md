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
`tts-chunks-v1`, it splits every selected chapter into non-empty segments of at
most 1000 characters, preferring whitespace/sentence opportunities and using a
hard boundary for long tokens. Concatenating a chapter's segments exactly
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
semantic chapter units. Assembly concatenates plan-ordered segment PCM and
aggregates exact frames into one M4B marker per selected chapter.

`workers="auto"` considers CPUs and chapter count; all requests are bounded by
chapter count and a hard cap of two. The scheduler bounds combined live and
completed-but-not-emitted work, validates per-segment and cumulative PCM budgets,
accepts completion out of order, and emits results/events strictly in plan order.

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
