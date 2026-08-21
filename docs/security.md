# Security boundaries

Kenkui treats EPUBs, worker results, cache contents, model files, voice prompts,
and native-tool output as untrusted. Stable public errors intentionally omit
local paths, subprocess output, and provider internals.

## EPUB limits

EPUBs are read directly as ZIP files and are never extracted. Before reading any
member, Kenkui rejects traversal/absolute/ambiguous paths, duplicate canonical
names, encrypted entries, malformed sizes, and unsafe compression ratios.
Default hard limits are:

| Resource | Limit |
| --- | ---: |
| ZIP members | 10,000 |
| Uncompressed bytes per member | 64 MiB |
| Total uncompressed bytes | 512 MiB |
| Compression ratio per nonempty member | 100:1 |
| Spine chapters | 10,000 |
| Total normalized speech characters | 50,000,000 |
| XML element depth | 256 |
| XML elements | 100,000 |
| XML attributes total / per element | 100,000 / 1,000 |

XML uses `defusedxml`, forbids DTDs, applies work limits while iterating, and
maps malformed/unsupported input to stable errors. Script, style, noscript,
template, hidden, and `aria-hidden` content is excluded from speech. XHTML text
is normalized before exact character counting. A spine item with no visible text
is skipped rather than failing the book, because image-only covers and title
pages are ordinary; `empty_chapter` is raised only when no spine item in the book
yields text. `archive_limit`, `unsafe_archive_path`, `malformed_epub`, and
`empty_chapter` are expected safety outcomes rather than requests to relax limits
for unknown files.

These controls reduce resource and path risks; they are not a general malware
scanner or a guarantee that arbitrary hostile input is harmless. Run untrusted
workloads with ordinary OS least privilege and independent resource isolation.

## Local Pocket assets

Pocket activation accepts no default model, online model identifier, URL, or
environment-discovered voice. It requires a complete local manifest of every
model relative path, exact byte size, and SHA-256; the manifest must exactly
match the model tree. The selected config must be a manifest member and may not
contain remote markers or unsafe path forms. Individual model files are limited
to 2 GiB, the model tree to 4 GiB, and config YAML to 1 MiB.

The voice must be a separate canonical local WAV (maximum 64 MiB) with an exact
SHA-256 and complete provenance, license identifier, rights statement,
commercial-use decision, and model-revision compatibility. Roots/files must be
owned by the current user, not group/world writable, regular/direct, and
single-link where required. Symlinks and tree mismatches fail closed.

Preflight verifies declarations and bytes without importing inference packages.
A worker re-verifies, copies an immutable private snapshot, enables offline
provider settings, denies socket audit events, replaces the provider downloader
with a local manifest allow-list, and passes a `Path` voice prompt to avoid URL
handling. These are defense-in-depth controls, not a claim that third-party code
is a formal sandbox.

## Spawned worker results and publication

A bounded set of reusable spawned workers each constructs one engine and handles
a bounded static segment batch serially. For each segment, the worker writes
bounded versioned metadata and PCM to its private result path, atomically renames
it, and waits for parent deletion as acknowledgement before continuing. Audio
never crosses a process pipe. The coordinator validates path containment,
identity, regular-file type, no-follow state, link count, size, JSON primitives,
task identity, and audio invariants before bounded reads, even while the worker
remains alive. Final success is withheld until worker exit and engine cleanup.
Worker startup/failure, timeout, malformed results, cancellation, and
native-tool failure all clean up without unbounded joins or publication.
