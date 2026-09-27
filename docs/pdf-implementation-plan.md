# PDF implementation plan

**Status: native cleanup and the isolated layout/OCR path are implemented.**
Furniture removal, conservative same-page footnote removal, paragraph reconstruction,
source-attested word repair, explicit numbered note-section removal and classified
visual omission are available. Milestone 5 has offline worker/routing coverage;
real-backend acceptance results are recorded separately. Full-book quality gates,
persistent caches and chapter detection remain unfinished. See
[PDF preparation](pdf-processing.md) for the supported API and limits.
This breaks the accepted [preprocessing design](pdf-preprocessing-design.md) into
ordered, reviewable changes. Each milestone includes its own tests; later
milestones cannot substitute for an earlier milestone's acceptance checks.

## Scope of the first release

Deliver automatic PDF preparation through `book(...).resolve()` and `.write()`,
explicit `.prepare()` for inspection, native-only mode, optional Docling/OCR, and
a configurable sequence of independently callable cleanup functions.

Keep current EPUB behavior and identities intact. Exclude general OCR spelling
rewriting, LLM correction, table/equation narration, automatic cover rendering,
experimental TTS chunking and full-book Whisper evaluation. The existing
experimental OCR retry is a subsequent milestone, not a release prerequisite.

## 1. Introduce the shared source boundary

**Purpose:** remove EPUB assumptions from orchestration before introducing PDFs.

- Add private `_source_adapter.py` for format dispatch and unselected source
  inspection. Keep `_source.py` responsible for bounded copying and hashing.
- Route `Pipeline.inspect()`, tuning-related `Pipeline.save()`,
  `_resolution.inspect_source()` and `_execution/coordinator.py` through the
  shared adapter. Audit other direct `inspect_epub()` calls.
- Keep selection above the adapter. Resolution must retain the full source basis
  used by planning, including `_planning_chapters` behavior.
- Preserve ownership: the shell creates and destroys workspaces; the adapter
  returns immutable values, not paths into an already deleted temporary directory.
- Keep render preflight before expensive source work. Preserve source snapshot
  hashing and the existing EPUB cover path during assembly.

**Acceptance:** existing EPUB inspections, chapter IDs, text/ranges, selections,
saved tuning, plans, fingerprints and covers remain identical. Extend
`test_resolution_inspection.py`, `test_pipeline.py`, sidecar/selection tests,
`test_write_preflight.py`, cover tests and import-boundary tests. Exercise both
direct writes and resolve-then-write, including changed sources and cancellation.

**Deliverable:** an EPUB-only refactor with no new optional dependencies or PDF
constructors. This is the first implementation change to make.

## 2. Establish PDF records, configuration and identity

**Purpose:** make the data and lifecycle contracts testable without an extractor.

Create these modules incrementally:

```text
src/kenkui/
  _source_adapter.py       shared source inspection/preparation dispatch
  pdf_processing.py       public records, built-in steps and registration surface
  _pdf/
    models.py             immutable evidence, narration, edit and issue records
    options.py            modes, recipe descriptors and validation
    identity.py           request/output identities and schema versions
    recipe.py             ordered execution, prerequisites and step registry
    inspection.py         normalized PDF narration -> BookInspection
    native.py             optional native extraction adapter
    docling.py            optional layout/OCR adapter, loaded inside worker
    worker.py             bounded extraction child and result protocol
    preparation.py        extraction, cleanup and final validation orchestration
    validation.py         evidence/output checks and quality dispositions
    steps/                independent pure cleanup functions
```

Avoid importing `Pipeline` from `_pdf` or importing effectful adapters into the
planning domain. Public `pdf_processing.py` must not import Docling at module load.

Define a private `PreparedSource` containing raw-byte hash, preparation identity,
full unselected inspection, format and PDF report. A pipeline owns an optional
prepared checkpoint. Keep `PdfDocument` separate: it carries the complete source
evidence and evolving narration projection used by transforms.

Define step descriptors with ID, algorithm version, serializable configuration,
required evidence and ordering constraints. Built-in functions are registered
descriptors' implementations, with the signature `PdfDocument -> PdfDocument`.
Validate `steps=None` versus `steps=()` explicitly. Pure transforms run in the
parent after extraction, so custom Python functions need not be pickled into
the model worker. They must return valid records and cannot mutate input evidence.

Identity contract:

| Identity | Meaning and consumers |
|---|---|
| Original-byte hash | Exact input snapshot; source-change checks and provenance |
| Preparation request key | Raw hash + mode + ordered recipe/configuration + language + backend/model/resource versions; extraction/preparation cache lookup |
| Prepared output digest | Canonical normalized chapters, ranges and structure; validates actual output |
| Prepared identity | Request key + output digest; PDF attribution, reviewed tuning, resolution and plan reuse |

Do not overload `source_bytes_hash` with a processed-text hash. Audit
`RosterCheckpoint`, `Resolved`, character attribution/store keys, sidecar
authoring snapshots, `_domain/planning.py` schema/fingerprint material and
`_execution/checkpoints.py`. PDF needs additional preparation identity; existing
EPUB fingerprint serialization must remain byte-for-byte unchanged when that
identity is absent. Use a PDF parser schema instead of labelling PDF output
`epub-visible-text-v1`.

Unregistered custom steps may execute locally with persistent preparation caching
disabled. They cannot be saved as portable recipes. An explicit prepared
checkpoint can still retain the exact result for its current pipeline instance.
Registered custom steps must supply stable version/configuration contracts.

**Acceptance:** immutable inputs, deterministic serialization, ordering and
prerequisite errors, policy/version/source invalidation, equivalent inputs giving
equivalent identities, custom-step behavior, and unchanged EPUB fingerprints.

## 3. Deliver native PDF preparation end to end

**Purpose:** establish one usable PDF path before introducing model processes.

- Add lazy `pdf()`/`book()` dispatch, source format/configuration fields,
  `.pdf_processing()`, `.prepare()` and `.pdf_report()`.
- Implement native extraction of pages, lines, characters, fonts, geometry and
  image coverage through the optional native dependency. Never rasterize in
  native mode. Validate magic/header and parser results, not just extension.
- Preserve every source page's disposition and cap source bytes, pages, native
  text/character output and serialized results. Reject unsupported encryption
  with an explicit error; leave password handling outside this release.
- Build the PDF inspection adapter using the same normalization as EPUB. Populate
  heading, emphasis and scene ranges only with supported evidence. Use one chapter
  if structural chapter boundaries cannot be established. Record PDF covers as
  unavailable while retaining explicit caller-supplied covers.
- Have `.prepare()` return a new pipeline with the frozen checkpoint. Inspection
  and script materialization on an unprepared PDF require preparation. Neither
  boundary may start model work implicitly.
- Make resolve/write invoke the same preparation service before selection and
  character inference. Never implement a different PDF path inside rendering.

During this intermediate milestone, auto mode must fail clearly as unavailable;
tests/examples use `.pdf_processing(mode="native")`. Do not change the eventual
default to native temporarily or imply that the complete default is available.

Lifecycle rules:

| Change/action | Required behavior |
|---|---|
| Change PDF mode, language or steps | Clear prepared, roster and resolved checkpoints |
| Change voice or speech style | Retain prepared source; invalidate downstream work as required |
| Change chapter selection | Retain full prepared source; apply selection to its chapter IDs |
| Change source bytes after PDF preparation | Reject stale prepared data on execution; explicit preparation can create a new checkpoint |
| Repeated inspection of a prepared source | Return the same canonical text, with no extraction or model calls |
| Cancellation/failure during preparation | Publish no prepared checkpoint or completed cache entry |

**Acceptance:** generated digital-PDF fixture reaches inspect, script and fake-TTS
write with identical canonical text. Test chapter fallback, explicit cover,
sidecar save, empty/corrupt/encrypted documents, image-only and mixed documents,
source changes, literal empty recipes and lazy dependency errors. Native mode
must make zero rasterization/OCR/layout-model calls.

## 4. Port the cleanup functions and freeze the standard recipe

**Purpose:** integrate the measured quality wins without importing spike scripts
or their dependency graphs into the library.

| Function/module | Required evidence | Difficult cases to protect |
|---|---|---|
| `remove_furniture` | Repetition + geometry/type; supported page-number offsets | Titles, chapter numbers, years, sparse pages, facing-page margins |
| `remove_notes` | Role corroborated by type/position; explicit endnote structure; marker geometry | Dialogue/verse falsely labelled notes, baseline digits, exponents, next chapter after notes |
| `omit_visual_material` | Supported block roles and caption evidence | Exercises falsely labelled captions, inline prose mentioning figures |
| `repair_reading_order` | Compatible column geometry and bounded adjacent inversion | Multi-column text, captions, overlapping fragments |
| `repair_line_break_words` | Original line ends + lexical/document support | Genuine compounds, names, unsupported languages, mid-line spaces |
| `reconstruct_paragraphs` | Indentation, font/margins, syntax and continuation constraints | New speakers, verse, headings, lists, drop caps, sparse chapter-ending pages |
| `remove_margin_marks` | Isolated symbols outside a supported prose area | Interior punctuation, section ornaments, short dialogue |

Canonical extraction preservation is mandatory and precedes this optional recipe.
Final accounting and failure checks are mandatory even when `steps=()`.

Port one step at a time with source evidence and edit logs. Start with furniture
and paragraph reconstruction for the largest narration benefit; freeze their
final execution order only after testing them together with note removal and
word repair. Preprocessing must not alter original archive records.

Tests should verify actual recovered prose and protected negatives, not merely
that the implementation logged an expected action. Require unchanged-letter
accounting for hyphen repair, explicit inventory for omissions, and logged
permutations for reordered paragraphs. Test each step and the complete recipe for
idempotence. Compare alternate recipes on the same extraction.

**Evidence sources:** the earlier ebook refinement, audio projection, prose
cleanup and iteration-two research under `spikes/`. Those scripts are reference
material, not distributable dependencies. Their EPUB references must never enter
cleanup decisions. Use generated fixtures or licensed excerpts in committed tests.

**Acceptance:** pure-step tests pass offline, standard recipe runs end to end on
native PDF fixtures, and canonical inspection/script/render input agrees.

## 5. Add optional Docling and OCR through an isolated worker

**Purpose:** complete auto mode without changing the core recipe or TTS workers.

- Resolve and lock native/layout extras across supported Python versions and the
  existing Pocket-TTS dependencies. Select compatible backend versions explicitly;
  the spike's environment is not automatically the production dependency set.
- Document executable/model asset provisioning separately from pip installation.
  Constructors, inspection and default offline tests must not download assets.
- Spawn a bounded extraction child with only source snapshot, validated options,
  resource limits and private output paths. Return versioned primitive records
  through a validated file protocol; keep model/backend objects out of IPC.
- Use Docling to classify blocks and route scanned pages to OCR. Batch large
  documents; retain source-page and block provenance and canonical text.
- Enable default auto mode only after its extraction path is usable. Never
  silently downgrade to native if dependencies/assets are missing.
- Emit extraction/layout/OCR/cleanup/validation events. Join and release the PDF
  worker before launching TTS workers. Bound cancellation, terminate/kill grace,
  retries, temporary storage and output size; do not reuse synthesis `workers`
  as the extraction concurrency setting.

Native mode cannot confidently resolve every image-dominated page. Unknown
content must produce an issue/failure rather than disappear. Layout/OCR can
classify a page as blank or intentionally omitted; extraction exceptions cannot
serve as blank-page evidence. Confirmed narrative extraction failures block speech.

**Acceptance:** fake-worker tests for success, cancellation, timeout, crash,
malformed/oversized output and missing assets. Test worker cleanup before fake TTS
startup and zero PDF dependency imports on EPUB-only paths. Add separately opted-in
real backend tests for digital, scanned and mixed PDFs using provisioned assets.

## 6. Persist preparation and expose release-ready diagnostics

**Purpose:** reuse work without losing source identity, transparency or bounds.

- Cache only validated artifacts with atomic publication, checksums, schema and
  request/output identities. Validate restored content and bounds before use.
- Define local cache ownership, cleanup/eviction and concurrent-writer behavior;
  do not leave full page images in a permanent cache by default.
- Integrate preparation with the existing optional job checkpoint facility using
  a separate versioned key. Incomplete durable checkpoints must not look complete;
  respect the current fatal handling of durable checkpoint storage failures.
- Expose a frozen report with page dispositions, edits, unresolved issues, backend
  versions, timings and retry counts. Export normalized text/HTML and optionally
  EPUB from the prepared representation without re-extraction.
- Update README, public API documentation, dependency installation instructions,
  examples, error documentation and the Unreleased changelog when the behavior
  is actually implemented. Keep design examples labelled as proposals until then.

**Acceptance:** cold/warm preparation yields identical text, stale/corrupt caches
cannot supply attribution offsets, concurrent writes cannot mix documents,
checkpoint restoration is validated, and reports remain bound to their text.

Milestones 1–6 constitute the first complete automatic-PDF feature. In-memory
prepared checkpoints allow incremental implementation before persistence exists;
do not imply cross-process reuse before this milestone.

## 7. Add targeted OCR recovery after broader validation

**Purpose:** recover exceptional bad pages without making every book more costly.

Port the reference-blind failure detector separately from the retry executor.
Treat the current English recognizability/embedded-layer comparison as one signal,
not a universal detector: it cannot evaluate scans without an embedded text layer.
Compare fresh high-resolution extraction against coverage and layout evidence.

The first experimental clipping rule removed short dialogue. Include that failure
as a regression test, alongside headers, footnotes, captions and multi-column
pages. Re-run cleanup and regenerate the whole prepared identity after replacement.
Cap attempts per page and total work. Preserve both candidates and rejection
reasons. A rejected retry must not turn a failed page into a successful omission.

**Acceptance:** broaden held-out page coverage, quantify losses/false triggers and
measure latency/RAM before enabling retry by default. No universal sub-0.1% prose
claim follows from the existing corpus.

## Verification and release gates

For each code milestone, run focused tests first, then the repository's applicable
checks: Ruff formatting/lint, strict mypy, offline pytest with its 90% coverage
gate, and strict documentation build. For dependency/public-package milestones,
also validate the frozen lock, build distributions, inspect wheel contents and
test base/native/layout installations. Exercise the supported Python-version
matrix. Native FFmpeg and real PDF-backend acceptance remain separately gated.

Whole-book research regression compares canonical narration, omission/edit logs,
paragraph behavior and runtime/resource use. Same-edition labels and new held-out
books are necessary to measure absolute prose quality; cross-edition EPUB word
disagreement and Whisper errors remain diagnostic measures.

Suggested initial review boundaries are one change per numbered milestone, with
milestone 4 split by cleanup function and milestone 5 split into worker isolation
and real-backend integration when needed. No deployment, billing change or TTS
policy change is part of these milestones.
