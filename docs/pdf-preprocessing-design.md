# PDF preprocessing design

**Status: target architecture, partially implemented.** Kenkui supports explicit
native preparation and optional isolated layout/OCR. The examples below also
include planned cleanup and persistence features; consult [the current API](pdf-processing.md)
before using them.

PDF preprocessing should be a source adapter composed of independently callable
steps. A standard recipe supplies the automatic behavior; callers can replace
that recipe without reimplementing extraction or audiobook processing.

The [implementation plan](pdf-implementation-plan.md) maps this design to ordered
code changes, test coverage and release gates.

## Public shape

```python
import kenkui as kk

# Record intent now; automatically prepare the PDF when execution begins.
book = kk.book("book.pdf")
book.assign_voice("eponine").tts().write("book.m4b")

# Disable page rasterization, OCR, and layout-model inference.
# Keep inexpensive, source-supported cleanup of native text.
native = kk.book("book.pdf").pdf_processing(mode="native")

# Perform source preparation explicitly, without starting TTS or casting.
prepared = book.prepare(on_event=print)
inspection = prepared.inspect()
script = prepared.script()
report = prepared.pdf_report()
```

Add `kk.pdf(path)` alongside `kk.epub(path)`. `kk.book()` dispatches `.pdf` and
`.epub` without opening the source. Do not infer a format by attempting every
parser. Validate the selected format against the bytes during preparation.

Use two extraction modes initially:

| Mode | Extraction behavior | Cleanup behavior |
|---|---|---|
| `auto` (PDF default) | Native extraction plus Docling layout analysis; OCR for scanned pages | Standard recipe |
| `native` | Embedded text and native geometry only; no rasterization or model calls | Same recipe, with explicit abstentions where evidence is unavailable |

`auto` describes page routing, not a silent fallback when dependencies are absent.
Missing layout/OCR dependencies produce an actionable error. Callers can choose
`native` explicitly. An image-only narrative page in native mode is an extraction
failure, not an empty successful chapter. Mixed PDFs must account for every page.

The existing full-book benchmarks used Docling across the document. Skipping
layout inference on apparently simple pages is a later optimization requiring its
own quality evaluation.

## Each cleanup step is a function

```python
# Proposed names; this module does not exist yet.
from kenkui.pdf_processing import (
    remove_furniture,
    omit_visual_material,
    repair_reading_order,
    repair_line_break_words,
    reconstruct_paragraphs,
    remove_margin_marks,
)

# Retain footnotes by omitting remove_notes from the standard recipe.
book = kk.book("book.pdf").pdf_processing(
    mode="auto",
    steps=(
        remove_furniture,
        omit_visual_material,
        repair_reading_order,
        repair_line_break_words,
        reconstruct_paragraphs,
        remove_margin_marks,
    ),
)

# No optional cleanup, but mandatory extraction validation still runs.
raw_native = kk.book("book.pdf").pdf_processing(mode="native", steps=())
```

`steps=None` means the versioned standard recipe. An explicit tuple replaces it;
an empty tuple does not select the default accidentally. The caller's order is
preserved. Validate missing prerequisites and incompatible order before doing
expensive work; do not silently reorder custom steps.

Built-in functions have the conceptual signature
`step(document: PdfDocument) -> PdfDocument`. They return immutable values and
append edit evidence and issues. They do not open PDFs, download models, run OCR,
make network calls, or invoke TTS. Functions should also be directly callable on
an extracted document in tests and advanced workflows. `remove_notes` may compose
separate footnote, explicit-endnote, and reference-marker helpers internally.

Keep these effectful functions outside the cleanup recipe:

- `extract_native(snapshot, limits, cancel)`;
- `classify_layout(snapshot, native, resources, cancel)`;
- `recognize_pages(snapshot, page_ids, resources, cancel)`;
- `retry_failed_pages(snapshot, document, resources, cancel)`.

Retrying OCR changes the evidence, so it must precede rerunning cleanup for affected
pages. It must not silently change words after selection, quote attribution, or
speech planning has already established character offsets.

The standard sequence is:

1. Snapshot and hash the original bytes; extract native evidence.
2. Classify layout and recognize scanned pages as required by the mode.
3. Validate extraction coverage and canonical model text, including overlapping
   provenance spans. Perform a bounded retry when an enabled policy supports it.
4. Remove corroborated furniture and notes from the narration projection; omit
   identified visual material under the default narration policy.
5. Repair supported reading-order inversions and line-break words; reconstruct
   paragraphs; remove isolated marginal scan marks.
6. Validate edit accounting and remaining unresolved narrative content.
7. Adapt ordered narration to Kenkui's existing book-inspection representation.

Word repair precedes paragraph reconstruction in this recipe so that a proven
split word cannot create a false paragraph start. Cross-page word repair records
a continuation constraint for the paragraph step; it never drops a second block
without recording where its remaining text goes.

Do not expose all research thresholds as initial public arguments. Infer body
font, margins, indentation and page-number offsets from the document. Language,
extraction mode and step selection are meaningful public policy; a universal
"crop top 10%" setting is not a safe default.

## Source representation and adaptation

Use immutable typed records under `_pdf`, separate from Docling's objects:

- `PdfDocument`: source identity, metadata, ordered pages/blocks, narration
  projection, capabilities, edits and issues;
- `PdfPage`: original page number, dimensions and extraction disposition;
- `PdfBlock`: stable ID, text, role, geometry, native/model provenance and decision;
- `PdfEdit`: step/version, affected IDs, before/after evidence and reason;
- `PdfIssue`: stable code, page/block IDs, severity and available recovery action.

The original extraction stays available even when a block is omitted from speech.
Use typed per-step evidence rather than an unrestricted dictionary as the public
contract. Keep capabilities explicit: native character geometry, reliable line
geometry and model classifications are different forms of evidence. English
lexical rules abstain for unsupported languages.

The adapter produces `BookInspection` and `ChapterInspection`, including the
existing normalized text, heading ranges, scene ranges and emphasis where
supported. Reuse existing normalization and planning. PDF-to-EPUB conversion is
not required internally; EPUB can be a separate export.

Choose chapter boundaries from corroborated structural headings. Repeated headers
cannot become chapters. If no reliable chapter boundaries exist, produce one
logical chapter, not a chapter per page. IDs are deterministic within the same
prepared source and recipe; references into another preparation are stale.

Initially report `cover_available=False` for PDF sources. Do not pass a PDF to the
existing EPUB cover extractor or silently rasterize page one in native mode.
Explicit caller-provided cover files continue to work.

## Lazy execution and the inspection contract

Current `inspect()` and `script()` promise no model calls. Preserve that contract:

- Constructors and `.pdf_processing(...)` only record immutable intent.
- `.validate()` remains cheap and does not parse/render a PDF or download assets.
- `.prepare()` is the explicit effectful source checkpoint. It returns a new
  pipeline with a frozen prepared inspection, identity and report. For EPUB it
  snapshots and parses through the existing parser without PDF dependencies.
- For an unprepared PDF, `.inspect()` and materializing `.script()` raise a clear
  preparation-required error. They must not inspect cheap native text and later
  render different, enhanced text. Both PDF modes use this explicit contract.
- `.resolve()` and `.write()` automatically prepare PDF sources before general
  book processing. Direct rendering therefore remains a one-chain operation.
- `.pdf_report()` reads an existing preparation; it does not trigger inference.
- PDF options on an EPUB source are rejected as inapplicable rather than ignored.

PDF preparation runs before chapter selection, character discovery, attribution,
spoken-form normalization and TTS. Selected chapters are validated against the
prepared chapter IDs. Metadata overrides retain their existing semantics.

Keep PDF configuration with source-preparation intent, not as an ordinary text
tuning operation that could appear after synthesis. Changing it invalidates
prepared, roster and resolved checkpoints. Existing `.pipe()` remains an
immediate user function call; it should not acquire hidden preprocessing semantics.

## Identity, persistence and cache correctness

Record original source SHA-256 separately from prepared text identity. A PDF's
bytes alone are insufficient: native mode, a different note policy or a new
paragraph algorithm may produce different text and offsets from the same PDF.

Prepared identity includes source hash, extraction mode, ordered step IDs and
versions, configuration, backend/model versions and relevant resource revisions,
language, and the normalized output structure/text digest. Separate the
deterministic extraction-request cache key from the resulting output digest.

Audit all consumers of source identity: resolved/roster reuse, attribution stores,
authoring sidecars, plan fingerprints, resume checkpoints and audio caches.
Selection and tuning authored against one prepared structure cannot be applied
silently to another. Preserve existing EPUB hashes and behavior; add the PDF
preparation identity where needed rather than changing EPUB identities gratuitously.

Built-in functions have registered stable IDs and versions; serialized recipes
use those descriptors, never Python pickles. Advanced custom functions may run
locally, but persistent caching and portable recipe serialization require an
explicit registered ID/version/configuration contract. Unregistered callables
disable persistent preparation caching and cannot be saved as portable recipes.
Do not hash a function's name or repr and call that a semantic identity.

Cache only completed, validated preparations using atomic publication. Preserve
the immutable prepared inspection in memory for repeated inspection/rendering;
never return a temporary-file path after its workspace has been removed. Bind
reports to the same identity as their text and invalidate them together.

## Dependencies, resources and failure handling

Keep PDF libraries lazy and optional. Proposed extras are `kenkui[pdf]` for native
extraction and lexical cleanup, and `kenkui[pdf-layout]` for that set plus Docling
and the chosen OCR backend. Verify supported Python versions and dependency
resolution before pinning them. Tesseract's executable and model files need
explicit installation/provisioning documentation; a pip extra alone is not enough.
Missing assets fail with instructions rather than triggering hidden provisioning.

Run layout/OCR in a bounded child process that exits before TTS workers start.
This fits the current coordinator, which already keeps TTS engines out of the
parent. Expose resource limits independently from synthesis `workers`; constrain
pages, rendered pixels, output bytes, threads, memory where enforceable, deadlines
and retries. Cancellation must stop the child and clean partial results.

Use existing stage events for `pdf.extract`, `pdf.layout`, `pdf.ocr`, `pdf.cleanup`
and `pdf.validate`. Carry page-level details in the PDF report; do not mislabel page
numbers as chapter IDs. Aggregate timings, page routes and retry counts for cost
measurement. A separate Modal service is a deployment choice, not a library API.

Distinguish confirmed blank/decorative pages from unreadable narrative pages.
Uncertain cleanup retains text with an issue. Failed narrative extraction blocks
rendering by default. Tables/figures explicitly omitted by policy are accounted
for separately from failures. Password-protected/unsupported PDFs receive explicit
errors in the first version; never return an apparently successful empty book.

The experimental two-page OCR retry is not a default production feature yet. Its
first clipping implementation lost short dialogue. Acceptance must cover full
body text, exclusion regions and short paragraphs; lexical scores and word-count
ratios alone cannot establish fidelity.

## Repository integration and implementation slices

The current seams needing changes are:

| Location | Required change |
|---|---|
| `api.py`, `pipeline.py:Source` | PDF dispatch and immutable source preparation policy |
| `Pipeline.inspect`, `script`, `save` | Prepared-source inspection; remove EPUB-only parse assumptions, including tuning sidecar saves |
| `_source.py` | Preserve bounded same-byte snapshot/hash guarantees for both formats |
| `_resolution.inspect_source` | Shared preparation adapter; currently names every snapshot `source.epub` |
| `_execution/coordinator.py` | Reuse preparation before planning; currently also snapshots directly as EPUB |
| `_resolution`, character stores, `_domain/sidecar.py` | Bind offsets/checkpoints to prepared text identity |
| `_domain/planning.py`, checkpoints/cache paths | Incorporate PDF recipe/output identity without changing existing EPUB identity |
| `_audio` cover handling | Avoid routing PDF sources through EPUB cover extraction |
| New `_pdf` package | Models, native adapter, optional Docling worker, pure transforms, validation and book adapter |
| New public `pdf_processing.py` | Typed document/step/report interface and built-in functions |

Implement in reviewable slices:

1. Refactor current EPUB inspection behind a shared source adapter with unchanged
   output/fingerprints. Test inspection, resolution, sidecar saving, covers and
   rendering against that boundary before adding PDF behavior.
2. Add native PDF preparation, the explicit checkpoint, format dispatch and
   prepared identity. Initially support native mode only; default auto support
   must not be advertised before its adapter exists.
3. Port cleanup steps individually from research with small, distributable fixtures
   and explicit positive/negative tests. Introduce the versioned standard recipe.
4. Add optional isolated Docling/OCR, automatic preparation during execution,
   dependency errors, resource bounds, progress and cancellation.
5. Add bounded retry only after broader recovery tests, then export/report tooling.

Acceptance includes offline fake backends, EPUB identity/regression preservation,
zero PDF-library imports on EPUB-only paths, literal empty-recipe behavior,
native mode making zero render/model calls, cleanup idempotence, exact edit
accounting, protected dialogue/verse, unreadable-page failures, custom-step cache
rules, source-change detection, policy-change invalidation, cancellation and
inspect/script/write text equality. Real Docling/OCR tests remain opt-in and must
not download assets during ordinary CI. Do not redistribute research books as
test fixtures; generate layout fixtures or use appropriately licensed excerpts.

This design promises auditable preprocessing, not 99.9% fidelity. Same-edition
prose labels and independent test books are still needed for that quality claim.
