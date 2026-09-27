# PDF preparation

Install the optional parser with `pip install 'kenkui[pdf]'`.

Native mode handles embedded text without rendering pages or loading models.
The default `auto` mode adds optional layout classification and selective OCR
in an isolated CPU worker; see the setup below.

```python
import kenkui as kk

book = kk.book("book.pdf").pdf_processing(mode="native")
prepared = book.prepare()

print(prepared.inspect().chapters[0].text)
report = prepared.pdf_report()
print(len(report.pages), len(report.edits), report.issues)

# Uses the existing voice provisioning and audiobook execution path.
prepared.assign_voice("eponine").tts().write("book.m4b")
```

Preparation returns a new pipeline; it does not modify the original. Constructors
remain lazy. An unprepared PDF's `inspect()`, `script()`, and `pdf_report()` raise
`pdf_preparation_required`. Calling `resolve()` or `write()` prepares automatically
using the recorded settings.

## Independent cleanup functions

The native recipe runs these independent functions in order:

1. `remove_furniture`: corroborates repeated headers/footers using position,
   font and size. A consistent page-number offset on at least five distinct pages
   supports top or bottom folios; attached running titles also need repetition.
2. `remove_notes`: omits numbered footer notes and their raised prose references
   together. It requires smaller type, a separated footer area, and a matching
   superscript after a prose word. Baseline digits, exponents, unverified note
   labels are retained. Unfinished note tails are retained;
   cross-page note chains are not followed.
3. `reconstruct_paragraphs`: joins compatible body lines, including supported
   continuations across page breaks. It preserves indentation and opening
   dialogue quotes and avoids monospaced or insufficiently supported layouts.
4. `repair_line_break_words`: repairs boundaries inside reconstructed prose when
   the source contains an unambiguous complete spelling, or the break contains
   an explicit soft hyphen. Attested compounds keep their hyphen. Unknown or
   conflicting spellings remain unchanged and generate
   `ambiguous_line_break_word` issues.

These rules need no model, dictionary download, or OCR substitution. Word repair
changes only the break separator, preserving every letter. It requires an exact
original-line projection; paragraphs already rewritten by a custom transform or
reference removal may be conservatively skipped. Note removal runs before
paragraph reconstruction and leaves already merged or edited note blocks alone.

Each function accepts an immutable `PdfDocument` and returns a new document with
a changed narration projection and appended edits. Original page, line, character
and bounding-box evidence remains accessible through `pdf_report()`.

```python
from kenkui.pdf_processing import PdfStep, remove_furniture

headers_only = book.pdf_processing(
    mode="native",
    steps=(
        PdfStep("remove_furniture", "2", remove_furniture, requires=("native_lines",)),
    ),
).prepare()

# Empty means no optional cleanup; extraction and final checks still run.
raw = book.pdf_processing(mode="native", steps=()).prepare()
```

Invalid recipe dependencies and non-callable steps are rejected before extraction.
Changing settings invalidates previous preparation and resolution. Versioned
`PdfStep` descriptors include explicit configuration and dependency requirements.
Plain document-to-document callables are also accepted; they must preserve source
evidence and append an edit when changing narration. They have no portable
persistent request identity. Generic recipe checks enforce the audit contract;
custom transforms remain responsible for the correctness of their edits.

See the [validation snapshot](pdf-cleanup-validation.md) for full-book checks
and the distinction between stability checks and measured prose accuracy.

## Current limits

This is a partial implementation of the
[implementation plan](pdf-implementation-plan.md), not the complete researched
cleanup pipeline. Current output uses one `pdf-body` chapter. Chapter detection,
cross-page note chains, persistent preparation caches and EPUB export remain
pending. Layout reading order is accepted only when its native text projection
passes the completeness checks. No measured
ebook-equivalence or prose error-rate guarantee applies to this implementation.

Native extraction rejects malformed or encrypted documents, documents exceeding the
page/text bounds, pages containing graphics without extractable text, and large
image pages with only sparse native text. A scanned body with an embedded running
header must not silently become a header-only audiobook. These checks are
conservative heuristics, not proof of extraction completeness; inspect the text
before using unfamiliar layouts. Native extraction currently runs in the caller
process with cancellation checks between pages; it has no hard time or memory
limit enforced by an isolated worker yet.

Preparation preserves the source-byte hash separately from the recipe/output
identity. A changed PDF cannot be rendered using stale prepared text; call
`prepare()` again. Prepared identity participates in attribution and plan
fingerprints. Checkpoints currently live in memory only.


## Optional layout and OCR

Install `pip install 'kenkui[pdf-layout]'` and provision models separately using
Docling's `docling-tools models download` command. The selected layout model is
`docling-project/docling-layout-heron`; the OCR backend is RapidOCR with
ONNX Runtime. Use `docling-tools models download --help` for the installed
version's model selectors. English OCR assets can be provisioned with:

```console
docling-tools models download layout rapidocr --rapidocr-backend-lang onnxruntime:en -o /models/pdf
```

The same asset root must contain `docling-project--docling-layout-heron/`
(with its configuration, preprocessing configuration and model weights) and,
for scanned pages, `RapidOcr/`. Preparation never downloads models. Set
`KENKUI_PDF_MODELS` or pass the path explicitly:

```python
from kenkui.pdf_processing import PdfLayoutOptions

prepared = (
    kk.book("book.pdf")
    .pdf_processing(
        layout=PdfLayoutOptions(
            artifacts_path="/models/pdf",
            threads=2,
            max_memory_mb=4096,
            timeout_seconds=1800,
            batch_pages=16,
        ),
    )
    .prepare()
)
```

The worker runs before downstream speech execution and exits before preparation
returns. Its wall time, combined process RSS and serialized output size are
monitored. RSS checks are sampled, not an operating-system memory reservation;
leave headroom for the parent process and any already loaded speech model.
Cancellation and failures terminate the worker process group. Missing packages,
missing assets, incomplete extraction, timeouts and invalid output have distinct
error codes. Auto mode does not silently downgrade to native mode.

Digital pages retain original text. Model layout is accepted only if every native
line is covered exactly once and the projected words agree; otherwise the page
retains native blocks and reports `layout_native_mismatch`. Scanned/image-heavy
pages use full-page OCR and must yield prose, not just an image placeholder.
OCR spelling accuracy is not guaranteed and no LLM correction runs.

The auto recipe also applies `remove_note_sections` and `omit_visual_material`.
Explicit Notes/Endnotes/Footnotes sections require at least two complete,
consecutively numbered entries. Classified tables, pictures and formulas are
omitted; captions require an explicit figure/table label adjacent to a visual.
Repeated classified edge headers and consistent folios can be removed from OCR
pages. Layout paragraphs keep their boundaries. Footnote removal inside native
layout blocks still requires the native geometry/reference evidence described
above; ambiguous and OCR-only footnotes remain.

All these functions are independently importable from `kenkui.pdf_processing`.
Pass a custom `steps` tuple to select them, or `steps=()` to retain extraction
output without optional omissions. Original native and layout records remain in
`pdf_report().pages`; each omission adds an audit edit.
