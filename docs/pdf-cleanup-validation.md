# Native PDF cleanup validation

Validation snapshot: 27 September 2026. This measures the initial library
implementation, not the earlier Docling research pipeline.

## Complete-book checks

The local research corpus supplied three existing PDFs. Each complete source was
hashed, extracted with pdfplumber 0.11.10, and passed through the native recipe:
furniture removal v2, note removal v1, paragraph reconstruction v2, and line-break
word repair v1. No EPUB reference was used to make cleanup decisions.

| Book | Pages | Extraction seconds | Cleanup seconds | Furniture blocks omitted | Paragraphs with word repairs | Ambiguous word-break paragraphs |
|---|---:|---:|---:|---:|---:|---:|
| Alice in Wonderland | 92 | 6.78 | 0.24 | 61 | 22 | 11 |
| Pride and Prejudice | 308 | 67.24 | 1.03 | 0 | 14 | 13 |
| Margin of Safety | 257 | 32.07 | 0.68 | 210 | 9 | 1 |

Times are single local wall-clock measurements, not Modal prices or performance
guarantees. Cleanup excludes extraction, TTS and repeated validation. A word
repair edit can change several boundaries in one paragraph; these counts are not
counts of individual words.

All three runs preserved the original page evidence, left no pages classified
as unresolved by the native adapter, and produced identical documents and audit
trails when the full recipe ran a second time. These are inventory and stability
checks, not proof of prose accuracy or complete extraction.

No footnotes were corroborated in these three editions. The strict native rule
has demonstrated its behavior on generated source PDFs and targeted geometry
fixtures; these book runs do not establish useful footnote recall. Broader
handling still needs layout evidence and more representative labelled examples.

The runs exposed a header limitation in the previous library recipe: top folios
and running titles containing changing page numbers were missed. The revised
rule requires a supported page-number offset on distinct pages; numbered titles
also need repeated position/font/size evidence. Tests retain years, numbers in
the body, larger title text, and custom-edited projections.

## Regression coverage

The preceding native-cleanup full-suite run passed 1,795 tests with 92.58% coverage (11 skipped,
7 deselected). Ruff, strict typing, the documentation build and package checks
also passed. This run did not evaluate synthesized audiobook quality.

Generated, project-authored fixtures cover:

- Real PDF extraction followed by note/reference removal, paragraph recovery and
  word repair; empty recipes preserve the original extraction.
- Hard-hyphen repair supported by an unambiguous spelling elsewhere in the same
  source; genuine compounds keep their hyphen and uncertain cases are reported.
- Discretionary soft hyphens, repeated text elsewhere in a paragraph, custom
  transformations, code blocks, and hyphenated words across facing-page margins.
- Same-page numbered notes, their continuations and attached superscripts.
  Baseline digits, exponents, mathematical fonts, dialogue, duplicate note
  numbers, unavailable note text and unfinished note tails are retained.
- Immutable original evidence, separator-only word edits, complete edit
  inventories, and repeat-run stability.

This does not certify the requested 0.1% prose-error threshold. Remaining work
includes broader note/marker handling and additional source-labelled quality
evaluation of layout and reading order.


## Layout and worker validation

The optional backend now uses Docling 2.128.0, PDFium, CPU layout and selective
RapidOCR/ONNX extraction in a disposable subprocess. Offline tests exercise
native/scanned routing, page batching, partial extraction rejection, missing
assets, memory and output limits, timeouts, cancellation, strict IPC records and
progress. A public-pipeline test verifies that the child is reaped before the
prepared book enters fake speech execution.

The adapter was also checked against the existing eight-page Docling samples
for Alice, Pride and Prejudice, Margin of Safety and Principles of Economics.
All four exports now map successfully (65, 10, 49 and 154 blocks respectively).
That check caught a real list-item compatibility problem: provenance spans can
include the marker in `orig` even though `text` has removed it. List adaptation
now preserves that original numbered text, with a regression fixture. These are
adapter checks on earlier exports, not new full-book accuracy measurements.

New cleanup fixtures protect central or isolated headers, uncertain note
sections and uncorroborated captions. They verify removal of repeated classified
edge furniture, supported footnotes inside layout paragraphs, consecutive
numbered note sections and classified visual material. Layout paragraphs are
not flattened by the native paragraph reconstruction step.

Real-model acceptance also passed for generated digital, image-only and mixed
PDFs, including one-page batches and original page-number mapping. Each fixture
preserved all three authored prose lines. The final run took 87.13, 24.92 and
28.68 seconds respectively, including worker startup, imports and model loading.
An earlier digital run hit the 240-second test timeout while cold-loading local
libraries. These tiny-fixture wall times are environment-sensitive and must not
be used to estimate whole-book cost or throughput. The local acceptance environment
used Docling 2.128.0 and OpenCV headless 4.13.0; it was the existing spike environment,
not a fresh installation of every transitive version in the project lock.

The final review regression run passed 1,840 tests (11 skipped, 10 deselected)
at 92.38% coverage. It includes the list-item regression, recipe validation before
extraction, invalid limit types and cancellation before custom cleanup. Native
PDF and process-monitor dependencies now belong to the development group so
ordinary CI exercises these paths. Ruff, strict typing, strict docs, distribution
checks and an installed-wheel public API smoke passed. Native FFmpeg acceptance
remains unverified on this host because FFmpeg is unavailable. No new
synthesized-audio/Whisper quality evaluation was run.
