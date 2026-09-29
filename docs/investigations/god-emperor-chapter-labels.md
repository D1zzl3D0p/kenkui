# God Emperor of Dune chapter-label investigation

Investigated 2026-09-26 using read-only production queries, the local Calibre
EPUB, and the EPUB parser and audio metadata writer. No production data or
audio was modified.

## Run evidence

The production inspection for source `6d2dc86e-c4be-417f-babc-8a3363828001`
contains 64 readable sections. Its successful job
`14f58425-2edc-4bf5-a7f4-385bd8c6e58f` selects all 64 IDs in source order.
Publication completed and an M4B artifact was recorded.

The local Calibre copy reproduces the stored 64 chapter IDs and titles,
including the anomalies below. Its archive SHA-256 differs from the uploaded
asset, so this is a matching structural reproduction, not verification of
byte-for-byte source identity. The finished M4B itself was not downloaded or
audited in this investigation.

## Causes

| Stored label | Position among readable sections | Explanation |
| --- | --- | --- |
| From the reading by Rebeth Vreeb: | 8 | An internal heading is used to name the entire opening Hadi Benotto speech. The TOC omits this file. |
| Chapter 1 | 9 | The actual TOC label. |
| —THE JOURNALS OF LETO II | 10 | An internal heading names a file beginning with the Hadi Benotto translation. The TOC omits this file. |
| Chapter 11 | 11 | The Welbeck Fragment has no TOC label, no heading, and an opaque document title matching its filename. The parser invents `Chapter {index + 1}`. |
| Chapter 2 | 12 | The actual TOC label. |
| Chapter 9 / Chapter 9 (part 2) | 19 / 20 | One numbered chapter is stored in two Calibre split files. Kenkui exposes both as separate chapter markers. |
| Chapter 11 | 22 | The actual TOC label, retained despite the earlier duplicate. |
| Chapter 50 | 61 | The last numbered chapter in the TOC. |
| Chapter 62 | 62 | An unlisted closing Hadi Benotto report gets the same positional fallback. |

The parser mixes publisher chapter numbers with physical section positions.
Front matter, unlisted narrative documents, and split files all count toward
the latter. The jump from 50 to 62 does not identify missing numbered chapters:
the local source TOC has chapters 1 through 50, and the production job selected
every stored section.

Names do not have to be unique. IDs hash the member path, occurrence, and
fragment; titles do not participate. Existing tests explicitly retain duplicate
titles. The audio metadata writer copies the planned titles without renaming
duplicates. There is no `Chapter 1` -> `Chapter 11` collision mechanism.

The reported extra `Chapter 1` before Dedication is not present in the stored
production inspection. Its opening is Praise, Other Books, Copyright,
Copyright (part 2), Dedication. Explaining a different player display requires
checking that file's embedded chapter metadata and the player's presentation.

## Recommended resolution

1. Replace invented `Chapter N` labels with a clearly distinct neutral fallback,
   such as `Untitled section N`. Retain authored TOC numbers verbatim.
2. Do not promote an arbitrary internal heading to the document title. Use
   heading position and EPUB semantics to identify credible opening titles;
   otherwise retain a meaningful document title or use the neutral fallback.
   Keep heading annotations and speech text unchanged.
3. Support explicit title overrides for unlisted sections. For this book,
   suitable editorial labels include `The Welbeck Fragment` and
   `Hadi Benotto — closing report`. These must be identified as editorial
   labels, not represented as publisher TOC entries.
4. Treat grouping Calibre split files into one audiobook marker as a separate
   change. Current `(part 2)` labels explain physical splits; grouping needs
   explicit mapping to preserve selection, offsets, checkpoint reuse, and order.
5. Add regression coverage for mixed front matter/numbered chapters/unlisted
   sections, duplicate authored titles, internal epigraph headings, and split
   continuations. Assert stable IDs, text, ordering, and annotations.

Correcting the parser affects future inspections and renders. Existing stored
inspection labels and completed M4B chapter metadata need a separate repair.
For an already-produced audiobook, a metadata-only remux can retain encoded
audio and timestamps without paying to synthesize it again. Verify chapter
count, timestamps, cover art, and audio preservation before replacing an
existing artifact.

## Code locations

- `src/kenkui/_epub/parser.py`: `_chapter_text`, `_fallback_title`,
  `_navigation_title`, and `_spine_chapters`.
- `src/kenkui/_epub/identifiers.py`: title-independent chapter IDs.
- `src/kenkui/_audio/production.py`: `_write_metadata`.
- `tests/test_epub_titles.py`: existing navigation and fallback tests.
- `tests/test_epub.py`: duplicate-title identity tests.

## Implemented resolution (2026-09-27)

The user selected TOC-based chapters. Valid TOC targets now delimit logical
chapters in source order, including fragment targets within a file. Intervening
files are joined, and annotations are shifted/clipped with the text. Missing or
unusable navigation uses opening headings, meaningful document titles, or
`Untitled section N`; consecutive Calibre split files still join. Internal
headings no longer supply fallback titles unless they start the visible text.

The local book now has 57 entries, including exactly Chapters 1 through 50.
The previous parser's 64 entries and the new entries have identical concatenated
text and identical ordered annotated text for emphasis, headings, and scenes.
The opening Hadi Benotto speech belongs to the introduction under TOC grouping;
the closing report belongs to Chapter 50. The promotional page after the author
biography remains in the final chapter. Image-only TOC entries have no audio.

Grouping changes chapter identity. Saved selections must be rebuilt from a new
inspection; existing stored inspections, running jobs, and completed artifacts
must not be silently relabelled across this change. This repository change does
not deploy the parser or repair the earlier production artifact. Editable title
overrides and automatic synthesis of descriptive editorial titles are not added;
the deterministic neutral fallback covers unnamed sections.


## Hosted rollout completed (2026-09-27)

Core PR #5 and server PR #3 are merged. Staging and production now run core
`97cba6b` and server `22c6ead`, with matching API and worker revisions. All clean
CI gates passed: 1,738 core tests with approximately 92.4% branch coverage on
Linux/macOS and Python 3.11–3.13, native FFmpeg and package smoke checks, plus
255 server tests and web integration. Both environments passed a real synthesis
canary turning three internal source files into two M4B chapters, full decode,
and exact R2 byte round-trip. Health/capability endpoints return HTTP 200 and
maintenance mode is off.

The server now returns HTTP 409 with an explicit re-upload instruction when an
immutable saved inspection has an older chapter layout. Re-upload the source to
use TOC chapters. Historical jobs and the existing audiobook are unchanged.
Deployment evidence and rollback references are recorded in the infra repository
at `releases/2026-09-27-toc-chapters.json`.
