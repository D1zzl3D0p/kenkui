# Grid Folds — Design

**Status:** Approved for implementation planning on 2026-09-10.

**Extends:** `docs/superpowers/specs/2026-09-08-grid-and-tuning-design.md`,
especially section 7, "The folds."

## Goal

Make the addressable grid the sole partition of canonical chapter text. Fold
quote detection, structural boundaries, and TTS chunking into that grid so
attribution, tuning, selection, preview, pauses, and rendering all share one
stable coordinate system.

This is one atomic migration. The layering cleanup and all three folds ship
together. The legacy chunker may exist temporarily as a test oracle during
development, but it is deleted before completion. There is no production
feature flag or dual runtime path.

## Scope

In scope:

- break the existing `_domain.grid` to `_characters` import cycle;
- move pure quote scanning and prefix-title data into `_domain`;
- make the grid own quote, block, line, sentence, and phrase boundaries;
- add a derived prefix-to-range index over the flat grid;
- migrate attribution to grid-provided dialogue ranges;
- replace `_chunk_span` and the general regex break ladder with hierarchical
  grid packing plus a pathological-leaf fallback;
- make structural gap reasons independent of pause settings;
- replace both chunking identity versions with `grid-v1`;
- prove exactness, determinism, tuning boundaries, silence placement,
  selection/cache behavior, and break quality on tests and the local corpus.

Out of scope:

- the deferred `pronounce()` / `spoken_form()` API split;
- manual grid split/merge operations;
- engine-token-based budgets;
- automatic deletion of unreachable cache files;
- changes to attribution storage or sidecar formats that are not required by
  changed grid paths.

## Current-state constraints

Phase 1 shipped on `fix/chunking-prosody-and-gender` at `d1c956b`. The design
must preserve its scoped pronunciation, effective span merge, manual silence,
selection, preview, and tier-introspection behavior.

The current dependency graph is circular:

```text
_domain.grid -> _characters.quotes -> _characters.__init__
             -> _domain.planning -> _domain.grid
```

`planning.py` is already large. New traversal and packing logic does not belong
there.

## Canonical representation

The canonical grid remains a flat, ordered tuple of immutable leaf `Unit`
values. Each leaf owns one exact half-open canonical range and a labeled path:

```text
chapter / paragraph / line / sentence / phrase
```

The hierarchy is not stored as a second mutable tree. A derived structural
index maps each path prefix to a contiguous leaf range:

```text
paragraph path -> [first leaf, past-last leaf)
line path      -> [first leaf, past-last leaf)
sentence path  -> [first leaf, past-last leaf)
phrase path    -> [first leaf, past-last leaf)
```

This index supplies tree-like traversal without duplicating offsets, text, or
flags. It is deterministic and derivable solely from the leaf tuple.

The safety invariant remains exact tiling: ordered leaf slices are non-empty,
adjacent, non-overlapping, and concatenate to the complete canonical chapter
text.

## Ownership and layering

`_domain` owns all pure partitioning:

- move the quote scanner out of `_characters.quotes`;
- move the prefix-title data needed by sentence splitting out of
  `_characters.identity`;
- update `_characters` consumers, then delete the private legacy quote module
  rather than retaining compatibility re-exports;
- keep `_domain.structure` as pure boundary discovery, with no pause settings;
- build leaves, dialogue flags, emphasis flags, structural ranges, and gap
  reasons once during grid construction.

After the fold:

- `_domain` does not import `_characters`;
- attribution reads grid dialogue ranges and never rescans quotes;
- planning never rescans blocks, lines, or quotes;
- pause settings translate already-discovered gap reasons into milliseconds;
- selection and scoped tuning continue to use canonical offsets.

## Structural gaps

A gap after a leaf records why a larger structural range closes there, such as
line, paragraph, or chapter. Quote edges are mandatory attribution boundaries,
not a new hierarchy level. Multiple reasons may apply to one gap.

Gap discovery is text- and structure-dependent only. `Pauses` selects the
effective duration later, and an explicit silence rule continues to replace
the derived value, including with zero.

## Hierarchical packing

Packing operates over the derived structural index:

1. Apply spoken-form transformation independently to grid regions while
   retaining canonical ranges and offset mappings.
2. Add mandatory cuts for speaker changes, manual silence, scoped
   pronunciation regions, selection edges, and any other phase-1 semantic
   boundary.
3. Attempt the largest structural ranges first.
4. When a range exceeds the existing 1,000-character spoken-text ceiling,
   descend paragraph to line to sentence to phrase.
5. Greedily combine adjacent fitting pieces without crossing a mandatory cut.
6. If one spoken phrase exceeds the ceiling, use an emergency leaf splitter.
   It prefers punctuation or hyphens, then whitespace, and uses a hard cut only
   for an indivisible token.

The general `_BREAK_TIERS`, `_CLEAN_BREAK_TIERS`, `MIN_BREAK_FILL`, and
`_chunk_span` machinery is deleted. The separator-free guard survives only if
needed inside the emergency leaf splitter.

Packing remains character-budgeted. Engine-token budgets are a separate
calibration and identity change.

## Spoken form and tuning

Grid identity stays in canonical coordinates. Spoken-form transformations may
expand or contract text, so fit decisions use transformed length while packed
ranges retain mappings back to canonical grid leaves.

Scoped pronunciations, effective attribution, and manual silence are inputs to
mandatory boundaries. A packer must never join across one of those boundaries.
The effective content, speaker, voice, and silence sequence must remain equal
to phase 1 even when segment boundaries improve.

## Selection and preview

Selection is an exact canonical range operation. Packing a selection uses the
same grid hierarchy and mandatory boundaries as a full render. Every segment
wholly contained by both plans must have the same identity. Only genuinely
clipped selection edges and the trailing selection-gap treatment may differ.

The fold must not reintroduce parsing outside a selection or invalidate the
resolved-attribution checkpoint.

## Identity and cache transition

One explicit `grid-v1` chunking schema replaces
`CHUNKING_SCHEMA_VERSION` and `STRUCTURAL_CHUNKING_SCHEMA_VERSION`. This
deliberately changes segment identities and makes old audio cache entries
unreachable. It does not delete them automatically.

Attribution records remain reusable. Existing tuning sidecars remain readable;
if grid path changes cause anchor or match-count drift, the existing warning
model reports it rather than silently applying a different correction.

Before a library render, check free space on the Data volume. Cache pruning is
an explicit operator action outside this migration.

## Implementation boundary

Create a focused `_domain/grid_packing.py` for hierarchical traversal,
budgeting, and emergency leaf splitting. `planning.py` prepares transformed
regions and semantic boundaries, invokes the packer, and constructs segments.
It does not discover textual boundaries.

The temporary legacy oracle must be test-only or clearly isolated. It exists
only until differential fixtures and measurements are established and is
removed in the same branch.

## Errors and invariants

Malformed grids and impossible packer results fail before synthesis. Required
invariants include:

- exact canonical tiling;
- a deterministic prefix-range index;
- exact transformed-text reconstruction;
- no empty segments;
- no segment above the configured hard character ceiling;
- no packing across a mandatory semantic boundary;
- gap silence attached to the preceding effective segment;
- stable ordering of speaker, voice, and pronunciation regions.

An emergency hard cut is permitted only when no natural boundary exists inside
one over-budget leaf. Tests and metrics distinguish these cuts from normal
grid-edge cuts.

## Verification

The implementation is accepted only when all of the following hold:

- **Grid exactness:** leaf text concatenates to every parseable corpus chapter.
- **Grid determinism:** pauses, voice/model, budget, selection, and tuning do
  not change the grid.
- **Scanner equivalence:** before legacy scanner deletion, relocated quote and
  structural discovery produces identical paths, offsets, and flags.
- **Spoken exactness:** packed text reconstructs transformed text region by
  region.
- **Hard bound:** every segment respects the character ceiling.
- **Mandatory cuts:** speaker, explicit silence, scoped lexicon, and selection
  boundaries are never crossed.
- **Gap equivalence:** derived and manual silences remain on the same canonical
  gaps.
- **Differential oracle:** legacy and new planning preserve spoken content,
  speaker/voice order, and effective silences; segment boundaries may differ.
- **Break quality:** every ordinary boundary is a grid edge; emergency
  within-leaf boundaries are counted and characterized separately.
- **Cache behavior:** all new IDs use `grid-v1`, and preview/full plans share
  every wholly contained segment identity.
- **Layering:** `_domain` has no `_characters` import, and attribution/planning
  do not rescan quotes or structure.
- **Project gate:** Ruff format/check, strict mypy, full pytest with at least
  90% coverage, the opt-in real-library grid corpus, and a Dune smoke test all
  pass.

Synthesized-audio comparison is not required. The semantic sequence and text
identity are the deterministic proof; waveform comparison would add engine and
hardware variability without improving the partitioning guarantee.

## Approved decisions

- Keep all folds in one plan.
- Use the legacy implementation temporarily as a test oracle, then delete it.
- Move pure scanners into `_domain`.
- Keep a flat canonical grid with a derived range index.
- Use hierarchical packing with an emergency pathological-leaf fallback.
- Keep character budgets for this migration.
- Ship `grid-v1` without a runtime rollback flag.
- Leave unreachable cache files in place.
- Exclude the `pronounce()` / `spoken_form()` split.
