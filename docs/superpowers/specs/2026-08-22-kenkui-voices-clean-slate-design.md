# kenkui-voices: clean-slate voice pack distribution

Status: design
Date: 2026-08-22
Supersedes: `2026-08-22-kenkui-voices-repo-design.md` (untracked draft; retained
for reference only)

## 1. Problem

The 95-voice pre-compiled pack has no single home and no reproducible build.
Its pieces are scattered across four locations, none of which references the
others.

| Piece | Currently lives in |
| --- | --- |
| Curated source manifest (95 voices) | `D1zzl3D0p/kenkui-voices` @ `33f00bc5` |
| Stale local copy (66 voices) | `Repos/tts-voices/kenkui_source_manifest.json` |
| `update_voices.py` orchestrator | same directory |
| `build_voice_pack.py` compiler | `kenkui/tools/` (the 2.3.6 app repo) |
| Compiled assets (597 MB) + previews (~68 MB) | `D1zzl3D0p/kenkui-voices` |
| Consumer pin | `PACK_REVISION` constant in `voices/registry.py` |
| Built-in kyutai voices (26) | hardcoded Python in `voices/registry.py` |

`Repos/tts-voices` is a symlink into the Hugging Face cache
(`models--kyutai--tts-voices/snapshots/323332d3…`). Everything written there is
untracked and unversioned; an `hf` cache prune discards it.

Four consequences follow.

**29 of the 95 voices cannot be rebuilt from anything published.** The source
manifest records 47 voices against relative upstream VCTK paths, 19 against
relative upstream EARS paths, and 29 against *absolute* paths into that cache:
`test-voice/ears-1/{speaker}.wav`. Those WAVs are locally trimmed and
Adobe-Podcast-enhanced derivations (24 kHz, ~8.7 s) that exist on one machine.

**The pack version and the runtime version drift silently.** `pyproject.toml`
pins `pocket-tts==2.1.0`; the shipped manifest declares
`"pocket_tts_version": "2.0.0"`. Nothing compares them, because the pack revision
is a constant in Python rather than resolved data.

**Provenance is not shipped.** The generated manifest drops `prompt_source`, so
the artifact kenkui vendors does not record what it was compiled from.

**Two catalogs, two shapes.** Pack voices arrive as JSON; the 26 kyutai built-ins
are Python objects. Merging is JSON-into-Python, so the two halves cannot be
validated, diffed, or regenerated the same way.

The failure mode being guarded against is silent: an incompatible embedding
loads without error and renders well-formed audio containing no words.

## 2. Decisions

Settled before design; recorded here with their rationale.

| Decision | Choice |
| --- | --- |
| Consumer | kenkui-v2 (`0.1.0`) only. 2.3.6 is untouched. |
| Tooling home | The Hugging Face dataset itself. No separate GitHub repo. |
| The 29 local prompts | Discarded. Recompile from upstream `_enhanced` audio. |
| Rebuild scope | All 95 under pocket-tts 2.1.0. |
| Version mismatch | Hard fail, naming the versions and the rebuild command. |
| Manifest access | Vendored into the package; assets fetched on demand. |
| Repo | `D1zzl3D0p/kenkui-voices` deleted and recreated. |

**Why upstream `_enhanced`.** All 29 speakers have both
`ears/{speaker}/freeform_speech_01.wav` (48 kHz, 10.0 s) and
`freeform_speech_01_enhanced.wav` (32 kHz, 10.0 s) upstream. The 19 EARS voices
already shipping use `_enhanced`. Adopting it for all 48 makes every EARS voice
uniform and every voice in the pack reproducible from
`kyutai/tts-voices` at a pinned revision alone. The local prompt tier disappears
rather than being versioned, which is the streamlining this project is for.

Cost, stated plainly: the 29 voices get new embeddings and new previews. They
are the same speakers reading the same passage, but the timbre will not be
bit-identical to what shipped. The 47 VCTK and 19 EARS voices compile from
unchanged inputs and must reproduce their published bytes exactly — §8 makes
that a gate.

**Why the pocket-tts major is the compatibility key.** Measured across releases,
2.0.0 and 2.1.0 produce byte-identical embeddings, while the 1.x companion
tensor differs in name, shape, and dtype and loads under 2.x without raising.
One pack per major; minors within a major share a pack and are recorded only
once verified.

## 3. Distribution repo

`D1zzl3D0p/kenkui-voices` (Hugging Face **dataset**) holds everything: source of
truth, binaries, and the tooling that produces them.

```
kenkui-voices/
├── README.md              dataset card: what this is, how to rebuild
├── voices.json            source of truth + generated build records
├── sources.lock.json      upstream pins
├── compiled/              95 × .safetensors  (~597 MB)
├── previews/              95 × .wav          (~68 MB)
└── tools/
    ├── pyproject.toml     standalone uv project: pocket-tts, huggingface_hub
    ├── rebuild.py         sync prompts → compile → preview → hash → stamp
    ├── verify.py          compile a sample, diff hashes, report
    └── publish.py         rebuild → upload → stamp assets.revision
```

Working from a clean slate drops four things the old repo carried: the split
`packs/{vctk,ears-research}-manifest.json` pair, `pack-index.json`, the separate
`kenkui_source_manifest.json`, and the per-voice fields that were constant
across all 95 records.

There is no `.cache/` tier and no committed symlink. `rebuild.py` materializes
prompts through `hf download` at the revision in `sources.lock.json` and works
out of a temporary directory.

## 4. Data formats

### 4.1 `voices.json` — one file, two halves

Each voice carries a hand-curated `source` block describing what it derives
from, and generated `compiled` / `preview` blocks describing what to fetch.
`rebuild.py` rewrites only the generated blocks. This is what lets one file
answer all three questions: what ships, what it came from, and how to remake it.

```json
{
  "schema_version": 2,
  "pocket_tts": ">=2.0.0,<3.0.0",
  "verified_versions": ["2.0.0", "2.1.0"],
  "built_with": "2.1.0",
  "preview_text": "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
  "assets": {
    "repo_id": "D1zzl3D0p/kenkui-voices",
    "repo_type": "dataset",
    "revision": "<sha of the commit carrying compiled/ and previews/>"
  },
  "voices": [
    {
      "voice_id": "boone-m-ears-p091-american",
      "display_name": "Boone",
      "dataset": "EARS",
      "speaker_id": "P091",
      "gender": "Male",
      "accent": "American",
      "language": "english",
      "pool_enabled": true,
      "license_id": "CC-BY-NC-4.0",
      "commercial_use_allowed": false,
      "voice_rights": "Derived from the EARS corpus. Treat as research-only/noncommercial unless your own review of the source terms concludes otherwise.",
      "source": {
        "repo_id": "kyutai/tts-voices",
        "path": "ears/p091/freeform_speech_01_enhanced.wav",
        "sha256": "…"
      },
      "compiled": {
        "path": "compiled/boone-m-ears-p091-american.safetensors",
        "sha256": "…",
        "size_bytes": 7472376
      },
      "preview": {
        "path": "previews/boone-m-ears-p091-american.wav",
        "sha256": "…",
        "duration_ms": 7520,
        "text": "It is a truth universally acknowledged, …"
      }
    }
  ]
}
```

`pocket_tts` is the range resolution matches against. `verified_versions`
records which releases were actually compiled and hash-checked, so a widened
range carries its own provenance. `built_with` is the release that produced the
bytes currently published.

Dropped from the old schema, each because it was constant across all 95 records
or is now expressed elsewhere: `origin` (`"kenkui_compiled"`), `asset_kind`
(`"safetensors"`), `status` (`"available"`), `voice_pack_format_version`
(superseded by `schema_version`), `pocket_tts_version` (superseded by the range
plus `built_with`), and the top-level `path` / `sha256` / `size_bytes` (moved
into `compiled`).

Within `source`, `repo_id` and `path` are hand-curated; `sha256` is not. It is
the hash of the upstream WAV as fetched at the locked revision, recorded by the
first rebuild that resolves the file and thereafter treated as an assertion. It
exists so a later rebuild that silently reads different input audio fails loudly
instead of publishing a different voice under the same ID.

**`assets.revision` is self-referential and cannot be written in one commit.**
`publish.py` resolves it in two: commit 1 uploads `compiled/` and `previews/`,
commit 2 writes `voices.json` with `assets.revision` set to commit 1's SHA. The
field pins the binaries, which is what needs pinning; the manifest commit is
always its descendant. §6 makes this explicit.

### 4.2 `sources.lock.json`

```json
{
  "schema_version": 1,
  "sources": [
    {
      "repo_id": "kyutai/tts-voices",
      "repo_type": "model",
      "revision": "323332d33f997de8394f24a193e1a76df720e01a"
    }
  ]
}
```

One entry today. It is a list because the prompt corpus is the kind of input
that acquires a second source, and a list costs nothing now.

### 4.3 `builtin.json` — the local manifest, in kenkui-v2

The 26 `CatalogEntry` objects in `registry.py` move to
`src/kenkui/voices/builtin.json`, using the same per-voice field names as
§4.1 minus the blocks that only apply to compiled assets.

```json
{
  "schema_version": 1,
  "embedding": {
    "repo_id": "kyutai/pocket-tts-without-voice-cloning",
    "revision": "e041936c75475d350b405bc870bcf7c22da4e9e6"
  },
  "voices": [
    {
      "voice_id": "anna",
      "display_name": "Anna",
      "language": "english",
      "origin_url": "hf://kyutai/tts-voices/vctk/p228_023_enhanced.wav",
      "license_id": "CC-BY-4.0",
      "commercial_use_allowed": false,
      "voice_rights": "Derived from the VCTK corpus via kyutai/tts-voices. …",
      "perceived_gender": "feminine"
    }
  ]
}
```

Built-ins have no `compiled` block: their embedding URL derives from
`embedding.repo_id`, the voice's `language`, and its `voice_id`, exactly as
`embedding_url()` does today. Extracting them is what makes the merge in §5 a
merge of two documents with one schema rather than JSON folded into Python.

This file is hand-maintained, tracked in kenkui-v2, and lists only voices
pocket-tts ships. It is deliberately not on the dataset: the dataset publishes
voices I compile, and these are not those.

## 5. Consumer changes in kenkui-v2

**Vendor `voices.json` as `src/kenkui/voices/pack.json`,** replacing
`manifest.json`. `load_pack()` keeps reading it as static package data with no
network, and keeps degrading to the built-in catalog when it is absent or
unreadable. That is what makes `list_voices` work offline and the adapter
fail-closed; it is preserved unchanged.

**Delete `PACK_REVISION` and `_PACK_REPO`.** Both become data:
`assets.revision` and `assets.repo_id`. A constant in Python source is what let
the pack and the runtime diverge in the first place.

**Resolve instead of assuming.** On load, `registry.py` compares the installed
pocket-tts against the manifest's `pocket_tts` range using
`packaging.specifiers.SpecifierSet`. No intersection raises
`VoiceError(ErrorCode.VOICE_INCOMPATIBLE)` naming the installed version, the
range the pack declares, and the rebuild command. Declare `packaging` as a
direct dependency — it is already present transitively, and `pyproject.toml`
already uses that rationale for `pyyaml`.

**Merge two documents.** `_catalog()` loads `builtin.json` and `pack.json` and
merges them with today's semantics, unchanged: built-ins win on ID collision,
and a pack voice whose lowercased `display_name` would shadow a built-in keeps
its full slug instead, because those are different speakers who happen to share
a name. `BUILT_IN_CATALOG` stays addressable so the upstream-drift guard still
compares against what kyutai ships rather than the merged catalog.

**Nothing else moves.** Provisioning, the pocket adapter, and the voice pool
consume `CATALOG` and `asset_url()`, whose shapes do not change.

## 6. Tooling

All three run from `tools/` with `uv run`, against a Hugging Face account that
has accepted the gated `kyutai/pocket-tts` terms.

**`rebuild.py [--voice ID]… [--all] [--check]`**
1. Read `voices.json` and `sources.lock.json`.
2. `hf download` the referenced prompt WAVs at the locked revision. Where
   `source.sha256` is present, verify and abort on mismatch; where it is absent,
   record it.
3. Compile each with the installed pocket-tts; render a preview from
   `preview_text`.
4. Hash both outputs; rewrite the `compiled` and `preview` blocks, fill any
   missing `source.sha256`, and set `built_with` to the installed version.
5. `--check` does steps 1–3 in a temporary directory and diffs against the
   recorded hashes without writing anything. Exit non-zero on any mismatch.

**`verify.py <version>`** — run when a new pocket-tts appears, not in CI.
Creates an isolated environment at that version, compiles one VCTK and one EARS
voice (the two corpora take different prompt shapes), and diffs against
`voices.json`. All match → print the `verified_versions` edit that widens the
range. Any differ → report which and exit non-zero, because semver is a promise
about API, not artifact format.

**`publish.py`** — upload `compiled/` and `previews/` as commit 1, capture its
SHA, write it to `assets.revision`, upload `voices.json` as commit 2, and print
the vendoring command for kenkui-v2.

## 7. Failure behavior

| Condition | Behavior |
| --- | --- |
| Installed pocket-tts outside `pocket_tts` | `VoiceError(VOICE_INCOMPATIBLE)` naming both versions and the rebuild command |
| `pack.json` missing or unreadable | Degrade to `builtin.json`; the pack is an addition, not a requirement |
| `builtin.json` missing or unreadable | Import fails. It is package data and its absence means a broken install |
| Asset SHA-256 mismatch on download | `VoiceError`; do not load |
| Upstream prompt hash differs at rebuild | Abort before compiling; the input changed under a pinned revision |

An unrecognized pocket-tts version resolves to nothing and fails loudly. There
is no fallback to the newest pack: the failure being guarded against is
inaudible to the code and nearly inaudible to a listener.

## 8. Migration

Ordered so that nothing irreversible happens before the replacement is proven.

1. **Back up what is published.** Full `hf download` of
   `D1zzl3D0p/kenkui-voices` to a durable directory outside the Hugging Face
   cache — `voices.json` predecessors, all 95 safetensors, all 95 previews.
   Nothing else in this plan may start until this exists.
2. **Author `voices.json`.** From the published 95-voice source manifest:
   rewrite the 29 absolute paths to `ears/{speaker}/freeform_speech_01_enhanced.wav`,
   nest the curated fields under `source`, and drop the constant fields per §4.1.
3. **Write `sources.lock.json`** pinning `kyutai/tts-voices` at `323332d3…`.
4. **Write `tools/`.** `build_voice_pack.py` from kenkui 2.3.6 and the surviving
   parts of `update_voices.py` are the starting point.
5. **Rebuild all 95** under pocket-tts 2.1.0, producing `compiled/` and
   `previews/`.
6. **Gate.** Assert that the 47 VCTK and 19 EARS voices whose prompts did not
   change reproduce the backed-up `sha256` values exactly. Any mismatch stops
   the migration and is investigated before anything is deleted — it would mean
   either the build is not reproducible or 2.1.0 is not byte-compatible with
   2.0.0, and both invalidate §2.
7. **Only after step 6 passes:** delete the dataset repo, recreate it, and
   `publish.py`.
8. **Update kenkui-v2** per §5: vendor `pack.json`, add `builtin.json`, rewrite
   `registry.py`, delete `PACK_REVISION`.
9. **Verify end to end.** Full test suite, plus rendering a real sample with one
   rebuilt EARS voice and one unchanged VCTK voice and listening to both.
10. **Retire the cache working tree.** Remove the `Repos/tts-voices` symlink and
    the untracked scripts inside the snapshot.

Step 6 is the gate on the migration. Step 7 is the only irreversible step, and
step 1 is its mitigation.

## 9. Testing

Against the vendored files, no network:

- **Schema** — every voice has the required fields, IDs are unique, `compiled`
  and `preview` paths agree with `voice_id`, no nulls in non-nullable fields.
- **Resolution** — a version inside `pocket_tts` resolves and yields the right
  asset URL; one outside raises `VOICE_INCOMPATIBLE` naming both versions.
- **Merge** — a built-in wins over a pack voice with the same ID; a pack voice
  whose display name shadows a built-in keeps its slug; the merged count is
  what the two documents imply.
- **Degradation** — a missing `pack.json` leaves the built-in catalog working.
- **Offline** — importing and listing voices opens no socket.

Round-trip reproducibility (`rebuild.py --check` on two voices) is a local,
on-demand check, not CI: it needs gated weights and a GPU-class compile.

## 10. Out of scope

- **The prompt-enhancement hook.** Compiling from locally enhanced audio
  (Adobe Podcast or a local equivalent) is a later project. §4.1's `source`
  block is where it would attach.
- **kenkui 2.3.6.** Its 431 MB of committed safetensors and its `pack-index.json`
  are untouched. Deleting the dataset invalidates the `b513c6aa` revision it
  pins; this is accepted, since it ships its assets in-package.
- **Backfilling a `pocket-1` pack.** No supported kenkui release depends on 1.x.
- **Changing which voices are in the pack.** Membership stays at 95.
- **Speaker-identity fidelity for the 29 rebuilt voices.** They change by
  design; §2 records why.
