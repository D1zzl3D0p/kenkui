# Voice provisioning and manifest varieties

Date: 2026-08-19
Status: Approved for planning

## 1. Problem

`Pipeline.write()` fails closed with `renderer_unavailable` unless
`KENKUI_POCKET_MANIFEST` names an approved local manifest. No manifest can be
produced without hand-authoring one, so no user has ever rendered anything. The
rendering implementation itself is complete; only activation is missing.

Three further gaps follow from the same root cause:

- `pocket-tts` is an optional extra, so a default install cannot render at all.
- The manifest models a voice as exactly one WAV audio prompt, which cannot
  express kyutai's predefined voice catalog or a distributed speaker embedding.
- Documentation states that no voice has passed the rights gate, which is
  accurate today but blocks the catalog that upstream already ships.

## 2. Upstream facts that shaped this design

Verified against the installed `pocket-tts==2.1.0` wheel.

The wheel contains Python code and twelve YAML configs. It bundles **no model
weights and no audio**. Every asset is fetched at runtime.

`utils/utils.py` defines `_ORIGINS_OF_PREDEFINED_VOICES`, mapping 26 names to
remote audio prompts; `eponine` is
`hf://kyutai/tts-voices/vctk/p262_023_enhanced.wav`. Twenty-one names are
English; `giovanni` (it), `lola` (es), `juergen` (de), `rafael` (pt), and
`estelle` (fr) are one each.

`utils/utils.py:46` defines `get_predefined_voice(language, name)`, returning
`hf://kyutai/pocket-tts-without-voice-cloning/languages/{language}/embeddings/{name}.safetensors`
at a pinned revision. The same names therefore exist as precompiled speaker
embeddings in the **ungated** repository. This is why `eponine` is usable
without accepting the gated model terms.

`models/tts_model.py:204-207` tries the gated `kyutai/pocket-tts` weights and
falls back to `weights_path_without_voice_cloning`, setting
`has_voice_cloning = False`. `tts_model.py:870` then rejects any audio-prompt
conditioning when that flag is false. **A WAV prompt is voice cloning and
requires the gated weights; a precompiled embedding does not.** The restriction
runs opposite to the intuitive reading.

`get_state_for_audio_prompt` has three branches: a `.safetensors` path loaded
via `_import_model_state`; a bare catalog name resolved through
`get_predefined_voice`; and an audio file encoded through Mimi. The bare-name
branch requires `self.origin.is_relative_to(CONFIGS_DIR)`, which never holds in
Kenkui because `_tts/pocket.py` loads from a hashed snapshot config. Kenkui
therefore resolves catalog names itself and never uses that branch.

`export_model_state` is public (`pocket_tts.__all__`) and is used by
pocket-tts's own `convert` command at `main.py:368-372` to turn an audio prompt
into a `.safetensors` file. Compiling a WAV ahead of render time is a supported
upstream flow.

## 3. Decisions

1. Provisioning lives in Kenkui, strictly separated from rendering. Suite spec
   05 already assigns `voices/registry.py` and voice loader/download to core.
2. Kenkui manages a default manifest at
   `~/Library/Caches/kenkui/v1/manifest.json` (macOS) or
   `${XDG_CACHE_HOME:-~/.cache}/kenkui/v1/manifest.json`.
   `KENKUI_POCKET_MANIFEST` becomes an operator override with unchanged
   semantics.
3. Provisioning is explicit and user-initiated. Rendering never downloads,
   never compiles, and never reaches the network.
4. All three varieties are supported: `built-in`, `pre-compiled`, `wav`.
5. Every renderable voice asset is a `.safetensors` embedding. WAVs are
   compiled during provisioning, so the render path has exactly one branch.
6. Built-in voices carry per-voice rights metadata with a conservative default
   of `commercial_use_allowed = false` wherever the source terms are not
   clearly permissive.
7. `pocket-tts==2.1.0` becomes a required dependency.
8. No CLI. Suite spec 00 section 18 assigns user-facing command-line surface to
   the separate `kentui` project, and the specs do not describe a CLI for core.
9. Engines are derived state, never user-chosen. They are provisioned when a
   voice needs one and pruned when the last `loaded` voice referencing them is
   unloaded or removed. There is no engine verb and no engine listing; engine
   data hangs off the voices that pull it in.
10. No bulk operations. Every provisioning verb takes exactly one voice ID, and
    bulk work composes over `list_voices()`. This keeps the blast radius of a
    destructive call visible at the call site and makes arbitrary filtering
    free, at the cost of a comprehension for multi-voice work.
11. `unload_voice` and `remove_voice` stay distinct verbs rather than one verb
    with a `keep_registration` flag. A boolean deciding whether hand-entered
    rights metadata survives is too easy to get wrong, and the two converge for
    built-in voices anyway.

## 4. Public API

```python
load_voice(voice_id: str) -> Voice
```

Makes a voice renderable. Idempotent; a fully provisioned and hash-verified
voice performs no network access. Accepts three cases:

- A built-in catalog name. Implicitly registered, so no prior call is needed.
  Provisions the ungated engine for the voice's language, downloads the
  embedding, hashes it, records a `loaded` entry.
- A registered `wav` voice. Provisions the cloning-capable engine for that
  language if absent, which requires accepted gated terms and local Hugging
  Face authentication, then compiles the prompt via
  `get_state_for_audio_prompt` and `export_model_state`. Missing gated access
  fails with `engine_not_cloning_capable` rather than a torch traceback.
- A registered `pre-compiled` voice. Verifies the hash and links the asset.

```python
add_voice(
    path: str | os.PathLike[str],
    *,
    voice_id: str,
    name: str,
    language: str,
    provenance: str,
    license_id: str,
    commercial_use_allowed: bool,
    voice_rights: str,
) -> Voice
```

Registers a local `.wav` or `.safetensors`. Hashes the file, records rights
metadata, writes a `registered` manifest entry. No model load, no network, no
compilation. Every rights field is required; Kenkui does not infer them.

`voice_id` must not collide with a built-in catalog name; collisions are
rejected with `voice_variety_invalid` rather than shadowing the catalog.
`language` is a pocket-tts config stem such as `english` or `italian`, not an
ISO code, because it selects both the engine config and the embedding path.

```python
unload_voice(voice_id: str) -> Voice
```

Reverts `loaded` to `registered`. Deletes the compiled or downloaded asset,
retains rights metadata so the voice can be reloaded without re-entering it,
and prunes the voice's engine if no other `loaded` voice references it. Engine
pruning is the point: at roughly 225 MB per language engine against roughly
6.5 MB per embedding, unloading without pruning reclaims a small fraction of
the actual footprint.

```python
remove_voice(voice_id: str) -> None
```

Deletes the manifest entry outright, including hand-entered rights metadata,
after performing the same asset deletion and engine pruning as `unload_voice`.
For a built-in voice this is equivalent to `unload_voice`, since catalog
registration cannot be deleted.

```python
list_voices() -> tuple[Voice, ...]
```

Returns every voice Kenkui knows about: the built-in catalog unioned with
manifest entries. No network, and no hashing — asset presence is a `stat`, not
a digest. This is the only enumeration primitive, and all multi-voice work
composes over it:

```python
loaded = [v for v in kk.list_voices() if v.state == "loaded"]
english = [v for v in kk.list_voices() if v.language == "english"]
stale = [v for v in kk.list_voices() if v.state == "missing"]

engines = {v.engine for v in kk.list_voices() if v.state == "loaded"}
disk = sum(e.size_bytes for e in engines) + sum(
    v.asset_bytes or 0 for v in kk.list_voices() if v.state == "loaded"
)
```

```python
Pipeline.assign_voice(voice: str | Voice) -> Pipeline
```

Accepts a resolved `Voice` or a bare ID, and always stores the ID, keeping
`Pipeline` intent-only per core spec section 24.

Invariant: **after `load_voice(x)` returns, `x` is renderable, and nothing else
produces that state.**

### 4.1 Voice state machine

```
(none) --add_voice--> registered --load_voice--> loaded
   ^                      ^  |                      |
   |                      |  +----unload_voice------+
   +----remove_voice------+
```

Built-in voices are implicitly registered by the catalog and therefore start at
`registered` with no manifest entry; they gain one on load. Local voices enter
at `registered` via `add_voice`. A third state, `missing`, is reported by
`list_voices()` when the manifest says `loaded` but the asset file is absent.
`missing` is computed at list time and never persisted, so the manifest schema
carries only `registered` and `loaded`. `load_voice` repairs a `missing` voice.

### 4.2 Voice and Engine types

`Voice` is extended beyond the rights metadata of core spec section 12, which
specifies a minimum rather than a maximum set. It gains `variety`, `state`,
`language`, `asset_bytes`, and `engine`. `engine` is `None` unless the voice is
`loaded`.

```python
@dataclass(frozen=True, slots=True)
class Engine:
    id: str
    language: str
    model_revision: str
    cloning_capable: bool
    size_bytes: int
```

`Engine` is returned only as `Voice.engine`. There is no top-level engine
function, per decision 9.

## 5. Manifest schema

Schema version becomes `kenkui-pocket-production-v2`.

`engine` becomes `engines`, a map keyed by engine ID (the language config
stem). Each entry keeps the existing fields and adds `cloning_capable`.

```json
{
  "schema_version": "kenkui-pocket-production-v2",
  "engines": {
    "english": {
      "model_root": "/abs/path",
      "config_path": "/abs/path/english.yaml",
      "model_revision": "...",
      "package_version": "2.1.0",
      "files": [{"relative_path": "...", "size": 0, "sha256": "..."}],
      "sample_rate_hz": 24000,
      "device": "cpu",
      "timeout_seconds": 300.0,
      "cloning_capable": false
    }
  },
  "voices": {
    "eponine": {
      "variety": "built-in",
      "state": "loaded",
      "name": "Eponine",
      "enabled": true,
      "language": "english",
      "engine_id": "english",
      "asset_path": "/abs/path/voices/english/eponine.safetensors",
      "asset_sha256": "...",
      "provenance": "...",
      "license_id": "...",
      "commercial_use_allowed": false,
      "voice_rights": "...",
      "compatible_model_revisions": ["..."]
    }
  }
}
```

`variety` is one of `built-in`, `pre-compiled`, `wav`. It records origin and
who asserts rights; it does not change the render path.

`state` is `registered` or `loaded`, and selects which exact key set the
validator enforces. `registered` entries carry `source_path` and
`source_sha256` and omit `asset_path`, `asset_sha256`, and
`compatible_model_revisions`. `loaded` entries carry the asset fields;
`wav`-variety `loaded` entries additionally retain `source_path` and
`source_sha256`, recording the reviewed original alongside the compiled
artifact.

`content_fingerprint` is removed as a manifest field. The public
`Voice.content_fingerprint` remains and is populated from `asset_sha256`.

Exact-type and exact-key-set validation, unknown-key rejection, and
absolute-path requirements are unchanged. The file security checks in
`_read_manifest` are unchanged.

## 6. Module layout

`src/kenkui/voices.py` becomes a package:

```
voices/__init__.py    re-exports Voice, Engine, add_voice, load_voice,
                      unload_voice, remove_voice, list_voices
voices/registry.py    static built-in catalog; no network, no I/O
voices/provision.py   the only network-touching module in Kenkui
voices/manifest.py    read, merge, atomic write of the managed manifest
```

`registry.py` holds one entry per built-in voice: ID, display name, language,
upstream origin URL, `license_id`, `commercial_use_allowed`, and a rights
statement. The embedding URL is derived from the language and name with the
revision pinned in Kenkui rather than read from pocket-tts, so a dependency
bump cannot silently change which bytes are fetched.

`manifest.py` writes through a temporary file, `fsync`, and `rename`, at mode
`0600` inside a `0700` directory, so output satisfies `_read_manifest`'s
ownership and permission checks. Concurrent provisioning is serialised with a
lock file beside the manifest.

## 7. Asset integrity

**Trust on first use is the accepted root of trust.** Trust has to root
somewhere, and for downloaded assets it roots at the first fetch of a pinned
upstream revision.

Kenkui cannot ship expected SHA-256 values for embeddings without downloading
every voice for every language first. Provisioning therefore hashes what it
fetches and pins the result in the manifest, which governs every subsequent
render.

`unload_voice` discards the pinned hash along with the asset, so a later
`load_voice` re-establishes trust from upstream rather than re-verifying
against the previously seen bytes. This is deliberate and consistent with the
model above: a reload is a first use. Retaining `asset_sha256` on the
`registered` entry to force byte-identical reloads was considered and rejected
as inconsistent ceremony.

Shipping known-good hashes later is additive hardening, not a redesign.

## 8. Adapter changes

`_tts/production.py` resolves the manifest as: `KENKUI_POCKET_MANIFEST` if set,
otherwise the managed default path, otherwise fail closed. It parses `engines`
as a map, resolves each voice's `engine_id`, validates `variety` and `state`,
and drops `content_fingerprint`.

`_tts/pocket.py`:

- `PocketEngineConfig` gains `voice_variety` and `cloning_capable`, and renames
  `voice_prompt_path` and `voice_prompt_sha256` to `voice_asset_path` and
  `voice_asset_sha256`.
- `semantic_material()` uses the renamed fields and includes `voice_variety`,
  which changes cache key material. No stale cache exists, since nothing has
  rendered.
- `preflight_pocket` validates the asset as a bounded `.safetensors` rather
  than a WAV. `_validate_wav` is retained for `add_voice` source validation.
- `synthesize()` builds the conditioning state once per engine instead of once
  per segment, and has no branch that encodes audio. Workers hold a reusable
  engine and process a batch serially, so per-segment state derivation is pure
  waste.

`pipeline.py`: `assign_voice` accepts `str | Voice`. `write()` verifies the
assigned voice is `loaded` before spawning workers.

The network posture of the render path is unchanged: `HF_HUB_OFFLINE=1`, the
`_deny_remote` allowlist, and the audit hook all remain. Kenkui always passes a
`Path` to `get_state_for_audio_prompt`, and that branch never calls
`download_if_necessary`.

## 9. Error codes

New stable codes:

- `voice_not_provisioned` — registered but not loaded. Message names the
  `load_voice` call that resolves it.
- `voice_unknown` — absent from both catalog and manifest.
- `engine_not_cloning_capable` — a `wav` voice compiled against, or assigned
  to, non-cloning weights.
- `voice_variety_invalid` — unrecognised `variety` or `state`.

`renderer_unavailable` is retained for a genuinely absent manifest.

## 10. Dependency change

`pocket-tts==2.1.0` moves into `dependencies`. The `pocket` extra is retained
as an empty alias so `kenkui[pocket]` installs keep working.

This pulls 17 required transitive dependencies including `torch>=2.5.0`,
`scipy`, `sentencepiece`, `numpy>=2`, `pydantic>=2`, and the unused
`fastapi`/`uvicorn`/`typer`/`python-multipart` demo-server stack. Install size
grows from roughly a few hundred kilobytes to torch scale. This is an accepted,
explicitly revisitable trade; the retained extra is the path back.

## 11. Testing

- Manifest schema validation for both key sets and every rejection path,
  extending the existing `test_production_manifest_branches.py` approach.
- Catalog drift: assert Kenkui's catalog keys equal
  `pocket_tts.utils.utils._ORIGINS_OF_PREDEFINED_VOICES` keys, so an upstream
  bump fails CI rather than degrading to `voice_unknown` at runtime.
- Provisioning against a local fake asset source, with no real network.
- `add_voice` then `load_voice` round trip, asserting the `registered` to
  `loaded` transition and both hashes for `wav`.
- `unload_voice` reverts to `registered`, deletes the asset, and retains every
  rights field.
- `remove_voice` deletes the entry, and on a built-in leaves the catalog entry
  reachable at `registered`.
- Engine pruning: unloading the last voice referencing an engine removes the
  engine entry and its files; unloading a voice while a sibling remains
  `loaded` leaves the engine intact.
- `list_voices()` reports `missing` when a `loaded` asset file is deleted
  underneath, without hashing, and `load_voice` repairs it.
- `list_voices()` union semantics: catalog-only voices appear as `registered`
  with no manifest entry.
- Atomic manifest write and concurrent provisioning.
- `write()` raising `voice_not_provisioned` for a registered-only voice.
- Real download, real compile, and real render remain opt-in behind explicit
  environment variables, matching `tests/test_pocket_tts_real.py`.

## 12. Documentation

- `models-and-voices.md`: largest rewrite. State the gated/ungated distinction
  correctly, replace the blanket EARS caveat with a per-voice rights table
  while keeping that caveat for `jean`, and supersede the "no voice fixture has
  passed the gate" framing for built-ins only.
- `usage.md`: rewrite "Current production gate" around `load_voice`.
- `installation.md`: state the torch-scale install.
- `README.md`: show the `load_voice` line in the example.
- `pocket-tts-adapter.md`: document the single-branch render path and the
  provisioning boundary.

## 13. Open risks

### STILL OPEN: gated-compiled embeddings under ungated weights

Compiling a WAV requires gated cloning-capable weights, and the exported state
contains flow-LM-specific tensors, so it is unclear whether such an embedding is
valid under the ungated model. **This remains unverified.**

An experiment on 2026-08-20 established only the weaker claim that the tensors
load. An embedding compiled with `kyutai/pocket-tts` imported cleanly into
`kyutai/pocket-tts-without-voice-cloning` — a genuinely different weight file,
confirmed by SHA-256 (`473f47d9…` versus `be9c6b48…`, both 219,029,196 bytes) —
and `generate_audio` returned 46,080 finite samples at RMS 0.058.

**That is not evidence of compatibility.** Weight incompatibility in this stack
fails *silently*: a mismatched pairing still produces finite, well-formed,
plausible-looking audio. Every signal the experiment measured — sample count,
finiteness, RMS, dynamic range — would look identical under an incompatible
pairing. The only reliable confirmation is listening to the output, which the
experiment did not do.

Concretely, what is known and what is not:

| Claim | Status |
| --- | --- |
| The embedding imports without error | verified |
| `generate_audio` returns well-formed finite audio | verified |
| The audio is intelligible speech | **not tested** |
| The audio preserves the cloned speaker's voice | **not tested** |
| The pairing is compatible | **unknown** |

Resolving this requires a human listening to output generated from a
gated-compiled embedding under ungated weights, ideally against the same
embedding under the gated weights as a reference.

The design stays conservative regardless, and nothing depends on the answer: the
compiling engine's revision is recorded in the voice's
`compatible_model_revisions`, and the check at `production.py:113-114` rejects a
mismatch. Relaxing that pinning must wait for a listening test.

### Resolved 2026-08-20: first real inference found three renderer defects

The end-to-end run surfaced three pre-existing bugs, all invisible to the
offline suite because every unit test stubs the model loader:

1. **Config paths contradicted themselves.** `_inspect_yaml` rejects absolute
   paths while `_deny_remote` required them, and the allowlist is built from a
   per-engine snapshot temp directory that does not exist when a config is
   written. Declared names are now relative, resolved against the snapshot root.
2. **The allowlist was bypassable.** Only three named modules had
   `download_if_necessary` patched; `pocket_tts.conditioners.text` holds its own
   reference and loads the tokenizer through it, so a stock `hf://` config would
   have reached the network despite the allowlist. Every loaded module holding
   the symbol is now patched, with a post-condition check.
3. **The snapshot root was never resolved.** The per-component symlink check
   compares a candidate against its own resolution, which only means "the final
   component is not a symlink" when ancestors are already resolved. On macOS the
   snapshot lives under `/var`, a symlink to `/private/var`, so every load
   failed.

Verified afterwards: a two-chapter EPUB renders to a 69,941-byte M4B, AAC mono
24 kHz, 7.68 s, both chapter markers correct, decoding with no errors.

### Remaining

Real inference is exercised only in the opt-in suite, not in CI. The tier-1
platform matrix, determinism, cancellation boundaries under real load, cold and
warm timing, and peak RSS remain unmeasured. Perceptual voice quality is
unassessed.

## 14. Out of scope

- A CLI or console script.
- Character voices and multi-voice assignment.
- Shipping known-good asset hashes.
- Making `pocket-tts` optional again.
- Bulk provisioning verbs, per decision 10.
- Any engine verb or engine enumeration, per decision 9.
