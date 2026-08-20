# Models, voices, provenance, and rights

Kenkui does not distribute model weights or voice prompts. Installing the
Apache-2.0 Kenkui package does not establish permission to use any separate
model, dataset, recording, or voice. Keep provenance and rights review scoped
to the exact immutable bytes and intended use.

## Which model each voice needs

Upstream publishes two model repositories, and the distinction is the opposite
of the intuitive reading:

| Repository | Gated | What it is for |
| --- | --- | --- |
| `kyutai/pocket-tts` | **yes** | voice cloning from an audio prompt |
| `kyutai/pocket-tts-without-voice-cloning` | no | precompiled speaker embeddings |

**A WAV prompt is voice cloning and requires the gated weights. A built-in or
pre-compiled embedding does not.** Built-in voices therefore work without
accepting any gated terms.

Kenkui compiles WAV prompts into embeddings during provisioning, so the render
path only ever loads an embedding and never performs voice cloning. Compiling
needs the gated weights; rendering the result does not.

## Voice varieties

| Variety | Asset | Engine requirement | Who asserts rights |
| --- | --- | --- | --- |
| `built-in` | embedding from the ungated catalog | any | Kenkui's static catalog |
| `pre-compiled` | embedding you supply | any | you |
| `wav` | audio prompt you supply | cloning-capable weights, at compile time | you |

## Required voice record

A production voice must be explicitly selected and must record all of:

- stable voice ID and display name;
- enabled/disabled state;
- exact asset SHA-256 (`content_fingerprint`);
- language and compatible immutable model revision(s);
- provenance describing origin and custody of the exact recording;
- license identifier and a separate rights statement retained by deployment;
- an explicit boolean commercial-use decision.

Missing metadata is not inferred. `add_voice` requires every rights field
explicitly. Planning fails as `voice_provenance_required`; disabled,
unresolved, and revision-incompatible voices have distinct stable errors.

A hash proves byte identity, not authorship, consent, license scope, or lawful
use. Operators remain responsible for reviewing their actual material and
jurisdiction. Kenkui's fields and checks are operational controls, not legal
advice or a legal conclusion.

## Built-in catalog

Kenkui ships metadata for the 26 predefined voices upstream publishes. The bytes
are downloaded on first `load_voice`; nothing is bundled in the package.

**Every built-in voice ships as `commercial_use_allowed = false`.** None of the
source terms were reviewed by this project, and a conservative default is the
only honest one. Setting a voice commercial is a deployment decision that
follows your own review of the upstream terms, speaker consent, and intended
use. Override it in your own manifest entry.

| Voice | Language | Upstream origin | License ID | Commercial use |
| --- | --- | --- | --- | --- |
| `alba` | english | `kyutai/tts-voices/alba-mackenna/casual.wav` | unreviewed | **no** |
| `anna` | english | `kyutai/tts-voices/vctk/p228_023_enhanced.wav` | CC-BY-4.0 | **no** |
| `azelma` | english | `kyutai/tts-voices/vctk/p303_023_enhanced.wav` | CC-BY-4.0 | **no** |
| `bill_boerst` | english | `kyutai/tts-voices/voice-zero/bill_boerst.wav` | unreviewed | **no** |
| `caro_davy` | english | `kyutai/tts-voices/voice-zero/caro_davy.wav` | unreviewed | **no** |
| `charles` | english | `kyutai/tts-voices/vctk/p254_023_enhanced.wav` | CC-BY-4.0 | **no** |
| `cosette` | english | `kyutai/tts-voices/expresso/ex04-ex02_confused_001_channel1_499s.wav` | CC-BY-NC-4.0 | **no** |
| `eponine` | english | `kyutai/tts-voices/vctk/p262_023_enhanced.wav` | CC-BY-4.0 | **no** |
| `eve` | english | `kyutai/tts-voices/vctk/p361_023_enhanced.wav` | CC-BY-4.0 | **no** |
| `fantine` | english | `kyutai/tts-voices/vctk/p244_023_enhanced.wav` | CC-BY-4.0 | **no** |
| `george` | english | `kyutai/tts-voices/vctk/p315_023_enhanced.wav` | CC-BY-4.0 | **no** |
| `jane` | english | `kyutai/tts-voices/vctk/p339_023_enhanced.wav` | CC-BY-4.0 | **no** |
| `javert` | english | `kyutai/tts-voices/voice-donations/Butter.wav` | unreviewed | **no** |
| `jean` | english | `kyutai/tts-voices/ears/p010/freeform_speech_01_enhanced.wav` | CC-BY-NC-4.0 | **no** |
| `marius` | english | `kyutai/tts-voices/voice-donations/Selfie.wav` | unreviewed | **no** |
| `mary` | english | `kyutai/tts-voices/vctk/p333_023_enhanced.wav` | CC-BY-4.0 | **no** |
| `michael` | english | `kyutai/tts-voices/vctk/p360_023_enhanced.wav` | CC-BY-4.0 | **no** |
| `paul` | english | `kyutai/tts-voices/vctk/p259_023_enhanced.wav` | CC-BY-4.0 | **no** |
| `peter_yearsley` | english | `kyutai/tts-voices/voice-zero/peter_yearsley.wav` | unreviewed | **no** |
| `stuart_bell` | english | `kyutai/tts-voices/voice-zero/stuart_bell.wav` | unreviewed | **no** |
| `vera` | english | `kyutai/tts-voices/vctk/p229_023_enhanced.wav` | CC-BY-4.0 | **no** |
| `estelle` | french_24l | `kyutai/tts-voices/unmute-prod-website/developpeuse-3.wav` | unreviewed | **no** |
| `juergen` | german | `kyutai/pocket-tts/de-DE-juergen.mp3` | unreviewed | **no** |
| `giovanni` | italian | `kyutai/pocket-tts/common_voice_it_36520747-enhanced-v2.mp3` | CC0-1.0 | **no** |
| `rafael` | portuguese | `kyutai/pocket-tts/g-Vi8PgmSY0-enhanced-v2.wav` | unreviewed | **no** |
| `lola` | spanish | `kyutai/pocket-tts/common_voice_es_19762977-enhanced-v2.mp3` | CC0-1.0 | **no** |

Two entries carry corpus-specific caveats that survive any review:

- **`jean`** derives from the EARS corpus. Treat it as
  research-only/noncommercial unless your own review of the applicable source
  terms concludes otherwise.
- **`cosette`** derives from the Expresso dataset, with the same caveat.

These are conservative engineering caveats, not statements about anyone's legal
rights, and not a substitute for reviewing the exact terms yourself.

The catalog is drift-tested against pocket-tts's own predefined-voice map, so an
upstream addition or removal fails the test suite rather than degrading to
`voice_unknown` at runtime.

## Asset integrity

Trust roots at first use. Kenkui cannot ship expected SHA-256 values for
embeddings without downloading every voice for every language first, so
provisioning hashes what it fetches at a pinned upstream revision and pins the
result in the manifest, which governs every subsequent render.

`unload_voice` discards the pinned hash along with the asset, so a later
`load_voice` re-establishes trust from upstream rather than re-verifying against
previously seen bytes. A reload is a first use.

## Provisioning and the render boundary

Provisioning is the only part of Kenkui that reaches the network, and it never
runs during a render. The renderer sets `HF_HUB_OFFLINE=1`, replaces
pocket-tts's downloader with a manifest allowlist, installs a socket audit hook,
and verifies every declared file by size and SHA-256 before loading anything.

A test asserts that no module under `_tts`, `_execution`, `_audio`, `_domain`,
or `_epub` imports the provisioning module.

## Local model and voice storage

Provisioned assets live under `~/Library/Caches/kenkui/v1` (macOS) or
`${XDG_CACHE_HOME:-~/.cache}/kenkui/v1` (Linux), alongside the managed
manifest. A per-language engine is roughly 225 MB; each embedding is roughly
6.5 MB. `unload_voice` and `remove_voice` prune an engine once no loaded voice
references it. Cache schema and location are not public API.
