# Models, voices, provenance, and rights

Kenkui does not distribute or download model weights or voice prompts. Installing
the Apache-2.0 Kenkui package, or the optional `pocket-tts` dependency, does not
establish permission to access or use any separate model, dataset, recording, or
voice. Keep provenance and rights review scoped to the exact immutable bytes and
intended use.

## Required voice record

A production voice must be explicitly selected and must record all of:

- stable voice ID and display name;
- enabled/disabled state;
- exact prompt-byte SHA-256 (`content_fingerprint`);
- language and compatible immutable model revision(s);
- provenance describing origin and custody of the exact recording;
- license identifier and a separate rights statement retained by deployment;
- an explicit boolean commercial-use decision.

Missing metadata is not inferred. Planning fails as `voice_provenance_required`;
disabled, unresolved, or revision-incompatible voices have distinct stable
errors. A hash proves byte identity, not authorship, consent, license scope, or
lawful use. Operators remain responsible for reviewing their actual material and
jurisdiction; Kenkui's fields and checks are operational controls, not legal
advice or a legal conclusion.

## EARS research/noncommercial caveat

If a prompt is derived from the EARS corpus, treat it as
**research-only/noncommercial unless your own review of the applicable source
terms and permissions concludes otherwise**. Record
`commercial_use_allowed=False` for that deployment and do not repurpose it as a
commercial voice merely because the bytes pass technical checks. This is a
conservative engineering caveat, not a statement about anyone's legal rights or
a substitute for reviewing the exact EARS terms, speaker consent, and intended
use.

No EARS audio is bundled with Kenkui and no project-owned redistributable voice
fixture has passed the real gate.

## Pocket package evidence versus inference evidence

The optional adapter is pinned to `pocket-tts==2.1.0`. Inspection of the
non-gated PyPI wheel established the local API used by the adapter and its Python
metadata; model-free import probes succeeded on one macOS arm64 machine with
CPython 3.12.13 and 3.13.15. A generated FFmpeg AAC/M4B probe also succeeded
there. These observations do not prove real inference, model/voice licensing,
spawned provider behavior, resource use, determinism, or the tier-1 platform
matrix.

The upstream default model is gated. Its terms were not accepted, it was not
accessed/downloaded, and a denied command was not retried. No real Pocket
inference was run. Retained non-secret spike evidence under
`spikes/pocket_tts_m1/` documents package inspection and the generated FFmpeg
probe only.

## Explicit unresolved real gate

The runtime activation file is an absolute path supplied only through
`KENKUI_POCKET_MANIFEST`. It must be an owner-controlled regular file (no
symlink/hardlink or group/world write), at most 1 MiB, containing exactly a
`kenkui-pocket-production-v1` `schema_version`, `engine`, and non-empty `voices`
map keyed by assigned voice ID. Engine fields are `model_root`, `config_path`,
`model_revision`, `package_version`, `files` (exact relative path, size, SHA-256),
`sample_rate_hz`, `device`, and `timeout_seconds`. Voice fields are `name`,
`enabled`, `provenance`, `license_id`, `commercial_use_allowed`, `language`,
`content_fingerprint`, `compatible_model_revisions`, `voice_prompt_path`,
`voice_prompt_sha256`, and `voice_rights`. Unknown keys, wrong exact JSON types,
relative asset paths, unknown IDs, and revision mismatches fail closed.

Valid activation binds FFmpeg and a private per-segment cache at
`~/Library/Caches/kenkui/v1` (macOS) or
`${XDG_CACHE_HOME:-~/.cache}/kenkui/v1` (Linux). Cache schema/location controls
are not public API. No secrets or asset paths are logged.

Real Pocket production is **NOT APPROVED / UNVERIFIED** pending all of:

1. an approved immutable model repository revision and complete local file
   manifest (relative paths, exact sizes, SHA-256 values, config, and weights);
2. reviewed model terms, license, and attribution obligations;
3. an authorized exact local voice fixture with complete provenance, license ID,
   rights statement, intended-use decision, checksum, and revision compatibility;
4. successful offline inference and spawned-child acceptance on every claimed
   tier-1 platform/Python combination, including PCM contract, determinism,
   failures, cancellation boundaries, cold/warm timing, and peak RSS; and
5. a reviewed pinned dependency/model manifest incorporating those results.

Without explicit approved assets the public renderer remains fail-closed.
Supplying a technically valid manifest activates the path but does not itself
satisfy this approval gate. Package imports, unit tests using doubles, and native
FFmpeg tests do not satisfy it.
