# Pocket-TTS 2.1.0 adapter and offline activation

The Pocket adapter is a private, fail-closed production binding. It is not an
online model manager: it downloads nothing, compiles nothing, and reads only a
manifest that [provisioning](usage.md#provisioning-voices) wrote.

`pocket-tts==2.1.0` is a required dependency and needs no separate install.
Installing it does not fetch or authorize any model or voice.

## Single-branch render path

Every renderable voice asset is a compiled `.safetensors` speaker embedding.
WAV prompts are compiled during provisioning, so the adapter has no
audio-encoding branch: `preflight_pocket` validates a bounded safetensors
header, and `synthesize` loads a conditioning state and nothing else.

The conditioning state is derived **once per engine**, not once per segment. A
worker holds a reusable engine and processes its batch serially, so
re-deriving per segment repeated a file load for every chunk of text.

The path passed to `get_state_for_audio_prompt` is always a `Path`, never a
`str`. Upstream calls `download_if_necessary` only for `str` input, so a `Path`
cannot reach the network even before the allowlist intervenes.

## Required local manifest

Activation requires an operator-reviewed, immutable local declaration containing:

- canonical absolute owner-controlled model root and selected local YAML path;
- immutable model revision and exact package version `2.1.0`;
- **every** model-tree file as a safe relative POSIX path, exact positive byte
  size, and lowercase SHA-256 (no extra or missing tree entries);
- canonical absolute prompt WAV outside the model root and its SHA-256;
- voice provenance, license identifier, rights statement, and explicit
  `commercial_use_allowed` boolean;
- expected sample rate, CPU device, and bounded timeout; and
- matching public voice content fingerprint, metadata, and compatible model
  revision.

No environment variable, cache entry, provider default, language shortcut, URL,
or Hugging Face identifier can substitute for this declaration. The selected
YAML must itself be in the manifest. Unsafe roots/files, symlinks, hardlinks,
permissions, hashes, sizes, remote markers, path forms, extra files, malformed
WAV, mismatched metadata, missing package, or wrong package version fail closed
with stable Pocket model/voice errors.

## Offline worker behavior

Preflight reads and verifies every declared byte without importing the inference
package. Actual construction is allowed only after the scheduler's spawned-worker
handshake. The child repeats preflight, copies a read-only private snapshot, sets
`HF_HUB_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`, and
`TRANSFORMERS_OFFLINE=1`, denies Python socket audit events, and replaces the
provider downloader with an allow-list that accepts only declared snapshot
files. The voice is passed as a local `Path`.

Each bounded spawned worker creates one private model snapshot and model instance,
then reuses them serially for its assigned segment batch. Voice conditioning is
rebuilt from the verified local prompt for each segment. No model is constructed
in the parent, workers do not share model state, and each worker removes its
snapshot when it exits. Provider exceptions are sanitized as stable load, voice,
inference, or invalid-audio errors.

## Package inspection basis

The adapter targets these inspected 2.1.0 interfaces:

- `TTSModel.load_model(config=<local Path>, quantize=False)`;
- `model.sample_rate` and `model.device`;
- `model.get_state_for_audio_prompt(<local Path>)`; and
- `model.generate_audio(state, text)` returning the tensor subsequently checked
  and converted to bounded mono PCM.

The inspected PyPI wheel SHA-256 was
`7b8f01d3e52aa7df84887b711994586bdc875e024a8b40a15f757feeeb29f752`.
Wheel/source inspection is not inference acceptance.

## Provisioning boundary

Provisioning is the only part of Kenkui that reaches the network, and it never
runs inside a render. `tests/test_import_boundaries.py` statically asserts that
no module under `_tts`, `_execution`, `_audio`, `_domain`, or `_epub` imports
the provisioning module.

Because the renderer replaces `download_if_necessary` with an allowlist
accepting only local absolute paths, a stock pocket-tts config full of `hf://`
URLs would be rejected at load time. Provisioning therefore writes a derived
config with every weight reference rewritten to a local path, and a test
asserts no remote scheme survives into it.

## Remaining gate

Real inference has not been exercised in CI. Unit tests use doubles, and the
end-to-end test that downloads real assets and renders a real M4B is opt-in
behind `KENKUI_RUN_PROVISIONING_REAL=1`. Fake or native FFmpeg acceptance
cannot be relabelled as real Pocket acceptance.

Whether a gated-compiled embedding is valid under the ungated model is **still
unverified**. An embedding compiled with `kyutai/pocket-tts` does import into
`kyutai/pocket-tts-without-voice-cloning` and produce finite audio, but that
proves only that the tensors load. Weight incompatibility here fails silently —
a mismatched pairing produces equally well-formed audio — so confirming this
requires listening to the output, which has not been done.

Kenkui pins conservatively: the compiling engine's revision goes into the
voice's `compatible_model_revisions` and the revision check rejects a mismatch.
That pinning must stay until a listening test settles the question.
