# Pocket-TTS 2.1.0 adapter and offline activation

The Pocket adapter is implemented as a private, fail-closed production binding.
It is not an online model manager and is not automatically selected by the
public pipeline. Install its Python dependency with:

```console
python -m pip install "kenkui[pocket]"
```

This installs the exact inspected adapter dependency, `pocket-tts==2.1.0`; it
does not fetch or authorize a model/voice and does not pass the real gate.

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

## Explicit blocker

The real gate is currently blocked on an **approved model revision plus complete
manifest/checksums** and a **selected authorized local voice with complete
provenance/license/rights metadata**. No approved assets are present, no
model/voice was downloaded, no credentials are used, and no real Pocket inference
runs in CI. The gate must remain separate and opt-in even after assets are
approved; fake/native FFmpeg acceptance cannot be relabeled as real Pocket
acceptance.
