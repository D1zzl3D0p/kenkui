#!/usr/bin/env python3
"""Compile the Kenkui voice pack from source WAV prompts.

Reads a source manifest describing each voice, compiles every prompt into a
`.safetensors` speaker embedding against the *installed* pocket-tts, optionally
renders a preview, and writes the pack manifest.

    python build_voice_pack.py kenkui_source_manifest.json OUT_DIR [--no-previews]

Invoked this way by `update_voices.py` in the tts-voices working directory.

Compiling a WAV prompt is voice cloning, which requires the gated
`kyutai/pocket-tts` weights. Accept the terms once and authenticate with
Hugging Face; without them pocket-tts silently falls back to the ungated
weights, refuses audio-prompt conditioning, and the failure surfaces far from
its cause. This script checks up front instead.

Rights metadata is derived per source dataset and recorded on every voice.
Kenkui refuses to render a voice that carries none, and `commercial_use_allowed`
stays false throughout: these corpora require the operator's own review, and a
conservative default is the only safe one to ship.

Re-running skips voices whose compiled asset already exists, so an interrupted
build resumes. Pass --force to recompile everything, which is what a pocket-tts
version bump calls for.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, NoReturn

VOICE_PACK_FORMAT_VERSION = 2
SCHEMA_VERSION = 1
# Matches the text the published pack was built with, so a rebuild does not
# silently change every preview. Override with --preview-text.
PREVIEW_TEXT = (
    "It is a truth universally acknowledged, that a single man in possession "
    "of a good fortune, must be in want of a wife."
)
PROMPT_REPO = "kyutai/tts-voices"
PACK_REPO = "D1zzl3D0p/kenkui-voices"

# Preview generation is stochastic and fails in a specific, silent way: the
# model runs to its generation limit without emitting EOS and returns audio
# that is well formed but contains no words. Duration is the cheap signal for
# it, so a preview far outside the plausible range for its phrase is retried
# rather than written.
PREVIEW_ATTEMPTS = 5
MIN_PREVIEW_MS = 500
MAX_PREVIEW_MS = 30_000
MAX_PREVIEW_CHARS_PER_SECOND = 30
# Full-scale in the float domain, and the share of it a real recording may hit
# before the result is distortion rather than loudness.
CLIP_LEVEL = 0.999
MAX_CLIPPED_FRACTION = 0.01

# Per-corpus terms. Both stay non-commercial by default: the licences differ,
# but neither substitutes for the operator's own review of speaker consent.
_RIGHTS: dict[str, tuple[str, str]] = {
    "VCTK": (
        "CC-BY-4.0",
        (
            "Derived from the VCTK corpus via kyutai/tts-voices. Review the "
            "VCTK terms and speaker consent for your intended use before "
            "commercial deployment."
        ),
    ),
    "EARS": (
        "CC-BY-NC-4.0",
        (
            "Derived from the EARS corpus. Treat as research-only or "
            "noncommercial unless your own review of the source terms "
            "concludes otherwise."
        ),
    ),
}


def fail(message: str) -> NoReturn:
    """Exit with a message that names the fix rather than the symptom."""
    sys.exit(f"ERROR: {message}")


def load_source(path: Path) -> list[dict[str, Any]]:
    """Read the source manifest and check the fields this build depends on."""
    if not path.is_file():
        fail(f"source manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    voices = payload.get("voices") if isinstance(payload, dict) else payload
    if not isinstance(voices, list) or not voices:
        fail(f"{path} contains no voices list")
    required = {"voice_id", "display_name", "dataset", "gender", "prompt_source"}
    for voice in voices:
        missing = required - set(voice)
        if missing:
            fail(f"voice {voice.get('voice_id', '?')} is missing {sorted(missing)}")
        if voice["dataset"] not in _RIGHTS:
            fail(
                f"voice {voice['voice_id']} has dataset {voice['dataset']!r}, "
                f"which has no recorded terms. Add it to _RIGHTS with the "
                f"corpus licence before shipping the voice."
            )
    return voices


def load_model() -> Any:
    """Load cloning-capable pocket-tts weights, or explain why they are absent.

    pocket-tts falls back to the ungated weights when the gated repository is
    unavailable and only refuses the audio prompt later, so the useful check is
    the capability flag right after loading.
    """
    try:
        from pocket_tts.models.tts_model import TTSModel
    except ImportError:  # pragma: no cover - environment problem, not logic
        fail("pocket-tts is not installed in this interpreter")
    model = TTSModel.load_model(language="english")
    if not getattr(model, "has_voice_cloning", False):
        fail(
            "loaded pocket-tts weights cannot clone voices, so no prompt can be "
            "compiled. Accept the terms at https://huggingface.co/kyutai/"
            "pocket-tts and authenticate, then re-run."
        )
    return model


def prompt_path(prompt_source: str) -> Path:
    """Resolve one prompt WAV, downloading it from the voices repo if needed."""
    local = Path(prompt_source)
    if local.is_file():
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(PROMPT_REPO, prompt_source))


def compile_voice(model: Any, source: Path, destination: Path) -> None:
    """Compile one prompt into a speaker embedding."""
    from pocket_tts import export_model_state

    destination.parent.mkdir(parents=True, exist_ok=True)
    state = model.get_state_for_audio_prompt(source)
    export_model_state(state, destination)


class RetryablePreviewError(ValueError):
    """A stochastic preview failure that a fresh generation may not repeat."""


def check_preview(samples: list[float], rate: int, text: str) -> None:
    """Reject a preview that cannot plausibly be the requested phrase."""
    if not samples:
        message = "preview audio is empty"
        raise RetryablePreviewError(message)
    if not all(math.isfinite(value) for value in samples):
        message = "preview audio contains non-finite samples"
        raise RetryablePreviewError(message)
    duration_ms = len(samples) * 1000 / rate
    # Speech cannot outrun this rate, so audio too short for its phrase is
    # truncated and audio far too long ran past the limit without EOS.
    floor_ms = max(MIN_PREVIEW_MS, len(text) * 1000 / MAX_PREVIEW_CHARS_PER_SECOND)
    if duration_ms < floor_ms:
        message = f"preview is {duration_ms:.0f}ms, too short for {len(text)} chars"
        raise RetryablePreviewError(message)
    if duration_ms > MAX_PREVIEW_MS:
        message = f"preview is {duration_ms:.0f}ms, likely generated without EOS"
        raise RetryablePreviewError(message)
    clipped = sum(1 for value in samples if abs(value) >= CLIP_LEVEL)
    if clipped / len(samples) > MAX_CLIPPED_FRACTION:
        message = "preview audio has obvious clipping"
        raise RetryablePreviewError(message)


def render_preview(
    model: Any, compiled: Path, destination: Path, text: str
) -> int:
    """Render the preview line in the compiled voice, returning its duration.

    Written with the standard library rather than soundfile: a preview is
    cosmetic, and a mono 16-bit WAV needs no dependency to produce.
    """
    import struct
    import wave

    destination.parent.mkdir(parents=True, exist_ok=True)
    state = model.get_state_for_audio_prompt(compiled)
    rate = int(model.sample_rate)
    last: RetryablePreviewError | None = None
    for _attempt in range(PREVIEW_ATTEMPTS):
        audio = model.generate_audio(state, text)
        samples = audio.detach().cpu().flatten().tolist()
        try:
            check_preview(samples, rate, text)
        except RetryablePreviewError as error:
            last = error
            continue
        break
    else:
        message = f"preview repeatedly failed validation: {last}"
        raise ValueError(message)
    frames = b"".join(
        struct.pack("<h", round(max(-1.0, min(float(value), 1.0)) * 32767.0))
        for value in samples
    )
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames)
    return len(samples) * 1000 // rate


def digest(path: Path) -> str:
    """Return the SHA-256 of a file, read in bounded chunks."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def pack_entry(
    voice: dict[str, Any], compiled: Path, *, preview: dict[str, Any] | None
) -> dict[str, Any]:
    """Build one manifest entry, carrying rights alongside identity."""
    license_id, rights = _RIGHTS[voice["dataset"]]
    entry: dict[str, Any] = {
        "voice_id": voice["voice_id"],
        "display_name": voice["display_name"],
        "dataset": voice["dataset"],
        "speaker_id": voice.get("speaker_id"),
        "gender": voice["gender"],
        "accent": voice.get("accent"),
        "pool_enabled": bool(voice.get("pool_enabled", True)),
        "asset_kind": "safetensors",
        "origin": "kenkui_compiled",
        "path": f"compiled/{compiled.name}",
        "sha256": digest(compiled),
        "size_bytes": compiled.stat().st_size,
        "status": "available",
        "license_id": license_id,
        "commercial_use_allowed": False,
        "voice_rights": rights,
    }
    if preview is not None:
        entry["preview"] = preview
    return entry


def require_version(expected: str | None) -> str:
    """Return the installed pocket-tts version, asserting it if one was named.

    A pack records the version it was compiled against, and a compiled
    embedding is only meaningfully pinned if that record is true. Passing the
    expected version turns a silent mismatch into a refusal.
    """
    from importlib import metadata

    installed = metadata.version("pocket-tts")
    if expected is not None and installed != expected:
        fail(f"pocket-tts {expected} required, but {installed} is installed")
    return installed


def sync_to_huggingface(output_dir: Path, repo_id: str) -> str:
    """Upload the built pack and return the resulting revision.

    Uses the library rather than the `hf` command, which is not always
    installed, and returns the revision so it can be pinned in the registry.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Rebuild voice pack",
    )
    return str(api.repo_info(repo_id, repo_type="dataset").sha)


def build(  # noqa: PLR0913 - one call site; each flag is an independent knob.
    source_manifest: Path,
    output_dir: Path,
    *,
    previews: bool,
    force: bool,
    preview_text: str = PREVIEW_TEXT,
    pocket_tts_version: str | None = None,
) -> int:
    """Compile every voice and write the pack manifest. Returns an exit code."""
    # Asserted before any compiling, so a wrong interpreter costs no work.
    require_version(pocket_tts_version)
    voices = load_source(source_manifest)
    compiled_dir = output_dir / "compiled"
    preview_dir = output_dir / "previews"
    compiled_dir.mkdir(parents=True, exist_ok=True)

    pending = [
        voice
        for voice in voices
        if force or not (compiled_dir / f"{voice['voice_id']}.safetensors").is_file()
    ]
    print(f"{len(voices)} voices, {len(pending)} to compile")

    # Loading the gated weights is the expensive step and the one most likely to
    # fail, so skip it entirely when every asset is already present.
    model = load_model() if pending or previews else None

    entries: list[dict[str, Any]] = []
    for index, voice in enumerate(voices, start=1):
        voice_id = voice["voice_id"]
        compiled = compiled_dir / f"{voice_id}.safetensors"
        if not compiled.is_file() or force:
            print(f"[{index}/{len(voices)}] compiling {voice_id}")
            try:
                compile_voice(model, prompt_path(voice["prompt_source"]), compiled)
            except Exception as error:
                # One bad voice must not abandon the other ninety-four; the
                # manifest simply records the gap.
                print(f"    FAILED: {type(error).__name__}: {error}")
                continue
        else:
            print(f"[{index}/{len(voices)}] reusing {voice_id}")

        preview_entry: dict[str, Any] | None = None
        if previews:
            preview = preview_dir / f"{voice_id}.wav"
            duration: int | None = None
            if not preview.is_file() or force:
                try:
                    duration = render_preview(model, compiled, preview, preview_text)
                except Exception as error:
                    # A missing preview is cosmetic; the voice still renders.
                    print(f"    preview failed: {type(error).__name__}: {error}")
            if preview.is_file():
                preview_entry = {
                    "path": f"previews/{preview.name}",
                    "sha256": digest(preview),
                    "text": preview_text,
                }
                if duration is not None:
                    preview_entry["duration_ms"] = duration
        entries.append(pack_entry(voice, compiled, preview=preview_entry))

    manifest = {
        "pocket_tts_version": require_version(pocket_tts_version),
        "preview_text": preview_text,
        "schema_version": SCHEMA_VERSION,
        "voice_pack_format_version": VOICE_PACK_FORMAT_VERSION,
        "voices": entries,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # Copied through so the pack is self-describing: the compiled assets and the
    # inputs that produced them travel together.
    (output_dir / source_manifest.name).write_text(
        source_manifest.read_text(encoding="utf-8"), encoding="utf-8"
    )

    missing = len(voices) - len(entries)
    print(f"\nwrote {output_dir / 'manifest.json'} with {len(entries)} voices")
    print(f"pocket-tts {manifest['pocket_tts_version']}")
    if missing:
        print(f"{missing} voice(s) failed to compile and are absent from the pack")
    return 1 if missing else 0


def main() -> int:
    """Parse arguments and run the build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--no-previews", action="store_true", help="skip preview rendering"
    )
    parser.add_argument(
        "--preview-text",
        default=PREVIEW_TEXT,
        help="line rendered for each preview; changing it re-renders all of them",
    )
    parser.add_argument(
        "--pocket-tts-version",
        help="refuse to build unless exactly this pocket-tts is installed",
    )
    parser.add_argument(
        "--sync-repo",
        nargs="?",
        const=PACK_REPO,
        help=f"upload the built pack; defaults to {PACK_REPO}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompile everything, as a pocket-tts version bump requires",
    )
    args = parser.parse_args()
    code = build(
        args.source_manifest,
        args.output_dir,
        previews=not args.no_previews,
        force=args.force,
        preview_text=args.preview_text,
        pocket_tts_version=args.pocket_tts_version,
    )
    if code == 0 and args.sync_repo:
        revision = sync_to_huggingface(args.output_dir, args.sync_repo)
        print(f"\nuploaded to {args.sync_repo}")
        print(f"pin this in kenkui/voices/registry.py:\n  PACK_REVISION = {revision!r}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
