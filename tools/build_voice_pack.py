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
import sys
from pathlib import Path
from typing import Any, NoReturn

VOICE_PACK_FORMAT_VERSION = 2
SCHEMA_VERSION = 1
PREVIEW_TEXT = (
    "The rain in Spain stays mainly in the plain. "
    "How wonderful it is to simply speak and be heard."
)
PROMPT_REPO = "kyutai/tts-voices"

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


def render_preview(model: Any, compiled: Path, destination: Path) -> None:
    """Render the preview line in the compiled voice.

    Written with the standard library rather than soundfile: a preview is
    cosmetic, and a mono 16-bit WAV needs no dependency to produce.
    """
    import struct
    import wave

    destination.parent.mkdir(parents=True, exist_ok=True)
    state = model.get_state_for_audio_prompt(compiled)
    audio = model.generate_audio(state, PREVIEW_TEXT)
    samples = audio.detach().cpu().flatten().tolist()
    frames = b"".join(
        struct.pack("<h", round(max(-1.0, min(float(value), 1.0)) * 32767.0))
        for value in samples
    )
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(model.sample_rate))
        handle.writeframes(frames)


def digest(path: Path) -> str:
    """Return the SHA-256 of a file, read in bounded chunks."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def pack_entry(
    voice: dict[str, Any], compiled: Path, *, preview: str | None
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
        "status": "ok",
        "license_id": license_id,
        "commercial_use_allowed": False,
        "voice_rights": rights,
    }
    if preview is not None:
        entry["preview"] = preview
    return entry


def build(
    source_manifest: Path,
    output_dir: Path,
    *,
    previews: bool,
    force: bool,
) -> int:
    """Compile every voice and write the pack manifest. Returns an exit code."""
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

        preview_name: str | None = None
        if previews:
            preview = preview_dir / f"{voice_id}.wav"
            if not preview.is_file() or force:
                try:
                    render_preview(model, compiled, preview)
                except Exception as error:
                    # A missing preview is cosmetic; the voice still renders.
                    print(f"    preview failed: {type(error).__name__}: {error}")
            if preview.is_file():
                preview_name = f"previews/{preview.name}"
        entries.append(pack_entry(voice, compiled, preview=preview_name))

    from importlib import metadata

    manifest = {
        "pocket_tts_version": metadata.version("pocket-tts"),
        "preview_text": PREVIEW_TEXT,
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
        "--force",
        action="store_true",
        help="recompile everything, as a pocket-tts version bump requires",
    )
    args = parser.parse_args()
    return build(
        args.source_manifest,
        args.output_dir,
        previews=not args.no_previews,
        force=args.force,
    )


if __name__ == "__main__":
    raise SystemExit(main())
