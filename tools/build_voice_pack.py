"""Build and optionally publish a kenkui compiled voice pack.

Input manifest entries may point at either an existing compiled ``.safetensors``
asset via ``path`` or a prompt source via ``prompt_source``.  Output is a
manifest plus compiled assets/previews suitable for the kenkui runtime catalog.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

import scipy.io.wavfile

from kenkui.voice_registry import PREVIEW_TEXT, VoiceCatalogEntry, validate_manifest
from kenkui.voice_compiler import (
    DEFAULT_VOICE_PACK_LANGUAGE,
    compile_audio_prompt_source,
    migrate_voice_asset,
)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_source_manifest(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    voices = raw.get("voices") if isinstance(raw, dict) else raw
    if not isinstance(voices, list):
        raise ValueError("source manifest must contain a voices list")
    return voices


def _assert_pocket_tts_version(expected: str | None) -> str:
    installed = importlib.metadata.version("pocket-tts")
    if expected is not None and installed != expected:
        raise RuntimeError(f"pocket-tts {expected} is required; found {installed}")
    return installed


def _copy_or_compile(entry: dict[str, Any], compiled_dir: Path, *, source_base: Path) -> Path:
    voice_id = entry["voice_id"]
    destination = compiled_dir / f"{voice_id}.safetensors"
    if entry.get("prompt_source"):
        return compile_audio_prompt_source(
            str(entry["prompt_source"]),
            destination,
            language=DEFAULT_VOICE_PACK_LANGUAGE,
            truncate=True,
        )
    source_path = Path(str(entry.get("path") or ""))
    if not source_path.is_absolute():
        source_path = source_base / source_path
    if source_path.suffix != ".safetensors" or not source_path.exists():
        raise ValueError(f"{voice_id!r} must define prompt_source or an existing .safetensors path")
    return migrate_voice_asset(source_path, destination, language=DEFAULT_VOICE_PACK_LANGUAGE)


def _generate_preview(asset_path: Path, output_path: Path) -> Path:
    try:
        from pocket_tts import TTSModel  # type: ignore
    except Exception as exc:
        raise RuntimeError("pocket_tts.TTSModel is required to generate previews") from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = TTSModel.load_model(language=DEFAULT_VOICE_PACK_LANGUAGE)
    state = model.get_state_for_audio_prompt(str(asset_path))
    audio = model.generate_audio(state, PREVIEW_TEXT, frames_after_eos=2).squeeze()
    scipy.io.wavfile.write(str(output_path), model.sample_rate, audio.cpu().numpy())
    return output_path


def _smoke_test(asset_path: Path) -> None:
    try:
        from pocket_tts import TTSModel  # type: ignore
    except Exception as exc:
        raise RuntimeError("pocket_tts.TTSModel is required for smoke tests") from exc
    model = TTSModel.load_model(language=DEFAULT_VOICE_PACK_LANGUAGE)
    state = model.get_state_for_audio_prompt(str(asset_path))
    audio = model.generate_audio(state, "Smoke test.", frames_after_eos=2)
    if audio.numel() == 0:
        raise RuntimeError("Smoke test generated no audio")


def build_voice_pack(
    source_manifest: Path,
    output_dir: Path,
    *,
    generate_previews: bool = True,
    smoke_test: bool = False,
    pocket_tts_version: str | None = None,
) -> Path:
    build_pocket_tts_version = _assert_pocket_tts_version(pocket_tts_version)
    compiled_dir = output_dir / "compiled"
    preview_dir = output_dir / "previews"
    output_dir.mkdir(parents=True, exist_ok=True)
    compiled_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[VoiceCatalogEntry] = []
    for source_entry in _load_source_manifest(source_manifest):
        asset_path = _copy_or_compile(source_entry, compiled_dir, source_base=source_manifest.parent)
        if smoke_test:
            _smoke_test(asset_path)

        preview = dict(source_entry.get("preview") or {})
        if generate_previews and not preview.get("path"):
            preview_path = _generate_preview(asset_path, preview_dir / f"{source_entry['voice_id']}.wav")
            preview = {
                "text": PREVIEW_TEXT,
                "path": str(preview_path.relative_to(output_dir)),
                "sha256": _hash_file(preview_path),
            }

        entry = VoiceCatalogEntry.from_dict(
            {
                **source_entry,
                "origin": "kenkui_compiled",
                "asset_kind": "safetensors",
                "path": str(asset_path.relative_to(output_dir)),
                "status": "available",
                "pool_enabled": source_entry.get("pool_enabled", True),
                "preview": preview,
                "sha256": _hash_file(asset_path),
                "size_bytes": asset_path.stat().st_size,
            },
            base_dir=output_dir,
        )
        manifest_entries.append(entry)

    manifest_path = output_dir / "manifest.json"
    data = {
        "schema_version": 1,
        "voice_pack_format_version": 2,
        "preview_text": PREVIEW_TEXT,
        "pocket_tts_version": build_pocket_tts_version,
        "voices": [entry.to_manifest_dict(base_dir=output_dir) for entry in manifest_entries],
    }
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate_manifest(manifest_path)
    return manifest_path


def sync_to_huggingface(output_dir: Path, repo_id: str, *, revision: str | None = None) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required to sync a voice pack")
    from huggingface_hub import HfApi

    HfApi(token=token).upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(output_dir),
        revision=revision,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--no-previews", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--pocket-tts-version", help="Require an exact Pocket TTS build version")
    parser.add_argument("--sync-repo")
    parser.add_argument("--revision")
    args = parser.parse_args()

    build_voice_pack(
        args.source_manifest,
        args.output_dir,
        generate_previews=not args.no_previews,
        smoke_test=args.smoke_test,
        pocket_tts_version=args.pocket_tts_version,
    )
    if args.sync_repo:
        sync_to_huggingface(args.output_dir, args.sync_repo, revision=args.revision)


if __name__ == "__main__":
    main()
