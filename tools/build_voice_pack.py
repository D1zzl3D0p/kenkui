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
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io.wavfile

from kenkui.voice_compiler import (
    DEFAULT_VOICE_PACK_LANGUAGE,
    compile_audio_prompt_source,
    migrate_voice_asset,
)
from kenkui.voice_registry import (
    PREVIEW_PHRASE_CATALOG,
    PREVIEW_TEXT,
    PreviewPhrase,
    VoiceCatalogEntry,
    validate_manifest,
)

PreviewGenerator = Callable[[Path, PreviewPhrase], tuple[int, np.ndarray]]
Mp3Encoder = Callable[[Path, Path], None]
MIN_PREVIEW_DURATION_MS = 500
MAX_PREVIEW_TOKENS = 200
PREVIEW_EOS_THRESHOLD = -2.0
MAX_PREVIEW_DURATION_MS = 30 * 1000
PREVIEW_GENERATION_ATTEMPTS = 5
MAX_PREVIEW_CHARACTERS_PER_SECOND = 30
MP3_BITRATE = "96k"


class RetryablePreviewError(ValueError):
    """A stochastic preview-generation failure that may succeed on retry."""


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
        prompt_path = Path(str(entry["prompt_source"]))
        if not prompt_path.is_absolute():
            prompt_path = source_base / prompt_path
        return compile_audio_prompt_source(
            str(prompt_path),
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


def _make_preview_generator() -> PreviewGenerator:
    try:
        from pocket_tts import TTSModel  # type: ignore
    except Exception as exc:
        raise RuntimeError("pocket_tts.TTSModel is required to generate previews") from exc
    model = TTSModel.load_model(
        language=DEFAULT_VOICE_PACK_LANGUAGE,
        eos_threshold=PREVIEW_EOS_THRESHOLD,
    )

    def generate(asset_path: Path, phrase: PreviewPhrase) -> tuple[int, np.ndarray]:
        state = model.get_state_for_audio_prompt(str(asset_path))
        audio = model.generate_audio(
            state,
            phrase.text,
            max_tokens=MAX_PREVIEW_TOKENS,
            frames_after_eos=2,
        ).squeeze()
        return model.sample_rate, audio.detach().cpu().numpy()

    return generate


def _encode_mp3(wav_path: Path, mp3_path: Path) -> None:
    from pydub import AudioSegment

    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    AudioSegment.from_wav(wav_path).set_channels(1).export(
        mp3_path, format="mp3", bitrate=MP3_BITRATE
    )


def validate_preview_audio(sample_rate: int, samples: np.ndarray) -> np.ndarray:
    audio = np.asarray(samples).squeeze()
    if sample_rate <= 0:
        raise ValueError("Preview audio has an invalid sample rate")
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Preview audio is empty or not mono")
    if not np.isfinite(audio).all():
        raise ValueError("Preview audio contains non-finite samples")
    duration_ms = audio.size * 1000 / sample_rate
    if duration_ms < MIN_PREVIEW_DURATION_MS:
        raise RetryablePreviewError("Preview audio is implausibly short")
    if duration_ms > MAX_PREVIEW_DURATION_MS:
        raise RetryablePreviewError("Preview audio is implausibly long")
    peak = float(np.max(np.abs(audio)))
    clipped_fraction = float(np.mean(np.abs(audio) >= 0.999))
    if clipped_fraction > 0.01:
        raise RetryablePreviewError("Preview audio has obvious clipping")
    if peak > 1.0:
        audio = audio * (0.99 / peak)
    return audio.astype(np.float32, copy=False)


def _generate_valid_preview(
    generator: PreviewGenerator,
    asset_path: Path,
    phrase: PreviewPhrase,
) -> tuple[int, np.ndarray]:
    minimum_duration_ms = max(
        MIN_PREVIEW_DURATION_MS,
        len(phrase.text) * 1000 / MAX_PREVIEW_CHARACTERS_PER_SECOND,
    )
    last_error: RetryablePreviewError | None = None
    for _attempt in range(PREVIEW_GENERATION_ATTEMPTS):
        sample_rate, raw_audio = generator(asset_path, phrase)
        try:
            audio = validate_preview_audio(sample_rate, raw_audio)
            if audio.size * 1000 / sample_rate < minimum_duration_ms:
                raise RetryablePreviewError(
                    "Preview audio is implausibly short for the requested phrase"
                )
        except RetryablePreviewError as exc:
            last_error = exc
            continue
        return sample_rate, audio
    assert last_error is not None
    raise ValueError(
        f"Preview audio repeatedly failed validation for {asset_path.stem}/{phrase.phrase_id}"
    ) from last_error


def _write_preview_matrix(
    entries: list[VoiceCatalogEntry],
    output_dir: Path,
    *,
    pocket_tts_version: str,
    preview_generator: PreviewGenerator | None,
    mp3_encoder: Mp3Encoder | None,
) -> dict[tuple[str, str], dict[str, Any]]:
    generator = preview_generator or _make_preview_generator()
    encoder = mp3_encoder or _encode_mp3
    enabled = sorted((entry for entry in entries if entry.pool_enabled), key=lambda item: item.voice_id)
    phrases = sorted(PREVIEW_PHRASE_CATALOG.phrases, key=lambda item: item.phrase_id)
    expected_pairs = {(entry.voice_id, phrase.phrase_id) for entry in enabled for phrase in phrases}
    assets: list[dict[str, Any]] = []
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}

    try:
        for entry in enabled:
            if entry.path is None:
                raise RuntimeError(f"Enabled voice {entry.voice_id!r} has no compiled asset")
            voice_asset_sha256 = _hash_file(entry.path)
            for phrase in phrases:
                wav_path = output_dir / "preview-wav" / entry.voice_id / f"{phrase.phrase_id}.wav"
                mp3_path = output_dir / "previews" / entry.voice_id / f"{phrase.phrase_id}.mp3"
                sample_rate, audio = _generate_valid_preview(generator, entry.path, phrase)
                wav_path.parent.mkdir(parents=True, exist_ok=True)
                mp3_path.parent.mkdir(parents=True, exist_ok=True)
                scipy.io.wavfile.write(wav_path, sample_rate, audio)
                encoder(wav_path, mp3_path)
                if not mp3_path.is_file() or mp3_path.stat().st_size == 0:
                    raise RuntimeError(f"MP3 encoder produced no audio for {entry.voice_id}/{phrase.phrase_id}")
                item = {
                    "voice_id": entry.voice_id,
                    "phrase_id": phrase.phrase_id,
                    "voice_asset_sha256": voice_asset_sha256,
                    "wav_path": wav_path.relative_to(output_dir).as_posix(),
                    "mp3_path": mp3_path.relative_to(output_dir).as_posix(),
                    "sha256": _hash_file(mp3_path),
                    "duration_ms": round(audio.size * 1000 / sample_rate),
                    "size_bytes": mp3_path.stat().st_size,
                    "content_type": "audio/mpeg",
                }
                assets.append(item)
                by_pair[(entry.voice_id, phrase.phrase_id)] = item
    except Exception as exc:
        raise RuntimeError("Failed to generate the complete preview matrix") from exc

    if set(by_pair) != expected_pairs:
        missing = sorted(expected_pairs - set(by_pair))
        raise RuntimeError(f"Failed to generate the complete preview matrix; missing {missing!r}")
    manifest = {
        "schema_version": 1,
        "kenkui_version": importlib.metadata.version("kenkui"),
        "pocket_tts_version": pocket_tts_version,
        "phrase_catalog_version": PREVIEW_PHRASE_CATALOG.version,
        "voice_count": len(enabled),
        "phrase_count": len(phrases),
        "asset_count": len(assets),
        "assets": assets,
    }
    (output_dir / "preview-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return by_pair


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
    preview_generator: PreviewGenerator | None = None,
    mp3_encoder: Mp3Encoder | None = None,
) -> Path:
    build_pocket_tts_version = _assert_pocket_tts_version(pocket_tts_version)
    compiled_dir = output_dir / "compiled"
    output_dir.mkdir(parents=True, exist_ok=True)
    compiled_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "preview-manifest.json").unlink(missing_ok=True)

    manifest_entries: list[VoiceCatalogEntry] = []
    for source_entry in _load_source_manifest(source_manifest):
        asset_path = _copy_or_compile(source_entry, compiled_dir, source_base=source_manifest.parent)
        if smoke_test:
            _smoke_test(asset_path)

        entry = VoiceCatalogEntry.from_dict(
            {
                **source_entry,
                "origin": "kenkui_compiled",
                "asset_kind": "safetensors",
                "path": str(asset_path.relative_to(output_dir)),
                "status": "available",
                "pool_enabled": source_entry.get("pool_enabled", True),
                "preview": dict(source_entry.get("preview") or {}),
                "sha256": _hash_file(asset_path),
                "size_bytes": asset_path.stat().st_size,
            },
            base_dir=output_dir,
        )
        manifest_entries.append(entry)

    if generate_previews:
        preview_assets = _write_preview_matrix(
            manifest_entries,
            output_dir,
            pocket_tts_version=build_pocket_tts_version,
            preview_generator=preview_generator,
            mp3_encoder=mp3_encoder,
        )
        default_id = PREVIEW_PHRASE_CATALOG.default_phrase_id
        manifest_entries = [
            replace(
                entry,
                preview=replace(
                    entry.preview,
                    text=PREVIEW_TEXT,
                    path=preview_assets[(entry.voice_id, default_id)]["wav_path"],
                    sha256=_hash_file(
                        output_dir / preview_assets[(entry.voice_id, default_id)]["wav_path"]
                    ),
                    duration_ms=preview_assets[(entry.voice_id, default_id)]["duration_ms"],
                ),
            )
            if entry.pool_enabled
            else entry
            for entry in manifest_entries
        ]

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
