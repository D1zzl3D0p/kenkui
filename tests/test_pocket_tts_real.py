"""Opt-in strictly local Pocket-TTS gate; skipped unless every variable is explicit."""
# ruff: noqa: D103, S101

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from kenkui._execution.process_pool import EngineSpecification, render_spawned
from kenkui._tts.pocket import (
    PocketEngineConfig,
    PocketManifestFile,
    VoiceAsset,
    preflight_pocket,
)
from kenkui._tts.protocols import SynthesisTask


@pytest.mark.pocket_real
def test_explicit_approved_local_pocket_install() -> None:
    if os.environ.get("KENKUI_RUN_POCKET_REAL") != "1":
        pytest.skip("set KENKUI_RUN_POCKET_REAL=1 for the strictly local real gate")
    manifest_name = os.environ.get("KENKUI_POCKET_MANIFEST")
    voice_name = os.environ.get("KENKUI_POCKET_VOICE_WAV")
    if not manifest_name or not voice_name:
        pytest.skip(
            "explicit KENKUI_POCKET_MANIFEST and KENKUI_POCKET_VOICE_WAV required"
        )
    assert manifest_name is not None and voice_name is not None
    payload = json.loads(Path(manifest_name).read_text(encoding="utf-8"))
    config = PocketEngineConfig(
        model_root=payload["model_root"],
        config_path=payload["config_path"],
        model_revision=payload["model_revision"],
        package_version=payload["package_version"],
        files=tuple(PocketManifestFile(**item) for item in payload["files"]),
        voices=(
            VoiceAsset(
                path=str(Path(voice_name).resolve(strict=True)),
                sha256=payload["voice_prompt_sha256"],
                variety="wav",
                provenance=payload["voice_provenance"],
                license_id=payload["voice_license_id"],
                rights=payload["voice_rights"],
                commercial_use_allowed=payload["commercial_use_allowed"],
            ),
        ),
        cloning_capable=True,
        sample_rate_hz=payload["sample_rate_hz"],
    )
    preflight_pocket(config)
    task = SynthesisTask(
        "real-gate",
        "real-gate",
        "Local test.",
        config.sample_rate_hz,
        1,
        8 * 1024 * 1024,
    )
    records = list(render_spawned((task,), EngineSpecification.pocket(config), 1, None))
    assert len(records) == 1
    audio = records[0].audio
    assert audio.pcm_s16le
