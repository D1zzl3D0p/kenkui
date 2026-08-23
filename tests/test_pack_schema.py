"""The vendored pack is well formed, self-consistent, and complete."""
# ruff: noqa: D103

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kenkui.voices import registry

_PACK = json.loads(
    Path(registry.__file__).with_name("pack.json").read_text(encoding="utf-8")
)
_REQUIRED = (
    "voice_id",
    "display_name",
    "dataset",
    "gender",
    "language",
    "pool_enabled",
    "license_id",
    "commercial_use_allowed",
    "voice_rights",
    "source",
    "compiled",
    "preview",
)
_EXPECTED_VOICES = 95
_SHA256_LENGTH = 64
_REVISION_LENGTH = 40
_SCHEMA_VERSION = 2


def test_the_document_declares_its_schema_and_range() -> None:
    assert _PACK["schema_version"] == _SCHEMA_VERSION
    assert _PACK["pocket_tts"] == ">=2.0.0,<3.0.0"
    assert _PACK["built_with"] in _PACK["verified_versions"]


def test_assets_are_pinned_to_a_full_revision() -> None:
    assets = _PACK["assets"]
    assert assets["repo_id"] == "D1zzl3D0p/kenkui-voices"
    assert assets["repo_type"] == "dataset"
    assert len(assets["revision"]) == _REVISION_LENGTH


def test_every_voice_is_present_and_published() -> None:
    assert len(_PACK["voices"]) == _EXPECTED_VOICES
    assert all(voice.get("compiled") for voice in _PACK["voices"])


@pytest.mark.parametrize("field", _REQUIRED)
def test_every_voice_carries_the_required_field(field: str) -> None:
    missing = [v["voice_id"] for v in _PACK["voices"] if v.get(field) is None]
    assert not missing


def test_voice_ids_are_unique() -> None:
    ids = [voice["voice_id"] for voice in _PACK["voices"]]
    assert len(set(ids)) == len(ids)


def test_asset_paths_agree_with_their_voice_id() -> None:
    """A path disagreeing with its ID would fetch another voice's bytes."""
    for voice in _PACK["voices"]:
        voice_id = voice["voice_id"]
        assert voice["compiled"]["path"] == f"compiled/{voice_id}.safetensors"
        assert voice["preview"]["path"] == f"previews/{voice_id}.wav"


def test_every_hash_is_a_sha256() -> None:
    for voice in _PACK["voices"]:
        assert len(voice["compiled"]["sha256"]) == _SHA256_LENGTH
        assert len(voice["preview"]["sha256"]) == _SHA256_LENGTH
        assert len(voice["source"]["sha256"]) == _SHA256_LENGTH


def test_no_prompt_resolves_to_a_machine_local_path() -> None:
    """The whole point of the migration: every prompt is upstream."""
    for voice in _PACK["voices"]:
        assert voice["source"]["repo_id"] == "kyutai/tts-voices"
        assert not voice["source"]["path"].startswith("/")


def test_no_voice_is_marked_commercially_usable() -> None:
    """These corpora need the operator's own review; the default stays no."""
    assert all(v["commercial_use_allowed"] is False for v in _PACK["voices"])
