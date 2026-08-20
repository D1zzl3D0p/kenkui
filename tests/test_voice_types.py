"""Public voice and engine metadata types."""
# ruff: noqa: D103

from __future__ import annotations

import dataclasses

import pytest

from kenkui import Engine, Voice

_ASSET_BYTES = 6_500_000


def _voice(**overrides: object) -> Voice:
    base: dict[str, object] = {
        "id": "eponine",
        "name": "Eponine",
        "enabled": True,
        "provenance": "kyutai catalog",
        "license_id": "CC-BY-4.0",
        "commercial_use_allowed": False,
        "language": "english",
        "content_fingerprint": "a" * 64,
        "compatible_model_revisions": ("revision-1",),
        "variety": "built-in",
        "state": "registered",
    }
    return Voice(**(base | overrides))  # type: ignore[arg-type]


def _engine() -> Engine:
    return Engine(
        id="english",
        language="english",
        model_revision="revision-1",
        cloning_capable=False,
        size_bytes=225_000_000,
    )


def test_voice_defaults_are_unloaded() -> None:
    voice = _voice()
    assert voice.asset_bytes is None
    assert voice.engine is None


def test_voice_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _voice().id = "other"  # type: ignore[misc]


def test_engine_is_hashable_for_set_deduplication() -> None:
    engine = _engine()
    assert len({engine, dataclasses.replace(engine)}) == 1


def test_loaded_voice_carries_its_engine() -> None:
    engine = _engine()
    voice = _voice(state="loaded", asset_bytes=_ASSET_BYTES, engine=engine)
    assert voice.engine is engine
    assert voice.asset_bytes == _ASSET_BYTES
