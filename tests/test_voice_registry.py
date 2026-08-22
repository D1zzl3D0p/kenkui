"""The built-in catalog mirrors upstream and records per-voice rights."""
# ruff: noqa: D103
# The upstream predefined-voice map is private, but it is the drift source.

from __future__ import annotations

import pytest
from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

from kenkui import ErrorCode, VoiceError
from kenkui.voices import registry

_NONCOMMERCIAL = ("jean", "cosette")


def test_catalog_covers_every_upstream_voice() -> None:
    """Drift guard: an upstream bump must fail here, not at render time.

    Compares the built-in catalog alone. The merged CATALOG also carries the
    voice pack, which upstream knows nothing about.
    """
    assert set(registry.BUILT_IN_CATALOG) == set(_ORIGINS_OF_PREDEFINED_VOICES)


def test_origin_urls_match_upstream_exactly() -> None:
    for voice_id, entry in registry.BUILT_IN_CATALOG.items():
        assert entry.origin_url == _ORIGINS_OF_PREDEFINED_VOICES[voice_id]


@pytest.mark.parametrize("voice_id", _NONCOMMERCIAL)
def test_research_only_corpora_are_not_commercial(voice_id: str) -> None:
    assert registry.CATALOG[voice_id].commercial_use_allowed is False


def test_every_entry_records_rights() -> None:
    for entry in registry.CATALOG.values():
        assert entry.license_id
        assert entry.voice_rights
        assert isinstance(entry.commercial_use_allowed, bool)


def test_embedding_url_pins_the_revision() -> None:
    url = registry.embedding_url("english", "eponine")
    assert url.startswith("hf://kyutai/pocket-tts-without-voice-cloning/")
    assert url.endswith(f"eponine.safetensors@{registry.EMBEDDING_REVISION}")


def test_catalog_voice_is_registered_and_built_in() -> None:
    voice = registry.catalog_voice("eponine")
    assert voice.id == "eponine"
    assert voice.variety == "built-in"
    assert voice.state == "registered"
    assert voice.language == "english"
    assert voice.engine is None


def test_catalog_voice_rejects_unknown_id() -> None:
    with pytest.raises(VoiceError) as excinfo:
        registry.catalog_voice("nobody")
    assert excinfo.value.code is ErrorCode.VOICE_UNKNOWN
