"""Perceived gender is sourced or absent, never inferred.

The gendered casting method filters on this trait. A voice whose trait is
unknown must not silently join a gendered pool, because the display names in
this catalog are Kenkui's own inventions and carry no information about the
speaker.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

import kenkui as kk
from kenkui._tts.production import default_cache_root as _cache_root

if TYPE_CHECKING:
    from pathlib import Path
from kenkui.voices.registry import CATALOG
from kenkui.voices.types import Voice


def test_voice_carries_the_trait() -> None:
    """The casting pool filter reads this field off Voice."""
    voice = Voice(
        id="v",
        name="V",
        enabled=True,
        provenance="p",
        license_id="CC0-1.0",
        commercial_use_allowed=False,
        perceived_gender="feminine",
    )
    assert voice.perceived_gender == "feminine"


def test_trait_defaults_to_unknown() -> None:
    """Absent is the honest default; a wrong guess is worse than no answer."""
    voice = Voice(
        id="v",
        name="V",
        enabled=True,
        provenance="p",
        license_id="CC0-1.0",
        commercial_use_allowed=False,
    )
    assert voice.perceived_gender is None


@pytest.mark.parametrize("entry_id", sorted(CATALOG))
def test_every_catalog_entry_declares_the_trait_explicitly(entry_id: str) -> None:
    """No entry may leave the field undeclared and drift into a pool by accident."""
    assert CATALOG[entry_id].perceived_gender in {"feminine", "masculine", None}


def test_catalog_traits_are_unsourced_for_now() -> None:
    """The upstream corpora ship speaker metadata this catalog cannot reach.

    kyutai's VCTK_Voice_Names.csv covers a different speaker selection than
    these entries, and speaker-info.txt is only distributed inside the full
    corpus download. Until a sourced trait exists, None is the correct value,
    and the pre-compiled voice pack is where sourced traits will come from.
    """
    assert all(entry.perceived_gender is None for entry in CATALOG.values())


_HEADER = b'{"v":{}}'
_EMBEDDING_BYTES = len(_HEADER).to_bytes(8, "little") + _HEADER + b"\0" * 16


def test_added_voice_round_trips_its_trait(tmp_path: Path) -> None:
    """A user-supplied voice must keep its trait across a manifest write."""
    source = tmp_path / "narrator.safetensors"
    source.write_bytes(_EMBEDDING_BYTES)
    added = kk.add_voice(
        source,
        voice_id="house_narrator",
        name="House Narrator",
        language="english",
        provenance="Recorded in-house with documented consent",
        license_id="CC0-1.0",
        commercial_use_allowed=True,
        voice_rights="Cleared for this project.",
        perceived_gender="masculine",
    )
    assert added.perceived_gender == "masculine"
    listed = {voice.id: voice for voice in kk.list_voices()}
    assert listed["house_narrator"].perceived_gender == "masculine"


def test_trait_is_optional_so_older_manifests_stay_readable(tmp_path: Path) -> None:
    """Absent means unsourced, not corrupt: no schema bump, no re-provisioning."""
    source = tmp_path / "plain.safetensors"
    source.write_bytes(_EMBEDDING_BYTES)
    kk.add_voice(
        source,
        voice_id="plain_voice",
        name="Plain",
        language="english",
        provenance="Recorded in-house with documented consent",
        license_id="CC0-1.0",
        commercial_use_allowed=True,
        voice_rights="Cleared for this project.",
    )
    manifest = json.loads((_cache_root() / "manifest.json").read_text())
    assert "perceived_gender" not in manifest["voices"]["plain_voice"]
    listed = {voice.id: voice for voice in kk.list_voices()}
    assert listed["plain_voice"].perceived_gender is None
