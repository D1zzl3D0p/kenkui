"""Perceived gender is sourced or absent, never inferred.

The gendered casting method filters on this trait. A voice whose trait is
unknown must not silently join a gendered pool, because the display names in
this catalog are Kenkui's own inventions and carry no information about the
speaker.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import kenkui as kk
from kenkui._domain.casting import CharacterProfile, candidates
from kenkui.voices.manifest import ManifestStore, VoiceRecord
from kenkui.voices.provision import _with_catalog_gender
from kenkui.voices.registry import BUILT_IN_CATALOG, CATALOG
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


def test_built_in_traits_match_the_sourced_catalog() -> None:
    """Every Pocket TTS preset keeps its reviewed catalog classification."""
    expected = {
        "alba": "feminine",
        "anna": "feminine",
        "azelma": "feminine",
        "bill_boerst": "masculine",
        "caro_davy": "feminine",
        "charles": "masculine",
        "cosette": "feminine",
        "eponine": "feminine",
        "estelle": "feminine",
        "eve": "feminine",
        "fantine": "feminine",
        "george": "masculine",
        "giovanni": "masculine",
        "jane": "feminine",
        "javert": "masculine",
        "jean": "masculine",
        "juergen": "masculine",
        "lola": "feminine",
        "marius": "masculine",
        "mary": "feminine",
        "michael": "masculine",
        "paul": "masculine",
        "peter_yearsley": "masculine",
        "rafael": "masculine",
        "stuart_bell": "masculine",
        "vera": "feminine",
    }
    assert {
        voice_id: entry.perceived_gender for voice_id, entry in BUILT_IN_CATALOG.items()
    } == expected


def test_built_in_trait_provenance_is_pinned() -> None:
    """The classification must stay auditable instead of becoming folklore."""
    payload = json.loads(
        (Path(__file__).parents[1] / "src/kenkui/voices/builtin.json").read_text()
    )
    source = payload["trait_provenance"]["perceived_gender"]
    assert "/blob/3602bfef412d6398cc518268a7ecc5de255f4991/" in source


def test_the_merged_catalog_does_carry_sourced_traits() -> None:
    """Without this the gendered method would have nothing to filter on."""
    sourced = [e for e in CATALOG.values() if e.perceived_gender is not None]
    assert len(sourced) >= _SOURCED_MINIMUM


_HEADER = b'{"v":{}}'
_EMBEDDING_BYTES = len(_HEADER).to_bytes(8, "little") + _HEADER + b"\0" * 16
# The voice pack supplies 95 sourced traits and the built-ins add 26; allow a
# little slack for pack churn while ensuring the built-ins remain represented.
_SOURCED_MINIMUM = 115


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


def test_trait_is_optional_so_older_manifests_stay_readable(
    tmp_path: Path, isolated_cache_root: Path
) -> None:
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
    manifest = json.loads((isolated_cache_root / "manifest.json").read_text())
    assert "perceived_gender" not in manifest["voices"]["plain_voice"]
    listed = {voice.id: voice for voice in kk.list_voices()}
    assert listed["plain_voice"].perceived_gender is None


def test_manifest_voice_inherits_catalog_gender(tmp_path: Path) -> None:
    """A manifest entry with no trait still enumerates with the catalog's.

    Manifests written before the field existed carry no trait at all, and
    reading the record alone discards what the catalog has always known.
    """
    voice_id = next(
        key
        for key, entry in sorted(CATALOG.items())
        if entry.perceived_gender == "feminine"
    )
    manifest = tmp_path / "manifest.json"
    ManifestStore(manifest).write(
        {},
        {
            voice_id: VoiceRecord(
                id=voice_id,
                variety="built-in",
                state="registered",
                name="Test",
                enabled=True,
                language="english",
                engine_id="english",
                provenance="test",
                license_id="CC-BY-4.0",
                commercial_use_allowed=False,
                voice_rights="test",
            )
        },
    )
    written = json.loads(manifest.read_text())["voices"][voice_id]
    assert "perceived_gender" not in written

    listed = {voice.id: voice for voice in kk.list_voices(manifest=manifest)}
    assert listed[voice_id].perceived_gender == "feminine"


def test_gendered_cast_admits_no_opposite_gender_voice() -> None:
    """A feminine character is never offered a masculine voice.

    The assertion that was missing: every unit test covered `candidates`
    against a hand-built pool, and none covered it against the pool the
    library actually enumerates. A trait the catalog knew and `list_voices`
    dropped therefore degraded gendered casting to random casting with a
    fully green suite.
    """
    pool = tuple(kk.list_voices())
    character = CharacterProfile(
        id="her",
        display_name="Her",
        gender="feminine",
        spoken_characters=100,
        chapter_ids=(),
    )
    admitted = candidates("gendered", character, pool)
    assert admitted, "the enumerated pool must contain at least one feminine voice"
    assert admitted != pool, "a pool equal to the whole pool is not gendered"
    assert not [v for v in admitted if v.perceived_gender == "masculine"]


def test_an_existing_ungendered_record_adopts_the_catalog_trait() -> None:
    """A record written before the field existed gains it on the next write.

    `_registered_from_catalog` already stamps the trait on a first
    registration, so a fresh install was never the gap. An operator whose
    manifest predates the field has a record, takes the `voices.get` path,
    and would otherwise keep writing it back ungendered forever.
    """
    voice_id = next(
        key
        for key, entry in sorted(CATALOG.items())
        if entry.perceived_gender == "masculine"
    )
    stale = VoiceRecord(
        id=voice_id,
        variety="built-in",
        state="registered",
        name="Stale",
        enabled=True,
        language="english",
        engine_id="english",
        provenance="test",
        license_id="CC-BY-4.0",
        commercial_use_allowed=False,
        voice_rights="test",
    )
    assert stale.perceived_gender is None
    assert _with_catalog_gender(stale).perceived_gender == "masculine"


def test_an_explicit_trait_is_never_overwritten() -> None:
    """The manifest is the authority when it has an answer."""
    voice_id = next(
        key
        for key, entry in sorted(CATALOG.items())
        if entry.perceived_gender == "masculine"
    )
    record = VoiceRecord(
        id=voice_id,
        variety="built-in",
        state="registered",
        name="Owned",
        enabled=True,
        language="english",
        engine_id="english",
        provenance="test",
        license_id="CC-BY-4.0",
        commercial_use_allowed=False,
        voice_rights="test",
        perceived_gender="feminine",
    )
    assert _with_catalog_gender(record).perceived_gender == "feminine"


def test_a_voice_the_catalog_does_not_know_is_left_alone() -> None:
    """An operator's own voice has no catalog answer to adopt."""
    record = VoiceRecord(
        id="house_voice",
        variety="pre-compiled",
        state="registered",
        name="House",
        enabled=True,
        language="english",
        engine_id="english",
        provenance="test",
        license_id="CC0-1.0",
        commercial_use_allowed=True,
        voice_rights="test",
    )
    assert _with_catalog_gender(record).perceived_gender is None
