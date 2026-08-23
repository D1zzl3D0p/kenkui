"""The bundled voice pack merges into the catalog, with built-ins winning.

The pack ships as a manifest inside the package; its compiled assets are
fetched on demand by load_voice, exactly as the kyutai built-ins are. Nothing
here reaches the network.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import TYPE_CHECKING

from kenkui.voices import registry
from kenkui.voices.registry import CATALOG, asset_url

if TYPE_CHECKING:
    import pytest

    from kenkui.voices.types import PerceivedGender

# 26 kyutai built-ins plus 95 pack voices, minus nothing: the one short-name
# clash keeps both entries under different ids.
_MERGED_MINIMUM = 100
_POOL_MINIMUM = 40
_SOURCED_MINIMUM = 90
_REVISION_LENGTH = 40


def test_pack_voices_join_the_catalog() -> None:
    """95 pack voices on top of the built-in catalog."""
    assert len(CATALOG) > _MERGED_MINIMUM


def test_pack_voices_carry_sourced_gender() -> None:
    """Sourced gender is what the built-in entries could not supply."""
    assert CATALOG["alasdair"].perceived_gender == "masculine"
    assert CATALOG["amara"].perceived_gender == "feminine"


def test_pack_voices_carry_rights() -> None:
    """Kenkui refuses to render a voice with no recorded terms."""
    assert CATALOG["alasdair"].license_id == "CC-BY-4.0"
    assert CATALOG["amara"].license_id == "CC-BY-NC-4.0"
    assert all(entry.commercial_use_allowed is False for entry in CATALOG.values())


def test_builtin_names_take_precedence_on_collision() -> None:
    """'vera' is a VCTK built-in and an unrelated EARS voice in the pack.

    They are different speakers that happen to share a display name, so the
    built-in keeps the short name and the pack voice stays addressable under
    its own slug rather than being dropped.
    """
    assert CATALOG["vera"].origin_url.endswith("p229_023_enhanced.wav")
    assert "vera-f-ears-p082-american" in CATALOG


def test_pack_voices_use_short_ids_where_unambiguous() -> None:
    """assign_voices(narrator="alasdair") beats the full slug for ergonomics."""
    assert "alasdair" in CATALOG
    assert "alasdair-m-vctk-p246-scottish" not in CATALOG


def test_every_catalog_id_is_unique_and_nonempty() -> None:
    """A key disagreeing with its entry id would break every lookup."""
    assert all(entry_id and entry_id == entry.id for entry_id, entry in CATALOG.items())


def test_pack_asset_urls_are_pinned() -> None:
    """A moving revision would silently change which bytes are fetched."""
    payload = json.loads(
        Path(registry.__file__).with_name("pack.json").read_text(encoding="utf-8")
    )
    revision = payload["assets"]["revision"]
    url = asset_url("alasdair", "english")
    assert url.startswith("hf://D1zzl3D0p/kenkui-voices/compiled/")
    assert url.endswith(f"@{revision}")
    assert len(revision) == _REVISION_LENGTH


def test_builtin_asset_urls_still_resolve_to_kyutai() -> None:
    """The pack must not divert the kyutai catalog voices."""
    assert "pocket-tts-without-voice-cloning" in asset_url("eponine", "english")


def test_gendered_pool_is_now_deep() -> None:
    """The shallow-pool risk the design recorded is retired by the pack."""
    pool: dict[PerceivedGender, int] = {"feminine": 0, "masculine": 0}
    for entry in CATALOG.values():
        if entry.perceived_gender in pool:
            pool[entry.perceived_gender] += 1
    assert pool["feminine"] >= _POOL_MINIMUM
    assert pool["masculine"] >= _POOL_MINIMUM


def test_loading_the_pack_touches_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """The catalog is static package data; only load_voice may fetch."""

    def deny(*_args: object, **_kwargs: object) -> None:
        message = "the registry must not open sockets"
        raise AssertionError(message)

    monkeypatch.setattr(socket, "socket", deny)
    registry.load_pack.cache_clear()
    assert registry.load_pack().entries


def test_a_missing_pack_degrades_to_the_built_ins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The pack is an addition; Kenkui must import and render without it."""
    monkeypatch.setattr(registry, "_PACK_MANIFEST", tmp_path / "absent.json")
    registry.load_pack.cache_clear()
    registry._withheld_pack_ids.cache_clear()  # noqa: SLF001
    try:
        assert registry.load_pack().entries == ()
    finally:
        registry.load_pack.cache_clear()
        registry._withheld_pack_ids.cache_clear()  # noqa: SLF001
