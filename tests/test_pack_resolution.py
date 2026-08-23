"""The pack resolves against the installed pocket-tts, or refuses to load."""
# ruff: noqa: D103

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from kenkui import ErrorCode, VoiceError
from kenkui.voices import registry

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

_PACK: dict[str, Any] = {
    "schema_version": 2,
    "pocket_tts": ">=2.0.0,<3.0.0",
    "verified_versions": ["2.0.0", "2.1.0"],
    "built_with": "2.1.0",
    "preview_text": "It is a truth universally acknowledged.",
    "assets": {
        "repo_id": "D1zzl3D0p/kenkui-voices",
        "repo_type": "dataset",
        "revision": "cafe1234cafe1234cafe1234cafe1234cafe1234",
    },
    "voices": [
        {
            "voice_id": "alasdair-m-vctk-p246-scottish",
            "display_name": "Alasdair",
            "dataset": "VCTK",
            "speaker_id": "P246",
            "gender": "Male",
            "accent": "Scottish",
            "language": "english",
            "pool_enabled": True,
            "license_id": "CC-BY-4.0",
            "commercial_use_allowed": False,
            "voice_rights": "Derived from the VCTK corpus.",
            "source": {
                "repo_id": "kyutai/tts-voices",
                "path": "vctk/p246.wav",
                "sha256": "a" * 64,
            },
            "compiled": {
                "path": "compiled/alasdair-m-vctk-p246-scottish.safetensors",
                "sha256": "b" * 64,
                "size_bytes": 7472376,
            },
            "preview": {
                "path": "previews/x.wav",
                "sha256": "c" * 64,
                "duration_ms": 7520,
                "text": "t",
            },
        }
    ],
}


@pytest.fixture
def pack_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[[dict[str, Any]], None]]:
    """Install a pack fixture and clear the caches around the test."""

    def install(document: dict[str, Any]) -> None:
        path = tmp_path / "pack.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        monkeypatch.setattr(registry, "_PACK_MANIFEST", path)
        registry.load_pack.cache_clear()
        registry._withheld_pack_ids.cache_clear()  # noqa: SLF001

    registry.load_pack.cache_clear()
    registry._withheld_pack_ids.cache_clear()  # noqa: SLF001
    yield install
    registry.load_pack.cache_clear()
    registry._withheld_pack_ids.cache_clear()  # noqa: SLF001


def test_a_pack_in_range_loads(pack_at: Callable[[dict[str, Any]], None]) -> None:
    pack_at(_PACK)
    loaded = registry.load_pack()
    assert loaded.incompatible is None
    assert len(loaded.entries) == 1


def test_asset_url_pins_the_revision_from_the_manifest(
    pack_at: Callable[[dict[str, Any]], None],
) -> None:
    pack_at(_PACK)
    entry = registry.load_pack().entries[0]
    assert entry.asset_url == (
        "hf://D1zzl3D0p/kenkui-voices/"
        "compiled/alasdair-m-vctk-p246-scottish.safetensors"
        "@cafe1234cafe1234cafe1234cafe1234cafe1234"
    )


def test_a_pack_out_of_range_yields_no_entries(
    pack_at: Callable[[dict[str, Any]], None],
) -> None:
    pack_at({**_PACK, "pocket_tts": ">=99.0.0"})
    loaded = registry.load_pack()
    assert loaded.entries == ()
    assert loaded.incompatible is not None


def test_the_incompatibility_names_both_versions(
    pack_at: Callable[[dict[str, Any]], None],
) -> None:
    """The message has to be actionable without reading the source."""
    pack_at({**_PACK, "pocket_tts": ">=99.0.0"})
    message = registry.load_pack().incompatible
    assert message is not None
    assert "2.1.0" in message
    assert ">=99.0.0" in message
    assert "rebuild.py" in message


def test_using_an_excluded_pack_voice_raises_incompatible(
    pack_at: Callable[[dict[str, Any]], None],
) -> None:
    pack_at({**_PACK, "pocket_tts": ">=99.0.0"})
    with pytest.raises(VoiceError) as excinfo:
        registry.pack_voice_guard("alasdair-m-vctk-p246-scottish")
    assert excinfo.value.code is ErrorCode.VOICE_INCOMPATIBLE
    assert "99.0.0" in str(excinfo.value)


def test_an_unknown_voice_is_not_claimed_by_the_guard(
    pack_at: Callable[[dict[str, Any]], None],
) -> None:
    pack_at({**_PACK, "pocket_tts": ">=99.0.0"})
    registry.pack_voice_guard("nobody")


def test_an_unreadable_range_is_still_refused(
    pack_at: Callable[[dict[str, Any]], None],
) -> None:
    """A malformed range must not resolve to 'compatible'."""
    pack_at({**_PACK, "pocket_tts": "not-a-range"})
    loaded = registry.load_pack()
    assert loaded.entries == ()
    assert loaded.incompatible is not None


def test_a_voice_with_no_compiled_block_is_skipped(
    pack_at: Callable[[dict[str, Any]], None],
) -> None:
    """An unpublished voice is a gap in the pack, not a crash."""
    unpublished = dict(_PACK["voices"][0])
    del unpublished["compiled"]
    pack_at({**_PACK, "voices": [unpublished]})
    assert registry.load_pack().entries == ()


def test_a_missing_pack_degrades_to_the_built_ins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(registry, "_PACK_MANIFEST", tmp_path / "absent.json")
    registry.load_pack.cache_clear()
    registry._withheld_pack_ids.cache_clear()  # noqa: SLF001
    try:
        loaded = registry.load_pack()
        assert loaded.entries == ()
        assert loaded.incompatible is None
    finally:
        registry.load_pack.cache_clear()
        registry._withheld_pack_ids.cache_clear()  # noqa: SLF001
