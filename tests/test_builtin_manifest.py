"""The built-in catalog is data, loaded the same way the pack is."""
# ruff: noqa: D103

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

from kenkui.voices import registry

_BUILTIN_PATH = Path(registry.__file__).with_name("builtin.json")


def test_builtin_json_ships_with_the_package() -> None:
    assert _BUILTIN_PATH.is_file()


def test_builtin_json_covers_every_upstream_voice() -> None:
    """Drift guard, now against the file rather than a Python tuple."""
    document = json.loads(_BUILTIN_PATH.read_text(encoding="utf-8"))
    ids = {voice["voice_id"] for voice in document["voices"]}
    assert ids == set(_ORIGINS_OF_PREDEFINED_VOICES)


def test_builtin_entries_carry_no_asset_url() -> None:
    """Built-ins derive their embedding URL; only pack voices pin one."""
    assert all(e.asset_url is None for e in registry.BUILT_IN_CATALOG.values())


def test_embedding_revision_comes_from_the_file() -> None:
    document = json.loads(_BUILTIN_PATH.read_text(encoding="utf-8"))
    assert document["embedding"]["revision"] == registry.EMBEDDING_REVISION


def test_an_unreadable_builtin_file_fails_loudly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unlike the pack, the built-ins are not optional: this is a broken install."""
    monkeypatch.setattr(registry, "_BUILTIN_MANIFEST", tmp_path / "absent.json")
    registry.load_builtin.cache_clear()
    try:
        with pytest.raises(OSError, match="builtin"):
            registry.load_builtin()
    finally:
        registry.load_builtin.cache_clear()
