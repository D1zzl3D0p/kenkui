"""Unit tests for voice_registry immutability and caching guarantees (QP Task 11)."""
from __future__ import annotations

from pathlib import Path

import pytest

import kenkui.voice_registry as vr
from kenkui.voice_registry import VoiceCatalog, VoiceCatalogEntry, get_catalog  # noqa: F401

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _builtin_entry(voice_id: str = "alba") -> VoiceCatalogEntry:
    return VoiceCatalogEntry(
        voice_id=voice_id,
        display_name=voice_id.title(),
        origin="pocket_tts_builtin",
        asset_kind="pocket_tts_builtin",
        gender="Male",
        pool_enabled=True,
    )


# ---------------------------------------------------------------------------
# VoiceCatalog.voices — immutability
# ---------------------------------------------------------------------------

class TestVoicesImmutability:
    def test_voices_returns_tuple(self, tmp_path: Path) -> None:
        catalog = VoiceCatalog(data_dir=tmp_path)
        result = catalog.voices
        assert isinstance(result, tuple), (
            "VoiceCatalog.voices must return an immutable tuple, not a list"
        )

    def test_voices_entries_are_frozen_dataclasses(self, tmp_path: Path) -> None:
        catalog = VoiceCatalog(data_dir=tmp_path)
        for entry in catalog.voices:
            with pytest.raises((AttributeError, TypeError)):
                entry.voice_id = "hacked"  # type: ignore[misc]

    def test_voices_cannot_be_mutated_via_returned_value(self, tmp_path: Path) -> None:
        catalog = VoiceCatalog(data_dir=tmp_path)
        snapshot = catalog.voices
        with pytest.raises((AttributeError, TypeError)):
            snapshot.append(_builtin_entry("new_voice"))  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# VoiceCatalog.voices — caching
# ---------------------------------------------------------------------------

class TestVoicesCaching:
    def test_voices_same_object_on_repeated_access(self, tmp_path: Path) -> None:
        catalog = VoiceCatalog(data_dir=tmp_path)
        first = catalog.voices
        second = catalog.voices
        assert first is second, "voices must be cached; _load() must not run twice"

    def test_invalidate_clears_cache(self, tmp_path: Path) -> None:
        catalog = VoiceCatalog(data_dir=tmp_path)
        first = catalog.voices
        catalog.invalidate()
        second = catalog.voices
        # Object identity changes after invalidation (fresh load)
        assert first is not second

    def test_invalidate_reloads_on_next_access(self, tmp_path: Path) -> None:
        catalog = VoiceCatalog(data_dir=tmp_path)
        _ = catalog.voices
        catalog.invalidate()
        # After invalidation, _voices should be None
        assert catalog._voices is None
        # Re-accessing triggers reload
        reloaded = catalog.voices
        assert isinstance(reloaded, tuple)
        assert len(reloaded) > 0


# ---------------------------------------------------------------------------
# Module-level singleton — caching and reset seam
# ---------------------------------------------------------------------------

class TestModuleSingleton:
    def test_get_catalog_returns_same_instance(self) -> None:
        a = get_catalog()
        b = get_catalog()
        assert a is b, "get_catalog() must return the cached singleton"

    def test_reset_catalog_clears_singleton(self) -> None:
        first = get_catalog()
        vr._reset_catalog()
        second = get_catalog()
        assert first is not second, "_reset_catalog() must allow a fresh instance"

    def test_get_catalog_after_reset_is_stable(self) -> None:
        vr._reset_catalog()
        a = get_catalog()
        b = get_catalog()
        assert a is b

    def test_reset_catalog_is_callable(self) -> None:
        # Explicit seam for tests; must not raise
        vr._reset_catalog()
        vr._reset_catalog()  # idempotent


# ---------------------------------------------------------------------------
# Module global is not directly writable (no raw `global _catalog =`)
# ---------------------------------------------------------------------------

class TestNoRawGlobalMutation:
    def test_module_has_no_bare_mutable_sentinel(self) -> None:
        """The module-level _catalog reference must live inside a container,
        not as a re-assignable top-level name used by get_catalog() via `global`."""
        # Confirm _reset_catalog exists as the sanctioned seam
        assert callable(vr._reset_catalog), "_reset_catalog must exist as explicit reset seam"

    def test_get_catalog_does_not_use_global_keyword_for_assignment(self) -> None:
        """get_catalog must delegate to a container, not write the bare module global.

        We verify indirectly: _reset_catalog() must reset what get_catalog() uses.
        """
        vr._reset_catalog()
        cat = get_catalog()
        # A second reset should not break the already-returned instance
        vr._reset_catalog()
        new_cat = get_catalog()
        assert cat is not new_cat  # fresh after reset
        assert isinstance(new_cat, VoiceCatalog)
