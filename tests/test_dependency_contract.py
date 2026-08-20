"""Pocket-TTS is a required runtime dependency at the pinned version."""

from __future__ import annotations

import importlib.metadata


def test_pocket_tts_is_installed_at_pinned_version() -> None:
    """The pocket-tts distribution installed matches the pinned version."""
    assert importlib.metadata.version("pocket-tts") == "2.1.0"


def test_pocket_tts_is_a_required_dependency() -> None:
    """pocket-tts is declared as a runtime dependency, not an optional extra."""
    requires = importlib.metadata.requires("kenkui") or []
    runtime = [item for item in requires if "extra ==" not in item]
    assert any(item.startswith("pocket-tts==2.1.0") for item in runtime)
