"""Shared test isolation.

Provisioning writes to a real per-user cache. Without redirecting it, results
depend on whether the developer has ever run `load_voice`, and a suite run can
read — or worse, mutate — real assets. Every test gets its own cache root.
"""
# ruff: noqa: TC003

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from kenkui._tts import production
from kenkui.voices import manifest as manifest_module


@pytest.fixture(autouse=True)
def isolated_cache_root(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Redirect the managed cache and manifest away from the real user cache."""
    root = tmp_path_factory.mktemp("kenkui-cache")
    monkeypatch.setattr(production, "default_cache_root", lambda: root)
    monkeypatch.setattr(
        manifest_module, "default_manifest_path", lambda: root / "manifest.json"
    )
    monkeypatch.delenv("KENKUI_POCKET_MANIFEST", raising=False)
    return root


@pytest.fixture
def _real_cache_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the autouse redirect for tests of default_cache_root itself."""
    monkeypatch.undo()

def log_field(record: logging.LogRecord, name: str) -> object:
    """Read one structured field that log_event attached through ``extra``.

    LogRecord declares no such attributes, so reading them directly is
    invisible to the type checker even though they exist at runtime.
    """
    return getattr(record, name)
