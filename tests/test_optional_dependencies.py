from __future__ import annotations

import builtins

import pytest

from kenkui.errors import KenkuiDependencyError
from kenkui.workers import _get_or_load_model


def test_get_or_load_model_raises_actionable_dependency_error_when_pocket_tts_missing(monkeypatch):
    def fake_import(name, *args, **kwargs):
        if name == "pocket_tts":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    original_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(KenkuiDependencyError, match="pocket-tts"):
        _get_or_load_model(0.7, 1, None)
