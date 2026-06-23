from __future__ import annotations

import pytest

from kenkui.models import AppConfig
from kenkui.services import runtime_service
from kenkui.services.runtime_service import RuntimeRegistrationError, register_configured_runtimes


class _EntryPoints(list):
    def select(self, *, group: str):
        assert group == "kenkui.runtime_providers"
        return self


class _EntryPoint:
    def __init__(self, name, register):
        self.name = name
        self._register = register

    def load(self):
        return self._register


def test_disabled_runtime_registration_does_not_discover_plugins(monkeypatch):
    def fail_entry_points():
        raise AssertionError("runtime plugins should not be discovered when modal is disabled")

    monkeypatch.setattr(runtime_service, "entry_points", fail_entry_points)

    register_configured_runtimes(AppConfig.from_dict({"modal_enabled": False}))


def test_enabled_runtime_registration_invokes_modal_entry_point(monkeypatch):
    calls = []

    def register(app_config=None):
        calls.append(app_config)

    monkeypatch.setattr(
        runtime_service,
        "entry_points",
        lambda: _EntryPoints([_EntryPoint("modal", register)]),
    )

    cfg = AppConfig.from_dict({"modal_enabled": True})
    register_configured_runtimes(cfg)

    assert calls == [cfg]


def test_enabled_runtime_registration_errors_when_modal_plugin_missing(monkeypatch):
    monkeypatch.setattr(runtime_service, "entry_points", lambda: _EntryPoints([]))

    with pytest.raises(RuntimeRegistrationError, match="kenkui-modal"):
        register_configured_runtimes(AppConfig.from_dict({"modal_enabled": True}))
