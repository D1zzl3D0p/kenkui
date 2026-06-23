from __future__ import annotations

import sys
import types

import pytest

from kenkui.models import AppConfig, AttributionExecutionMode, ExtractionTool, NlpExecutionMode
from kenkui.nlp.providers._factory import get_attribution_provider, get_extraction_provider
from kenkui.nlp_config import NLPConfig
from kenkui.services.execution_service import get_tts_execution_provider
from kenkui.services.runtime_service import register_configured_runtimes


class _DummyJob:
    tts_execution_mode = type("Mode", (), {"value": "modal"})()


class _DummyQueueItem:
    job = _DummyJob()


@pytest.fixture(autouse=True)
def reset_runtime_hooks(monkeypatch):
    import kenkui.nlp.providers._factory as factory
    import kenkui.services.execution_service as execution

    monkeypatch.setattr(factory, "_ExtractionExt", None)
    monkeypatch.setattr(factory, "_AttributionExt", None)
    monkeypatch.setitem(execution._TTS_PROVIDERS, "local", execution.LocalTTSProvider)
    execution._TTS_PROVIDERS.pop("modal", None)
    yield
    monkeypatch.setattr(factory, "_ExtractionExt", None)
    monkeypatch.setattr(factory, "_AttributionExt", None)
    execution._TTS_PROVIDERS.pop("modal", None)


def test_disabled_runtime_registration_does_not_import_modal_runtime(monkeypatch):
    def fail_import(name, *args, **kwargs):
        if name == "kenkui.modal_runtime":
            raise AssertionError("modal runtime should not be imported when disabled")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", fail_import)

    register_configured_runtimes(AppConfig.from_dict({"modal_enabled": False}))


def test_enabled_runtime_registration_invokes_modal_register(monkeypatch):
    calls = []
    module = types.ModuleType("kenkui.modal_runtime")
    module.register_modal_runtime = lambda app_config=None: calls.append(app_config)
    monkeypatch.setitem(sys.modules, "kenkui.modal_runtime", module)

    cfg = AppConfig.from_dict({"modal_enabled": True})
    register_configured_runtimes(cfg)

    assert calls == [cfg]


def test_modal_runtime_registers_tts_and_nlp_providers():
    from kenkui.modal_runtime import register_modal_runtime
    from kenkui.modal_runtime.nlp import ModalAttributionProvider, ModalExtractionProvider
    from kenkui.modal_runtime.tts import ModalTTSProvider

    register_modal_runtime(AppConfig.from_dict({"modal_enabled": True}))

    assert isinstance(get_tts_execution_provider(_DummyQueueItem()), ModalTTSProvider)
    extraction = get_extraction_provider(
        NLPConfig(extraction_tool=ExtractionTool.OLLAMA, extraction_mode=NlpExecutionMode.MODAL)
    )
    attribution = get_attribution_provider(
        NLPConfig(attribution_mode=AttributionExecutionMode.MODAL)
    )
    assert isinstance(extraction, ModalExtractionProvider)
    assert isinstance(attribution, ModalAttributionProvider)
