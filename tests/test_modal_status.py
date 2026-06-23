from __future__ import annotations

from kenkui.models import AppConfig, NlpExecutionMode
from kenkui.services import runtime_service
from kenkui.services.application_service import KenkuiService


class _EntryPoints(list):
    def select(self, *, group: str):
        assert group == "kenkui.runtime_providers"
        return self


class _EntryPoint:
    name = "modal"

    @staticmethod
    def load():
        return lambda app_config=None: None


def test_multivoice_status_modal_mode_does_not_require_local_spacy(monkeypatch, tmp_path):
    def fail_spacy_import(name, *args, **kwargs):
        if name == "spacy.util" or name.startswith("spacy"):
            raise AssertionError("local spaCy should not be imported for Modal NLP status")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(runtime_service, "entry_points", lambda: _EntryPoints([_EntryPoint()]))
    original_import = __import__
    monkeypatch.setattr("builtins.__import__", fail_spacy_import)

    service = KenkuiService(
        queue_file=tmp_path / "queue.toml",
        app_config=AppConfig.from_dict(
            {"modal_enabled": True, "nlp_execution_mode": NlpExecutionMode.MODAL.value}
        ),
    )

    status = service.multivoice_status()

    assert status.spacy_ok is True
    assert status.spacy_model == "modal"
    assert "Modal" in status.message
