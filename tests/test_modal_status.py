from __future__ import annotations

import sys

from kenkui.models import AppConfig, NlpExecutionMode
from kenkui.services.application_service import KenkuiService


def test_multivoice_status_modal_mode_does_not_require_local_spacy(monkeypatch, tmp_path):
    def fail_spacy_import(name, *args, **kwargs):
        if name == "spacy.util" or name.startswith("spacy"):
            raise AssertionError("local spaCy should not be imported for Modal NLP status")
        return original_import(name, *args, **kwargs)

    module = type(sys)("kenkui.modal_runtime")
    module.register_modal_runtime = lambda app_config=None: None
    monkeypatch.setitem(sys.modules, "kenkui.modal_runtime", module)
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
