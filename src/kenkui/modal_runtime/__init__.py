"""Optional Modal execution runtime registration."""
from __future__ import annotations

from typing import Any

from kenkui.models import AttributionExecutionMode, NlpExecutionMode
from kenkui.nlp.providers._factory import register_nlp_extension
from kenkui.services.execution_service import register_tts_execution_provider


def register_modal_runtime(app_config: Any | None = None) -> None:
    """Register Modal-backed TTS and NLP provider factories.

    Importing this package must stay lightweight; concrete Modal SDK imports are
    delayed until a provider actually invokes a remote client.
    """
    from kenkui.modal_runtime.nlp import ModalAttributionProvider, ModalExtractionProvider
    from kenkui.modal_runtime.tts import ModalTTSProvider

    register_tts_execution_provider("modal", lambda: ModalTTSProvider(app_config=app_config))

    def extraction_factory(config):
        if config.extraction_mode == NlpExecutionMode.MODAL:
            return ModalExtractionProvider(config=config, app_config=app_config)
        return None

    def attribution_factory(config):
        if config.attribution_mode == AttributionExecutionMode.MODAL:
            return ModalAttributionProvider(config=config, app_config=app_config)
        return None

    register_nlp_extension(extraction_factory, attribution_factory)


__all__ = ["register_modal_runtime"]
