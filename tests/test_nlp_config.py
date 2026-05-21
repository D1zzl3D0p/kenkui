"""Tests for NLPConfig."""
import os
import pytest
from kenkui.models import AttributionExecutionMode, AttributionTool, ExtractionTool, NlpExecutionMode
from kenkui.nlp_config import NLPConfig


class TestNLPConfigDefaults:
    def test_default_extraction_tool_is_ollama(self):
        cfg = NLPConfig()
        assert cfg.extraction_tool == ExtractionTool.OLLAMA

    def test_default_attribution_tool_is_ollama(self):
        cfg = NLPConfig()
        assert cfg.attribution_tool == AttributionTool.OLLAMA

    def test_default_extraction_mode_is_local(self):
        cfg = NLPConfig()
        assert cfg.extraction_mode == NlpExecutionMode.LOCAL

    def test_default_attribution_mode_is_local(self):
        cfg = NLPConfig()
        assert cfg.attribution_mode == AttributionExecutionMode.LOCAL

    def test_default_models_are_llama(self):
        cfg = NLPConfig()
        assert cfg.extraction_model == "llama3.2"
        assert cfg.attribution_model == "llama3.2"

    def test_default_ollama_url(self):
        cfg = NLPConfig()
        assert cfg.ollama_url == "http://localhost:11434"

    def test_default_retry_policy(self):
        cfg = NLPConfig()
        assert cfg.retry_max_attempts == 3
        assert cfg.retry_backoff_base == 2.0


class TestNLPConfigFromEnv:
    def test_extraction_tool_from_env(self, monkeypatch):
        monkeypatch.setenv("KENKUI_NLP_EXTRACTION_TOOL", "booknlp")
        cfg = NLPConfig()
        assert cfg.extraction_tool == ExtractionTool.BOOKNLP

    def test_extraction_model_from_env(self, monkeypatch):
        monkeypatch.setenv("KENKUI_NLP_EXTRACTION_MODEL", "mistral")
        cfg = NLPConfig()
        assert cfg.extraction_model == "mistral"

    def test_ollama_url_from_env(self, monkeypatch):
        monkeypatch.setenv("KENKUI_NLP_OLLAMA_URL", "http://remote:11434")
        cfg = NLPConfig()
        assert cfg.ollama_url == "http://remote:11434"

    def test_retry_attempts_from_env(self, monkeypatch):
        monkeypatch.setenv("KENKUI_NLP_RETRY_MAX_ATTEMPTS", "5")
        cfg = NLPConfig()
        assert cfg.retry_max_attempts == 5

    def test_attribution_mode_from_env(self, monkeypatch):
        monkeypatch.setenv("KENKUI_NLP_ATTRIBUTION_MODE", "modal")
        cfg = NLPConfig()
        assert cfg.attribution_mode == AttributionExecutionMode.MODAL


class TestExtractionToolEnum:
    def test_booknlp_value(self):
        assert ExtractionTool.BOOKNLP.value == "booknlp"

    def test_ollama_value(self):
        assert ExtractionTool.OLLAMA.value == "ollama"

    def test_from_string(self):
        assert ExtractionTool("booknlp") == ExtractionTool.BOOKNLP


class TestAttributionToolEnum:
    def test_all_values(self):
        assert {t.value for t in AttributionTool} == {"booknlp", "ollama"}


class TestNLPConfigFromAppConfig:
    def test_ollama_provider_maps_correctly(self):
        from kenkui.models import AppConfig
        app_cfg = AppConfig.from_dict({"nlp_provider": "ollama", "nlp_model": "llama3.2"})
        nlp_cfg = NLPConfig.from_app_config(app_cfg)
        assert nlp_cfg.extraction_tool == ExtractionTool.OLLAMA
        assert nlp_cfg.extraction_model == "llama3.2"

    def test_booknlp_provider_maps_correctly(self):
        from kenkui.models import AppConfig
        app_cfg = AppConfig.from_dict({"nlp_provider": "booknlp", "nlp_model": "small"})
        nlp_cfg = NLPConfig.from_app_config(app_cfg)
        assert nlp_cfg.extraction_tool == ExtractionTool.BOOKNLP

    def test_attribution_provider_override(self):
        from kenkui.models import AppConfig
        app_cfg = AppConfig.from_dict({
            "nlp_provider": "ollama",
            "nlp_model": "llama3.2",
            "nlp_attribution_provider": "booknlp",
        })
        nlp_cfg = NLPConfig.from_app_config(app_cfg)
        assert nlp_cfg.extraction_tool == ExtractionTool.OLLAMA
        assert nlp_cfg.attribution_tool == AttributionTool.BOOKNLP

    def test_unknown_provider_falls_back_to_ollama(self):
        from kenkui.models import AppConfig
        app_cfg = AppConfig.from_dict({"nlp_provider": "unknown_tool"})
        nlp_cfg = NLPConfig.from_app_config(app_cfg)
        assert nlp_cfg.extraction_tool == ExtractionTool.OLLAMA
