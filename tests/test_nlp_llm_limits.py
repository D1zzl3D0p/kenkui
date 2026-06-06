"""Regression tests for LLM token-limit configuration.

These tests pin the specific failures that caused:
  WARNING: LLM response truncated at 35831 chars (roster building)
  WARNING: LLM response truncated at 26801 chars (dialogue attribution)

Root cause: num_predict=8192 was too small. Both responses exceeded ~8k tokens.
The tests below fail on the old defaults and pass only after the fix.
"""
from __future__ import annotations

import importlib
import logging
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel


class _SimpleSchema(BaseModel):
    value: str


def _make_mock_response(content: str, eval_count: int | None = None) -> MagicMock:
    """Return a minimal fake ollama.chat() response object."""
    resp = MagicMock()
    resp.message.content = content
    resp.prompt_eval_count = None
    if eval_count is not None:
        resp.eval_count = eval_count
    else:
        # Simulate the field being absent (as in older Ollama versions)
        del resp.eval_count
    return resp


# ---------------------------------------------------------------------------
# Token limit defaults — these directly pin the regression
# ---------------------------------------------------------------------------


class TestNumPredictDefault:
    """num_predict must default to 32768, not 8192."""

    def test_default_num_predict_is_32768(self, monkeypatch):
        """Regression: old default was 8192 — too small for large roster/attribution responses."""
        monkeypatch.delenv("KENKUI_NLP_OLLAMA_NUM_PREDICT", raising=False)
        import kenkui.nlp.llm as llm_mod
        importlib.reload(llm_mod)
        assert llm_mod._LLM_OPTIONS["num_predict"] == 32768, (
            f"num_predict={llm_mod._LLM_OPTIONS['num_predict']} is too small; "
            "large character rosters (~35k chars) and long chapters (~27k chars) "
            "will be truncated. Must be 32768."
        )

    def test_default_num_ctx_is_65536(self, monkeypatch):
        """num_ctx must accommodate prompt + response; 16384 is too tight for large books."""
        monkeypatch.delenv("KENKUI_NLP_OLLAMA_NUM_CTX", raising=False)
        import kenkui.nlp.llm as llm_mod
        importlib.reload(llm_mod)
        assert llm_mod._LLM_OPTIONS["num_ctx"] == 65536, (
            f"num_ctx={llm_mod._LLM_OPTIONS['num_ctx']} — must be at least 65536 "
            "so prompt_tokens + num_predict fits within the context window."
        )

    def test_env_var_overrides_num_predict(self, monkeypatch):
        """KENKUI_NLP_OLLAMA_NUM_PREDICT must override the default (12-factor)."""
        monkeypatch.setenv("KENKUI_NLP_OLLAMA_NUM_PREDICT", "16384")
        import kenkui.nlp.llm as llm_mod
        importlib.reload(llm_mod)
        assert llm_mod._LLM_OPTIONS["num_predict"] == 16384

    def test_env_var_overrides_num_ctx(self, monkeypatch):
        """KENKUI_NLP_OLLAMA_NUM_CTX must override the default (12-factor)."""
        monkeypatch.setenv("KENKUI_NLP_OLLAMA_NUM_CTX", "131072")
        import kenkui.nlp.llm as llm_mod
        importlib.reload(llm_mod)
        assert llm_mod._LLM_OPTIONS["num_ctx"] == 131072


# ---------------------------------------------------------------------------
# Truncation warning must include eval_count for diagnostics
# ---------------------------------------------------------------------------


class TestTruncationWarningIncludesEvalCount:
    """Truncation warning must include eval_count so future occurrences are diagnosable.

    The original WARNING said only 'truncated at N chars' — no token count,
    no limit shown. This makes it hard to know how close to the ceiling we ran.
    """

    def _trigger_truncation(self, caplog, eval_count=None):
        """Send a truncated JSON response that can't be recovered, expect a WARNING.

        Injects a fake ollama module into sys.modules so the lazy ``import ollama``
        inside LLMClient.generate() picks up our mock without needing Ollama installed.
        The module is reloaded inside the injection context so _LLM_OPTIONS is fresh.
        """
        import sys
        truncated = '{"value": "hel'
        mock_resp = _make_mock_response(truncated, eval_count=eval_count)

        fake_ollama = MagicMock()
        fake_ollama.chat.return_value = mock_resp

        original = sys.modules.get("ollama")
        sys.modules["ollama"] = fake_ollama
        try:
            import kenkui.nlp.llm as llm_mod
            importlib.reload(llm_mod)
            client = llm_mod.LLMClient("test-model")
            with caplog.at_level(logging.WARNING, logger="kenkui.nlp.llm"):
                with pytest.raises(Exception):
                    client.generate("some prompt", _SimpleSchema)
        finally:
            if original is None:
                sys.modules.pop("ollama", None)
            else:
                sys.modules["ollama"] = original

    def test_warning_includes_eval_count_when_available(self, caplog):
        """When eval_count is present in the response, the WARNING must mention it."""
        self._trigger_truncation(caplog, eval_count=8111)
        messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("8111" in m or "eval_count" in m for m in messages), (
            f"WARNING must include eval_count=8111 for diagnostics. Got: {messages}"
        )

    def test_warning_includes_token_limit(self, caplog):
        """The WARNING must show the active num_predict ceiling so the fix is obvious."""
        self._trigger_truncation(caplog, eval_count=9000)
        messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        # The limit (32768 or whatever _num_predict is) must appear in the warning
        import kenkui.nlp.llm as llm_mod
        importlib.reload(llm_mod)
        limit = str(llm_mod._LLM_OPTIONS["num_predict"])
        assert any(limit in m for m in messages), (
            f"WARNING must include the num_predict limit ({limit}). Got: {messages}"
        )

    def test_warning_fires_even_without_eval_count(self, caplog):
        """Older Ollama versions may not return eval_count — warning must still fire."""
        self._trigger_truncation(caplog, eval_count=None)
        assert any(r.levelno >= logging.WARNING for r in caplog.records), (
            "A truncation WARNING must fire even when eval_count is absent from response."
        )
