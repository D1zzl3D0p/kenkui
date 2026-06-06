"""Tests for kenkui.nlp._retry."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from kenkui.errors import KenkuiDependencyError
from kenkui.nlp._retry import with_retry


class TestWithRetry:
    def test_success_on_first_attempt(self):
        fn = MagicMock(return_value=42)
        result = with_retry(fn, max_attempts=3)()
        assert result == 42
        assert fn.call_count == 1

    def test_retries_on_os_error(self):
        fn = MagicMock(side_effect=[OSError("connection reset"), 99])
        result = with_retry(fn, max_attempts=3, backoff_base=0)()
        assert result == 99
        assert fn.call_count == 2

    def test_retries_on_timeout_error(self):
        fn = MagicMock(side_effect=[TimeoutError(), "done"])
        result = with_retry(fn, max_attempts=3, backoff_base=0)()
        assert result == "done"

    def test_raises_after_max_attempts(self):
        fn = MagicMock(side_effect=OSError("persistent error"))
        with pytest.raises(OSError, match="persistent error"):
            with_retry(fn, max_attempts=3, backoff_base=0)()
        assert fn.call_count == 3

    def test_does_not_retry_value_error(self):
        fn = MagicMock(side_effect=ValueError("bad value"))
        with pytest.raises(ValueError, match="bad value"):
            with_retry(fn, max_attempts=3, backoff_base=0)()
        assert fn.call_count == 1

    def test_does_not_retry_type_error(self):
        fn = MagicMock(side_effect=TypeError("wrong type"))
        with pytest.raises(TypeError):
            with_retry(fn, max_attempts=3, backoff_base=0)()
        assert fn.call_count == 1

    def test_does_not_retry_keyboard_interrupt(self):
        fn = MagicMock(side_effect=KeyboardInterrupt())
        with pytest.raises(KeyboardInterrupt):
            with_retry(fn, max_attempts=3, backoff_base=0)()
        assert fn.call_count == 1

    def test_preserves_function_name(self):
        def my_function():
            pass
        wrapped = with_retry(my_function)
        assert wrapped.__name__ == "my_function"

    def test_passes_args_and_kwargs(self):
        fn = MagicMock(return_value="ok")
        with_retry(fn)(1, 2, key="val")
        fn.assert_called_once_with(1, 2, key="val")

    def test_ollama_missing_llama_server_is_short_non_retryable_error(self, caplog):
        message = (
            "error starting llama-server: llama-server binary not found "
            "(checked: /opt/homebrew/Cellar/ollama/0.30.4/libexec/lib/ollama/llama-server, "
            "/opt/homebrew/Cellar/ollama/0.30.4/libexec/llama-server). "
            "Run 'cmake -S llama/server --preset cpu && cmake --build --preset cpu' first "
            "(status code: 500)"
        )
        fn = MagicMock(side_effect=RuntimeError(message))

        with pytest.raises(KenkuiDependencyError) as exc_info, caplog.at_level("WARNING"):
            with_retry(fn, max_attempts=3, backoff_base=0)()

        assert fn.call_count == 1
        assert str(exc_info.value) == (
            "Ollama cannot start because its llama-server binary is missing. "
            "Reinstall or upgrade Ollama, then restart the Ollama service."
        )
        warning_messages = [record.message for record in caplog.records if record.levelname == "WARNING"]
        assert warning_messages == [str(exc_info.value)]
        assert "/opt/homebrew/Cellar" not in warning_messages[0]
