"""Tests for kenkui.nlp._retry."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

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
