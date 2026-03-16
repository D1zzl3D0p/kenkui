"""Tests for the HuggingFace authentication module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

from kenkui.huggingface_auth import (
    GATED_MODELS,
    HF_SIGNUP_URL,
    HF_TOKEN_URL,
    AuthStatus,
    check_auth_status,
    do_login,
    is_custom_voice,
    is_model_gated,
    open_model_page,
    open_signup_page,
    open_token_page,
    verify_access,
)
from kenkui.utils import DEFAULT_VOICES


def _fake_response() -> httpx.Response:
    """Build a minimal httpx.Response for use in HF error constructors."""
    return httpx.Response(200, request=httpx.Request("GET", "https://huggingface.co"))


class TestIsCustomVoice:
    def test_default_voices_not_custom(self):
        bundled = []
        for voice in DEFAULT_VOICES:
            assert is_custom_voice(voice, bundled) is False

    def test_bundled_voices_are_custom(self):
        bundled = ["custom1.wav", "custom2.wav"]
        assert is_custom_voice("custom1", bundled) is True
        assert is_custom_voice("custom2", bundled) is True

    def test_hf_urls_are_custom(self):
        assert is_custom_voice("hf://user/repo/voice.wav", []) is True

    def test_unknown_voices_are_custom(self):
        assert is_custom_voice("unknown_voice", []) is True


class TestIsModelGated:
    def test_gated_model_returns_true(self):
        assert is_model_gated("kyutai/pocket-tts") is True

    def test_ungated_model_returns_false(self):
        assert is_model_gated("unknown/model") is False

    def test_gated_models_dict_is_configurable(self):
        GATED_MODELS["test/model"] = True
        assert is_model_gated("test/model") is True
        del GATED_MODELS["test/model"]
        assert is_model_gated("test/model") is False


class TestAuthStatus:
    def test_auth_status_values(self):
        assert AuthStatus.OK.value == "ok"
        assert AuthStatus.NO_TOKEN.value == "no_token"
        assert AuthStatus.NEEDS_TERMS.value == "needs_terms"
        assert AuthStatus.NOT_FOUND.value == "not_found"
        assert AuthStatus.ERROR.value == "error"


class TestCheckAuthStatus:
    def test_ungated_model_returns_ok(self):
        status = check_auth_status("ungated/model")
        assert status == AuthStatus.OK

    def test_no_token_returns_no_token(self):
        from huggingface_hub.errors import LocalTokenNotFoundError

        with patch("kenkui.huggingface_auth.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.model_info.side_effect = LocalTokenNotFoundError()
            mock_api_cls.return_value = mock_api
            status = check_auth_status("kyutai/pocket-tts")
        assert status == AuthStatus.NO_TOKEN

    def test_gated_repo_returns_needs_terms(self):
        from huggingface_hub.errors import GatedRepoError

        with patch("kenkui.huggingface_auth.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.model_info.side_effect = GatedRepoError("gated", response=_fake_response())
            mock_api_cls.return_value = mock_api
            status = check_auth_status("kyutai/pocket-tts")
        assert status == AuthStatus.NEEDS_TERMS

    def test_model_accessible_returns_ok(self):
        with patch("kenkui.huggingface_auth.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.model_info.return_value = MagicMock()
            mock_api_cls.return_value = mock_api
            status = check_auth_status("kyutai/pocket-tts")
        assert status == AuthStatus.OK


class TestDoLogin:
    def test_empty_token_fails(self):
        ok, msg = do_login("")
        assert ok is False
        assert "No token" in msg

    def test_valid_token_succeeds(self):
        with patch("kenkui.huggingface_auth.login") as mock_login:
            mock_login.return_value = None
            ok, msg = do_login("hf_abc123")
        assert ok is True

    def test_bad_token_raises_returns_false(self):
        with patch("kenkui.huggingface_auth.login", side_effect=Exception("bad token")):
            ok, msg = do_login("hf_bad")
        assert ok is False
        assert "failed" in msg.lower()


class TestVerifyAccess:
    def test_accessible_model_returns_true(self):
        with patch("kenkui.huggingface_auth.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.model_info.return_value = MagicMock()
            mock_api_cls.return_value = mock_api
            ok, msg = verify_access("kyutai/pocket-tts")
        assert ok is True
        assert "granted" in msg.lower()

    def test_still_gated_returns_false(self):
        from huggingface_hub.errors import GatedRepoError

        with patch("kenkui.huggingface_auth.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.model_info.side_effect = GatedRepoError("gated", response=_fake_response())
            mock_api_cls.return_value = mock_api
            ok, msg = verify_access("kyutai/pocket-tts")
        assert ok is False


class TestBrowserHelpers:
    """Browser-open helpers should call webbrowser.open with correct URLs."""

    def test_open_signup_page(self):
        with patch("kenkui.huggingface_auth.webbrowser.open") as mock_open:
            open_signup_page()
            mock_open.assert_called_once_with(HF_SIGNUP_URL)

    def test_open_token_page(self):
        with patch("kenkui.huggingface_auth.webbrowser.open") as mock_open:
            open_token_page()
            mock_open.assert_called_once_with(HF_TOKEN_URL)

    def test_open_model_page(self):
        with patch("kenkui.huggingface_auth.webbrowser.open") as mock_open:
            open_model_page("kyutai/pocket-tts")
            mock_open.assert_called_once_with("https://huggingface.co/kyutai/pocket-tts")

    def test_browser_failure_does_not_raise(self):
        """webbrowser.open failure should be caught silently."""
        with patch("kenkui.huggingface_auth.webbrowser.open", side_effect=Exception("no browser")):
            open_signup_page()  # should not raise
            open_token_page()  # should not raise
            open_model_page()  # should not raise
