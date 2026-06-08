"""Tests for the HuggingFace authentication module.

Coverage:
- is_custom_voice / is_model_gated — pure logic
- AuthStatus enum values
- check_auth_status — all five outcomes (OK, NO_TOKEN, NEEDS_TERMS, NOT_FOUND, ERROR)
- do_login — success, empty token, bad token, non-hf_ prefix
- verify_access — OK, still-gated, unexpected error
- open_signup/token/model_page — correct URLs, silent failure on browser error
- ensure_huggingface_access — noninteractive compatibility status check
- check_voice_access — built-in voice (no auth needed), custom voice (auth called)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

from kenkui.huggingface_auth import (
    GATED_MODELS,
    HF_SIGNUP_URL,
    HF_TOKEN_URL,
    AuthStatus,
    check_auth_status,
    check_voice_access,
    do_login,
    ensure_huggingface_access,
    is_custom_voice,
    is_model_gated,
    open_model_page,
    open_signup_page,
    open_token_page,
    verify_access,
)
from kenkui.voice_registry import BUILTIN_VOICE_NAMES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_response() -> httpx.Response:
    """Minimal httpx.Response for HF error constructors that require one."""
    return httpx.Response(200, request=httpx.Request("GET", "https://huggingface.co"))


def _gated_error() -> Exception:
    from huggingface_hub.errors import GatedRepoError

    return GatedRepoError("gated model", response=_fake_response())


def _not_found_error() -> Exception:
    from huggingface_hub.errors import RepositoryNotFoundError

    return RepositoryNotFoundError("not found", response=_fake_response())


def _no_token_error() -> Exception:
    from huggingface_hub.errors import LocalTokenNotFoundError

    return LocalTokenNotFoundError()


# ---------------------------------------------------------------------------
# is_custom_voice
# ---------------------------------------------------------------------------


class TestIsCustomVoice:
    def test_every_builtin_voice_not_custom(self):
        for voice in BUILTIN_VOICE_NAMES:
            assert is_custom_voice(voice) is False, f"{voice!r} should not be custom"

    def test_hf_url_is_custom(self):
        assert is_custom_voice("hf://user/repo/voice.wav") is True

    def test_local_existing_file_needs_no_auth(self, tmp_path):
        # User already has the file — no HF auth required
        wav = tmp_path / "voice.wav"
        wav.write_bytes(b"RIFF")
        assert is_custom_voice(str(wav)) is False

    def test_unknown_name_needs_auth(self):
        assert is_custom_voice("unknown_voice") is True

    def test_known_compiled_catalog_voice_needs_no_auth(self, tmp_path):
        from kenkui.voice_registry import VoiceCatalogEntry

        asset = tmp_path / "voice.safetensors"
        asset.write_bytes(b"compiled")
        catalog = MagicMock()
        catalog.resolve.return_value = VoiceCatalogEntry(
            voice_id="compiled_voice",
            display_name="Compiled Voice",
            origin="kenkui_compiled",
            asset_kind="safetensors",
            gender="Female",
            pool_enabled=True,
            path=asset,
        )
        with patch("kenkui.voice_registry.get_catalog", return_value=catalog):
            assert is_custom_voice("compiled_voice") is False


# ---------------------------------------------------------------------------
# is_model_gated
# ---------------------------------------------------------------------------


class TestIsModelGated:
    def test_pocket_tts_is_gated(self):
        assert is_model_gated("kyutai/pocket-tts") is True

    def test_unknown_model_not_gated(self):
        assert is_model_gated("some/model") is False

    def test_adding_and_removing_from_dict(self):
        GATED_MODELS["test/temp"] = True
        assert is_model_gated("test/temp") is True
        del GATED_MODELS["test/temp"]
        assert is_model_gated("test/temp") is False


# ---------------------------------------------------------------------------
# AuthStatus
# ---------------------------------------------------------------------------


class TestAuthStatus:
    def test_all_values(self):
        assert AuthStatus.OK.value == "ok"
        assert AuthStatus.NO_TOKEN.value == "no_token"
        assert AuthStatus.NEEDS_TERMS.value == "needs_terms"
        assert AuthStatus.NOT_FOUND.value == "not_found"
        assert AuthStatus.ERROR.value == "error"

    def test_comparison(self):
        assert AuthStatus.OK != AuthStatus.NO_TOKEN
        assert AuthStatus.OK == AuthStatus.OK


# ---------------------------------------------------------------------------
# check_auth_status
# ---------------------------------------------------------------------------


class TestCheckAuthStatus:
    def test_ungated_model_returns_ok_without_network(self):
        # Should short-circuit before any HfApi call
        with patch("huggingface_hub.HfApi") as mock_cls:
            status = check_auth_status("totally/unknown")
        mock_cls.assert_not_called()
        assert status == AuthStatus.OK

    def test_accessible_returns_ok(self):
        with patch("huggingface_hub.HfApi") as mock_cls:
            mock_cls.return_value.model_info.return_value = MagicMock()
            assert check_auth_status("kyutai/pocket-tts") == AuthStatus.OK

    def test_no_token_returns_no_token(self):
        with patch("huggingface_hub.HfApi") as mock_cls:
            mock_cls.return_value.model_info.side_effect = _no_token_error()
            assert check_auth_status("kyutai/pocket-tts") == AuthStatus.NO_TOKEN

    def test_gated_returns_needs_terms(self):
        with patch("huggingface_hub.HfApi") as mock_cls:
            mock_cls.return_value.model_info.side_effect = _gated_error()
            assert check_auth_status("kyutai/pocket-tts") == AuthStatus.NEEDS_TERMS

    def test_repo_not_found_returns_not_found(self):
        with patch("huggingface_hub.HfApi") as mock_cls:
            mock_cls.return_value.model_info.side_effect = _not_found_error()
            assert check_auth_status("kyutai/pocket-tts") == AuthStatus.NOT_FOUND

    def test_unexpected_exception_returns_no_token(self):
        # Generic exception → assume no token so user can try logging in
        with patch("huggingface_hub.HfApi") as mock_cls:
            mock_cls.return_value.model_info.side_effect = RuntimeError("network down")
            assert check_auth_status("kyutai/pocket-tts") == AuthStatus.NO_TOKEN


# ---------------------------------------------------------------------------
# do_login
# ---------------------------------------------------------------------------


class TestDoLogin:
    def test_empty_string_fails_immediately(self):
        ok, msg = do_login("")
        assert ok is False
        assert "No token" in msg

    def test_whitespace_only_attempts_login(self):
        # "   " is truthy so do_login bypasses the empty-string guard and calls
        # login(); the call will fail (no real HF server) so ok must be False.
        with patch(
            "huggingface_hub.login",
            side_effect=ValueError("invalid token"),
        ):
            ok, msg = do_login("   ")
        assert ok is False
        assert "failed" in msg.lower()

    def test_valid_hf_token_succeeds(self):
        with patch("huggingface_hub.login") as mock_login:
            mock_login.return_value = None
            ok, msg = do_login("hf_validtoken123")
        assert ok is True
        assert "accepted" in msg.lower()
        mock_login.assert_called_once_with(token="hf_validtoken123", add_to_git_credential=False)

    def test_non_hf_prefix_still_attempts_login(self):
        """Tokens not starting with hf_ trigger a warning but still try to login."""
        with patch("huggingface_hub.login") as mock_login:
            mock_login.return_value = None
            ok, msg = do_login("sk_someothertoken")
        assert ok is True  # login call succeeded (mocked)

    def test_login_exception_returns_false_with_message(self):
        with patch("huggingface_hub.login", side_effect=ValueError("invalid")):
            ok, msg = do_login("hf_bad")
        assert ok is False
        assert "failed" in msg.lower()

    def test_login_network_error_returns_false(self):
        with patch("huggingface_hub.login", side_effect=ConnectionError("timeout")):
            ok, msg = do_login("hf_abc")
        assert ok is False
        assert len(msg) > 0


# ---------------------------------------------------------------------------
# verify_access
# ---------------------------------------------------------------------------


class TestVerifyAccess:
    def test_accessible_model_ok(self):
        with patch("huggingface_hub.HfApi") as mock_cls:
            mock_cls.return_value.model_info.return_value = MagicMock()
            ok, msg = verify_access("kyutai/pocket-tts")
        assert ok is True
        assert "granted" in msg.lower()

    def test_still_gated_returns_false_with_hint(self):
        with patch("huggingface_hub.HfApi") as mock_cls:
            mock_cls.return_value.model_info.side_effect = _gated_error()
            ok, msg = verify_access("kyutai/pocket-tts")
        assert ok is False
        assert "processing" in msg.lower() or "not yet" in msg.lower()

    def test_unexpected_error_returns_false(self):
        with patch("huggingface_hub.HfApi") as mock_cls:
            mock_cls.return_value.model_info.side_effect = RuntimeError("boom")
            ok, msg = verify_access("kyutai/pocket-tts")
        assert ok is False
        assert "error" in msg.lower()

    def test_default_model_id(self):
        """verify_access() should default to pocket-tts."""
        with patch("huggingface_hub.HfApi") as mock_cls:
            mock_cls.return_value.model_info.return_value = MagicMock()
            ok, _ = verify_access()
        assert ok is True


# ---------------------------------------------------------------------------
# Browser helpers
# ---------------------------------------------------------------------------


class TestBrowserHelpers:
    def test_signup_opens_correct_url(self):
        with patch("kenkui.huggingface_auth.webbrowser.open") as mock_open:
            open_signup_page()
        mock_open.assert_called_once_with(HF_SIGNUP_URL)

    def test_token_opens_correct_url(self):
        with patch("kenkui.huggingface_auth.webbrowser.open") as mock_open:
            open_token_page()
        mock_open.assert_called_once_with(HF_TOKEN_URL)

    def test_model_page_opens_correct_url(self):
        with patch("kenkui.huggingface_auth.webbrowser.open") as mock_open:
            open_model_page("kyutai/pocket-tts")
        mock_open.assert_called_once_with("https://huggingface.co/kyutai/pocket-tts")

    def test_model_page_default_model(self):
        with patch("kenkui.huggingface_auth.webbrowser.open") as mock_open:
            open_model_page()
        mock_open.assert_called_once_with("https://huggingface.co/kyutai/pocket-tts")

    def test_browser_exception_does_not_propagate(self):
        err = Exception("no browser available")
        with patch("kenkui.huggingface_auth.webbrowser.open", side_effect=err):
            open_signup_page()  # must not raise
            open_token_page()  # must not raise
            open_model_page()  # must not raise


# ---------------------------------------------------------------------------
# ensure_huggingface_access — noninteractive compatibility check
# ---------------------------------------------------------------------------


class TestEnsureHuggingfaceAccess:
    def test_already_ok_returns_true_without_interaction(self):
        with (
            patch("kenkui.huggingface_auth.check_auth_status", return_value=AuthStatus.OK),
            patch("builtins.input") as mock_input,
            patch("builtins.print") as mock_print,
        ):
            assert ensure_huggingface_access() is True
        mock_input.assert_not_called()
        mock_print.assert_not_called()

    def test_not_found_returns_false(self):
        with (
            patch("kenkui.huggingface_auth.check_auth_status", return_value=AuthStatus.NOT_FOUND),
            patch("builtins.input") as mock_input,
            patch("builtins.print") as mock_print,
        ):
            assert ensure_huggingface_access() is False
        mock_input.assert_not_called()
        mock_print.assert_not_called()

    def test_not_ok_statuses_return_false_without_prompting(self):
        for status in (AuthStatus.NO_TOKEN, AuthStatus.NEEDS_TERMS, AuthStatus.ERROR):
            with (
                patch("kenkui.huggingface_auth.check_auth_status", return_value=status),
                patch("builtins.input") as mock_input,
                patch("builtins.print") as mock_print,
            ):
                result = ensure_huggingface_access(skip_if_no_interaction=False)
            assert result is False, f"Expected False for status {status}"
            mock_input.assert_not_called()
            mock_print.assert_not_called()

    def test_ungated_model_always_ok(self):
        # Should never call HfApi at all
        with patch("huggingface_hub.HfApi") as mock_cls:
            result = ensure_huggingface_access("ungated/model")
        mock_cls.assert_not_called()
        assert result is True


# ---------------------------------------------------------------------------
# check_voice_access
# ---------------------------------------------------------------------------


class TestCheckVoiceAccess:
    def test_builtin_voice_no_auth_needed(self):
        # Builtin voices are not custom → return True without calling HfApi
        with patch("huggingface_hub.HfApi") as mock_cls:
            result = check_voice_access("alba")
        mock_cls.assert_not_called()
        assert result is True

    def test_unknown_voice_triggers_auth_check(self):
        with patch(
            "kenkui.huggingface_auth.ensure_huggingface_access", return_value=True
        ) as mock_efa:
            result = check_voice_access("MyVoice")
        mock_efa.assert_called_once_with("kyutai/pocket-tts")
        assert result is True

    def test_auth_failure_propagates(self):
        with patch("kenkui.huggingface_auth.ensure_huggingface_access", return_value=False):
            result = check_voice_access("MyVoice")
        assert result is False

    def test_hf_url_triggers_auth_check(self):
        with patch(
            "kenkui.huggingface_auth.ensure_huggingface_access", return_value=True
        ) as mock_efa:
            check_voice_access("hf://user/repo/voice.wav")
        mock_efa.assert_called_once()
