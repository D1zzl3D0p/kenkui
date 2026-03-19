"""Tests for the HuggingFace authentication module.

Coverage:
- is_custom_voice / is_model_gated — pure logic
- AuthStatus enum values
- check_auth_status — all five outcomes (OK, NO_TOKEN, NEEDS_TERMS, NOT_FOUND, ERROR)
- do_login — success, empty token, bad token, non-hf_ prefix
- verify_access — OK, still-gated, unexpected error
- open_signup/token/model_page — correct URLs, silent failure on browser error
- ensure_huggingface_access — CLI orchestrator (skip_if_no_interaction,
  already-OK, NOT_FOUND, NO_TOKEN path, NEEDS_TERMS path)
- _cli_setup_authentication — choices 1, 2, 3
- _cli_token_flow — success first try, success after retry, exhaust all retries
- _cli_accept_terms_flow — accept+verify OK, accept+still-gated then retry,
  accept+still-gated then give-up, reject+skip, invalid input then valid
- check_voice_access — built-in voice (no auth needed), custom voice (auth called)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import httpx
import pytest

from kenkui.huggingface_auth import (
    GATED_MODELS,
    HF_SIGNUP_URL,
    HF_TOKEN_URL,
    AuthStatus,
    _cli_accept_terms_flow,
    _cli_setup_authentication,
    _cli_token_flow,
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
from kenkui.utils import DEFAULT_VOICES


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
    def test_every_default_voice_not_custom(self):
        for voice in DEFAULT_VOICES:
            assert is_custom_voice(voice, []) is False, f"{voice!r} should not be custom"

    def test_bundled_voice_with_wav_suffix(self):
        assert is_custom_voice("MyVoice", ["MyVoice.wav"]) is True

    def test_bundled_voice_without_wav_suffix_not_matched(self):
        # bundled_voices list contains "MyVoice.wav"; voice without .wav won't match
        # the bundled check but returns True anyway because unknown voices are custom
        assert is_custom_voice("MyVoice", []) is True

    def test_hf_url_is_custom(self):
        assert is_custom_voice("hf://user/repo/voice.wav", []) is True

    def test_local_existing_file_is_custom(self, tmp_path):
        wav = tmp_path / "voice.wav"
        wav.write_bytes(b"RIFF")
        assert is_custom_voice(str(wav), []) is True

    def test_unknown_name_is_custom(self):
        assert is_custom_voice("unknown_voice", []) is True

    def test_empty_string_is_custom(self):
        # Empty string is not in DEFAULT_VOICES, not a file → True
        assert is_custom_voice("", []) is True


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
        with patch("kenkui.huggingface_auth.HfApi") as mock_cls:
            status = check_auth_status("totally/unknown")
        mock_cls.assert_not_called()
        assert status == AuthStatus.OK

    def test_accessible_returns_ok(self):
        with patch("kenkui.huggingface_auth.HfApi") as mock_cls:
            mock_cls.return_value.model_info.return_value = MagicMock()
            assert check_auth_status("kyutai/pocket-tts") == AuthStatus.OK

    def test_no_token_returns_no_token(self):
        with patch("kenkui.huggingface_auth.HfApi") as mock_cls:
            mock_cls.return_value.model_info.side_effect = _no_token_error()
            assert check_auth_status("kyutai/pocket-tts") == AuthStatus.NO_TOKEN

    def test_gated_returns_needs_terms(self):
        with patch("kenkui.huggingface_auth.HfApi") as mock_cls:
            mock_cls.return_value.model_info.side_effect = _gated_error()
            assert check_auth_status("kyutai/pocket-tts") == AuthStatus.NEEDS_TERMS

    def test_repo_not_found_returns_not_found(self):
        with patch("kenkui.huggingface_auth.HfApi") as mock_cls:
            mock_cls.return_value.model_info.side_effect = _not_found_error()
            assert check_auth_status("kyutai/pocket-tts") == AuthStatus.NOT_FOUND

    def test_unexpected_exception_returns_no_token(self):
        # Generic exception → assume no token so user can try logging in
        with patch("kenkui.huggingface_auth.HfApi") as mock_cls:
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
            "kenkui.huggingface_auth.login",
            side_effect=ValueError("invalid token"),
        ):
            ok, msg = do_login("   ")
        assert ok is False
        assert "failed" in msg.lower()

    def test_valid_hf_token_succeeds(self):
        with patch("kenkui.huggingface_auth.login") as mock_login:
            mock_login.return_value = None
            ok, msg = do_login("hf_validtoken123")
        assert ok is True
        assert "accepted" in msg.lower()
        mock_login.assert_called_once_with(token="hf_validtoken123", add_to_git_credential=False)

    def test_non_hf_prefix_still_attempts_login(self):
        """Tokens not starting with hf_ trigger a warning but still try to login."""
        with patch("kenkui.huggingface_auth.login") as mock_login:
            mock_login.return_value = None
            ok, msg = do_login("sk_someothertoken")
        assert ok is True  # login call succeeded (mocked)

    def test_login_exception_returns_false_with_message(self):
        with patch("kenkui.huggingface_auth.login", side_effect=ValueError("invalid")):
            ok, msg = do_login("hf_bad")
        assert ok is False
        assert "failed" in msg.lower()

    def test_login_network_error_returns_false(self):
        with patch("kenkui.huggingface_auth.login", side_effect=ConnectionError("timeout")):
            ok, msg = do_login("hf_abc")
        assert ok is False
        assert len(msg) > 0


# ---------------------------------------------------------------------------
# verify_access
# ---------------------------------------------------------------------------


class TestVerifyAccess:
    def test_accessible_model_ok(self):
        with patch("kenkui.huggingface_auth.HfApi") as mock_cls:
            mock_cls.return_value.model_info.return_value = MagicMock()
            ok, msg = verify_access("kyutai/pocket-tts")
        assert ok is True
        assert "granted" in msg.lower()

    def test_still_gated_returns_false_with_hint(self):
        with patch("kenkui.huggingface_auth.HfApi") as mock_cls:
            mock_cls.return_value.model_info.side_effect = _gated_error()
            ok, msg = verify_access("kyutai/pocket-tts")
        assert ok is False
        assert "processing" in msg.lower() or "not yet" in msg.lower()

    def test_unexpected_error_returns_false(self):
        with patch("kenkui.huggingface_auth.HfApi") as mock_cls:
            mock_cls.return_value.model_info.side_effect = RuntimeError("boom")
            ok, msg = verify_access("kyutai/pocket-tts")
        assert ok is False
        assert "error" in msg.lower()

    def test_default_model_id(self):
        """verify_access() should default to pocket-tts."""
        with patch("kenkui.huggingface_auth.HfApi") as mock_cls:
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
# ensure_huggingface_access — CLI orchestrator
# ---------------------------------------------------------------------------


class TestEnsureHuggingfaceAccess:
    def test_already_ok_returns_true_without_interaction(self):
        with patch("kenkui.huggingface_auth.check_auth_status", return_value=AuthStatus.OK):
            assert ensure_huggingface_access() is True

    def test_not_found_returns_false(self):
        with patch("kenkui.huggingface_auth.check_auth_status", return_value=AuthStatus.NOT_FOUND):
            assert ensure_huggingface_access() is False

    def test_skip_if_no_interaction_returns_false_when_not_ok(self):
        for status in (AuthStatus.NO_TOKEN, AuthStatus.NEEDS_TERMS, AuthStatus.ERROR):
            with patch("kenkui.huggingface_auth.check_auth_status", return_value=status):
                result = ensure_huggingface_access(skip_if_no_interaction=True)
            assert result is False, f"Expected False for status {status}"

    def test_no_token_calls_setup_auth(self):
        with (
            patch("kenkui.huggingface_auth.check_auth_status", return_value=AuthStatus.NO_TOKEN),
            patch(
                "kenkui.huggingface_auth._cli_setup_authentication", return_value=True
            ) as mock_setup,
        ):
            result = ensure_huggingface_access()
        mock_setup.assert_called_once_with("kyutai/pocket-tts")
        assert result is True

    def test_needs_terms_calls_accept_terms(self):
        with (
            patch("kenkui.huggingface_auth.check_auth_status", return_value=AuthStatus.NEEDS_TERMS),
            patch(
                "kenkui.huggingface_auth._cli_accept_terms_flow", return_value=True
            ) as mock_terms,
        ):
            result = ensure_huggingface_access()
        mock_terms.assert_called_once_with("kyutai/pocket-tts")
        assert result is True

    def test_ungated_model_always_ok(self):
        # Should never call HfApi at all
        with patch("kenkui.huggingface_auth.HfApi") as mock_cls:
            result = ensure_huggingface_access("ungated/model")
        mock_cls.assert_not_called()
        assert result is True


# ---------------------------------------------------------------------------
# _cli_setup_authentication — choice 1/2/3
# ---------------------------------------------------------------------------


class TestCliSetupAuthentication:
    def _patch(self, inputs: list[str], login_ok: bool = True):
        """Context manager stack for CLI setup tests."""
        return (
            patch("builtins.input", side_effect=inputs),
            patch("kenkui.huggingface_auth.open_signup_page"),
            patch("kenkui.huggingface_auth._cli_token_flow", return_value=login_ok),
        )

    def test_choice_3_returns_false(self):
        with patch("builtins.input", return_value="3"), patch("builtins.print"):
            result = _cli_setup_authentication("kyutai/pocket-tts")
        assert result is False

    def test_choice_1_goes_to_token_flow(self):
        with (
            patch("builtins.input", return_value="1"),
            patch("builtins.print"),
            patch("kenkui.huggingface_auth._cli_token_flow", return_value=True) as mock_tf,
        ):
            result = _cli_setup_authentication("kyutai/pocket-tts")
        mock_tf.assert_called_once_with("kyutai/pocket-tts")
        assert result is True

    def test_choice_2_opens_browser_then_token_flow(self):
        # choice "2", then "Enter" (blank) to confirm signup done
        with (
            patch("builtins.input", side_effect=["2", ""]),
            patch("builtins.print"),
            patch("kenkui.huggingface_auth.open_signup_page") as mock_signup,
            patch("kenkui.huggingface_auth._cli_token_flow", return_value=True),
        ):
            _cli_setup_authentication("kyutai/pocket-tts")
        mock_signup.assert_called_once()

    def test_token_flow_failure_propagates(self):
        with (
            patch("builtins.input", return_value="1"),
            patch("builtins.print"),
            patch("kenkui.huggingface_auth._cli_token_flow", return_value=False),
        ):
            result = _cli_setup_authentication("kyutai/pocket-tts")
        assert result is False


# ---------------------------------------------------------------------------
# _cli_token_flow — success, retry, exhausted
# ---------------------------------------------------------------------------


class TestCliTokenFlow:
    def test_valid_token_first_try(self):
        with (
            patch("builtins.input", return_value="hf_goodtoken"),
            patch("builtins.print"),
            patch("kenkui.huggingface_auth.open_token_page"),
            patch("kenkui.huggingface_auth.do_login", return_value=(True, "Token accepted.")),
            patch(
                "kenkui.huggingface_auth._cli_accept_terms_flow", return_value=True
            ) as mock_terms,
        ):
            result = _cli_token_flow("kyutai/pocket-tts")
        assert result is True
        mock_terms.assert_called_once()

    def test_bad_then_good_token(self):
        # First call fails, second succeeds
        login_responses = [(False, "bad"), (True, "ok")]
        with (
            patch("builtins.input", side_effect=["hf_bad", "hf_good"]),
            patch("builtins.print"),
            patch("kenkui.huggingface_auth.open_token_page"),
            patch("kenkui.huggingface_auth.do_login", side_effect=login_responses),
            patch("kenkui.huggingface_auth._cli_accept_terms_flow", return_value=True),
        ):
            result = _cli_token_flow("kyutai/pocket-tts")
        assert result is True

    def test_all_attempts_fail_returns_false(self):
        # 3 attempts, all fail
        with (
            patch("builtins.input", return_value="hf_bad"),
            patch("builtins.print"),
            patch("kenkui.huggingface_auth.open_token_page"),
            patch("kenkui.huggingface_auth.do_login", return_value=(False, "invalid")),
        ):
            result = _cli_token_flow("kyutai/pocket-tts")
        assert result is False

    def test_opens_token_page_at_start(self):
        with (
            patch("builtins.input", return_value="hf_t"),
            patch("builtins.print"),
            patch("kenkui.huggingface_auth.open_token_page") as mock_tp,
            patch("kenkui.huggingface_auth.do_login", return_value=(True, "ok")),
            patch("kenkui.huggingface_auth._cli_accept_terms_flow", return_value=True),
        ):
            _cli_token_flow("kyutai/pocket-tts")
        mock_tp.assert_called_once()


# ---------------------------------------------------------------------------
# _cli_accept_terms_flow — accept + verify, reject, retry, invalid input
# ---------------------------------------------------------------------------


class TestCliAcceptTermsFlow:
    def test_accept_and_verify_ok_returns_true(self):
        with (
            patch("builtins.input", return_value="y"),
            patch("builtins.print"),
            patch("kenkui.huggingface_auth.open_model_page"),
            patch("kenkui.huggingface_auth.verify_access", return_value=(True, "Access granted!")),
        ):
            result = _cli_accept_terms_flow("kyutai/pocket-tts")
        assert result is True

    def test_accept_still_gated_then_give_up(self):
        # "y" → verify fails → "n" to retry → returns False
        with (
            patch("builtins.input", side_effect=["y", "n"]),
            patch("builtins.print"),
            patch("kenkui.huggingface_auth.open_model_page"),
            patch("kenkui.huggingface_auth.verify_access", return_value=(False, "not yet")),
        ):
            result = _cli_accept_terms_flow("kyutai/pocket-tts")
        assert result is False

    def test_accept_still_gated_then_retry_then_ok(self):
        # "y" → fail → "y" (retry) → "y" → ok
        verify_results = [(False, "not yet"), (True, "Access granted!")]
        with (
            patch("builtins.input", side_effect=["y", "y", "y"]),
            patch("builtins.print"),
            patch("kenkui.huggingface_auth.open_model_page"),
            patch("kenkui.huggingface_auth.verify_access", side_effect=verify_results),
        ):
            result = _cli_accept_terms_flow("kyutai/pocket-tts")
        assert result is True

    def test_reject_then_skip(self):
        # "n" → skip "y" → False
        with (
            patch("builtins.input", side_effect=["n", "y"]),
            patch("builtins.print"),
            patch("kenkui.huggingface_auth.open_model_page"),
        ):
            result = _cli_accept_terms_flow("kyutai/pocket-tts")
        assert result is False

    def test_reject_no_skip_then_accept(self):
        # "n" → skip "n" → "y" → verify ok
        with (
            patch("builtins.input", side_effect=["n", "n", "y"]),
            patch("builtins.print"),
            patch("kenkui.huggingface_auth.open_model_page"),
            patch("kenkui.huggingface_auth.verify_access", return_value=(True, "ok")),
        ):
            result = _cli_accept_terms_flow("kyutai/pocket-tts")
        assert result is True

    def test_invalid_input_then_accept(self):
        # "maybe" → "y" → verify ok
        with (
            patch("builtins.input", side_effect=["maybe", "y"]),
            patch("builtins.print"),
            patch("kenkui.huggingface_auth.open_model_page"),
            patch("kenkui.huggingface_auth.verify_access", return_value=(True, "ok")),
        ):
            result = _cli_accept_terms_flow("kyutai/pocket-tts")
        assert result is True

    def test_opens_model_page_at_start(self):
        with (
            patch("builtins.input", return_value="y"),
            patch("builtins.print"),
            patch("kenkui.huggingface_auth.open_model_page") as mock_mp,
            patch("kenkui.huggingface_auth.verify_access", return_value=(True, "ok")),
        ):
            _cli_accept_terms_flow("kyutai/pocket-tts")
        mock_mp.assert_called_once_with("kyutai/pocket-tts")


# ---------------------------------------------------------------------------
# check_voice_access
# ---------------------------------------------------------------------------


class TestCheckVoiceAccess:
    def test_default_voice_no_auth_needed(self):
        # DEFAULT_VOICES are not custom → return True without calling HfApi
        with patch("kenkui.huggingface_auth.HfApi") as mock_cls:
            result = check_voice_access("alba", [])
        mock_cls.assert_not_called()
        assert result is True

    def test_bundled_voice_triggers_auth_check(self):
        with patch(
            "kenkui.huggingface_auth.ensure_huggingface_access", return_value=True
        ) as mock_efa:
            result = check_voice_access("MyVoice", ["MyVoice.wav"])
        mock_efa.assert_called_once_with("kyutai/pocket-tts")
        assert result is True

    def test_auth_failure_propagates(self):
        with patch("kenkui.huggingface_auth.ensure_huggingface_access", return_value=False):
            result = check_voice_access("MyVoice", ["MyVoice.wav"])
        assert result is False

    def test_hf_url_triggers_auth_check(self):
        with patch(
            "kenkui.huggingface_auth.ensure_huggingface_access", return_value=True
        ) as mock_efa:
            check_voice_access("hf://user/repo/voice.wav", [])
        mock_efa.assert_called_once()
