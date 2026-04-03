"""Tests for kenkui.services.auth_service."""

from __future__ import annotations

from unittest.mock import patch

from kenkui.huggingface_auth import AuthStatus
from kenkui.services.auth_service import (
    HFAuthStatus,
    HFLoginResult,
    _get_username,
    get_hf_status,
    login,
)


# ---------------------------------------------------------------------------
# get_hf_status tests
# ---------------------------------------------------------------------------


def test_get_hf_status_ok():
    with (
        patch("kenkui.services.auth_service.check_auth_status", return_value=AuthStatus.OK),
        patch("kenkui.services.auth_service._get_username", return_value="alice"),
    ):
        result = get_hf_status()

    assert isinstance(result, HFAuthStatus)
    assert result.authenticated is True
    assert result.has_pocket_tts_access is True
    assert result.username == "alice"


def test_get_hf_status_needs_terms():
    with (
        patch("kenkui.services.auth_service.check_auth_status", return_value=AuthStatus.NEEDS_TERMS),
        patch("kenkui.services.auth_service._get_username", return_value="bob"),
    ):
        result = get_hf_status()

    assert result.authenticated is True
    assert result.has_pocket_tts_access is False
    assert result.username == "bob"


def test_get_hf_status_no_token():
    with (
        patch("kenkui.services.auth_service.check_auth_status", return_value=AuthStatus.NO_TOKEN),
        patch("kenkui.services.auth_service._get_username") as mock_username,
    ):
        result = get_hf_status()

    assert result.authenticated is False
    assert result.has_pocket_tts_access is False
    mock_username.assert_not_called()


def test_get_hf_status_not_found():
    with (
        patch("kenkui.services.auth_service.check_auth_status", return_value=AuthStatus.NOT_FOUND),
        patch("kenkui.services.auth_service._get_username") as mock_username,
    ):
        result = get_hf_status()

    assert result.authenticated is False
    assert result.has_pocket_tts_access is False
    mock_username.assert_not_called()


def test_get_hf_status_error():
    with (
        patch("kenkui.services.auth_service.check_auth_status", return_value=AuthStatus.ERROR),
        patch("kenkui.services.auth_service._get_username") as mock_username,
    ):
        result = get_hf_status()

    assert result.authenticated is False
    assert result.has_pocket_tts_access is False
    mock_username.assert_not_called()


def test_get_hf_status_username_none_on_unauthenticated():
    """For any non-OK/NEEDS_TERMS status, username must be None without calling _get_username."""
    for status in (AuthStatus.NO_TOKEN, AuthStatus.ERROR, AuthStatus.NOT_FOUND):
        with (
            patch("kenkui.services.auth_service.check_auth_status", return_value=status),
            patch("kenkui.services.auth_service._get_username") as mock_username,
        ):
            result = get_hf_status()

        assert result.username is None, f"Expected None username for {status}"
        mock_username.assert_not_called()


# ---------------------------------------------------------------------------
# login tests
# ---------------------------------------------------------------------------


def test_login_success():
    with (
        patch("kenkui.services.auth_service.do_login", return_value=(True, "Token accepted.")),
        patch("kenkui.services.auth_service._get_username", return_value="carol"),
    ):
        result = login("hf_testtoken")

    assert isinstance(result, HFLoginResult)
    assert result.authenticated is True
    assert result.username == "carol"
    assert result.error is None


def test_login_failure():
    error_msg = "Token validation failed: 401 Unauthorized"
    with (
        patch("kenkui.services.auth_service.do_login", return_value=(False, error_msg)),
        patch("kenkui.services.auth_service._get_username"),
    ):
        result = login("hf_badtoken")

    assert result.authenticated is False
    assert result.username is None
    assert result.error == error_msg


def test_login_no_username_on_failure():
    with (
        patch("kenkui.services.auth_service.do_login", return_value=(False, "Token validation failed: bad")),
        patch("kenkui.services.auth_service._get_username") as mock_username,
    ):
        result = login("hf_badtoken")

    assert result.username is None
    mock_username.assert_not_called()


# ---------------------------------------------------------------------------
# _get_username tests
# ---------------------------------------------------------------------------


def test_get_username_returns_none_on_import_error():
    with patch.dict("sys.modules", {"huggingface_hub": None}):
        result = _get_username()
    assert result is None
