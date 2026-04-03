"""auth_service — structured wrappers around HuggingFace authentication.

Public API:
  get_hf_status(model_id) -> HFAuthStatus
  login(token) -> HFLoginResult
"""

from __future__ import annotations

from dataclasses import dataclass

from kenkui.huggingface_auth import AuthStatus, check_auth_status, do_login


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------


@dataclass
class HFAuthStatus:
    authenticated: bool
    username: str | None
    has_pocket_tts_access: bool


@dataclass
class HFLoginResult:
    authenticated: bool
    username: str | None
    error: str | None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _get_username() -> str | None:
    """Return the HuggingFace username of the currently logged-in user.

    Returns None on any exception (huggingface_hub not installed, not logged
    in, network error, etc.).
    """
    try:
        from huggingface_hub import whoami
        return whoami()["name"]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_hf_status(model_id: str = "kyutai/pocket-tts") -> HFAuthStatus:
    """Return structured HuggingFace authentication status for *model_id*.

    No prompts are issued; this is a pure inspection function.
    """
    status = check_auth_status(model_id)

    if status == AuthStatus.OK:
        return HFAuthStatus(
            authenticated=True,
            username=_get_username(),
            has_pocket_tts_access=True,
        )
    if status == AuthStatus.NEEDS_TERMS:
        return HFAuthStatus(
            authenticated=True,
            username=_get_username(),
            has_pocket_tts_access=False,
        )
    # AuthStatus.NO_TOKEN, AuthStatus.ERROR, AuthStatus.NOT_FOUND
    return HFAuthStatus(authenticated=False, username=None, has_pocket_tts_access=False)


def login(token: str) -> HFLoginResult:
    """Attempt to log in with *token* and return a structured result.

    No prompts are issued; callers supply the token directly.
    """
    success, message = do_login(token)

    if success:
        return HFLoginResult(
            authenticated=True,
            username=_get_username(),
            error=None,
        )
    return HFLoginResult(authenticated=False, username=None, error=message)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "HFAuthStatus",
    "HFLoginResult",
    "get_hf_status",
    "login",
]
