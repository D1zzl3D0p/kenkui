"""HuggingFace authentication helpers for custom voice access.

The library core exposes status, login, verification, and URL-opening helpers.
Callers own any interactive setup UI.
"""

from __future__ import annotations

import logging
import webbrowser
from enum import Enum
from pathlib import Path

# huggingface_hub is an optional dependency (kenkui[custom-voices]).
# All imports are lazy (inside functions) so the base install works without it.

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GATED_MODELS: dict[str, bool] = {
    "kyutai/pocket-tts": True,
}

HF_SIGNUP_URL = "https://huggingface.co/join"
HF_TOKEN_URL = "https://huggingface.co/settings/tokens/new?tokenType=read&name=KenkuiVoices"


class AuthStatus(Enum):
    """Result of a HuggingFace authentication check."""

    OK = "ok"  # Token present and model accessible
    NO_TOKEN = "no_token"  # No local token found
    NEEDS_TERMS = "needs_terms"  # Token OK but model terms not accepted
    NOT_FOUND = "not_found"  # Model does not exist
    ERROR = "error"  # Unexpected error


# ---------------------------------------------------------------------------
# Programmatic API
# ---------------------------------------------------------------------------


def is_model_gated(model_id: str) -> bool:
    """Return True if the model requires HF authentication."""
    return GATED_MODELS.get(model_id, False)


def is_custom_voice(voice: str) -> bool:
    """Return True if this voice requires HuggingFace authentication.

    Runtime voices are catalog IDs.  Known catalog voices are either built-in
    Pocket TTS IDs or local compiled assets and do not require HuggingFace auth.
    - Unknown voices are treated as requiring auth (safe default).
    """
    if voice.startswith("hf://"):
        return True
    if Path(voice).exists():
        return False

    from .voice_registry import get_catalog
    meta = get_catalog().resolve(voice)
    if meta is None:
        return True  # Unknown voice — be safe
    return False


def check_auth_status(model_id: str = "kyutai/pocket-tts") -> AuthStatus:
    """Check whether the user can access *model_id* right now.

    Returns an :class:`AuthStatus` indicating what (if anything) the user
    needs to do before the model can be downloaded.
    """
    if not is_model_gated(model_id):
        return AuthStatus.OK

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import (
            GatedRepoError,
            LocalTokenNotFoundError,
            RepositoryNotFoundError,
        )
    except ImportError:
        logger.warning("huggingface_hub not installed — install kenkui[custom-voices] to use custom voices")
        return AuthStatus.NO_TOKEN

    api = HfApi()
    try:
        api.model_info(model_id)
        return AuthStatus.OK
    except LocalTokenNotFoundError:
        return AuthStatus.NO_TOKEN
    except GatedRepoError:
        return AuthStatus.NEEDS_TERMS
    except RepositoryNotFoundError:
        logger.error("Model %r not found on HuggingFace", model_id)
        return AuthStatus.NOT_FOUND
    except Exception as exc:
        logger.warning("Could not check model access: %s", exc)
        return AuthStatus.NO_TOKEN  # Assume token missing and let user try


def do_login(token: str) -> tuple[bool, str]:
    """Attempt to log in with *token*.

    Returns ``(success, message)`` where *message* is a human-readable
    explanation on failure.
    """
    if not token:
        return False, "No token provided."
    if not token.startswith("hf_"):
        logger.warning("Token does not start with 'hf_' — proceeding anyway")

    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        logger.debug("HuggingFace login succeeded")
        return True, "Token accepted."
    except ImportError:
        return False, "huggingface_hub not installed — run: pip install kenkui[custom-voices]"
    except Exception as exc:
        return False, f"Token validation failed: {exc}"


def verify_access(model_id: str = "kyutai/pocket-tts") -> tuple[bool, str]:
    """Verify that the current token grants access to *model_id*.

    Returns ``(success, message)``.  Call this after the user has accepted
    the model's terms of use on the HuggingFace website.
    """
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import GatedRepoError
    except ImportError:
        return False, "huggingface_hub not installed — run: pip install kenkui[custom-voices]"

    api = HfApi()
    try:
        api.model_info(model_id)
        return True, "Access granted! Custom voices are now available."
    except GatedRepoError:
        return False, (
            "Access not yet confirmed. The terms acceptance may still be processing — "
            "please wait a moment and try again."
        )
    except Exception as exc:
        return False, f"Error verifying access: {exc}"


def open_signup_page() -> None:
    """Open the HuggingFace account creation page in the default browser."""
    try:
        webbrowser.open(HF_SIGNUP_URL)
    except Exception as exc:
        logger.warning("Could not open browser: %s", exc)


def open_token_page() -> None:
    """Open the HuggingFace token creation page in the default browser."""
    try:
        webbrowser.open(HF_TOKEN_URL)
    except Exception as exc:
        logger.warning("Could not open browser: %s", exc)


def open_model_page(model_id: str = "kyutai/pocket-tts") -> None:
    """Open the model page so the user can accept terms of use."""
    try:
        webbrowser.open(f"https://huggingface.co/{model_id}")
    except Exception as exc:
        logger.warning("Could not open browser: %s", exc)


# ---------------------------------------------------------------------------
# Noninteractive compatibility helpers
# ---------------------------------------------------------------------------


def ensure_huggingface_access(
    model_id: str = "kyutai/pocket-tts",
    skip_if_no_interaction: bool = False,
) -> bool:
    """Return True when the current environment can access *model_id*.

    This compatibility helper is intentionally noninteractive. Applications that
    receive ``False`` should inspect :func:`check_auth_status` and present their
    own login or terms-acceptance flow.
    """
    _ = skip_if_no_interaction
    status = check_auth_status(model_id)
    if status != AuthStatus.OK:
        logger.info("HuggingFace access for %s is not ready: %s", model_id, status.value)
    return status == AuthStatus.OK


def check_voice_access(voice: str) -> bool:
    """Check if the user has access needed for a specific voice (CLI path)."""
    if not is_custom_voice(voice):
        return True
    return ensure_huggingface_access("kyutai/pocket-tts")


__all__ = [
    "AuthStatus",
    "GATED_MODELS",
    "HF_SIGNUP_URL",
    "HF_TOKEN_URL",
    "is_model_gated",
    "is_custom_voice",
    "check_auth_status",
    "do_login",
    "verify_access",
    "open_signup_page",
    "open_token_page",
    "open_model_page",
    "ensure_huggingface_access",
    "check_voice_access",
]
