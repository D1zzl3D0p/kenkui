"""HuggingFace authentication helpers for custom voice access.

This module provides two interfaces:

1. **Programmatic API** (``check_auth_status``, ``do_login``, ``verify_access``) —
   pure functions that return structured results, suitable for calling from a TUI.

2. **Legacy CLI helpers** (``ensure_huggingface_access``) — kept for headless /
   script usage, wraps the programmatic API with stdin/stdout interaction.
"""

from __future__ import annotations

import logging
import webbrowser
from enum import Enum
from pathlib import Path

from huggingface_hub import HfApi, login
from huggingface_hub.errors import (
    GatedRepoError,
    LocalTokenNotFoundError,
    RepositoryNotFoundError,
)

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


def is_custom_voice(voice: str, bundled_voices: list[str]) -> bool:
    """Return True if the voice is not a built-in default."""
    from .utils import DEFAULT_VOICES

    if voice in DEFAULT_VOICES:
        return False
    if Path(voice).exists():
        return True
    if voice.startswith("hf://"):
        return True
    if f"{voice}.wav" in bundled_voices:
        return True
    return True


def check_auth_status(model_id: str = "kyutai/pocket-tts") -> AuthStatus:
    """Check whether the user can access *model_id* right now.

    Returns an :class:`AuthStatus` indicating what (if anything) the user
    needs to do before the model can be downloaded.
    """
    if not is_model_gated(model_id):
        return AuthStatus.OK

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
        login(token=token, add_to_git_credential=False)
        logger.debug("HuggingFace login succeeded")
        return True, "Token accepted."
    except Exception as exc:
        return False, f"Token validation failed: {exc}"


def verify_access(model_id: str = "kyutai/pocket-tts") -> tuple[bool, str]:
    """Verify that the current token grants access to *model_id*.

    Returns ``(success, message)``.  Call this after the user has accepted
    the model's terms of use on the HuggingFace website.
    """
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
# Legacy CLI helper (kept for headless / script usage)
# ---------------------------------------------------------------------------


def ensure_huggingface_access(
    model_id: str = "kyutai/pocket-tts",
    skip_if_no_interaction: bool = False,
) -> bool:
    """Ensure the user has access to *model_id*, prompting via stdin/stdout.

    This is the legacy CLI flow.  In TUI mode use :class:`HuggingFaceAuthModal`
    from ``kenkui.widgets`` instead.
    """
    status = check_auth_status(model_id)

    if status == AuthStatus.OK:
        return True
    if status == AuthStatus.NOT_FOUND:
        print(f"Error: Model '{model_id}' not found on HuggingFace.")
        return False
    if skip_if_no_interaction:
        return False

    if status == AuthStatus.NO_TOKEN:
        return _cli_setup_authentication(model_id)
    if status == AuthStatus.NEEDS_TERMS:
        return _cli_accept_terms_flow(model_id)
    return False


def _cli_setup_authentication(model_id: str) -> bool:
    print()
    print("=== HuggingFace Setup Required ===")
    print()
    print("Custom voices require a free HuggingFace account.")
    print()
    print("[1] I have an account")
    print("[2] I need to create one  (opens browser)")
    print("[3] Skip (custom voices won't work)")
    choice = input("\nEnter 1, 2, or 3: ").strip()

    if choice == "3":
        return False

    if choice == "2":
        print("\nOpening signup page...")
        open_signup_page()
        print("Complete signup in your browser, then return here.")
        input("Press Enter when your account is ready...")

    return _cli_token_flow(model_id)


def _cli_token_flow(model_id: str) -> bool:
    print("\n--- Step 1: Create an Access Token ---")
    print("Opening token creation page...")
    open_token_page()
    print("\nInstructions:")
    print("  1. Click 'Create token' in the browser (select 'Read' type)")
    print("  2. Copy the token (starts with 'hf_')")
    print("  3. Paste it below")

    for attempt in range(3):
        print()
        token = input("Token: ").strip()
        ok, msg = do_login(token)
        print(msg)
        if ok:
            return _cli_accept_terms_flow(model_id)
        if attempt < 2:
            print("Please try again.")

    print("Could not authenticate. Custom voices will not be available.")
    return False


def _cli_accept_terms_flow(model_id: str) -> bool:
    print("\n--- Step 2: Accept Terms of Use ---")
    print("Opening model page...")
    open_model_page(model_id)
    print("\nInstructions:")
    print("  1. Find the 'Access repository' / 'Gated model' section")
    print("  2. Click 'Agree and access repository'")
    print("  3. Return here")

    while True:
        print()
        response = input("Have you accepted the terms? (y/n): ").strip().lower()
        if response in ("y", "yes"):
            ok, msg = verify_access(model_id)
            print(msg)
            if ok:
                return True
            retry = input("Try again? (y/n): ").strip().lower()
            if retry not in ("y", "yes"):
                return False
        elif response in ("n", "no"):
            skip = input("Skip for now? (y/n): ").strip().lower()
            if skip in ("y", "yes"):
                return False
        else:
            print("Please enter 'y' or 'n'.")


def check_voice_access(voice: str, bundled_voices: list[str]) -> bool:
    """Check if the user has access needed for a specific voice (CLI path)."""
    if not is_custom_voice(voice, bundled_voices):
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
