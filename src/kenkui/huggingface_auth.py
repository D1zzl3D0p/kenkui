"""HuggingFace authentication flow for custom voice access.

This module provides a user-friendly, step-by-step guide for users to:
1. Create a HuggingFace account (if needed)
2. Generate an access token
3. Accept terms for gated models
4. Download custom voice models

Designed for users with no prior HuggingFace knowledge.
"""

from __future__ import annotations

import webbrowser
from pathlib import Path

from huggingface_hub import HfApi, login
from huggingface_hub.errors import (
    GatedRepoError,
    LocalTokenNotFoundError,
    RepositoryNotFoundError,
)


def _print(msg: str = ""):
    """Print message, stripping Rich markup."""
    msg = msg.replace("[bold ", "").replace("[/bold]", "")
    msg = msg.replace("[cyan]", "").replace("[/cyan]", "")
    msg = msg.replace("[red]", "").replace("[/red]", "")
    msg = msg.replace("[yellow]", "").replace("[/yellow]", "")
    msg = msg.replace("[dim]", "").replace("[/dim]", "")
    msg = msg.replace("[green]", "").replace("[/green]", "")
    msg = msg.replace("[blue]", "").replace("[/blue]", "")
    print(msg)


def _input(prompt: str = "") -> str:
    """Get input from user."""
    return input(prompt)


# Model configuration - tracks which models are gated
GATED_MODELS = {
    "kyutai/pocket-tts": True,
}


def is_model_gated(model_id: str) -> bool:
    """Check if a model requires authentication/terms acceptance."""
    return GATED_MODELS.get(model_id, False)


def is_custom_voice(voice: str, bundled_voices: list[str]) -> bool:
    """Determine if a voice requires custom model access.

    Returns True if voice is NOT a default built-in voice.
    """
    from .utils import DEFAULT_VOICES

    if voice in DEFAULT_VOICES:
        return False

    voice_path = Path(voice)
    if voice_path.exists():
        return True

    if voice.startswith("hf://"):
        return True

    voice_filename = f"{voice}.wav"
    if voice_filename in bundled_voices:
        return True

    return True


def ensure_huggingface_access(
    model_id: str = "kyutai/pocket-tts",
    skip_if_no_interaction: bool = False,
) -> bool:
    """Ensure user has access to HuggingFace model with user-friendly flow."""
    if not is_model_gated(model_id):
        return True

    api = HfApi()

    try:
        api.model_info(model_id)
        return True
    except LocalTokenNotFoundError:
        return _setup_authentication(model_id)
    except GatedRepoError:
        return _accept_terms_flow(model_id)
    except RepositoryNotFoundError:
        _print(f"Error: Model '{model_id}' not found on HuggingFace")
        return False
    except Exception as e:
        _print(f"Warning: Could not check model access: {e}")
        return _setup_authentication(model_id)


def _setup_authentication(model_id: str) -> bool:
    """Guide user through creating account and token."""
    _print()
    _print("=== Welcome to Custom Voice Setup ===")
    _print()
    _print("To use custom voices, we need to connect to HuggingFace")
    _print("(a platform that hosts AI models). This is free and takes about 2 minutes.")
    _print()
    _print("Why is this needed?")
    _print("The custom voice technology is hosted on HuggingFace and requires")
    _print("accepting their terms of use.")

    _print("\nDo you have a HuggingFace account?")
    _print("[1] Yes, I have an account")
    _print("[2] No, I need to create one")
    _print("[3] Skip for now (custom voices won't work)")

    choice = _input("\nEnter 1, 2, or 3: ").strip()

    if choice == "3":
        _print("\nSkipping authentication. Custom voices will not be available.")
        return False

    if choice == "2":
        _print("\nLet's create your account:")
        _print("1. Opening signup page in your browser...")

        try:
            webbrowser.open("https://huggingface.co/join")
        except Exception:
            _print("   Could not open browser. Please visit:")
            _print("   https://huggingface.co/join")

        _print("\n2. Complete the signup form in your browser")
        _print("3. Once done, return here and press Enter")

        input("\nPress Enter when you've created your account...")

    return _create_token_flow(model_id)


def _create_token_flow(model_id: str) -> bool:
    """Guide user through creating and entering an access token."""
    _print("\n" + "=" * 60)
    _print("Step 1: Create an Access Token")
    _print("=" * 60)

    _print("\nWe need to create a 'read' token so kenkui can download voice models.")
    _print("Opening token creation page...")

    token_url = (
        "https://huggingface.co/settings/tokens/new?tokenType=read&name=KenkuiVoices"
    )

    try:
        webbrowser.open(token_url)
    except Exception:
        _print("Could not open browser. Please visit:")
        _print(token_url)

    _print("\nInstructions:")
    _print("1. In the browser, click 'Create token' (it will be a 'Read' token)")
    _print("2. Copy the token (it starts with 'hf_')")
    _print("3. Paste it below")

    max_attempts = 3
    for attempt in range(max_attempts):
        _print()
        token = _input("Enter your token: ").strip()

        if not token:
            _print("No token entered. Please try again.")
            continue

        if not token.startswith("hf_"):
            _print("Warning: Token should start with 'hf_'. Please check your token.")

        try:
            login(token=token, add_to_git_credential=False)
            _print("Token accepted!")

            return _accept_terms_flow(model_id)

        except Exception as e:
            _print(f"Token validation failed: {e}")
            if attempt < max_attempts - 1:
                _print("Please try again or press Ctrl+C to cancel.")

    _print("\nCould not authenticate after multiple attempts.")
    _print("Custom voices will not be available.")
    return False


def _accept_terms_flow(model_id: str) -> bool:
    """Guide user through accepting model terms of use."""
    _print("\n" + "=" * 60)
    _print("Step 2: Accept Terms of Use")
    _print("=" * 60)

    _print("\nThe voice model requires accepting terms of use.")
    _print("Opening model page...")

    model_url = f"https://huggingface.co/{model_id}"

    try:
        webbrowser.open(model_url)
    except Exception:
        _print("Could not open browser. Please visit:")
        _print(model_url)

    _print("\nInstructions:")
    _print("1. Look for a section about 'Access repository' or 'Gated model'")
    _print("2. Read the terms and click 'Agree and access repository'")
    _print("3. Return here when done")

    while True:
        _print()
        response = _input("Have you accepted the terms? (y/n): ").strip().lower()

        if response in ("y", "yes"):
            try:
                api = HfApi()
                api.model_info(model_id)
                _print("Access granted! Custom voices are now available.")
                return True
            except GatedRepoError:
                _print("Access not yet granted. The terms may still be processing.")
                _print(
                    "Please wait a moment and try again, or refresh the browser page."
                )
                retry = _input("Try again? (y/n): ").strip().lower()
                if retry not in ("y", "yes"):
                    return False
            except Exception as e:
                _print(f"Error verifying access: {e}")
                return False

        elif response in ("n", "no"):
            _print("\nYou must accept the terms to use custom voices.")
            skip = _input("Skip for now? (y/n): ").strip().lower()
            if skip in ("y", "yes"):
                return False

        else:
            _print("Please enter 'y' or 'n'")


def check_voice_access(voice: str, bundled_voices: list[str]) -> bool:
    """Check if user has access needed for a specific voice."""
    if not is_custom_voice(voice, bundled_voices):
        return True

    model_id = "kyutai/pocket-tts"
    if not is_model_gated(model_id):
        return True

    return ensure_huggingface_access(model_id)


__all__ = [
    "ensure_huggingface_access",
    "is_model_gated",
    "is_custom_voice",
    "check_voice_access",
    "GATED_MODELS",
]
