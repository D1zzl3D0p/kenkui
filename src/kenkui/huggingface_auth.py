"""HuggingFace authentication flow for custom voice access.

This module provides a user-friendly, step-by-step guide for users to:
1. Create a HuggingFace account (if needed)
2. Generate an access token
3. Accept terms for gated models
4. Download custom voice models

Designed for users with no prior HuggingFace knowledge.
"""

import sys
import webbrowser
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box
from huggingface_hub import HfApi, login
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from huggingface_hub.utils import LocalTokenNotFoundError


# Model configuration - tracks which models are gated
GATED_MODELS = {
    "kyutai/pocket-tts": True,  # The main TTS model is gated
}


def is_model_gated(model_id: str) -> bool:
    """Check if a model requires authentication/terms acceptance."""
    return GATED_MODELS.get(model_id, False)


def is_custom_voice(voice: str, bundled_voices: list[str]) -> bool:
    """Determine if a voice requires custom model access.

    Returns True if voice is NOT a default built-in voice.
    """
    from .utils import DEFAULT_VOICES

    # Check if it's a default built-in voice
    if voice in DEFAULT_VOICES:
        return False

    # Check if it's a local file path
    voice_path = Path(voice)
    if voice_path.exists():
        return True  # Local custom voice files need model access

    # Check if it's an hf:// URL
    if voice.startswith("hf://"):
        return True  # HuggingFace voices need model access

    # Check if it's a bundled voice
    voice_filename = f"{voice}.wav"
    if voice_filename in bundled_voices:
        return True  # Bundled custom voices need model access

    # Unknown voice - assume it might need custom model
    return True


def ensure_huggingface_access(
    model_id: str = "kyutai/pocket-tts",
    console: Optional[Console] = None,
    skip_if_no_interaction: bool = False,
) -> bool:
    """Ensure user has access to HuggingFace model with user-friendly flow.

    This function guides users through the entire authentication process:
    - Account creation (if needed)
    - Token generation
    - Terms acceptance

    Args:
        model_id: The HuggingFace model ID to check access for
        console: Rich console for output (creates one if None)
        skip_if_no_interaction: If True and not in interactive mode, return False

    Returns:
        True if access is granted, False otherwise
    """
    if console is None:
        console = Console()

    # First, check if model is actually gated
    if not is_model_gated(model_id):
        return True

    # Check current access status
    api = HfApi()

    try:
        # Try to access the model
        api.model_info(model_id)
        return True  # Already have access
    except LocalTokenNotFoundError:
        # No token set up yet - guide through setup
        return _setup_authentication(model_id, console)
    except GatedRepoError:
        # Have token but need to accept terms
        return _accept_terms_flow(model_id, console)
    except RepositoryNotFoundError:
        console.print(f"[red]Error: Model '{model_id}' not found on HuggingFace[/red]")
        return False
    except Exception as e:
        console.print(f"[yellow]Warning: Could not check model access: {e}[/yellow]")
        # Assume we need auth to be safe
        return _setup_authentication(model_id, console)


def _setup_authentication(model_id: str, console: Console) -> bool:
    """Guide user through creating account and token."""
    console.print()
    console.print(
        Panel(
            Text.from_markup(
                "[bold cyan]Welcome to Custom Voice Setup[/bold cyan]\n\n"
                "To use custom voices, we need to connect to HuggingFace "
                "(a platform that hosts AI models). This is free and takes about 2 minutes.\n\n"
                "[dim]Why is this needed?[/dim]\n"
                "The custom voice technology is hosted on HuggingFace and requires "
                "accepting their terms of use."
            ),
            border_style="cyan",
            box=box.ROUNDED,
        )
    )

    # Check if user already has an account
    console.print("\n[bold]Do you have a HuggingFace account?[/bold]")
    console.print("[1] Yes, I have an account")
    console.print("[2] No, I need to create one")
    console.print("[3] Skip for now (custom voices won't work)")

    choice = console.input("\nEnter 1, 2, or 3: ").strip()

    if choice == "3":
        console.print(
            "\n[yellow]Skipping authentication. Custom voices will not be available.[/yellow]"
        )
        return False

    if choice == "2":
        # Guide to create account
        console.print("\n[bold]Let's create your account:[/bold]")
        console.print("1. Opening signup page in your browser...")

        try:
            webbrowser.open("https://huggingface.co/join")
        except Exception:
            console.print("   [yellow]Could not open browser. Please visit:[/yellow]")
            console.print("   [blue]https://huggingface.co/join[/blue]")

        console.print("\n2. Complete the signup form in your browser")
        console.print("3. Once done, return here and press Enter")

        input("\nPress Enter when you've created your account...")

    # Now guide to create token
    return _create_token_flow(model_id, console)


def _create_token_flow(model_id: str, console: Console) -> bool:
    """Guide user through creating and entering an access token."""
    console.print("\n" + "=" * 60)
    console.print("[bold]Step 1: Create an Access Token[/bold]")
    console.print("=" * 60)

    console.print(
        "\nWe need to create a 'read' token so kenkui can download voice models."
    )
    console.print("Opening token creation page...")

    # Pre-fill token URL with name
    token_url = (
        "https://huggingface.co/settings/tokens/new?tokenType=read&name=KenkuiVoices"
    )

    try:
        webbrowser.open(token_url)
    except Exception:
        console.print("[yellow]Could not open browser. Please visit:[/yellow]")
        console.print(f"[blue]{token_url}[/blue]")

    console.print("\n[bold]Instructions:[/bold]")
    console.print("1. In the browser, click 'Create token' (it will be a 'Read' token)")
    console.print("2. Copy the token (it starts with 'hf_')")
    console.print("3. Paste it below")

    # Get token from user
    max_attempts = 3
    for attempt in range(max_attempts):
        console.print()
        token = console.input("Enter your token: ").strip()

        if not token:
            console.print("[yellow]No token entered. Please try again.[/yellow]")
            continue

        if not token.startswith("hf_"):
            console.print(
                "[yellow]Warning: Token should start with 'hf_'. Please check your token.[/yellow]"
            )

        # Try to login with this token
        try:
            login(token=token, add_to_git_credential=False)
            console.print("[green]✓ Token accepted![/green]")

            # Now check if we need to accept terms
            return _accept_terms_flow(model_id, console)

        except Exception as e:
            console.print(f"[red]✗ Token validation failed: {e}[/red]")
            if attempt < max_attempts - 1:
                console.print("Please try again or press Ctrl+C to cancel.")

    console.print("\n[red]Could not authenticate after multiple attempts.[/red]")
    console.print("Custom voices will not be available.")
    return False


def _accept_terms_flow(model_id: str, console: Console) -> bool:
    """Guide user through accepting model terms of use."""
    console.print("\n" + "=" * 60)
    console.print("[bold]Step 2: Accept Terms of Use[/bold]")
    console.print("=" * 60)

    console.print(f"\nThe voice model requires accepting terms of use.")
    console.print("Opening model page...")

    model_url = f"https://huggingface.co/{model_id}"

    try:
        webbrowser.open(model_url)
    except Exception:
        console.print("[yellow]Could not open browser. Please visit:[/yellow]")
        console.print(f"[blue]{model_url}[/blue]")

    console.print("\n[bold]Instructions:[/bold]")
    console.print("1. Look for a section about 'Access repository' or 'Gated model'")
    console.print("2. Read the terms and click 'Agree and access repository'")
    console.print("3. Return here when done")

    while True:
        console.print()
        response = console.input("Have you accepted the terms? (y/n): ").strip().lower()

        if response in ("y", "yes"):
            # Verify access
            try:
                api = HfApi()
                api.model_info(model_id)
                console.print(
                    "[green]✓ Access granted! Custom voices are now available.[/green]"
                )
                return True
            except GatedRepoError:
                console.print(
                    "[yellow]Access not yet granted. The terms may still be processing.[/yellow]"
                )
                console.print(
                    "Please wait a moment and try again, or refresh the browser page."
                )
                retry = console.input("Try again? (y/n): ").strip().lower()
                if retry not in ("y", "yes"):
                    return False
            except Exception as e:
                console.print(f"[red]Error verifying access: {e}[/red]")
                return False

        elif response in ("n", "no"):
            console.print(
                "\n[yellow]You must accept the terms to use custom voices.[/yellow]"
            )
            skip = console.input("Skip for now? (y/n): ").strip().lower()
            if skip in ("y", "yes"):
                return False

        else:
            console.print("Please enter 'y' or 'n'")


def check_voice_access(voice: str, bundled_voices: list[str]) -> bool:
    """Check if user has access needed for a specific voice.

    This is a convenience wrapper that only shows auth flow for custom voices.

    Args:
        voice: The voice identifier
        bundled_voices: List of bundled custom voice filenames

    Returns:
        True if voice can be used, False otherwise
    """
    # Check if this is a custom voice that needs auth
    if not is_custom_voice(voice, bundled_voices):
        return True  # Built-in voices don't need auth

    # Check if TTS model is gated
    model_id = "kyutai/pocket-tts"
    if not is_model_gated(model_id):
        return True  # Model not gated, no auth needed

    # Run the full auth flow
    console = Console()
    return ensure_huggingface_access(model_id, console)


__all__ = [
    "ensure_huggingface_access",
    "is_model_gated",
    "is_custom_voice",
    "check_voice_access",
    "GATED_MODELS",
]
