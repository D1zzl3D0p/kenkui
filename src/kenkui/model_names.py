"""Canonical model-name normalization for LLM providers.

Providers advertise the same underlying model under several slug shapes.
OpenRouter, for example, exposes Anthropic canonical slugs such as
``anthropic/claude-4.5-sonnet-20250929`` alongside the shorter routing id
``anthropic/claude-sonnet-4.5`` that LiteLLM's OpenRouter integration expects.
First-party Anthropic slugs use a ``claude-<name>-<major>-<minor>-<date>``
shape.  ``normalize_model_for_provider`` rewrites user-entered or restored
canonical slugs into the id each provider actually routes on.

This lives in kenkui (rather than the UI layer) so every consumer normalizes
model ids identically at the boundary where a provider id is chosen.
"""

from __future__ import annotations

import re

_ANTHROPIC_CANONICAL_OPENROUTER_RE = re.compile(
    r"^(?P<prefix>(?:openrouter/)?anthropic/)claude-"
    r"(?P<major>\d+)\.(?P<minor>\d+)-(?P<name>[a-z][a-z0-9-]*)-(?P<date>\d{8})$"
)
_ANTHROPIC_FIRST_PARTY_RE = re.compile(
    r"^(?P<prefix>(?:anthropic/)?)claude-"
    r"(?P<major>\d+)\.(?P<minor>\d+)-(?P<name>[a-z][a-z0-9-]*)-(?P<date>\d{8})$"
)


def normalize_model_for_provider(provider: str, model: str) -> str:
    """Normalize known provider model aliases before routing.

    OpenRouter/LiteLLM OpenRouter canonical Anthropic slugs are rewritten to
    the shorter routing id; first-party Anthropic canonical slugs are rewritten
    to the dashed ``claude-<name>-<major>-<minor>-<date>`` form.  Unknown
    providers and unrecognised slugs are returned unchanged (trimmed).
    """
    normalized_provider = provider.lower().strip()
    value = (model or "").strip()
    if not value:
        return value

    if normalized_provider == "openrouter" or (
        normalized_provider == "litellm" and value.startswith("openrouter/")
    ):
        match = _ANTHROPIC_CANONICAL_OPENROUTER_RE.match(value)
        if match:
            groups = match.groupdict()
            return (
                f"{groups['prefix']}claude-{groups['name']}-"
                f"{groups['major']}.{groups['minor']}"
            )

    if normalized_provider in {"anthropic", "litellm"}:
        match = _ANTHROPIC_FIRST_PARTY_RE.match(value)
        if match:
            groups = match.groupdict()
            return (
                f"{groups['prefix']}claude-{groups['name']}-"
                f"{groups['major']}-{groups['minor']}-{groups['date']}"
            )

    return value


__all__ = ["normalize_model_for_provider"]
