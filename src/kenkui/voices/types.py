"""Public voice and engine metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

VoiceVariety = Literal["built-in", "pre-compiled", "wav"]
VoiceState = Literal["registered", "loaded", "missing"]
# How the rendered voice is generally heard, used only to build casting pools.
# None means unsourced, and an unsourced voice never joins a gendered pool:
# a display name is Kenkui's own invention and says nothing about the speaker.
PerceivedGender = Literal["feminine", "masculine"] | None


@dataclass(frozen=True, slots=True)
class Engine:
    """A provisioned per-language synthesis engine."""

    id: str
    language: str
    model_revision: str
    cloning_capable: bool
    size_bytes: int


@dataclass(frozen=True, slots=True)
class Voice:
    """Reusable voice identity, content, compatibility, and rights metadata."""

    id: str
    name: str
    enabled: bool
    provenance: str | None
    license_id: str | None
    commercial_use_allowed: bool | None
    language: str | None = None
    content_fingerprint: str | None = None
    compatible_model_revisions: tuple[str, ...] = ()
    variety: VoiceVariety = "built-in"
    state: VoiceState = "registered"
    asset_bytes: int | None = None
    engine: Engine | None = None
    perceived_gender: PerceivedGender = None
