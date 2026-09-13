"""Public voice and engine metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

VoiceVariety = Literal["built-in", "pre-compiled", "wav"]
VoiceState = Literal["registered", "loaded", "missing"]
# How the rendered voice is generally heard, used only to build casting pools.
# None means unsourced, not a gender match. Such voices remain available when
# no matching voice exists; a display name is not evidence about the speaker.
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
    # The terms statement that accompanies license_id. Catalog voices inherit
    # it from their source dataset; added voices carry what their owner declared.
    voice_rights: str | None = None
