"""Public voice and engine metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

VoiceVariety = Literal["built-in", "pre-compiled", "wav"]
VoiceState = Literal["registered", "loaded", "missing"]


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
