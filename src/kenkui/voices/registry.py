"""Bundled local voice metadata without provider or network discovery."""

from dataclasses import dataclass
from types import MappingProxyType

from ..errors import ErrorCode, VoiceError


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


_VOICES = (
    Voice(
        id="fixture-voice",
        name="Fixture Voice",
        enabled=True,
        provenance="bundled local metadata",
        license_id="CC0-1.0",
        commercial_use_allowed=True,
        language="en",
        content_fingerprint="0" * 64,
        compatible_model_revisions=("fixture-v1",),
    ),
)
_BY_ID = MappingProxyType({voice.id: voice for voice in _VOICES})


def list_voices() -> tuple[Voice, ...]:
    """Return render-eligible voice metadata known to this local installation."""
    return _VOICES


def get_voice(voice_id: str) -> Voice:
    """Return one local voice by stable ID without accessing a provider."""
    if not isinstance(voice_id, str) or not voice_id.strip():
        raise VoiceError(ErrorCode.INVALID_VOICE)
    try:
        return _BY_ID[voice_id]
    except KeyError:
        raise VoiceError(ErrorCode.VOICE_UNRESOLVED) from None
