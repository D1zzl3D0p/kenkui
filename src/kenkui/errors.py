"""Typed exceptions raised by the kenkui library core."""

from __future__ import annotations


class KenkuiError(Exception):
    """Base class for actionable kenkui failures."""


class KenkuiConfigError(KenkuiError):
    """Configuration loading, validation, or persistence failed."""


class KenkuiReaderError(KenkuiError):
    """Book reader selection or parsing failed."""


class KenkuiNLPError(KenkuiError):
    """NLP extraction, attribution, or provider orchestration failed."""


class KenkuiCacheError(KenkuiError):
    """Cache loading or persistence failed."""


class KenkuiVoiceError(KenkuiError):
    """Voice lookup, assignment, or synthesis failed."""


class KenkuiRenderingError(KenkuiError):
    """Audiobook rendering, stitching, or post-processing failed."""


class KenkuiDependencyError(KenkuiError):
    """A required optional dependency or external tool is unavailable."""


__all__ = [
    "KenkuiCacheError",
    "KenkuiConfigError",
    "KenkuiDependencyError",
    "KenkuiError",
    "KenkuiNLPError",
    "KenkuiReaderError",
    "KenkuiRenderingError",
    "KenkuiVoiceError",
]

