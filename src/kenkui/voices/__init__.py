"""Public voice metadata.

Deliberately thin: importing `kenkui.voices.types` executes this module first,
and `kenkui._tts.production` depends on those types. Re-exporting `provision`
here would drag the network-touching provisioning module into every render
import. The provisioning verbs are exported from the package root instead, and
tests/test_import_boundaries.py enforces the split.
"""

from .registry import CatalogEntry
from .types import Engine, Voice, VoiceState, VoiceVariety

__all__ = ["CatalogEntry", "Engine", "Voice", "VoiceState", "VoiceVariety"]
