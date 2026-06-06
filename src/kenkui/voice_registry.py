"""Voice registry — single source of truth for voice discovery and metadata.

Provides :class:`VoiceMetadata` (structured per-voice info parsed from filenames)
and :class:`VoiceRegistry` (lazy-scanned catalog of all available voices).

Voice sources, in display priority order:
1. **compiled** — ``.safetensors`` files in ``kenkui/voices/compiled-voices/``.
   Filename schema: ``{Name}-{Gender}-{Dataset}-{SpeakerId}-{Accent}.safetensors``
   These require no HuggingFace authentication.
2. **builtin** — The 21 pocket-tts built-in voice names (``alba``, ``cosette``, …).
   No file path; passed as a string directly to the TTS model.
3. **uncompiled** — ``.wav`` audio-prompt files, either from the package
   ``kenkui/voices/uncompiled-voices/`` directory or from the XDG user data dir
   ``~/.local/share/kenkui/voices/uncompiled/``.
   These require HuggingFace authentication (gated pocket-tts model).
"""

from __future__ import annotations

import importlib.resources
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in voice metadata (pocket-tts defaults, no file needed)
# ---------------------------------------------------------------------------

_BUILTIN_VOICE_DATA: dict[str, dict[str, str | None]] = {
    # Voice-donations / single-source voices
    "alba":           {"gender": "Male",   "accent": "American", "dataset": "Alba-Mackenna", "speaker_id": "casual"},
    "marius":         {"gender": "Male",   "accent": "American", "dataset": "Voice Donation", "speaker_id": None},
    "javert":         {"gender": "Male",   "accent": "American", "dataset": "Voice Donation", "speaker_id": None},
    "cosette":        {"gender": "Female", "accent": "American", "dataset": "Expresso",      "speaker_id": "ex04-ex02_confused_001_channel1_499s"},
    # EARS
    "jean":           {"gender": "Male",   "accent": "Southern", "dataset": "EARS", "speaker_id": "P010"},
    # VCTK (original three, now with full metadata)
    "fantine":        {"gender": "Female", "accent": "British",  "dataset": "VCTK", "speaker_id": "P244"},
    "eponine":        {"gender": "Female", "accent": "British",  "dataset": "VCTK", "speaker_id": "P262"},
    "azelma":         {"gender": "Female", "accent": "American", "dataset": "VCTK", "speaker_id": "P303"},
    # VCTK (new defaults)
    "anna":           {"gender": "Female", "accent": "Scottish", "dataset": "VCTK", "speaker_id": "P228"},
    "vera":           {"gender": "Female", "accent": "English",  "dataset": "VCTK", "speaker_id": "P229"},
    "charles":        {"gender": "Male",   "accent": "English",  "dataset": "VCTK", "speaker_id": "P254"},
    "paul":           {"gender": "Male",   "accent": "British",  "dataset": "VCTK", "speaker_id": "P259"},
    "george":         {"gender": "Male",   "accent": "American", "dataset": "VCTK", "speaker_id": "P315"},
    "mary":           {"gender": "Female", "accent": "American", "dataset": "VCTK", "speaker_id": "P333"},
    "jane":           {"gender": "Female", "accent": "American", "dataset": "VCTK", "speaker_id": "P339"},
    "michael":        {"gender": "Male",   "accent": "American", "dataset": "VCTK", "speaker_id": "P360"},
    "eve":            {"gender": "Female", "accent": "American", "dataset": "VCTK", "speaker_id": "P361"},
    # Voice-zero (gender inferred from given names — NOT verified from audio)
    "bill_boerst":    {"gender": "Male",   "accent": None,       "dataset": "Voice Zero", "speaker_id": None},
    "caro_davy":      {"gender": "Female", "accent": None,       "dataset": "Voice Zero", "speaker_id": None},
    "peter_yearsley": {"gender": "Male",   "accent": None,       "dataset": "Voice Zero", "speaker_id": None},
    "stuart_bell":    {"gender": "Male",   "accent": None,       "dataset": "Voice Zero", "speaker_id": None},
}

BUILTIN_VOICE_NAMES: list[str] = list(_BUILTIN_VOICE_DATA.keys())


# ---------------------------------------------------------------------------
# VoiceMetadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VoiceMetadata:
    """Structured metadata for a single voice.

    All fields except ``name``, ``file_path``, and ``source`` may be ``None``
    when the voice format does not encode that information (e.g. uncompiled WAVs).
    """

    name: str
    """Display name / lookup key (e.g. ``"Alasdair"``, ``"alba"``)."""

    file_path: Path | None
    """Absolute path to the voice file, or ``None`` for pocket-tts built-ins."""

    source: Literal["compiled", "uncompiled", "builtin"]
    """Where this voice comes from."""

    gender: str | None
    """``"Male"`` or ``"Female"``, or ``None`` if unknown."""

    dataset: str | None
    """Source dataset identifier (e.g. ``"VCTK"``, ``"EARS"``), or ``None``."""

    speaker_id: str | None
    """Dataset speaker ID (e.g. ``"P246"``), or ``None``."""

    accent: str | None
    """Accent / region descriptor (e.g. ``"Scottish"``, ``"English-Yorkshire"``), or ``None``."""

    is_default: bool = False
    """True for voices shipped as pocket-tts built-ins (no file path required)."""

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    @property
    def description(self) -> str:
        """Short human-readable description of this voice."""
        if self.source == "builtin":
            parts = []
            if self.gender:
                parts.append(self.gender)
            if self.accent:
                parts.append(self.accent)
            if self.dataset:
                parts.append(self.dataset)
            parts.append("Default")
            return " · ".join(parts)
        if self.source == "compiled":
            parts = []
            if self.gender:
                parts.append(self.gender)
            if self.accent:
                parts.append(self.accent)
            if self.dataset:
                parts.append(self.dataset)
            return " · ".join(parts) if parts else "Compiled voice"
        return "Custom voice"

    @property
    def display_label(self) -> str:
        """Name + description in a single string for list display."""
        desc = self.description
        if desc:
            return f"{self.name:<20} {desc}"
        return self.name


# ---------------------------------------------------------------------------
# Filename parsers (pure functions — no I/O)
# ---------------------------------------------------------------------------

def parse_compiled_filename(path: Path) -> VoiceMetadata:
    """Parse a compiled voice filename into :class:`VoiceMetadata`.

    Expected format: ``{Name}-{Gender}-{Dataset}-{SpeakerId}-{Accent}.safetensors``

    The accent field may itself contain hyphens (e.g. ``English-Yorkshire``).
    ``split("-", 4)`` is used so all trailing parts are captured as the accent.
    """
    stem = path.stem
    parts = stem.split("-", 4)

    name = parts[0] if len(parts) > 0 else stem
    gender_code = parts[1] if len(parts) > 1 else None
    dataset = parts[2] if len(parts) > 2 else None
    speaker_id = parts[3] if len(parts) > 3 else None
    accent = parts[4] if len(parts) > 4 else None

    gender_map = {"M": "Male", "F": "Female"}
    gender = gender_map.get(gender_code or "")

    return VoiceMetadata(
        name=name,
        file_path=path,
        source="compiled",
        gender=gender,
        dataset=dataset,
        speaker_id=speaker_id,
        accent=accent,
    )


def parse_uncompiled_filename(path: Path) -> VoiceMetadata:
    """Parse an uncompiled (legacy .wav) voice filename into :class:`VoiceMetadata`.

    Old format is plain PascalCase with no structured metadata.
    """
    return VoiceMetadata(
        name=path.stem,
        file_path=path,
        source="uncompiled",
        gender=None,
        dataset=None,
        speaker_id=None,
        accent=None,
    )


# ---------------------------------------------------------------------------
# VoiceRegistry
# ---------------------------------------------------------------------------

class VoiceRegistry:
    """Lazy-scanned catalog of all available voices.

    Scans voice directories on first access and caches results.  Call
    :meth:`invalidate` to force a re-scan (e.g. after downloading new voices).
    """

    def __init__(self) -> None:
        self._voices: list[VoiceMetadata] | None = None

    # ------------------------------------------------------------------
    # Internal scan
    # ------------------------------------------------------------------

    def _scan(self) -> list[VoiceMetadata]:
        results: list[VoiceMetadata] = []

        # 1. Compiled voices (package bundled .safetensors)
        results.extend(self._scan_compiled())

        # 2. Built-in pocket-tts voices
        results.extend(self._builtin_voices())

        # 3. Uncompiled voices (package bundled .wav)
        results.extend(self._scan_uncompiled_pkg())

        # 4. User-downloaded uncompiled voices (XDG data dir)
        results.extend(self._scan_uncompiled_user())

        return results

    def _scan_compiled(self) -> list[VoiceMetadata]:
        voices: list[VoiceMetadata] = []
        try:
            compiled_dir = importlib.resources.files("kenkui") / "voices" / "compiled-voices"
            if compiled_dir.is_dir():
                for entry in sorted(compiled_dir.iterdir(), key=lambda e: e.name):
                    if entry.is_file() and entry.name.endswith(".safetensors"):
                        voices.append(parse_compiled_filename(Path(str(entry))))
        except Exception as exc:
            logger.debug("Could not scan compiled voices: %s", exc)
        # Also scan user-downloaded compiled voices
        user_compiled = Path.home() / ".local" / "share" / "kenkui" / "voices" / "compiled"
        if user_compiled.exists():
            for p in sorted(user_compiled.glob("*.safetensors")):
                meta = parse_compiled_filename(p)
                if meta:
                    voices.append(meta)
        return voices

    def _scan_uncompiled_pkg(self) -> list[VoiceMetadata]:
        voices: list[VoiceMetadata] = []
        try:
            uncompiled_dir = importlib.resources.files("kenkui") / "voices" / "uncompiled-voices"
            if uncompiled_dir.is_dir():
                for entry in sorted(uncompiled_dir.iterdir(), key=lambda e: e.name):
                    if entry.is_file() and entry.name.endswith(".wav"):
                        voices.append(parse_uncompiled_filename(Path(str(entry))))
        except Exception as exc:
            logger.debug("Could not scan uncompiled package voices: %s", exc)
        return voices

    def _scan_uncompiled_user(self) -> list[VoiceMetadata]:
        voices: list[VoiceMetadata] = []
        try:
            xdg_data = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
            user_dir = Path(xdg_data) / "kenkui" / "voices" / "uncompiled"
            if user_dir.is_dir():
                for wav_file in sorted(user_dir.glob("*.wav")):
                    voices.append(parse_uncompiled_filename(wav_file))
        except Exception as exc:
            logger.debug("Could not scan user uncompiled voices: %s", exc)
        return voices

    def _builtin_voices(self) -> list[VoiceMetadata]:
        return [
            VoiceMetadata(
                name=name,
                file_path=None,
                source="builtin",
                gender=data["gender"],
                dataset=data.get("dataset"),
                speaker_id=data.get("speaker_id"),
                accent=data.get("accent"),
                is_default=True,
            )
            for name, data in _BUILTIN_VOICE_DATA.items()
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def voices(self) -> list[VoiceMetadata]:
        """All available voices (compiled first, then builtin, then uncompiled)."""
        if self._voices is None:
            self._voices = self._scan()
        return self._voices

    def resolve(self, name: str) -> VoiceMetadata | None:
        """Find a voice by name (case-insensitive stem match).

        Priority: compiled > builtin > uncompiled.
        """
        name_lower = name.lower()
        # Strip extension if provided (e.g. "RafeBeckley.wav" → "rafebeckley")
        if "." in name_lower:
            name_lower = name_lower.rsplit(".", 1)[0]

        for voice in self.voices:
            if voice.name.lower() == name_lower:
                return voice
        return None

    def filter(
        self,
        *,
        gender: str | None = None,
        accent: str | None = None,
        dataset: str | None = None,
        source: str | None = None,
        is_default: bool | None = None,
    ) -> list[VoiceMetadata]:
        """Return voices matching all specified criteria.

        Comparisons are case-insensitive.  ``None`` criteria are ignored.
        """
        def matches(v: VoiceMetadata) -> bool:
            if source and v.source != source:
                return False
            if gender and (v.gender or "").lower() != gender.lower():
                return False
            if dataset and (v.dataset or "").lower() != dataset.lower():
                return False
            if accent and (v.accent or "").lower() != accent.lower():
                return False
            if is_default is not None and v.is_default != is_default:
                return False
            return True

        return [v for v in self.voices if matches(v)]

    def invalidate(self) -> None:
        """Force a re-scan on next access (e.g. after downloading new voices)."""
        self._voices = None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: VoiceRegistry | None = None


def get_registry() -> VoiceRegistry:
    """Return the shared :class:`VoiceRegistry` singleton."""
    global _registry
    if _registry is None:
        _registry = VoiceRegistry()
    return _registry


def get_bundled_voices() -> list[str]:
    """Return names of all compiled + uncompiled voices, sorted alphabetically."""
    return sorted(
        v.name
        for v in get_registry().voices
        if v.source in ("compiled", "uncompiled")
    )


__all__ = [
    "BUILTIN_VOICE_NAMES",
    "VoiceMetadata",
    "VoiceRegistry",
    "get_registry",
    "get_bundled_voices",
    "parse_compiled_filename",
    "parse_uncompiled_filename",
]
