"""voice_service — high-level service functions for voice management.

Public API:
  list_voices(gender, accent, dataset, source, config_path) -> list[VoiceInfo]
  get_voice(name, config_path) -> VoiceInfo | None
  exclude_voice(name, config_path) -> ExcludeResult
  include_voice(name, config_path) -> IncludeResult
  synthesize_preview(voice_name, text, config_path, progress_callback) -> AudioPreviewResult
  gender_from_pronoun(pronoun) -> str
  top_gender_matched_voice(characters, excluded, default_voice) -> str
  sort_cast(speaker_voices) -> list[tuple[str, str]]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kenkui.voice_registry import get_registry, VoiceMetadata
from kenkui.config import load_app_config, save_app_config, DEFAULT_CONFIG_PATH


# ---------------------------------------------------------------------------
# Default audition text
# ---------------------------------------------------------------------------

DEFAULT_AUDITION_TEXT = (
    "The rain in Spain stays mainly in the plain. "
    "How wonderful it is to simply speak and be heard."
)


# ---------------------------------------------------------------------------
# Return type dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VoiceInfo:
    name: str
    source: str          # "compiled" | "builtin" | "uncompiled"
    gender: str | None
    accent: str | None
    dataset: str | None
    speaker_id: str | None
    description: str
    display_label: str
    excluded: bool


@dataclass
class ExcludeResult:
    excluded_voices: list[str]
    warning: str | None   # non-None when an entire gender pool is now excluded


@dataclass
class IncludeResult:
    excluded_voices: list[str]


@dataclass
class AudioPreviewResult:
    audio_path: str      # absolute path to the saved WAV file
    duration_ms: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _voice_metadata_to_info(v: VoiceMetadata, excluded: bool) -> VoiceInfo:
    return VoiceInfo(
        name=v.name,
        source=v.source,
        gender=v.gender,
        accent=v.accent,
        dataset=v.dataset,
        speaker_id=v.speaker_id,
        description=v.description,
        display_label=v.display_label,
        excluded=excluded,
    )


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def list_voices(
    gender: str | None = None,
    accent: str | None = None,
    dataset: str | None = None,
    source: str | None = None,
    config_path=None,
) -> list[VoiceInfo]:
    """Return all voices matching the given filters, with excluded flag set.

    Voices are returned in registry order: compiled, builtin, uncompiled.
    """
    registry = get_registry()
    voices = registry.filter(gender=gender, accent=accent, dataset=dataset, source=source)

    config = load_app_config(config_path)
    excluded_set = set(config.excluded_voices)

    return [_voice_metadata_to_info(v, v.name in excluded_set) for v in voices]


def get_voice(name: str, config_path=None) -> VoiceInfo | None:
    """Look up a single voice by name. Returns None if not found."""
    meta = get_registry().resolve(name)
    if meta is None:
        return None

    config = load_app_config(config_path)
    excluded_set = set(config.excluded_voices)
    return _voice_metadata_to_info(meta, meta.name in excluded_set)


def exclude_voice(name: str, config_path=None) -> ExcludeResult:
    """Add a voice to the excluded-from-auto-assignment list.

    Does not raise if the voice is not in the registry.
    Returns ExcludeResult with a warning string if an entire gender pool is
    now excluded.
    """
    config = load_app_config(config_path)
    dest = config_path if config_path is not None else DEFAULT_CONFIG_PATH

    if name not in config.excluded_voices:
        config.excluded_voices = list(config.excluded_voices) + [name]

    registry = get_registry()
    male_names = {v.name for v in registry.filter(gender="Male")}
    female_names = {v.name for v in registry.filter(gender="Female")}
    excluded_set = set(config.excluded_voices)

    if male_names and male_names <= excluded_set:
        warning = "All Male voices are now excluded from auto-assignment"
    elif female_names and female_names <= excluded_set:
        warning = "All Female voices are now excluded from auto-assignment"
    else:
        warning = None

    save_app_config(config, dest)
    return ExcludeResult(excluded_voices=list(config.excluded_voices), warning=warning)


def include_voice(name: str, config_path=None) -> IncludeResult:
    """Remove a voice from the excluded list, restoring it to auto-assignment.

    If the voice is not in the excluded list, returns unchanged list (no-op).
    """
    config = load_app_config(config_path)

    if name not in config.excluded_voices:
        return IncludeResult(excluded_voices=list(config.excluded_voices))

    config.excluded_voices = [v for v in config.excluded_voices if v != name]
    dest = config_path if config_path is not None else DEFAULT_CONFIG_PATH
    save_app_config(config, dest)
    return IncludeResult(excluded_voices=list(config.excluded_voices))


def synthesize_preview(
    voice_name: str,
    text: str | None = None,
    config_path=None,
    progress_callback=None,
) -> AudioPreviewResult:
    """Synthesize a short audio preview for a voice and save it to disk.

    Saves output to ~/.cache/kenkui/previews/{voice_name}.wav.
    Raises on failure (model load error, voice load error, synthesis failure).

    Progress callback receives (percent: int, message: str) tuples:
      (0, "Loading model"), (40, "Loading voice"),
      (70, "Synthesizing"), (100, "Done")
    """
    from kenkui.workers import _get_or_load_model, _render_text
    from kenkui.voice_loader import load_voice

    if text is None:
        text = DEFAULT_AUDITION_TEXT

    out_path = Path.home() / ".cache" / "kenkui" / "previews" / f"{voice_name}.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _cb(percent: int, message: str) -> None:
        if progress_callback is not None:
            progress_callback(percent, message)

    _cb(0, "Loading model")
    config = load_app_config(config_path)
    model = _get_or_load_model(
        config.temp,
        config.lsd_decode_steps,
        config.noise_clamp,
        config.eos_threshold,
    )

    _cb(40, "Loading voice")
    voice_resolved = load_voice(voice_name)
    voice_state = model.get_state_for_audio_prompt(voice_resolved)

    _cb(70, "Synthesizing")
    seg = _render_text(
        model,
        voice_state,
        text,
        log_message=lambda _: None,
        pid=0,
        batch_idx=0,
        total_batches=1,
        frames_after_eos=0,
    )

    if seg is None:
        raise RuntimeError(f"Synthesis returned no audio for voice '{voice_name}'")

    seg.export(str(out_path), format="wav")

    _cb(100, "Done")
    return AudioPreviewResult(
        audio_path=str(out_path),
        duration_ms=int(seg.duration_seconds * 1000),
    )


# ---------------------------------------------------------------------------
# Voice pool helpers
# ---------------------------------------------------------------------------


def gender_from_pronoun(pronoun: str | None) -> str:
    """Map a gender_pronoun string to 'male', 'female', or 'they'."""
    raw = (pronoun or "").strip().lower()
    if not raw:
        return "they"
    for segment in raw.split("/"):
        segment = segment.strip()
        if segment in ("he", "him", "his", "male"):
            return "male"
        if segment in ("she", "her", "hers", "female"):
            return "female"
    return "they"


def top_gender_matched_voice(
    characters: list,          # list[CharacterInfo]
    excluded: list[str],
    default_voice: str,
) -> str:
    """Return the best auto-selected voice based on the dominant character gender.

    Sorts characters by prominence descending, finds the top male/female character,
    and returns a voice matching the dominant gender. Falls back to default_voice
    if no gender-matched voices are available.
    """
    by_quotes = sorted(characters, key=lambda c: c.prominence, reverse=True)

    top_male_quotes = 0
    top_female_quotes = 0
    for ch in by_quotes:
        g = gender_from_pronoun(ch.gender_pronoun)
        if g == "male":
            top_male_quotes = ch.prominence
            break
        if g == "female":
            top_female_quotes = ch.prominence
            break

    registry = get_registry()
    male_voices = [v.name for v in registry.filter(gender="Male") if v.name not in excluded]
    female_voices = [v.name for v in registry.filter(gender="Female") if v.name not in excluded]

    if top_female_quotes > top_male_quotes and female_voices:
        return female_voices[0]
    if male_voices:
        return male_voices[0]
    return default_voice


def sort_cast(speaker_voices: dict) -> list[tuple[str, str]]:
    """Sort cast alphabetically with NARRATOR pinned last."""
    return sorted(
        speaker_voices.items(),
        key=lambda kv: ("~" if kv[0] == "NARRATOR" else kv[0].lower()),
    )


__all__ = [
    "DEFAULT_AUDITION_TEXT",
    "VoiceInfo",
    "ExcludeResult",
    "IncludeResult",
    "AudioPreviewResult",
    "list_voices",
    "get_voice",
    "exclude_voice",
    "include_voice",
    "synthesize_preview",
    "gender_from_pronoun",
    "top_gender_matched_voice",
    "sort_cast",
]
