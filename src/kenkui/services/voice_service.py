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

import logging
from collections import defaultdict
from collections.abc import Callable
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


@dataclass
class SuggestCastResult:
    speaker_voices: dict[str, str]   # character_name → voice_name
    warnings: list[str]


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
    config_path: str | None = None,
) -> list[VoiceInfo]:
    """Return all voices matching the given filters, with excluded flag set.

    Voices are returned in registry order: compiled, builtin, uncompiled.
    """
    registry = get_registry()
    voices = registry.filter(gender=gender, accent=accent, dataset=dataset, source=source)

    config = load_app_config(config_path)
    excluded_set = set(config.excluded_voices)

    return [_voice_metadata_to_info(v, v.name in excluded_set) for v in voices]


def get_voice(name: str, config_path: str | None = None) -> VoiceInfo | None:
    """Look up a single voice by name. Returns None if not found."""
    meta = get_registry().resolve(name)
    if meta is None:
        return None

    config = load_app_config(config_path)
    excluded_set = set(config.excluded_voices)
    return _voice_metadata_to_info(meta, meta.name in excluded_set)


def exclude_voice(name: str, config_path: str | None = None) -> ExcludeResult:
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

    warnings = []
    if male_names and male_names <= excluded_set:
        warnings.append("All Male voices are now excluded from auto-assignment")
    if female_names and female_names <= excluded_set:
        warnings.append("All Female voices are now excluded from auto-assignment")
    warning = "; ".join(warnings) if warnings else None

    save_app_config(config, dest)
    return ExcludeResult(excluded_voices=list(config.excluded_voices), warning=warning)


def include_voice(name: str, config_path: str | None = None) -> IncludeResult:
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
    config_path: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> AudioPreviewResult:
    """Synthesize a short audio preview for a voice and save it to disk.

    Saves output to ~/.cache/kenkui/previews/{voice_name}.wav.
    Raises on failure (model load error, voice load error, synthesis failure).

    Preview files are saved to ``~/.cache/kenkui/previews/`` (persistent, unlike /tmp)
    so repeated calls for the same voice reuse the file.

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


# ---------------------------------------------------------------------------
# suggest_cast helpers
# ---------------------------------------------------------------------------


def _get_chapter_cooccurrence_from_paragraphs(chapters) -> "dict[int, set[str]]":
    """Build {chapter_index: set of speaker names} from chapter paragraphs.

    Accepts chapter objects that expose either a `paragraphs` attribute
    (each paragraph having .speaker and .is_spoken) or a `segments` attribute
    (each segment having .speaker).
    """
    result: dict[int, set[str]] = {}
    EXCLUDED = {"NARRATOR", "SCENE_BREAK", "Unknown"}
    for idx, ch in enumerate(chapters):
        speakers: set[str] = set()
        if hasattr(ch, "paragraphs"):
            for p in ch.paragraphs:
                sp = getattr(p, "speaker", None)
                is_spoken = getattr(p, "is_spoken", False)
                if sp and is_spoken and sp not in EXCLUDED:
                    speakers.add(sp)
        elif hasattr(ch, "segments"):
            for s in ch.segments:
                sp = getattr(s, "speaker", None)
                if sp and sp not in EXCLUDED and not getattr(s, "is_scene_break", False):
                    speakers.add(sp)
        if speakers:
            result[idx] = speakers
    return result


def _resolve_cast_conflicts(
    speaker_voices: "dict[str, str]",
    char_prominence: "dict[str, int]",
    char_gender: "dict[str, str]",
    chapters,
    male_pool: "list[str]",
    female_pool: "list[str]",
    narrator_voice: str,
) -> "tuple[dict[str, str], list[str]]":
    """Ensure no two characters sharing a chapter are assigned the same voice.

    Returns (updated_speaker_voices, warnings).
    """
    logger = logging.getLogger(__name__)

    cooccurrence = _get_chapter_cooccurrence_from_paragraphs(chapters)
    warnings: list[str] = []
    unresolved_seen: set[frozenset] = set()

    changed = True
    while changed:
        changed = False
        for ch_idx, ch_speakers in cooccurrence.items():
            voice_to_chars: dict[str, list[str]] = defaultdict(list)
            for sp in ch_speakers:
                v = speaker_voices.get(sp)
                if v:
                    voice_to_chars[v].append(sp)

            for voice, chars in voice_to_chars.items():
                if len(chars) <= 1:
                    continue
                chars_sorted = sorted(chars, key=lambda c: char_prominence.get(c, 0), reverse=True)
                chapter_voices_used = {
                    speaker_voices[sp] for sp in ch_speakers if sp in speaker_voices
                }
                for char_to_reassign in chars_sorted[1:]:
                    gender = char_gender.get(char_to_reassign, "they")
                    pool = male_pool if gender == "male" else female_pool
                    new_voice = next(
                        (v for v in pool if v not in chapter_voices_used and v != narrator_voice),
                        None,
                    )
                    if new_voice:
                        speaker_voices[char_to_reassign] = new_voice
                        chapter_voices_used.add(new_voice)
                        changed = True
                    else:
                        pair_key = frozenset({chars_sorted[0], char_to_reassign})
                        if pair_key not in unresolved_seen:
                            unresolved_seen.add(pair_key)
                            warnings.append(
                                f"Voice conflict: {chars_sorted[0]!r} and "
                                f"{char_to_reassign!r} share voice {voice!r} in chapter "
                                f"{ch_idx} — no spare voice available."
                            )
                            logger.warning(
                                "Voice conflict: %r and %r share voice %r in chapter %d "
                                "and no spare voice is available.",
                                chars_sorted[0], char_to_reassign, voice, ch_idx,
                            )

    return speaker_voices, warnings


def suggest_cast(
    *,
    roster: list,
    excluded_voices: list[str],
    default_voice: str,
    chapters: list | None = None,
    config_path: str | None = None,
) -> SuggestCastResult:
    """Assign voices to characters using round-robin pool + conflict resolution.

    No interactive I/O, no print/Rich/sys.exit. File reads occur via list_voices
    (voice registry + config).
    Moved from cli/add.py._auto_assign_character_voices and
    _resolve_chapter_voice_conflicts.

    Each item in ``roster`` must have:
      - .character_id  (str) — used as key in the returned speaker_voices dict
      - .gender_pronoun (str | None)
      - .prominence    (int property) — used for sorting

    Returns SuggestCastResult(speaker_voices, warnings).
    """
    warnings: list[str] = []

    # Fetch all voices from registry (respects config excluded list internally,
    # but we apply our own excluded_voices filter on top).
    all_voices = list_voices(config_path=config_path)
    excluded_set = set(excluded_voices or [])

    # Build gender pools.
    _all_male = [v.name for v in all_voices if (v.gender or "").lower() == "male"]
    _all_female = [v.name for v in all_voices if (v.gender or "").lower() == "female"]

    male_voices = [v for v in _all_male if v not in excluded_set] or _all_male
    female_voices = [v for v in _all_female if v not in excluded_set] or _all_female

    # Exclude narrator voice from character pools.
    male_pool = [v for v in male_voices if v != default_voice] or male_voices
    female_pool = [v for v in female_voices if v != default_voice] or female_voices

    if not male_pool and not female_pool:
        warnings.append(
            "No voices available in the pool; all characters assigned to default_voice."
        )

    male_idx, female_idx = 0, 0
    male_quotes, female_quotes = 0, 0
    speaker_voices: dict[str, str] = {}

    char_prominence: dict[str, int] = {}
    char_gender: dict[str, str] = {}

    for ch in sorted(roster, key=lambda c: c.prominence, reverse=True):
        gender = gender_from_pronoun(ch.gender_pronoun)
        char_prominence[ch.character_id] = ch.prominence
        char_gender[ch.character_id] = gender

        if gender == "male":
            if male_pool:
                voice = male_pool[male_idx % len(male_pool)]
                male_idx += 1
            else:
                voice = default_voice
                warnings.append(f"No male voices available; assigned default to {ch.character_id!r}.")
            male_quotes += ch.prominence
        elif gender == "female":
            if female_pool:
                voice = female_pool[female_idx % len(female_pool)]
                female_idx += 1
            else:
                voice = default_voice
                warnings.append(f"No female voices available; assigned default to {ch.character_id!r}.")
            female_quotes += ch.prominence
        else:
            # they/them — pick the pool with fewer total quotes so far
            if male_quotes <= female_quotes:
                if male_pool:
                    voice = male_pool[male_idx % len(male_pool)]
                    male_idx += 1
                elif female_pool:
                    voice = female_pool[female_idx % len(female_pool)]
                    female_idx += 1
                else:
                    voice = default_voice
                male_quotes += ch.prominence
            else:
                if female_pool:
                    voice = female_pool[female_idx % len(female_pool)]
                    female_idx += 1
                elif male_pool:
                    voice = male_pool[male_idx % len(male_pool)]
                    male_idx += 1
                else:
                    voice = default_voice
                female_quotes += ch.prominence

        speaker_voices[ch.character_id] = voice

    # Resolve chapter co-occurrence conflicts if chapters provided.
    if chapters:
        speaker_voices, conflict_warnings = _resolve_cast_conflicts(
            speaker_voices=speaker_voices,
            char_prominence=char_prominence,
            char_gender=char_gender,
            chapters=chapters,
            male_pool=male_pool,
            female_pool=female_pool,
            narrator_voice=default_voice,
        )
        warnings.extend(conflict_warnings)

    return SuggestCastResult(speaker_voices=speaker_voices, warnings=warnings)


__all__ = [
    "DEFAULT_AUDITION_TEXT",
    "VoiceInfo",
    "ExcludeResult",
    "IncludeResult",
    "AudioPreviewResult",
    "SuggestCastResult",
    "list_voices",
    "get_voice",
    "exclude_voice",
    "include_voice",
    "synthesize_preview",
    "gender_from_pronoun",
    "top_gender_matched_voice",
    "sort_cast",
    "suggest_cast",
]
