"""High-level voice catalog services."""

from __future__ import annotations

import hashlib
import logging
import tempfile
import urllib.request
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from kenkui.voice_compiler import compile_audio_prompt_source
from kenkui.voice_loader import load_voice_conditioning_source
from kenkui.voice_registry import (
    PREVIEW_TEXT,
    VoiceCatalogEntry,
    get_catalog,
    preview_cache_dir,
)

if TYPE_CHECKING:
    from kenkui.voice_pool import VoicePoolTemplate

logger = logging.getLogger(__name__)

DEFAULT_AUDITION_TEXT = PREVIEW_TEXT

_VOICE_ORIGIN_SORT_ORDER = {
    "pocket_tts_builtin": 0,
    "kenkui_compiled": 1,
    "custom_compiled": 2,
}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class VoiceInfo:
    voice_id: str
    display_name: str
    origin: str
    asset_kind: str
    gender: str
    pool_enabled: bool
    status: str
    path: str | None
    preview_path: str | None
    preview_url: str | None
    accent: str | None
    dataset: str | None
    speaker_id: str | None
    license: str | None
    tags: tuple[str, ...]
    notes: str | None
    description: str
    display_label: str


@dataclass
class PoolUpdateResult:
    voice_id: str
    pool_enabled: bool


@dataclass
class AudioPreviewResult:
    voice_id: str
    audio_path: str
    duration_ms: int | None = None


@dataclass
class CustomVoiceImportResult:
    voice: VoiceInfo
    manifest_path: str


@dataclass
class SuggestCastResult:
    speaker_voices: dict[str, str]
    warnings: list[str]


def _entry_to_info(v: VoiceCatalogEntry) -> VoiceInfo:
    return VoiceInfo(
        voice_id=v.voice_id,
        display_name=v.display_name,
        origin=v.origin,
        asset_kind=v.asset_kind,
        gender=v.gender,
        pool_enabled=v.pool_enabled,
        status=v.status,
        path=str(v.path) if v.path is not None else None,
        preview_path=v.preview.path,
        preview_url=v.preview.url,
        accent=v.accent,
        dataset=v.dataset,
        speaker_id=v.speaker_id,
        license=v.license,
        tags=v.tags,
        notes=v.notes,
        description=v.description,
        display_label=v.display_label,
    )


def list_voices(
    gender: str | None = None,
    accent: str | None = None,
    dataset: str | None = None,
    origin: str | None = None,
    asset_kind: str | None = None,
    pool_enabled: bool | None = None,
    status: str | None = None,
    config_path: str | None = None,
) -> list[VoiceInfo]:
    """Return catalog voices matching explicit metadata filters."""
    del config_path
    voices = get_catalog().filter(
        gender=gender,
        accent=accent,
        dataset=dataset,
        origin=origin,
        asset_kind=asset_kind,
        pool_enabled=pool_enabled,
        status=status,
    )
    infos = [_entry_to_info(v) for v in voices]
    return sorted(
        infos,
        key=lambda v: (
            _VOICE_ORIGIN_SORT_ORDER.get(v.origin, len(_VOICE_ORIGIN_SORT_ORDER)),
            v.display_name.lower(),
            v.voice_id.lower(),
        ),
    )


def get_voice(voice_id: str, config_path: str | None = None) -> VoiceInfo | None:
    """Look up a voice by canonical ``voice_id``."""
    del config_path
    entry = get_catalog().resolve(voice_id)
    return _entry_to_info(entry) if entry is not None else None


def set_voice_pool_enabled(voice_id: str, enabled: bool) -> PoolUpdateResult:
    """Enable or disable a catalog voice for automatic cast assignment."""
    entry = get_catalog().set_pool_enabled(voice_id, enabled)
    return PoolUpdateResult(voice_id=entry.voice_id, pool_enabled=entry.pool_enabled)


def _download_preview(entry: VoiceCatalogEntry, out_path: Path) -> bool:
    if not entry.preview.url:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(entry.preview.url, timeout=30) as response:  # noqa: S310
        out_path.write_bytes(response.read())
    if entry.preview.sha256 is not None and _hash_file(out_path) != entry.preview.sha256:
        raise RuntimeError(f"Preview hash does not match manifest for {entry.voice_id!r}")
    return True


def _synthesize_preview(entry: VoiceCatalogEntry, out_path: Path, text: str) -> None:
    from kenkui.config import load_app_config
    from kenkui.workers import _get_or_load_model, _render_text

    config = load_app_config(None)
    model = _get_or_load_model(
        config.temp,
        config.lsd_decode_steps,
        config.noise_clamp,
        config.eos_threshold,
    )
    voice_state = model.get_state_for_audio_prompt(load_voice_conditioning_source(entry.voice_id))
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
        raise RuntimeError(f"Synthesis returned no audio for voice_id {entry.voice_id!r}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seg.export(str(out_path), format="wav")


def prepare_voice_preview(
    voice_id: str,
    *,
    text: str | None = None,
    force: bool = False,
) -> AudioPreviewResult:
    """Return a local playable preview path for ``voice_id``.

    Hosted previews are preferred for kenkui compiled voices.  Built-in and
    custom voices synthesize/cache a local preview when no manifest preview path
    is already available.
    """
    entry = get_catalog().resolve(voice_id)
    if entry is None:
        raise KeyError(f"Unknown voice_id: {voice_id}")

    if entry.preview.path:
        path = Path(entry.preview.path)
        if path.exists() and not force:
            return AudioPreviewResult(voice_id=voice_id, audio_path=str(path), duration_ms=entry.preview.duration_ms)

    out_path = preview_cache_dir() / f"{voice_id}.wav"
    if out_path.exists() and not force:
        return AudioPreviewResult(voice_id=voice_id, audio_path=str(out_path), duration_ms=entry.preview.duration_ms)

    if entry.preview.url:
        try:
            if _download_preview(entry, out_path):
                return AudioPreviewResult(voice_id=voice_id, audio_path=str(out_path), duration_ms=entry.preview.duration_ms)
        except Exception as exc:
            logger.warning("Failed to download hosted preview for %s: %s", voice_id, exc)

    _synthesize_preview(entry, out_path, text or entry.preview.text)
    return AudioPreviewResult(voice_id=voice_id, audio_path=str(out_path), duration_ms=None)


def _default_compile_voice(source: str, output_path: Path) -> Path:
    """Compile a prompt source to a Pocket TTS voice-state safetensors file."""
    return compile_audio_prompt_source(source, output_path)


def import_custom_voice(
    *,
    source: str,
    voice_id: str,
    display_name: str,
    gender: str,
    pool_enabled: bool = False,
    accent: str | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    compiler: Callable[[str, Path], Path] | None = None,
    preview_generator: Callable[[str, Path], Path] | None = None,
) -> CustomVoiceImportResult:
    """Compile a user prompt source and add it as a local custom catalog entry."""
    if not display_name.strip():
        raise ValueError("display_name is required")
    if gender not in ("Male", "Female", "Nonbinary", "Unknown"):
        raise ValueError("gender must be Male, Female, Nonbinary, or Unknown")

    with tempfile.TemporaryDirectory(prefix="kenkui-voice-import-") as td:
        compiled_target = Path(td) / f"{voice_id}.safetensors"
        compiled = (compiler or _default_compile_voice)(source, compiled_target)
        if compiled.suffix != ".safetensors":
            raise ValueError("Custom voice compiler must produce a .safetensors file")

        preview_path = None
        if preview_generator is not None:
            preview_target = Path(td) / f"{voice_id}.wav"
            preview_path = preview_generator(str(compiled), preview_target)

        entry = get_catalog().add_custom_voice(
            voice_id=voice_id,
            display_name=display_name,
            gender=gender,
            compiled_path=compiled,
            pool_enabled=pool_enabled,
            accent=accent,
            tags=tags,
            notes=notes,
            preview_path=preview_path,
        )
    return CustomVoiceImportResult(
        voice=_entry_to_info(entry),
        manifest_path=str(get_catalog().custom_manifest_path),
    )


def gender_from_pronoun(pronoun: str | None) -> str:
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
    characters: list,
    default_voice: str,
) -> str:
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

    pool = get_catalog().pool()
    male_voices = [v.voice_id for v in pool if v.gender.lower() == "male" and v.voice_id != default_voice]
    female_voices = [v.voice_id for v in pool if v.gender.lower() == "female" and v.voice_id != default_voice]

    if top_female_quotes > top_male_quotes and female_voices:
        return female_voices[0]
    if male_voices:
        return male_voices[0]
    return default_voice


def assign_simple_cast(
    *,
    roster: list,
    narrator_voice: str,
    male_voice: str,
    female_voice: str,
) -> dict[str, str]:
    speaker_voices: dict[str, str] = {"NARRATOR": narrator_voice}
    for ch in roster:
        gender = gender_from_pronoun(getattr(ch, "gender_pronoun", None))
        if gender == "male":
            speaker_voices[ch.character_id] = male_voice
        elif gender == "female":
            speaker_voices[ch.character_id] = female_voice
        else:
            speaker_voices[ch.character_id] = narrator_voice
    return speaker_voices


def build_roster_payload(characters: list) -> list[dict]:
    return [
        {
            "name": character.character_id,
            "pronoun": character.gender_pronoun or None,
            "quote_count": character.quote_count,
            "mention_count": character.mention_count,
        }
        for character in characters
    ]


def merge_speaker_voices(
    base_assignments: dict[str, str],
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    merged = dict(base_assignments)
    if overrides:
        merged.update(overrides)
    return merged


def format_character_review_label(
    character,
    voice: str,
    pinned: set[str] | None = None,
    series_name: str | None = None,
) -> str:
    pinned = pinned or set()
    gender = getattr(character, "gender_pronoun", None) or "?"
    prominence = getattr(character, "prominence", 0)
    base = f"{voice:<20}  {character.display_name}  ({prominence} mentions, {gender})"
    if character.character_id in pinned and series_name:
        base += f"  [series: {series_name}]"
    return base


def build_voice_users(
    speaker_voices: dict[str, str],
    characters: list,
) -> dict[str, list[str]]:
    character_names = {c.character_id: c.display_name for c in characters}
    users: dict[str, list[str]] = defaultdict(list)
    for character_id, voice_id in speaker_voices.items():
        if character_id == "NARRATOR":
            continue
        users[voice_id].append(character_names.get(character_id, character_id))
    return dict(users)


def annotate_voice_choices(
    voice_choices: list[dict],
    voice_users: dict[str, list[str]],
    *,
    exclude_char_name: str | None = None,
) -> list[dict]:
    result: list[dict] = []
    for choice in voice_choices:
        if choice.get("value") == "__custom__":
            result.append(choice)
            continue
        voice_id = choice["value"]
        users = [u for u in voice_users.get(voice_id, []) if u != exclude_char_name]
        suffix = f"  <- {', '.join(users[:2])}" if users else ""
        result.append({**choice, "name": choice["name"] + suffix})
    return result


def format_unresolved_conflict_warnings(
    unresolved_conflicts: list[tuple[str, str]] | None,
    pinned: set[str] | None = None,
) -> list[str]:
    warnings: list[str] = []
    pinned = pinned or set()
    for char_a, char_b in unresolved_conflicts or []:
        inherited = char_a if char_a in pinned else char_b if char_b in pinned else None
        if inherited:
            warnings.append(
                f"{char_a!r} and {char_b!r} share a chapter with the same voice. "
                f"{inherited!r} is inherited from the series; change the other if needed."
            )
        else:
            warnings.append(
                f"{char_a!r} and {char_b!r} share a chapter with the same voice "
                "and no spare voice exists."
            )
    return warnings


def build_character_review_choices(
    characters: list,
    speaker_voices: dict[str, str],
    narrator_voice: str,
    *,
    pinned: set[str] | None = None,
    series_name: str | None = None,
) -> list[dict[str, str]]:
    pinned = pinned or set()
    review_choices: list[dict[str, str]] = []
    for character in sorted(characters, key=lambda c: c.prominence, reverse=True):
        current_voice = speaker_voices.get(character.character_id, narrator_voice)
        review_choices.append(
            {
                "name": format_character_review_label(
                    character,
                    current_voice,
                    pinned=pinned,
                    series_name=series_name,
                ),
                "value": character.character_id,
            }
        )
    return review_choices


def apply_voice_pool_template(
    roster: list,
    template: VoicePoolTemplate,
    series_voices: dict[str, str],
    narrator_voice: str,
    roster_roles: dict[str, str] | None = None,
) -> dict[str, str]:
    from collections import defaultdict

    from kenkui.voice_pool import _normalize_gender, _normalize_role

    if template.is_empty():
        unmatched = [c for c in roster if c.character_id not in series_voices]
        return suggest_cast(roster=unmatched, default_voice=narrator_voice).speaker_voices if unmatched else {}

    roles = roster_roles or {}
    buckets: dict[tuple[str, str], list] = defaultdict(list)
    for ch in sorted(roster, key=lambda c: c.prominence, reverse=True):
        if ch.character_id in series_voices:
            continue
        role_n = _normalize_role(roles.get(ch.character_id, "supporting"))
        gender_n = _normalize_gender(getattr(ch, "gender_pronoun", "") or "")
        buckets[(role_n, gender_n)].append(ch)

    assigned: dict[str, str] = {}
    pool_counters: dict[tuple, int] = {}
    for bucket_key, chars in buckets.items():
        role_n, gender_n = bucket_key
        slot = template.get_slot(role_n, gender_n)
        for rank, ch in enumerate(chars, start=1):
            voice = slot.pick(rank, pool_counters, bucket_key)
            if voice is not None and voice != narrator_voice:
                assigned[ch.character_id] = voice

    uncovered = [
        c for c in roster
        if c.character_id not in series_voices and c.character_id not in assigned
    ]
    if uncovered:
        assigned.update(suggest_cast(roster=uncovered, default_voice=narrator_voice).speaker_voices)
    return assigned


def sort_cast(speaker_voices: dict) -> list[tuple[str, str]]:
    return sorted(
        speaker_voices.items(),
        key=lambda kv: ("~" if kv[0] == "NARRATOR" else kv[0].lower()),
    )


def _get_chapter_cooccurrence_from_paragraphs(chapters) -> dict[int, set[str]]:
    result: dict[int, set[str]] = {}
    excluded = {"NARRATOR", "SCENE_BREAK", "Unknown"}
    for idx, ch in enumerate(chapters):
        speakers: set[str] = set()
        if hasattr(ch, "paragraphs"):
            for p in ch.paragraphs:
                sp = getattr(p, "speaker", None)
                if sp and getattr(p, "is_spoken", False) and sp not in excluded:
                    speakers.add(sp)
        elif hasattr(ch, "segments"):
            for s in ch.segments:
                sp = getattr(s, "speaker", None)
                if sp and sp not in excluded and not getattr(s, "is_scene_break", False):
                    speakers.add(sp)
        if speakers:
            result[idx] = speakers
    return result


def _resolve_cast_conflicts(
    speaker_voices: dict[str, str],
    char_prominence: dict[str, int],
    char_gender: dict[str, str],
    chapters,
    male_pool: list[str],
    female_pool: list[str],
    narrator_voice: str,
) -> tuple[dict[str, str], list[str]]:
    cooccurrence = _get_chapter_cooccurrence_from_paragraphs(chapters)
    warnings: list[str] = []
    unresolved_seen: set[frozenset] = set()

    changed = True
    while changed:
        changed = False
        for ch_idx, ch_speakers in cooccurrence.items():
            voice_to_chars: dict[str, list[str]] = defaultdict(list)
            for sp in ch_speakers:
                if sp in speaker_voices:
                    voice_to_chars[speaker_voices[sp]].append(sp)

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
                                f"{ch_idx}; no spare voice available."
                            )
    return speaker_voices, warnings


def suggest_cast(
    *,
    roster: list,
    default_voice: str,
    excluded_voices: list[str] | None = None,
    chapters: list | None = None,
    config_path: str | None = None,
) -> SuggestCastResult:
    """Assign voices from catalog entries where ``pool_enabled`` is true."""
    # External clients may lag the breaking API and still send excluded_voices.
    # Catalog pool_enabled state remains the only assignment filter.
    del config_path, excluded_voices
    warnings: list[str] = []
    pool = get_catalog().pool()
    male_pool = [v.voice_id for v in pool if v.gender.lower() == "male" and v.voice_id != default_voice]
    female_pool = [v.voice_id for v in pool if v.gender.lower() == "female" and v.voice_id != default_voice]

    if not male_pool and not female_pool:
        warnings.append("No voices are enabled in the assignment pool; using the default voice.")

    male_idx = female_idx = 0
    male_quotes = female_quotes = 0
    speaker_voices: dict[str, str] = {}
    char_prominence: dict[str, int] = {}
    char_gender: dict[str, str] = {}

    for ch in sorted(roster, key=lambda c: c.prominence, reverse=True):
        gender = gender_from_pronoun(ch.gender_pronoun)
        char_prominence[ch.character_id] = ch.prominence
        char_gender[ch.character_id] = gender
        if gender == "male":
            voice = male_pool[male_idx % len(male_pool)] if male_pool else default_voice
            male_idx += 1 if male_pool else 0
            male_quotes += ch.prominence
        elif gender == "female":
            voice = female_pool[female_idx % len(female_pool)] if female_pool else default_voice
            female_idx += 1 if female_pool else 0
            female_quotes += ch.prominence
        elif male_quotes <= female_quotes and male_pool:
            voice = male_pool[male_idx % len(male_pool)]
            male_idx += 1
            male_quotes += ch.prominence
        elif female_pool:
            voice = female_pool[female_idx % len(female_pool)]
            female_idx += 1
            female_quotes += ch.prominence
        else:
            voice = default_voice
        speaker_voices[ch.character_id] = voice

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
    "PoolUpdateResult",
    "AudioPreviewResult",
    "CustomVoiceImportResult",
    "SuggestCastResult",
    "list_voices",
    "get_voice",
    "set_voice_pool_enabled",
    "prepare_voice_preview",
    "import_custom_voice",
    "apply_voice_pool_template",
    "assign_simple_cast",
    "build_roster_payload",
    "merge_speaker_voices",
    "format_character_review_label",
    "build_voice_users",
    "annotate_voice_choices",
    "format_unresolved_conflict_warnings",
    "build_character_review_choices",
    "gender_from_pronoun",
    "top_gender_matched_voice",
    "sort_cast",
    "suggest_cast",
]
