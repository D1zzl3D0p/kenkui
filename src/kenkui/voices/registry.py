"""Static built-in voice catalog with per-voice rights metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

from kenkui.errors import ErrorCode, VoiceError
from kenkui.voices.types import PerceivedGender, Voice

_BUILTIN_MANIFEST: Final = Path(__file__).with_name("builtin.json")


def _embedding_pin() -> tuple[str, str]:
    """Return the (repo_id, revision) the ungated embeddings are pinned to."""
    payload = json.loads(_BUILTIN_MANIFEST.read_text(encoding="utf-8"))
    return payload["embedding"]["repo_id"], payload["embedding"]["revision"]


_EMBEDDING_REPO, EMBEDDING_REVISION = _embedding_pin()

# The pre-compiled Kenkui voice pack. Pinned so a repository update cannot
# silently change which bytes a load fetches, exactly as EMBEDDING_REVISION
# pins the kyutai embeddings.
PACK_REVISION: Final = "33f00bc5d8608de365592edc789d1aee2de4815a"
_PACK_REPO: Final = "D1zzl3D0p/kenkui-voices"
_PACK_MANIFEST: Final = Path(__file__).with_name("manifest.json")
# Every pack voice is English; the corpora behind it are VCTK and EARS.
_PACK_LANGUAGE: Final = "english"
_GENDERS: Final[dict[str, PerceivedGender]] = {
    "Male": "masculine",
    "Female": "feminine",
}

@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """One upstream predefined voice and the rights Kenkui records for it."""

    id: str
    name: str
    language: str
    origin_url: str
    license_id: str
    commercial_use_allowed: bool
    voice_rights: str
    # Set for pack voices, which are fetched from the Kenkui voice pack rather
    # than derived from a kyutai catalog name. None means derive the kyutai
    # embedding URL from the language and ID.
    asset_url: str | None = None
    # Unsourced for every built-in entry below. kyutai's VCTK_Voice_Names.csv covers a
    # different speaker selection than these voices, and VCTK's speaker-info.txt
    # ships only inside the full corpus download. Sourced traits arrive with the
    # pre-compiled voice pack; guessing from the display names above -- which
    # Kenkui invented -- would be worse than admitting the gap.
    perceived_gender: PerceivedGender = None


def _builtin_entry(voice: dict[str, Any]) -> CatalogEntry:
    """Convert one builtin.json record into a catalog entry."""
    return CatalogEntry(
        id=voice["voice_id"],
        name=voice["display_name"],
        language=voice["language"],
        origin_url=voice["origin_url"],
        license_id=voice["license_id"],
        commercial_use_allowed=bool(voice["commercial_use_allowed"]),
        voice_rights=voice["voice_rights"],
        perceived_gender=voice["perceived_gender"],
    )


@lru_cache(maxsize=1)
def load_builtin() -> tuple[CatalogEntry, ...]:
    """Read the kyutai predefined voices that ship with the package.

    Unlike the pack, these are not optional. A missing or unreadable file is a
    broken install, not a degraded one, so the error propagates.
    """
    payload = json.loads(_BUILTIN_MANIFEST.read_text(encoding="utf-8"))
    return tuple(_builtin_entry(voice) for voice in payload["voices"])


def _pack_entry(voice: dict[str, Any], voice_id: str) -> CatalogEntry:
    """Convert one pack manifest record into a catalog entry."""
    slug = voice["voice_id"]
    return CatalogEntry(
        id=voice_id,
        name=voice["display_name"],
        language=_PACK_LANGUAGE,
        origin_url=f"hf://{_PACK_REPO}/{voice['path']}@{PACK_REVISION}",
        license_id=voice["license_id"],
        commercial_use_allowed=bool(voice["commercial_use_allowed"]),
        voice_rights=voice["voice_rights"],
        asset_url=f"hf://{_PACK_REPO}/compiled/{slug}.safetensors@{PACK_REVISION}",
        perceived_gender=_GENDERS.get(voice["gender"]),
    )


@lru_cache(maxsize=1)
def load_pack() -> tuple[CatalogEntry, ...]:
    """Read the bundled voice pack, or return nothing if it is not shipped.

    Static package data, not a network or user source: no request is made and
    no path comes from a caller. A missing or unreadable manifest degrades to
    the built-in catalog rather than failing an import, because the pack is an
    addition and Kenkui must work without it.
    """
    try:
        payload = json.loads(_PACK_MANIFEST.read_text(encoding="utf-8"))
        voices = payload["voices"]
    except (OSError, ValueError, KeyError, TypeError):
        return ()
    built_in = {entry.id for entry in load_builtin()}
    entries: list[CatalogEntry] = []
    for voice in voices:
        # Short display names read far better at a call site than the full
        # slug. Where one would shadow a built-in the pack voice keeps its
        # slug: they are different speakers who happen to share a name, so
        # dropping either would lose a voice.
        short = str(voice["display_name"]).lower()
        entries.append(
            _pack_entry(voice, voice["voice_id"] if short in built_in else short)
        )
    return tuple(entries)


def _catalog() -> dict[str, CatalogEntry]:
    """Merge the built-in catalog with the pack, built-ins winning."""
    merged = {entry.id: entry for entry in load_pack()}
    merged.update({entry.id: entry for entry in load_builtin()})
    return merged


# The kyutai predefined voices alone. Kept addressable so the upstream-drift
# guard compares against what upstream actually ships, rather than against the
# merged catalog the voice pack also feeds.
BUILT_IN_CATALOG: Final[dict[str, CatalogEntry]] = {
    entry.id: entry for entry in load_builtin()
}
CATALOG: Final[dict[str, CatalogEntry]] = _catalog()


def embedding_url(language: str, name: str) -> str:
    """Return the pinned ungated embedding URL for one catalog voice."""
    return (
        f"hf://{_EMBEDDING_REPO}/languages/{language}/embeddings/"
        f"{name}.safetensors@{EMBEDDING_REVISION}"
    )


def asset_url(voice_id: str, language: str) -> str:
    """Return where one catalog voice's embedding is fetched from.

    Pack voices carry their own pinned URL; kyutai built-ins derive theirs
    from the language and catalog name.
    """
    entry = CATALOG.get(voice_id)
    if entry is not None and entry.asset_url is not None:
        return entry.asset_url
    return embedding_url(language, voice_id)


def catalog_voice(voice_id: str) -> Voice:
    """Return catalog metadata for one built-in voice as a registered Voice."""
    entry = CATALOG.get(voice_id)
    if entry is None:
        raise VoiceError(ErrorCode.VOICE_UNKNOWN)
    return Voice(
        id=entry.id,
        name=entry.name,
        enabled=True,
        provenance=entry.origin_url,
        license_id=entry.license_id,
        commercial_use_allowed=entry.commercial_use_allowed,
        language=entry.language,
        variety="built-in",
        state="registered",
        perceived_gender=entry.perceived_gender,
    )
