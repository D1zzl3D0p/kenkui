"""Static built-in voice catalog with per-voice rights metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

from kenkui.errors import ErrorCode, VoiceError
from kenkui.voices.types import PerceivedGender, Voice

EMBEDDING_REVISION: Final = "e041936c75475d350b405bc870bcf7c22da4e9e6"
_EMBEDDING_REPO: Final = "kyutai/pocket-tts-without-voice-cloning"

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

_VCTK_RIGHTS: Final = (
    "Derived from the VCTK corpus via kyutai/tts-voices. Review the VCTK terms "
    "and speaker consent for your intended use before commercial deployment."
)
_EARS_RIGHTS: Final = (
    "Derived from the EARS corpus. Treat as research-only/noncommercial unless "
    "your own review of the source terms concludes otherwise."
)
_EXPRESSO_RIGHTS: Final = (
    "Derived from the Expresso dataset. Treat as research-only/noncommercial "
    "unless your own review of the source terms concludes otherwise."
)
_DONATION_RIGHTS: Final = (
    "Voice donation distributed by kyutai. Confirm the donor's permission "
    "scope for your intended use."
)
_COMMON_VOICE_RIGHTS: Final = (
    "Derived from Common Voice via kyutai/pocket-tts. Review the Common Voice "
    "terms for your intended use before commercial deployment."
)


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


def _vctk(voice_id: str, name: str, filename: str) -> CatalogEntry:
    return CatalogEntry(
        id=voice_id,
        name=name,
        language="english",
        origin_url=f"hf://kyutai/tts-voices/vctk/{filename}",
        license_id="CC-BY-4.0",
        commercial_use_allowed=False,
        voice_rights=_VCTK_RIGHTS,
    )


def _zero(voice_id: str, name: str) -> CatalogEntry:
    return CatalogEntry(
        id=voice_id,
        name=name,
        language="english",
        origin_url=f"hf://kyutai/tts-voices/voice-zero/{voice_id}.wav",
        license_id="unreviewed",
        commercial_use_allowed=False,
        voice_rights=_DONATION_RIGHTS,
    )


def _donation(voice_id: str, name: str, filename: str) -> CatalogEntry:
    return CatalogEntry(
        id=voice_id,
        name=name,
        language="english",
        origin_url=f"hf://kyutai/tts-voices/voice-donations/{filename}",
        license_id="unreviewed",
        commercial_use_allowed=False,
        voice_rights=_DONATION_RIGHTS,
    )


_POCKET_PIN: Final = "@64ab7d24c479d736a83b8cc666c4a776fca30fda"

_BUILT_IN: Final[tuple[CatalogEntry, ...]] = (
    _vctk("anna", "Anna", "p228_023_enhanced.wav"),
    _vctk("vera", "Vera", "p229_023_enhanced.wav"),
    _vctk("fantine", "Fantine", "p244_023_enhanced.wav"),
    _vctk("charles", "Charles", "p254_023_enhanced.wav"),
    _vctk("paul", "Paul", "p259_023_enhanced.wav"),
    _vctk("eponine", "Eponine", "p262_023_enhanced.wav"),
    _vctk("azelma", "Azelma", "p303_023_enhanced.wav"),
    _vctk("george", "George", "p315_023_enhanced.wav"),
    _vctk("mary", "Mary", "p333_023_enhanced.wav"),
    _vctk("jane", "Jane", "p339_023_enhanced.wav"),
    _vctk("michael", "Michael", "p360_023_enhanced.wav"),
    _vctk("eve", "Eve", "p361_023_enhanced.wav"),
    _zero("bill_boerst", "Bill Boerst"),
    _zero("peter_yearsley", "Peter Yearsley"),
    _zero("stuart_bell", "Stuart Bell"),
    _zero("caro_davy", "Caro Davy"),
    _donation("marius", "Marius", "Selfie.wav"),
    _donation("javert", "Javert", "Butter.wav"),
    CatalogEntry(
        id="cosette",
        name="Cosette",
        language="english",
        origin_url=(
            "hf://kyutai/tts-voices/expresso/"
            "ex04-ex02_confused_001_channel1_499s.wav"
        ),
        license_id="CC-BY-NC-4.0",
        commercial_use_allowed=False,
        voice_rights=_EXPRESSO_RIGHTS,
    ),
    CatalogEntry(
        id="jean",
        name="Jean",
        language="english",
        origin_url=(
            "hf://kyutai/tts-voices/ears/p010/freeform_speech_01_enhanced.wav"
        ),
        license_id="CC-BY-NC-4.0",
        commercial_use_allowed=False,
        voice_rights=_EARS_RIGHTS,
    ),
    CatalogEntry(
        id="alba",
        name="Alba",
        language="english",
        origin_url="hf://kyutai/tts-voices/alba-mackenna/casual.wav",
        license_id="unreviewed",
        commercial_use_allowed=False,
        voice_rights=_DONATION_RIGHTS,
    ),
    CatalogEntry(
        id="giovanni",
        name="Giovanni",
        language="italian",
        origin_url=(
            "hf://kyutai/pocket-tts/"
            f"common_voice_it_36520747-enhanced-v2.mp3{_POCKET_PIN}"
        ),
        license_id="CC0-1.0",
        commercial_use_allowed=False,
        voice_rights=_COMMON_VOICE_RIGHTS,
    ),
    CatalogEntry(
        id="lola",
        name="Lola",
        language="spanish",
        origin_url=(
            "hf://kyutai/pocket-tts/"
            f"common_voice_es_19762977-enhanced-v2.mp3{_POCKET_PIN}"
        ),
        license_id="CC0-1.0",
        commercial_use_allowed=False,
        voice_rights=_COMMON_VOICE_RIGHTS,
    ),
    CatalogEntry(
        id="juergen",
        name="Juergen",
        language="german",
        origin_url=f"hf://kyutai/pocket-tts/de-DE-juergen.mp3{_POCKET_PIN}",
        license_id="unreviewed",
        commercial_use_allowed=False,
        voice_rights=_DONATION_RIGHTS,
    ),
    CatalogEntry(
        id="rafael",
        name="Rafael",
        language="portuguese",
        origin_url=(
            f"hf://kyutai/pocket-tts/g-Vi8PgmSY0-enhanced-v2.wav{_POCKET_PIN}"
        ),
        license_id="unreviewed",
        commercial_use_allowed=False,
        voice_rights=_DONATION_RIGHTS,
    ),
    CatalogEntry(
        id="estelle",
        name="Estelle",
        language="french_24l",
        origin_url=(
            "hf://kyutai/tts-voices/unmute-prod-website/developpeuse-3.wav"
            "@1fc7395b7e012e2bbebfca14b942a4ef62ccc899"
        ),
        license_id="unreviewed",
        commercial_use_allowed=False,
        voice_rights=_DONATION_RIGHTS,
    ),
)


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
    built_in = {entry.id for entry in _BUILT_IN}
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
    merged.update({entry.id: entry for entry in _BUILT_IN})
    return merged


# The kyutai predefined voices alone. Kept addressable so the upstream-drift
# guard compares against what upstream actually ships, rather than against the
# merged catalog the voice pack also feeds.
BUILT_IN_CATALOG: Final[dict[str, CatalogEntry]] = {
    entry.id: entry for entry in _BUILT_IN
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
