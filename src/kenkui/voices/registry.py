"""Static built-in voice catalog with per-voice rights metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from kenkui.errors import ErrorCode, VoiceError
from kenkui.voices.types import Voice

EMBEDDING_REVISION: Final = "e041936c75475d350b405bc870bcf7c22da4e9e6"
_EMBEDDING_REPO: Final = "kyutai/pocket-tts-without-voice-cloning"

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

CATALOG: Final[dict[str, CatalogEntry]] = {
    entry.id: entry
    for entry in (
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
}


def embedding_url(language: str, name: str) -> str:
    """Return the pinned ungated embedding URL for one catalog voice."""
    return (
        f"hf://{_EMBEDDING_REPO}/languages/{language}/embeddings/"
        f"{name}.safetensors@{EMBEDDING_REVISION}"
    )


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
    )
