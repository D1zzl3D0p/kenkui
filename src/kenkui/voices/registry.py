"""Static built-in voice catalog; each voice's terms come from its dataset."""

from __future__ import annotations

import json
import logging
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any, Final

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from kenkui.errors import ErrorCode, VoiceError
from kenkui.observability import get_logger, log_event
from kenkui.voices.datasets import (
    Dataset,
    dataset_for_origin,
    dataset_for_pack,
    rights_for,
)
from kenkui.voices.types import PerceivedGender, Voice

_LOGGER = get_logger(__name__)

_BUILTIN_MANIFEST: Final = Path(__file__).with_name("builtin.json")


def _embedding_pin() -> tuple[str, str]:
    """Return the (repo_id, revision) the ungated embeddings are pinned to."""
    payload = json.loads(_BUILTIN_MANIFEST.read_text(encoding="utf-8"))
    return payload["embedding"]["repo_id"], payload["embedding"]["revision"]


_EMBEDDING_REPO, EMBEDDING_REVISION = _embedding_pin()

# The pre-compiled Kenkui voice pack. Where it lives and which revision it
# pins are data in the manifest, not constants here: a constant in Python
# source is what previously let the pack and the runtime diverge in silence.
_PACK_MANIFEST: Final = Path(__file__).with_name("pack.json")
_REBUILD_HINT: Final = (
    "Rebuild the pack for this pocket-tts: "
    "hf download D1zzl3D0p/kenkui-voices --repo-type dataset "
    "--local-dir kenkui-voices && cd kenkui-voices/tools && uv sync && "
    "uv run python rebuild.py --all --force"
)
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
    # The source corpus. License, rights statement, and the commercial default
    # are derived from it rather than recorded per voice.
    dataset: Dataset
    # Set for pack voices, which are fetched from the Kenkui voice pack rather
    # than derived from a kyutai catalog name. None means derive the kyutai
    # embedding URL from the language and ID.
    asset_url: str | None = None
    # Built-in traits are declared in builtin.json, together with a pinned
    # provenance URL. None remains the honest default for custom or future
    # entries whose perceived presentation has not been documented.
    perceived_gender: PerceivedGender = None

    @property
    def license_id(self) -> str:
        """The dataset's license identifier."""
        return rights_for(self.dataset).license_id

    @property
    def voice_rights(self) -> str:
        """The dataset's rights statement."""
        return rights_for(self.dataset).voice_rights

    @property
    def commercial_use_allowed(self) -> bool:
        """The dataset's commercial-use default; False for every catalog voice."""
        return rights_for(self.dataset).commercial_use_allowed


def _builtin_entry(voice: dict[str, Any]) -> CatalogEntry:
    """Convert one builtin.json record into a catalog entry."""
    return CatalogEntry(
        id=voice["voice_id"],
        name=voice["display_name"],
        language=voice["language"],
        origin_url=voice["origin_url"],
        dataset=dataset_for_origin(voice["origin_url"]),
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


@dataclass(frozen=True, slots=True)
class PackLoad:
    """The pack as resolved against the installed pocket-tts.

    `incompatible` is None when the pack loaded or was simply absent, and a
    message naming both versions when the pack exists but cannot be used. The
    distinction matters: an absent pack is a degraded install, an incompatible
    one is a misconfiguration that must not fail quietly.
    """

    entries: tuple[CatalogEntry, ...]
    incompatible: str | None


def installed_pocket_version() -> str:
    """Return the installed pocket-tts version."""
    return metadata.version("pocket-tts")


def _incompatibility(declared: str) -> str | None:
    """Return why the installed pocket-tts cannot use this pack, or None."""
    installed = installed_pocket_version()
    try:
        if Version(installed) in SpecifierSet(declared):
            return None
    except (InvalidSpecifier, InvalidVersion):
        return (
            f"the voice pack declares an unreadable pocket-tts range "
            f"{declared!r}; installed is {installed}. {_REBUILD_HINT}"
        )
    return (
        f"the voice pack was compiled for pocket-tts {declared} but "
        f"{installed} is installed. Its embeddings would load without error "
        f"and render audio containing no words. {_REBUILD_HINT}"
    )


def _pack_entry(
    voice: dict[str, Any], voice_id: str, assets: dict[str, str]
) -> CatalogEntry:
    """Convert one pack manifest record into a catalog entry."""
    repo = assets["repo_id"]
    revision = assets["revision"]
    return CatalogEntry(
        id=voice_id,
        name=voice["display_name"],
        language=voice["language"],
        origin_url=f"hf://{voice['source']['repo_id']}/{voice['source']['path']}",
        # The pack's own license fields are ignored: the dataset decides, and
        # a test holds the bundled pack to agreeing with it.
        dataset=dataset_for_pack(voice["dataset"]),
        # A dataset resolve URL, not hf://: Pocket-TTS reads hf:// as a model
        # repository, and the pack is a dataset, so hf:// cannot be fetched.
        # The revision in the path still pins the exact bytes.
        asset_url=(
            f"https://huggingface.co/datasets/{repo}/resolve/{revision}/"
            f"{voice['compiled']['path']}"
        ),
        perceived_gender=_GENDERS.get(voice["gender"]),
    )


@lru_cache(maxsize=1)
def load_pack() -> PackLoad:
    """Read the bundled voice pack and resolve it against pocket-tts.

    Static package data, not a network or user source: no request is made and
    no path comes from a caller. A missing or unreadable manifest degrades to
    the built-in catalog rather than failing an import, because the pack is an
    addition and Kenkui must work without it. A pack that is present but
    compiled for a different pocket-tts major is a different matter: its
    voices are withheld and the reason is kept, so using one raises instead of
    rendering silence.
    """
    try:
        payload = json.loads(_PACK_MANIFEST.read_text(encoding="utf-8"))
        voices = payload["voices"]
        assets = payload["assets"]
        declared = payload["pocket_tts"]
    except (OSError, ValueError, KeyError, TypeError):
        return PackLoad((), None)

    incompatible = _incompatibility(declared)
    if incompatible is not None:
        return PackLoad((), incompatible)

    built_in = {entry.id for entry in load_builtin()}
    entries: list[CatalogEntry] = []
    try:
        for voice in voices:
            # A voice with no compiled block is unpublished: a gap in the pack,
            # not a reason to abandon the other ninety-four.
            if not voice.get("compiled"):
                continue
            # Short display names read far better at a call site than the full
            # slug. Where one would shadow a built-in the pack voice keeps its
            # slug: they are different speakers who happen to share a name, so
            # dropping either would lose a voice.
            short = str(voice["display_name"]).lower()
            entries.append(
                _pack_entry(
                    voice, voice["voice_id"] if short in built_in else short, assets
                )
            )
    except (KeyError, TypeError, AttributeError):
        # One malformed record makes the whole pack untrustworthy, and the
        # catalog is built at import: raising here would break `import kenkui`
        # rather than degrade to the built-in voices this promises. Logged
        # because the returned shape is identical to an absent pack, and an
        # operator whose ninety-five voices vanished needs to know why.
        log_event(
            _LOGGER,
            "voice_pack_malformed",
            level=logging.WARNING,
            context={"boundary": "voices", "path": _PACK_MANIFEST.name},
        )
        return PackLoad((), None)
    return PackLoad(tuple(entries), None)


@lru_cache(maxsize=1)
def _withheld_pack_ids() -> frozenset[str]:
    """Return the voice IDs an incompatible pack is keeping out of the catalog."""
    if load_pack().incompatible is None:
        return frozenset()
    try:
        payload = json.loads(_PACK_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return frozenset()
    ids: set[str] = set()
    # Per record rather than around the loop: aborting on the first malformed
    # entry would leave every later voice unguarded, so an incompatible pack
    # would raise VOICE_UNKNOWN for them instead of VOICE_INCOMPATIBLE and its
    # rebuild hint.
    for voice in _iterable(payload.get("voices", ())):
        with suppress(KeyError, TypeError, AttributeError):
            ids.add(voice["voice_id"])
            ids.add(str(voice["display_name"]).lower())
    return frozenset(ids)


def _iterable(value: object) -> tuple[Any, ...]:
    """Return a safely iterable view of a manifest collection."""
    return tuple(value) if isinstance(value, list) else ()


def pack_voice_guard(voice_id: str) -> None:
    """Raise if this ID names a voice an incompatible pack is withholding."""
    reason = load_pack().incompatible
    if reason is not None and voice_id in _withheld_pack_ids():
        raise VoiceError(ErrorCode.VOICE_INCOMPATIBLE, reason)


def _catalog() -> dict[str, CatalogEntry]:
    """Merge the built-in catalog with the pack, built-ins winning."""
    merged = {entry.id: entry for entry in load_pack().entries}
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
    pack_voice_guard(voice_id)
    entry = CATALOG.get(voice_id)
    if entry is not None and entry.asset_url is not None:
        return entry.asset_url
    return embedding_url(language, voice_id)


def catalog_voice(voice_id: str) -> Voice:
    """Return catalog metadata for one built-in voice as a registered Voice."""
    pack_voice_guard(voice_id)
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
        voice_rights=entry.voice_rights,
    )
