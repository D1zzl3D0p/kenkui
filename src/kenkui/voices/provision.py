"""Explicit, user-initiated voice provisioning.

This is the only module in Kenkui permitted to reach the network. Nothing under
`kenkui._execution` or `kenkui._tts` may import it, and it never runs inside a
render: provisioning downloads and compiles, rendering only reads a manifest.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Final

from kenkui.errors import ErrorCode, VoiceError
from kenkui.voices.manifest import ManifestStore, VoiceRecord, default_manifest_path
from kenkui.voices.registry import CATALOG
from kenkui.voices.types import Voice, VoiceVariety

if TYPE_CHECKING:
    import os

_HASH_CHUNK_BYTES: Final = 1024 * 1024
_SUFFIX_VARIETY: Final[dict[str, VoiceVariety]] = {
    ".wav": "wav",
    ".safetensors": "pre-compiled",
}


def _store(manifest: Path | None) -> ManifestStore:
    return ManifestStore(manifest if manifest is not None else default_manifest_path())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_HASH_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def add_voice(  # noqa: PLR0913
    # The seven rights fields are deliberately separate required keyword
    # arguments: the spec forbids inferring any of them, and collapsing them
    # into a container would let a caller omit one by passing a partial object.
    path: str | os.PathLike[str],
    *,
    voice_id: str,
    name: str,
    language: str,
    provenance: str,
    license_id: str,
    commercial_use_allowed: bool,
    voice_rights: str,
    manifest: Path | None = None,
) -> Voice:
    """Register a local WAV or safetensors voice with explicit rights metadata.

    Cheap and offline: hashes the source, records the rights fields verbatim,
    and writes a `registered` entry. Call `load_voice` to make it renderable.
    """
    if voice_id in CATALOG:
        raise VoiceError(ErrorCode.VOICE_VARIETY_INVALID)
    source = Path(path).resolve()
    variety = _SUFFIX_VARIETY.get(source.suffix.lower())
    if variety is None:
        raise VoiceError(ErrorCode.VOICE_VARIETY_INVALID)
    if not source.is_file():
        raise VoiceError(ErrorCode.POCKET_VOICE_INVALID)
    record = VoiceRecord(
        id=voice_id,
        variety=variety,
        state="registered",
        name=name,
        enabled=True,
        language=language,
        engine_id=language,
        provenance=provenance,
        license_id=license_id,
        commercial_use_allowed=commercial_use_allowed,
        voice_rights=voice_rights,
        source_path=str(source),
        source_sha256=_sha256(source),
    )
    store = _store(manifest)
    with store.lock():
        engines, voices = store.read()
        voices[voice_id] = record
        store.write(engines, voices)
    return Voice(
        id=voice_id,
        name=name,
        enabled=True,
        provenance=provenance,
        license_id=license_id,
        commercial_use_allowed=commercial_use_allowed,
        language=language,
        variety=variety,
        state="registered",
    )
