"""Explicit, user-initiated voice provisioning.

This is the only module in Kenkui permitted to reach the network. Nothing under
`kenkui._execution` or `kenkui._tts` may import it, and it never runs inside a
render: provisioning downloads and compiles, rendering only reads a manifest.
"""

from __future__ import annotations

import dataclasses
import hashlib
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import yaml

from kenkui.errors import ErrorCode, VoiceError
from kenkui.voices.manifest import (
    EngineRecord,
    FileRecord,
    ManifestStore,
    VoiceRecord,
    default_manifest_path,
)
from kenkui.voices.registry import CATALOG, catalog_voice, embedding_url
from kenkui.voices.types import Engine, Voice, VoiceVariety

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


def _fetch(url: str) -> Path:
    """Resolve one remote or local asset reference to a local file.

    The single network seam in Kenkui. Tests monkeypatch this symbol.
    """
    from pocket_tts.utils.utils import download_if_necessary  # noqa: PLC0415

    return Path(download_if_necessary(url))


def _assets_root(manifest_path: Path) -> Path:
    return manifest_path.parent


def _revision_of(url: str) -> str:
    return url.rsplit("@", 1)[-1] if "@" in url else "unpinned"


def _place(source: Path, destination: Path) -> FileRecord:
    """Copy one fetched asset into the engine root and record its identity."""
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(0o600)
    return FileRecord(
        relative_path=destination.name,
        size=destination.stat().st_size,
        sha256=_sha256(destination),
    )


def _provision_engine(language: str, *, cloning: bool, root: Path) -> EngineRecord:
    """Download one language engine and write a fully local derived config.

    The renderer replaces pocket-tts's `download_if_necessary` with an allowlist
    that accepts only absolute local paths, so a stock config full of `hf://`
    URLs would be rejected at load time. The derived config substitutes local
    paths for all three weight references.
    """
    from pocket_tts.utils.config import CONFIGS_DIR  # noqa: PLC0415

    model_root = root / "engines" / language
    model_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    stock: dict[str, Any] = yaml.safe_load(
        (CONFIGS_DIR / f"{language}.yaml").read_text(encoding="utf-8")
    )
    weights_url = (
        stock["weights_path"]
        if cloning
        else stock["weights_path_without_voice_cloning"]
    )
    files = [_place(_fetch(weights_url), model_root / "model.safetensors")]

    derived = dict(stock)
    derived["weights_path"] = str(model_root / "model.safetensors")
    derived.pop("weights_path_without_voice_cloning", None)
    derived["flow_lm"] = dict(stock["flow_lm"])
    derived["mimi"] = dict(stock["mimi"])

    # tokenizer and Mimi weights are optional: Mimi's are commented out upstream
    # for the distilled models, where they ride inside the main safetensors.
    lookup = stock["flow_lm"].get("lookup_table")
    if lookup is not None and lookup.get("tokenizer_path"):
        derived["flow_lm"]["lookup_table"] = dict(lookup)
        files.append(
            _place(_fetch(lookup["tokenizer_path"]), model_root / "tokenizer.model")
        )
        derived["flow_lm"]["lookup_table"]["tokenizer_path"] = str(
            model_root / "tokenizer.model"
        )
    if stock["mimi"].get("weights_path"):
        files.append(
            _place(
                _fetch(stock["mimi"]["weights_path"]), model_root / "mimi.safetensors"
            )
        )
        derived["mimi"]["weights_path"] = str(model_root / "mimi.safetensors")
    if stock["flow_lm"].get("weights_path"):
        files.append(
            _place(
                _fetch(stock["flow_lm"]["weights_path"]),
                model_root / "flow_lm.safetensors",
            )
        )
        derived["flow_lm"]["weights_path"] = str(model_root / "flow_lm.safetensors")

    config_path = model_root / f"{language}.yaml"
    config_path.write_text(yaml.safe_dump(derived, sort_keys=True), encoding="utf-8")
    config_path.chmod(0o600)
    files.append(
        FileRecord(
            relative_path=config_path.name,
            size=config_path.stat().st_size,
            sha256=_sha256(config_path),
        )
    )

    return EngineRecord(
        id=language,
        language=language,
        model_root=str(model_root),
        config_path=str(config_path),
        model_revision=_revision_of(weights_url),
        package_version="2.1.0",
        files=tuple(files),
        sample_rate_hz=int(stock["mimi"]["sample_rate"]),
        device="cpu",
        timeout_seconds=300.0,
        cloning_capable=cloning,
    )


def _engine_view(record: EngineRecord) -> Engine:
    return Engine(
        id=record.id,
        language=record.language,
        model_revision=record.model_revision,
        cloning_capable=record.cloning_capable,
        size_bytes=sum(item.size for item in record.files),
    )


def _loaded_view(record: VoiceRecord, engine: EngineRecord) -> Voice:
    asset = Path(record.asset_path or "")
    return Voice(
        id=record.id,
        name=record.name,
        enabled=record.enabled,
        provenance=record.provenance,
        license_id=record.license_id,
        commercial_use_allowed=record.commercial_use_allowed,
        language=record.language,
        content_fingerprint=record.asset_sha256,
        compatible_model_revisions=record.compatible_model_revisions,
        variety=record.variety,
        state="loaded",
        asset_bytes=asset.stat().st_size if asset.is_file() else None,
        engine=_engine_view(engine),
    )


def _registered_from_catalog(voice_id: str) -> VoiceRecord:
    entry = CATALOG[voice_id]
    return VoiceRecord(
        id=entry.id,
        variety="built-in",
        state="registered",
        name=entry.name,
        enabled=True,
        language=entry.language,
        engine_id=entry.language,
        provenance=entry.origin_url,
        license_id=entry.license_id,
        commercial_use_allowed=entry.commercial_use_allowed,
        voice_rights=entry.voice_rights,
    )


def _compile_wav(source: Path, engine: EngineRecord, destination: Path) -> None:
    """Compile one audio prompt into a speaker embedding via pocket-tts.

    Voice cloning needs the gated weights, so this runs only at provision time
    and never during a render.
    """
    from pocket_tts import export_model_state  # noqa: PLC0415
    from pocket_tts.models.tts_model import TTSModel  # noqa: PLC0415

    if not engine.cloning_capable:
        raise VoiceError(ErrorCode.ENGINE_NOT_CLONING_CAPABLE)
    try:
        model = TTSModel.load_model(config=Path(engine.config_path))
        state = model.get_state_for_audio_prompt(
            audio_conditioning=source, truncate=True
        )
        export_model_state(state, destination)
    except VoiceError:
        raise
    except Exception:  # noqa: BLE001 — sanitize any torch/pocket-tts failure
        # Public errors are stable and must not leak third-party internals or
        # local paths, so every downstream failure collapses to one code.
        raise VoiceError(ErrorCode.POCKET_VOICE_LOAD_FAILED) from None


def _materialize(record: VoiceRecord, engine: EngineRecord, root: Path) -> VoiceRecord:
    """Produce the safetensors asset for one registered voice."""
    destination = root / "voices" / record.language / f"{record.id}.safetensors"
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if record.variety == "built-in":
        fetched = _fetch(embedding_url(record.language, record.id))
        shutil.copyfile(fetched, destination)
    else:
        source = Path(record.source_path or "")
        if not source.is_file() or _sha256(source) != record.source_sha256:
            raise VoiceError(ErrorCode.POCKET_VOICE_INVALID)
        if record.variety == "wav":
            _compile_wav(source, engine, destination)
        else:
            shutil.copyfile(source, destination)
    destination.chmod(0o600)
    return VoiceRecord(
        id=record.id,
        variety=record.variety,
        state="loaded",
        name=record.name,
        enabled=record.enabled,
        language=record.language,
        engine_id=engine.id,
        provenance=record.provenance,
        license_id=record.license_id,
        commercial_use_allowed=record.commercial_use_allowed,
        voice_rights=record.voice_rights,
        source_path=record.source_path,
        source_sha256=record.source_sha256,
        asset_path=str(destination),
        asset_sha256=_sha256(destination),
        compatible_model_revisions=(engine.model_revision,),
    )


def load_voice(voice_id: str, *, manifest: Path | None = None) -> Voice:
    """Make one voice renderable, downloading or compiling only if required.

    Idempotent: a voice whose asset is present and hash-verified performs no
    network access at all.
    """
    store = _store(manifest)
    root = _assets_root(store.path)
    with store.lock():
        engines, voices = store.read()
        record = voices.get(voice_id)
        if record is None and voice_id not in CATALOG:
            raise VoiceError(ErrorCode.VOICE_UNKNOWN)
        if record is not None and record.state == "loaded":
            asset = Path(record.asset_path or "")
            if asset.is_file() and _sha256(asset) == record.asset_sha256:
                return _loaded_view(record, engines[record.engine_id])
        if record is None:
            record = _registered_from_catalog(voice_id)
        engine = engines.get(record.engine_id)
        cloning = record.variety == "wav"
        if engine is None or (cloning and not engine.cloning_capable):
            engine = _provision_engine(record.language, cloning=cloning, root=root)
            engines[engine.id] = engine
        loaded = _materialize(record, engine, root)
        voices[voice_id] = loaded
        store.write(engines, voices)
        return _loaded_view(loaded, engine)


def _registered_view(record: VoiceRecord) -> Voice:
    return Voice(
        id=record.id,
        name=record.name,
        enabled=record.enabled,
        provenance=record.provenance,
        license_id=record.license_id,
        commercial_use_allowed=record.commercial_use_allowed,
        language=record.language,
        variety=record.variety,
        state="registered",
    )


def _unloaded(record: VoiceRecord) -> VoiceRecord:
    return VoiceRecord(
        id=record.id,
        variety=record.variety,
        state="registered",
        name=record.name,
        enabled=record.enabled,
        language=record.language,
        engine_id=record.engine_id,
        provenance=record.provenance,
        license_id=record.license_id,
        commercial_use_allowed=record.commercial_use_allowed,
        voice_rights=record.voice_rights,
        source_path=record.source_path,
        source_sha256=record.source_sha256,
    )


def _prune_engines(
    engines: dict[str, EngineRecord], voices: dict[str, VoiceRecord]
) -> None:
    """Delete engines no loaded voice references, reclaiming their files.

    An engine is roughly 225 MB against roughly 6.5 MB per embedding, so this
    is where unloading actually reclaims disk.
    """
    referenced = {
        record.engine_id for record in voices.values() if record.state == "loaded"
    }
    for engine_id in list(engines):
        if engine_id in referenced:
            continue
        shutil.rmtree(Path(engines[engine_id].model_root), ignore_errors=True)
        del engines[engine_id]


def _discard_asset(record: VoiceRecord) -> None:
    if record.asset_path is not None:
        Path(record.asset_path).unlink(missing_ok=True)


def unload_voice(voice_id: str, *, manifest: Path | None = None) -> Voice:
    """Delete a voice's asset, keep its rights metadata, and prune its engine."""
    store = _store(manifest)
    with store.lock():
        engines, voices = store.read()
        record = voices.get(voice_id)
        if record is None:
            if voice_id in CATALOG:
                return catalog_voice(voice_id)
            raise VoiceError(ErrorCode.VOICE_UNKNOWN)
        _discard_asset(record)
        reverted = _unloaded(record)
        voices[voice_id] = reverted
        _prune_engines(engines, voices)
        store.write(engines, voices)
        return _registered_view(reverted)


def remove_voice(voice_id: str, *, manifest: Path | None = None) -> None:
    """Delete a voice entry entirely, including hand-entered rights metadata.

    Equivalent to `unload_voice` for a built-in, whose catalog registration
    cannot be deleted.
    """
    store = _store(manifest)
    with store.lock():
        engines, voices = store.read()
        record = voices.pop(voice_id, None)
        if record is None:
            if voice_id in CATALOG:
                return
            raise VoiceError(ErrorCode.VOICE_UNKNOWN)
        _discard_asset(record)
        _prune_engines(engines, voices)
        store.write(engines, voices)


def list_voices(*, manifest: Path | None = None) -> tuple[Voice, ...]:
    """Return every known voice, unioning the catalog with the manifest.

    No network and no hashing: asset presence is a stat. This is the only
    enumeration primitive, so multi-voice work composes over it.
    """
    engines, voices = _store(manifest).read()
    known: dict[str, Voice] = {
        voice_id: catalog_voice(voice_id) for voice_id in CATALOG
    }
    for voice_id, record in voices.items():
        if record.state != "loaded":
            known[voice_id] = _registered_view(record)
            continue
        asset = Path(record.asset_path or "")
        engine = engines.get(record.engine_id)
        if not asset.is_file() or engine is None:
            known[voice_id] = dataclasses.replace(
                _registered_view(record), state="missing"
            )
            continue
        known[voice_id] = _loaded_view(record, engine)
    return tuple(known[key] for key in sorted(known))
