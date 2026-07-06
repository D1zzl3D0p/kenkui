"""Manifest-driven voice catalog.

The catalog is the single source of truth for usable voices.  Raw prompts and
filename metadata are import inputs only; render-time voice resolution uses a
canonical ``voice_id``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from .voice_compiler import is_legacy_audio_prompt_asset

logger = logging.getLogger(__name__)

VoiceOrigin = Literal["pocket_tts_builtin", "kenkui_compiled", "custom_compiled"]
VoiceAssetKind = Literal["pocket_tts_builtin", "safetensors"]
VoiceStatus = Literal["available", "missing", "downloadable"]

DEFAULT_VOICE_PACK_REPO = "D1zzl3D0p/kenkui-voices"
DEFAULT_VOICE_PACK_REVISION = "main"
VOICE_PACK_FORMAT_VERSION = 2
MANIFEST_FILENAMES = ("manifest.json", "voice_manifest.json")
CUSTOM_MANIFEST_FILENAME = "custom_manifest.json"
PREVIEW_TEXT = (
    "The rain in Spain stays mainly in the plain. "
    "How wonderful it is to simply speak and be heard."
)

_VOICE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")

_BUILTIN_VOICE_DATA: dict[str, dict[str, str | None]] = {
    "alba": {"gender": "Male", "accent": "American", "dataset": "Alba-Mackenna", "speaker_id": "casual"},
    "marius": {"gender": "Male", "accent": "American", "dataset": "Voice Donation", "speaker_id": None},
    "javert": {"gender": "Male", "accent": "American", "dataset": "Voice Donation", "speaker_id": None},
    "cosette": {
        "gender": "Female",
        "accent": "American",
        "dataset": "Expresso",
        "speaker_id": "ex04-ex02_confused_001_channel1_499s",
    },
    "jean": {"gender": "Male", "accent": "Southern", "dataset": "EARS", "speaker_id": "P010"},
    "fantine": {"gender": "Female", "accent": "British", "dataset": "VCTK", "speaker_id": "P244"},
    "eponine": {"gender": "Female", "accent": "British", "dataset": "VCTK", "speaker_id": "P262"},
    "azelma": {"gender": "Female", "accent": "American", "dataset": "VCTK", "speaker_id": "P303"},
    "anna": {"gender": "Female", "accent": "Scottish", "dataset": "VCTK", "speaker_id": "P228"},
    "vera": {"gender": "Female", "accent": "English", "dataset": "VCTK", "speaker_id": "P229"},
    "charles": {"gender": "Male", "accent": "English", "dataset": "VCTK", "speaker_id": "P254"},
    "paul": {"gender": "Male", "accent": "British", "dataset": "VCTK", "speaker_id": "P259"},
    "george": {"gender": "Male", "accent": "American", "dataset": "VCTK", "speaker_id": "P315"},
    "mary": {"gender": "Female", "accent": "American", "dataset": "VCTK", "speaker_id": "P333"},
    "jane": {"gender": "Female", "accent": "American", "dataset": "VCTK", "speaker_id": "P339"},
    "michael": {"gender": "Male", "accent": "American", "dataset": "VCTK", "speaker_id": "P360"},
    "eve": {"gender": "Female", "accent": "American", "dataset": "VCTK", "speaker_id": "P361"},
    "bill_boerst": {"gender": "Male", "accent": None, "dataset": "Voice Zero", "speaker_id": None},
    "caro_davy": {"gender": "Female", "accent": None, "dataset": "Voice Zero", "speaker_id": None},
    "peter_yearsley": {"gender": "Male", "accent": None, "dataset": "Voice Zero", "speaker_id": None},
    "stuart_bell": {"gender": "Male", "accent": None, "dataset": "Voice Zero", "speaker_id": None},
}

BUILTIN_VOICE_NAMES: list[str] = list(_BUILTIN_VOICE_DATA)


class VoiceCatalogError(ValueError):
    """Raised when a voice manifest is invalid."""


@dataclass(frozen=True)
class PreviewInfo:
    text: str = PREVIEW_TEXT
    path: str | None = None
    url: str | None = None
    sha256: str | None = None
    duration_ms: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PreviewInfo:
        if not data:
            return cls()
        return cls(
            text=str(data.get("text") or PREVIEW_TEXT),
            path=data.get("path"),
            url=data.get("url"),
            sha256=data.get("sha256"),
            duration_ms=data.get("duration_ms"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True)
class VoiceCatalogEntry:
    voice_id: str
    display_name: str
    origin: VoiceOrigin
    asset_kind: VoiceAssetKind
    gender: str
    pool_enabled: bool
    path: Path | None = None
    status: VoiceStatus = "available"
    preview: PreviewInfo = field(default_factory=PreviewInfo)
    accent: str | None = None
    dataset: str | None = None
    speaker_id: str | None = None
    license: str | None = None
    tags: tuple[str, ...] = ()
    notes: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None

    @property
    def description(self) -> str:
        parts = [self.gender]
        if self.accent:
            parts.append(self.accent)
        if self.dataset:
            parts.append(self.dataset)
        parts.append(self.origin.replace("_", " "))
        return " · ".join(parts)

    @property
    def display_label(self) -> str:
        return f"{self.display_name:<20} {self.description}"

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, base_dir: Path | None = None) -> VoiceCatalogEntry:
        voice_id = str(data.get("voice_id") or "").strip()
        display_name = str(data.get("display_name") or "").strip()
        origin = data.get("origin")
        asset_kind = data.get("asset_kind")
        gender = str(data.get("gender") or "").strip()
        if not voice_id or not _VOICE_ID_RE.match(voice_id):
            raise VoiceCatalogError(f"Invalid voice_id: {voice_id!r}")
        if not display_name:
            raise VoiceCatalogError(f"Voice {voice_id!r} is missing display_name")
        if origin not in ("pocket_tts_builtin", "kenkui_compiled", "custom_compiled"):
            raise VoiceCatalogError(f"Voice {voice_id!r} has invalid origin {origin!r}")
        if asset_kind not in ("pocket_tts_builtin", "safetensors"):
            raise VoiceCatalogError(f"Voice {voice_id!r} has invalid asset_kind {asset_kind!r}")
        if gender not in ("Male", "Female", "Nonbinary", "Unknown"):
            raise VoiceCatalogError(f"Voice {voice_id!r} has invalid gender {gender!r}")

        raw_path = data.get("path")
        path = Path(raw_path) if raw_path else None
        if path is not None and not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        status = str(data.get("status") or "available")
        if status not in ("available", "missing", "downloadable"):
            raise VoiceCatalogError(f"Voice {voice_id!r} has invalid status {status!r}")
        if asset_kind == "safetensors" and path is None and status != "downloadable":
            raise VoiceCatalogError(f"Voice {voice_id!r} is missing a compiled asset path")
        if origin == "pocket_tts_builtin" and asset_kind != "pocket_tts_builtin":
            raise VoiceCatalogError(f"Voice {voice_id!r} has inconsistent built-in asset_kind")
        if origin != "pocket_tts_builtin" and asset_kind != "safetensors":
            raise VoiceCatalogError(f"Voice {voice_id!r} must use a safetensors asset")

        if path is not None and asset_kind == "safetensors" and path.exists():
            if is_legacy_audio_prompt_asset(path):
                status = "missing"
        elif path is not None and asset_kind == "safetensors":
            status = "missing"

        return cls(
            voice_id=voice_id,
            display_name=display_name,
            origin=origin,
            asset_kind=asset_kind,
            gender=gender,
            pool_enabled=bool(data.get("pool_enabled", origin != "custom_compiled")),
            path=path,
            status=status,  # type: ignore[arg-type]
            preview=PreviewInfo.from_dict(data.get("preview")),
            accent=data.get("accent"),
            dataset=data.get("dataset"),
            speaker_id=data.get("speaker_id"),
            license=data.get("license"),
            tags=tuple(data.get("tags") or ()),
            notes=data.get("notes"),
            sha256=data.get("sha256"),
            size_bytes=data.get("size_bytes"),
        )

    def to_manifest_dict(self, *, base_dir: Path | None = None) -> dict[str, Any]:
        path = self.path
        path_value: str | None = None
        if path is not None:
            try:
                path_value = str(path.relative_to(base_dir)) if base_dir else str(path)
            except ValueError:
                path_value = str(path)
        data: dict[str, Any] = {
            "voice_id": self.voice_id,
            "display_name": self.display_name,
            "origin": self.origin,
            "asset_kind": self.asset_kind,
            "gender": self.gender,
            "pool_enabled": self.pool_enabled,
            "status": self.status,
        }
        if path_value:
            data["path"] = path_value
        for key in ("accent", "dataset", "speaker_id", "license", "notes", "sha256", "size_bytes"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        if self.tags:
            data["tags"] = list(self.tags)
        preview = self.preview.to_dict()
        if preview:
            data["preview"] = preview
        return data


def _xdg_data_home() -> Path:
    raw = os.environ.get("XDG_DATA_HOME")
    return Path(raw) if raw else Path.home() / ".local" / "share"


def voice_data_dir() -> Path:
    return _xdg_data_home() / "kenkui" / "voices"


def compiled_voices_dir() -> Path:
    return voice_data_dir() / "compiled"


def bundled_voice_manifest_path() -> Path:
    return Path(__file__).resolve().parent / "voices" / "manifest.json"


def custom_voices_dir() -> Path:
    return voice_data_dir() / "custom"


def preview_cache_dir() -> Path:
    raw = os.environ.get("XDG_CACHE_HOME")
    root = Path(raw) if raw else Path.home() / ".cache"
    return root / "kenkui" / "previews"


def builtin_catalog_entries() -> list[VoiceCatalogEntry]:
    entries: list[VoiceCatalogEntry] = []
    for voice_id, data in _BUILTIN_VOICE_DATA.items():
        entries.append(
            VoiceCatalogEntry(
                voice_id=voice_id,
                display_name=voice_id.replace("_", " ").title(),
                origin="pocket_tts_builtin",
                asset_kind="pocket_tts_builtin",
                gender=str(data["gender"] or "Unknown"),
                pool_enabled=True,
                accent=data.get("accent"),
                dataset=data.get("dataset"),
                speaker_id=data.get("speaker_id"),
            )
        )
    return entries


def load_manifest(path: Path) -> list[VoiceCatalogEntry]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VoiceCatalogError(f"Invalid JSON in {path}") from exc
    voices = raw.get("voices") if isinstance(raw, dict) else raw
    if not isinstance(voices, list):
        raise VoiceCatalogError(f"{path} must contain a voices list")
    entries = [VoiceCatalogEntry.from_dict(item, base_dir=path.parent) for item in voices]
    _validate_unique_voice_ids(entries, source=str(path))
    return entries


def _load_manifest_version(path: Path) -> int | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    version = raw.get("voice_pack_format_version")
    if version is None:
        return None
    try:
        return int(version)
    except (TypeError, ValueError):
        return None


def voice_pack_manifest_is_current(path: Path) -> bool:
    version = _load_manifest_version(path)
    return version is not None and version >= VOICE_PACK_FORMAT_VERSION


def write_manifest(path: Path, entries: list[VoiceCatalogEntry]) -> Path:
    _validate_unique_voice_ids(entries, source=str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": 1,
        "voices": [entry.to_manifest_dict(base_dir=path.parent) for entry in entries],
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_manifest(path: Path) -> list[VoiceCatalogEntry]:
    return load_manifest(path)


def verify_manifest_assets(entries: list[VoiceCatalogEntry]) -> list[str]:
    """Verify local manifest assets against declared size and SHA-256 metadata.

    Files that fail verification are deleted so they will be treated as missing
    on the next catalog load. Returns the voice_ids of any removed files.
    """
    failed: list[str] = []
    for entry in entries:
        if entry.path is not None and entry.path.exists():
            if entry.size_bytes is not None and entry.path.stat().st_size != entry.size_bytes:
                logger.warning(
                    "Voice %r asset size does not match manifest; removing",
                    entry.voice_id,
                )
                entry.path.unlink(missing_ok=True)
                failed.append(entry.voice_id)
                continue
            if entry.sha256 is not None and _hash_file(entry.path) != entry.sha256:
                logger.warning(
                    "Voice %r asset hash does not match manifest; removing",
                    entry.voice_id,
                )
                entry.path.unlink(missing_ok=True)
                failed.append(entry.voice_id)
                continue
        if entry.preview.path:
            preview_path = Path(entry.preview.path)
            if (
                preview_path.exists()
                and entry.preview.sha256 is not None
                and _hash_file(preview_path) != entry.preview.sha256
            ):
                logger.warning(
                    "Voice %r preview hash does not match manifest; removing preview",
                    entry.voice_id,
                )
                preview_path.unlink(missing_ok=True)
    return failed


def _validate_unique_voice_ids(entries: list[VoiceCatalogEntry], *, source: str) -> None:
    seen: set[str] = set()
    for entry in entries:
        if entry.voice_id in seen:
            raise VoiceCatalogError(f"Duplicate voice_id {entry.voice_id!r} in {source}")
        seen.add(entry.voice_id)


def _manifest_paths(root: Path) -> list[Path]:
    return [root / name for name in MANIFEST_FILENAMES]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VoiceCatalog:
    """Lazy manifest-backed catalog.

    The ``voices`` property returns an immutable ``tuple`` snapshot loaded once
    on first access and cached until :meth:`invalidate` is called.  All
    :class:`VoiceCatalogEntry` objects are ``frozen=True`` dataclasses, so the
    entire structure is deeply immutable between invalidations.
    """

    def __init__(self, *, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or voice_data_dir()
        self._voices: tuple[VoiceCatalogEntry, ...] | None = None

    @property
    def voices(self) -> tuple[VoiceCatalogEntry, ...]:
        """Return an immutable snapshot of all catalog entries."""
        if self._voices is None:
            self._voices = self._load()
        return self._voices

    def _load(self) -> tuple[VoiceCatalogEntry, ...]:
        entries = builtin_catalog_entries()
        for path in _manifest_paths(self.data_dir):
            if path.exists():
                version = _load_manifest_version(path)
                if version is None or version < VOICE_PACK_FORMAT_VERSION:
                    logger.warning(
                        "Ignoring stale voice pack manifest %s (format %s < %s)",
                        path,
                        version if version is not None else "missing",
                        VOICE_PACK_FORMAT_VERSION,
                    )
                    continue
                entries = self._merge_entries(entries, load_manifest(path))
                break
        else:
            packaged_manifest = bundled_voice_manifest_path()
            if packaged_manifest.exists():
                version = _load_manifest_version(packaged_manifest)
                if version is not None and version >= VOICE_PACK_FORMAT_VERSION:
                    entries = self._merge_entries(entries, load_manifest(packaged_manifest))

        custom_manifest = self.custom_manifest_path
        if custom_manifest.exists():
            entries = self._merge_entries(entries, load_manifest(custom_manifest))

        _validate_unique_voice_ids(entries, source="voice catalog")
        return tuple(entries)

    @staticmethod
    def _merge_entries(
        base: list[VoiceCatalogEntry],
        overlay: list[VoiceCatalogEntry],
    ) -> list[VoiceCatalogEntry]:
        merged = list(base)
        index = {entry.voice_id: i for i, entry in enumerate(merged)}
        for entry in overlay:
            if entry.voice_id in index:
                merged[index[entry.voice_id]] = entry
            else:
                index[entry.voice_id] = len(merged)
                merged.append(entry)
        return merged

    @property
    def custom_manifest_path(self) -> Path:
        return self.data_dir / "custom" / CUSTOM_MANIFEST_FILENAME

    def resolve(self, voice_id: str) -> VoiceCatalogEntry | None:
        for voice in self.voices:
            if voice.voice_id == voice_id:
                return voice
        return None

    def filter(
        self,
        *,
        gender: str | None = None,
        accent: str | None = None,
        dataset: str | None = None,
        origin: str | None = None,
        asset_kind: str | None = None,
        pool_enabled: bool | None = None,
        status: str | None = None,
    ) -> list[VoiceCatalogEntry]:
        def matches(v: VoiceCatalogEntry) -> bool:
            if origin and v.origin != origin:
                return False
            if asset_kind and v.asset_kind != asset_kind:
                return False
            if status and v.status != status:
                return False
            if pool_enabled is not None and v.pool_enabled != pool_enabled:
                return False
            if gender and v.gender.lower() != gender.lower():
                return False
            if dataset and (v.dataset or "").lower() != dataset.lower():
                return False
            if accent and (v.accent or "").lower() != accent.lower():
                return False
            return True

        return [v for v in self.voices if matches(v)]

    def pool(self) -> list[VoiceCatalogEntry]:
        return [
            v for v in self.voices
            if v.pool_enabled and v.status == "available"
        ]

    def set_pool_enabled(self, voice_id: str, enabled: bool) -> VoiceCatalogEntry:
        existing = self.resolve(voice_id)
        if existing is None:
            raise KeyError(f"Unknown voice_id: {voice_id}")
        custom_manifest = self.custom_manifest_path
        custom_entries = load_manifest(custom_manifest) if custom_manifest.exists() else []
        entries_by_id = {entry.voice_id: entry for entry in custom_entries}
        entries_by_id[voice_id] = replace(existing, pool_enabled=enabled)
        write_manifest(custom_manifest, list(entries_by_id.values()))
        self.invalidate()
        updated = self.resolve(voice_id)
        if updated is None:
            raise RuntimeError(f"Failed to reload voice_id: {voice_id}")
        return updated

    def add_custom_voice(
        self,
        *,
        voice_id: str,
        display_name: str,
        gender: str,
        compiled_path: Path,
        pool_enabled: bool = False,
        accent: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        preview_path: Path | None = None,
    ) -> VoiceCatalogEntry:
        destination = self.data_dir / "custom" / f"{voice_id}.safetensors"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if compiled_path.resolve() != destination.resolve():
            destination.write_bytes(compiled_path.read_bytes())
        preview = PreviewInfo()
        if preview_path:
            cached_preview = preview_cache_dir() / f"{voice_id}.wav"
            cached_preview.parent.mkdir(parents=True, exist_ok=True)
            if preview_path.resolve() != cached_preview.resolve():
                cached_preview.write_bytes(preview_path.read_bytes())
            preview = PreviewInfo(path=str(cached_preview), sha256=_hash_file(cached_preview))
        entry = VoiceCatalogEntry.from_dict(
            {
                "voice_id": voice_id,
                "display_name": display_name,
                "origin": "custom_compiled",
                "asset_kind": "safetensors",
                "gender": gender,
                "pool_enabled": pool_enabled,
                "path": str(destination),
                "preview": preview.to_dict(),
                "accent": accent,
                "tags": tags or [],
                "notes": notes,
                "sha256": _hash_file(destination),
                "size_bytes": destination.stat().st_size,
            }
        )
        custom_manifest = self.custom_manifest_path
        entries = load_manifest(custom_manifest) if custom_manifest.exists() else []
        entries = [e for e in entries if e.voice_id != voice_id] + [entry]
        write_manifest(custom_manifest, entries)
        self.invalidate()
        return entry

    def invalidate(self) -> None:
        self._voices = None


class _CatalogRef:
    """Container for the module-level catalog singleton.

    Encapsulates mutation so that ``get_catalog()`` never writes a bare module
    global.  :func:`_reset_catalog` is the explicit injection seam for tests.
    """

    def __init__(self) -> None:
        self._instance: VoiceCatalog | None = None

    def get(self) -> VoiceCatalog:
        if self._instance is None:
            self._instance = VoiceCatalog()
        return self._instance

    def reset(self) -> None:
        self._instance = None


_CATALOG_REF = _CatalogRef()


def get_catalog() -> VoiceCatalog:
    return _CATALOG_REF.get()


def _reset_catalog() -> None:
    """Reset the module-level catalog singleton.

    Intended for tests and tooling that need a clean slate between runs.
    Production code should call :meth:`VoiceCatalog.invalidate` instead.
    """
    _CATALOG_REF.reset()


def get_registry() -> VoiceCatalog:
    """Return the shared catalog.

    Kept as an internal convenience for modules that used the old registry name.
    It still exposes voice_id-only resolution.
    """
    return get_catalog()


__all__ = [
    "BUILTIN_VOICE_NAMES",
    "DEFAULT_VOICE_PACK_REPO",
    "DEFAULT_VOICE_PACK_REVISION",
    "VOICE_PACK_FORMAT_VERSION",
    "PREVIEW_TEXT",
    "PreviewInfo",
    "VoiceCatalog",
    "VoiceCatalogEntry",
    "VoiceCatalogError",
    "builtin_catalog_entries",
    "compiled_voices_dir",
    "custom_voices_dir",
    "bundled_voice_manifest_path",
    "get_catalog",
    "get_registry",
    "_reset_catalog",
    "load_manifest",
    "preview_cache_dir",
    "validate_manifest",
    "voice_pack_manifest_is_current",
    "verify_manifest_assets",
    "voice_data_dir",
    "write_manifest",
]
