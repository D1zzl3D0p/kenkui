"""Read, merge, and atomically write the Kenkui-managed production manifest."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from kenkui._tts.production import default_cache_root
from kenkui.errors import ErrorCode, ModelError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from kenkui.voices.types import VoiceVariety

MANIFEST_SCHEMA_VERSION: Final = "kenkui-pocket-production-v2"

_ENGINE_KEYS: Final = (
    "language",
    "model_root",
    "config_path",
    "model_revision",
    "package_version",
    "sample_rate_hz",
    "device",
    "timeout_seconds",
    "cloning_capable",
)
_VOICE_COMMON_KEYS: Final = (
    "variety",
    "state",
    "name",
    "enabled",
    "language",
    "engine_id",
    "provenance",
    "license_id",
    "commercial_use_allowed",
    "voice_rights",
)


@dataclass(frozen=True, slots=True)
class FileRecord:
    """One verified engine asset file."""

    relative_path: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class EngineRecord:
    """One provisioned per-language engine."""

    id: str
    language: str
    model_root: str
    config_path: str
    model_revision: str
    package_version: str
    files: tuple[FileRecord, ...]
    sample_rate_hz: int
    device: str
    timeout_seconds: float
    cloning_capable: bool


@dataclass(frozen=True, slots=True)
class VoiceRecord:
    """One registered or loaded voice."""

    id: str
    variety: VoiceVariety
    state: str
    name: str
    enabled: bool
    language: str
    engine_id: str
    provenance: str
    license_id: str
    commercial_use_allowed: bool
    voice_rights: str
    source_path: str | None = None
    source_sha256: str | None = None
    asset_path: str | None = None
    asset_sha256: str | None = None
    compatible_model_revisions: tuple[str, ...] = ()


def default_manifest_path() -> Path:
    """Return the Kenkui-managed manifest location for this operating system."""
    return default_cache_root() / "manifest.json"


class ManifestStore:
    """Owner-private manifest persistence with atomic replacement."""

    __slots__ = ("path",)

    def __init__(self, path: Path) -> None:
        """Bind the store to one manifest path."""
        self.path = path

    @contextmanager
    def lock(self) -> Iterator[None]:
        """Serialize provisioning against other processes on this machine."""
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(".lock")
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def read(self) -> tuple[dict[str, EngineRecord], dict[str, VoiceRecord]]:
        """Return stored engines and voices, or empty mappings when absent.

        A corrupt, truncated, or foreign-schema manifest raises a stable public
        error rather than leaking a JSON or key error from the internals.
        """
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}, {}
        except OSError:
            raise ModelError(ErrorCode.POCKET_MODEL_INVALID) from None
        try:
            payload = json.loads(raw)
            _require_schema(payload)
            engines = {
                key: _engine_from(key, value)
                for key, value in payload.get("engines", {}).items()
            }
            voices = {
                key: _voice_from(key, value)
                for key, value in payload.get("voices", {}).items()
            }
        except ModelError:
            raise
        except (AttributeError, KeyError, TypeError, ValueError):
            raise ModelError(ErrorCode.POCKET_MODEL_INVALID) from None
        return engines, voices

    def write(
        self,
        engines: dict[str, EngineRecord],
        voices: dict[str, VoiceRecord],
    ) -> None:
        """Replace the manifest atomically with owner-private permissions."""
        payload = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "engines": {key: _engine_to(value) for key, value in engines.items()},
            "voices": {key: _voice_to(value) for key, value in voices.items()},
        }
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path.parent.chmod(0o700)
        descriptor, temporary = tempfile.mkstemp(
            dir=self.path.parent, prefix=".manifest-", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            Path(temporary).chmod(0o600)
            Path(temporary).replace(self.path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise


def _require_schema(payload: object) -> None:
    """Reject anything that is not a v2 manifest object."""
    if type(payload) is not dict or payload.get("schema_version") != (
        MANIFEST_SCHEMA_VERSION
    ):
        raise ModelError(ErrorCode.POCKET_MODEL_INVALID)


def _engine_to(record: EngineRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {key: getattr(record, key) for key in _ENGINE_KEYS}
    payload["files"] = [
        {
            "relative_path": item.relative_path,
            "size": item.size,
            "sha256": item.sha256,
        }
        for item in record.files
    ]
    return payload


def _engine_from(engine_id: str, payload: dict[str, Any]) -> EngineRecord:
    return EngineRecord(
        id=engine_id,
        files=tuple(
            FileRecord(item["relative_path"], item["size"], item["sha256"])
            for item in payload["files"]
        ),
        **{key: payload[key] for key in _ENGINE_KEYS},
    )


def _voice_to(record: VoiceRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {key: getattr(record, key) for key in _VOICE_COMMON_KEYS}
    # Source keys are present exactly when the voice came from a local file,
    # for both local varieties and in both states. The reader's exact-key-set
    # check mirrors this rule; the two must agree or a voice that loaded
    # successfully becomes unrenderable.
    if record.source_path is not None:
        payload["source_path"] = record.source_path
        payload["source_sha256"] = record.source_sha256
    if record.state == "loaded":
        payload["asset_path"] = record.asset_path
        payload["asset_sha256"] = record.asset_sha256
        payload["compatible_model_revisions"] = list(record.compatible_model_revisions)
    return payload


def _voice_from(voice_id: str, payload: dict[str, Any]) -> VoiceRecord:
    return VoiceRecord(
        id=voice_id,
        source_path=payload.get("source_path"),
        source_sha256=payload.get("source_sha256"),
        asset_path=payload.get("asset_path"),
        asset_sha256=payload.get("asset_sha256"),
        compatible_model_revisions=tuple(payload.get("compatible_model_revisions", ())),
        **{key: payload[key] for key in _VOICE_COMMON_KEYS},
    )
