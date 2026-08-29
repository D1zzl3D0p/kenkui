"""Fail-closed, spawned-worker-only Pocket-TTS 2.1.0 adapter."""
# ruff: noqa: ANN401, B009, B010, BLE001, C901, E501, PLC0415, PLR0912, PLR0913, PLR0915, PLW0603, TRY004, TRY300, TRY301

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import math
import os
import re
import shutil
import stat
import struct
import sys
import tempfile
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Final

from kenkui._tts.protocols import SynthesisTask, SynthesizedAudio
from kenkui.errors import ErrorCode, ModelError, RenderError, VoiceError

if TYPE_CHECKING:
    from kenkui._domain.planning import VoicePlan

POCKET_PACKAGE_VERSION: Final = "2.1.0"
MAX_MODEL_FILE_BYTES: Final = 2 * 1024 * 1024 * 1024
MAX_MODEL_TOTAL_BYTES: Final = 4 * 1024 * 1024 * 1024
MAX_CONFIG_BYTES: Final = 1024 * 1024
MAX_VOICE_BYTES: Final = 64 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES: Final = 16 * 1024 * 1024
MAX_MANIFEST_ENTRIES: Final = 4096
MAX_DIRECTORY_ENTRIES: Final = 8192
MAX_RELATIVE_PATH_DEPTH: Final = 32
MAX_RELATIVE_NAME_LENGTH: Final = 255
MAX_OUTPUT_SAMPLES: Final = 32 * 1024 * 1024
# Neural vocoders routinely overshoot full scale by a fraction of a percent.
# Hard-limiting that back to the int16 domain is ordinary audio practice, but a
# sample far outside the band is corruption rather than overshoot, so the band
# keeps the fail-closed signal that a bare clamp would discard.
SAMPLE_TOLERANCE: Final = 2.0
MAX_TIMEOUT_SECONDS: Final = 3600.0
_TENSOR_CHUNK_SAMPLES: Final = 64 * 1024
_HASH_CHUNK_BYTES: Final = 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REMOTE_MARKERS = (b"http://", b"https://", b"hf://")
_WORKER_TOKEN = object()
_worker_marker: object | None = None
_audit_installed = False


@dataclass(frozen=True, slots=True)
class PocketManifestFile:
    relative_path: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class VoiceAsset:
    """One renderable speaker embedding and the rights recorded against it."""

    path: str
    sha256: str
    variety: str
    provenance: str
    license_id: str
    rights: str
    commercial_use_allowed: bool

    def semantic_material(self) -> dict[str, object]:
        """Return semantic identity without machine-specific absolute paths."""
        return {
            "commercial_use_allowed": self.commercial_use_allowed,
            "license_id": self.license_id,
            "provenance": self.provenance,
            "rights": self.rights,
            "sha256": self.sha256,
            "variety": self.variety,
        }


@dataclass(frozen=True, slots=True)
class PocketEngineConfig:
    model_root: str
    config_path: str
    model_revision: str
    package_version: str
    files: tuple[PocketManifestFile, ...]
    voices: tuple[VoiceAsset, ...]
    cloning_capable: bool
    sample_rate_hz: int
    device: str = "cpu"
    timeout_seconds: float = 300.0

    def voice_by_sha(self, sha256: str) -> VoiceAsset:
        """Resolve one voice asset by content digest."""
        for voice in self.voices:
            if voice.sha256 == sha256:
                return voice
        raise KeyError(sha256)

    def semantic_material(self) -> dict[str, object]:
        """Return semantic identity without machine-specific absolute paths."""
        selected = _selected_config_identity(self)
        return {
            "config_manifest_path": selected,
            "device": self.device,
            "files": tuple(
                (item.relative_path, item.size, item.sha256) for item in self.files
            ),
            "model_revision": self.model_revision,
            "package_version": self.package_version,
            "sample_rate_hz": self.sample_rate_hz,
            # Sorted by digest so cast ordering cannot change the cache key.
            "voices": tuple(
                voice.semantic_material()
                for voice in sorted(self.voices, key=lambda item: item.sha256)
            ),
        }


def _model_failure() -> ModelError:
    return ModelError(ErrorCode.POCKET_MODEL_INVALID)


def _voice_failure() -> VoiceError:
    return VoiceError(ErrorCode.POCKET_VOICE_INVALID)


def _nonempty_string(value: object, *, voice: bool = False) -> str:
    if type(value) is not str or not value.strip() or len(value) > 4096:
        raise _voice_failure() if voice else _model_failure()
    return value


def _safe_relative(value: object) -> PurePosixPath:
    if type(value) is not str:
        raise _model_failure()
    relative = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or relative.is_absolute()
        or relative.as_posix() != value
        or any(part in ("", ".", "..") for part in relative.parts)
        or len(relative.parts) > MAX_RELATIVE_PATH_DEPTH
        or any(len(part) > MAX_RELATIVE_NAME_LENGTH for part in relative.parts)
    ):
        raise _model_failure()
    return relative


def _validate_fields(config: object) -> PocketEngineConfig:
    if type(config) is not PocketEngineConfig:
        raise _model_failure()
    _nonempty_string(config.model_root)
    _nonempty_string(config.config_path)
    _nonempty_string(config.model_revision)
    if type(config.package_version) is not str:
        raise _model_failure()
    if (
        type(config.files) is not tuple
        or not config.files
        or len(config.files) > MAX_MANIFEST_ENTRIES
    ):
        raise _model_failure()
    total = 0
    for item in config.files:
        if type(item) is not PocketManifestFile:
            raise _model_failure()
        _safe_relative(item.relative_path)
        if type(item.size) is not int or not 1 <= item.size <= MAX_MODEL_FILE_BYTES:
            raise _model_failure()
        if type(item.sha256) is not str or _SHA256.fullmatch(item.sha256) is None:
            raise _model_failure()
        total += item.size
        if total > MAX_MODEL_TOTAL_BYTES:
            raise _model_failure()
    if type(config.device) is not str or config.device != "cpu":
        raise _model_failure()
    if (
        type(config.sample_rate_hz) is not int
        or not 8_000 <= config.sample_rate_hz <= 192_000
    ):
        raise _model_failure()
    if (
        type(config.timeout_seconds) is not float
        or not math.isfinite(config.timeout_seconds)
        or not 0.0 < config.timeout_seconds <= MAX_TIMEOUT_SECONDS
    ):
        raise _model_failure()
    if type(config.cloning_capable) is not bool:
        raise _model_failure()
    if (
        type(config.voices) is not tuple
        or not config.voices
    ):
        raise _voice_failure()
    digests: set[str] = set()
    for voice in config.voices:
        if type(voice) is not VoiceAsset:
            raise _voice_failure()
        _nonempty_string(voice.path, voice=True)
        if type(voice.sha256) is not str or _SHA256.fullmatch(voice.sha256) is None:
            raise _voice_failure()
        # Two assets sharing a digest would make voice_by_sha ambiguous, and the
        # worker resolves its conditioning state by exactly that key.
        if voice.sha256 in digests:
            raise _voice_failure()
        digests.add(voice.sha256)
        _nonempty_string(voice.variety, voice=True)
        _nonempty_string(voice.provenance, voice=True)
        _nonempty_string(voice.license_id, voice=True)
        _nonempty_string(voice.rights, voice=True)
        if type(voice.commercial_use_allowed) is not bool:
            raise _voice_failure()
    return config


def _canonical_absolute(value: str, *, voice: bool = False) -> Path:
    failure = _voice_failure if voice else _model_failure
    try:
        path = Path(value)
        if not path.is_absolute() or path != path.resolve(strict=True):
            raise OSError
    except (OSError, RuntimeError, ValueError, TypeError):
        raise failure() from None
    return path


def _selected_config_identity(config: PocketEngineConfig) -> str:
    _validate_fields(config)
    root = _canonical_absolute(config.model_root)
    selected = _canonical_absolute(config.config_path)
    try:
        return selected.relative_to(root).as_posix()
    except ValueError:
        raise _model_failure() from None


def _safe_mode(info: os.stat_result, *, directory: bool = False) -> bool:
    return (
        info.st_uid == os.getuid()
        and not info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        and (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))
    )


def _validate_root(path: Path, *, voice: bool = False) -> None:
    failure = _voice_failure if voice else _model_failure
    try:
        info = path.lstat()
        if not _safe_mode(info, directory=True) or stat.S_ISLNK(info.st_mode):
            raise OSError
    except OSError:
        raise failure() from None


def _open_source(
    path: Path,
    expected_size: int,
    expected_hash: str,
    limit: int,
    *,
    voice: bool = False,
) -> tuple[int, os.stat_result]:
    failure = _voice_failure if voice else _model_failure
    descriptor = -1
    try:
        named = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        opened = os.fstat(descriptor)
        if (
            not _safe_mode(opened)
            or stat.S_ISLNK(named.st_mode)
            or opened.st_nlink != 1
            or opened.st_size != expected_size
            or not 1 <= opened.st_size <= limit
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise OSError
        if type(expected_hash) is not str or _SHA256.fullmatch(expected_hash) is None:
            raise OSError
        return descriptor, opened
    except (OSError, ValueError, TypeError):
        if descriptor >= 0:
            os.close(descriptor)
        raise failure() from None


def _verify_source(
    path: Path,
    expected_size: int,
    expected_hash: str,
    limit: int,
    *,
    voice: bool = False,
    capture_limit: int | None = None,
) -> bytes | None:
    """Hash from a fixed descriptor, retaining bytes only for bounded parsers."""
    failure = _voice_failure if voice else _model_failure
    descriptor, before = _open_source(
        path, expected_size, expected_hash, limit, voice=voice
    )
    try:
        if capture_limit is not None and not 1 <= expected_size <= capture_limit:
            raise OSError
        digest = hashlib.sha256()
        captured = bytearray() if capture_limit is not None else None
        remaining = expected_size
        while remaining:
            chunk = os.read(descriptor, min(_HASH_CHUNK_BYTES, remaining))
            if not chunk:
                raise OSError
            if captured is not None:
                captured.extend(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if (
            os.read(descriptor, 1)
            or digest.hexdigest() != expected_hash
            or (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
        ):
            raise OSError
        return bytes(captured) if captured is not None else None
    except OSError:
        raise failure() from None
    finally:
        os.close(descriptor)


def _inspect_yaml(data: bytes) -> None:
    if not 1 <= len(data) <= MAX_CONFIG_BYTES or any(
        marker in data.lower() for marker in _REMOTE_MARKERS
    ):
        raise _model_failure()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise _model_failure() from None
    for raw in text.splitlines():
        value = raw.split("#", 1)[0]
        if re.search(
            r"(?:^|[\s\[{:,'\"])(?:/|~[/\\]|\.\.[/\\]|file:|ftp:|[a-zA-Z]:[/\\])",
            value,
            re.IGNORECASE,
        ):
            raise _model_failure()
        if ":" not in value:
            continue
        scalar = value.split(":", 1)[1].strip().strip("'\"")
        if not scalar or scalar[0] in "[{":
            continue
        if any(part == ".." for part in PurePosixPath(scalar).parts):
            raise _model_failure()


def _validate_safetensors(data: bytes) -> None:
    """Validate a bounded safetensors header without importing torch."""
    prefix = 8
    if len(data) < prefix:
        raise _voice_failure()
    declared = int.from_bytes(data[:prefix], "little")
    if (
        declared <= 0
        or declared > MAX_SAFETENSORS_HEADER_BYTES
        or declared > len(data) - prefix
    ):
        raise _voice_failure()
    if not data[prefix : prefix + declared].lstrip().startswith(b"{"):
        raise _voice_failure()


def _scan_manifest(root: Path, expected: dict[str, PocketManifestFile]) -> None:
    """Traverse incrementally without following links or retaining an actual set."""
    entries = 0
    files = 0
    pending = [root]
    try:
        while pending:
            current = pending.pop()
            _validate_root(current)
            with os.scandir(current) as children:
                for entry in children:
                    entries += 1
                    if entries > MAX_DIRECTORY_ENTRIES:
                        raise OSError
                    relative = (current / entry.name).relative_to(root)
                    if (
                        len(entry.name) > MAX_RELATIVE_NAME_LENGTH
                        or len(relative.parts) > MAX_RELATIVE_PATH_DEPTH
                    ):
                        raise OSError
                    info = entry.stat(follow_symlinks=False)
                    if stat.S_ISLNK(info.st_mode):
                        raise OSError
                    if stat.S_ISDIR(info.st_mode):
                        if not _safe_mode(info, directory=True):
                            raise OSError
                        pending.append(Path(entry.path))
                    elif stat.S_ISREG(info.st_mode):
                        if relative.as_posix() not in expected:
                            raise OSError
                        files += 1
                    else:
                        raise OSError
    except (OSError, RuntimeError, ValueError):
        raise _model_failure() from None
    if files != len(expected):
        raise _model_failure()


def _manifest(
    config: PocketEngineConfig,
) -> tuple[Path, Path, dict[str, PocketManifestFile], dict[str, int]]:
    """Verify every declared byte, returning voice sizes keyed by digest."""
    config = _validate_fields(config)
    root = _canonical_absolute(config.model_root)
    selected = _canonical_absolute(config.config_path)
    _validate_root(root)
    if not selected.is_relative_to(root):
        raise _model_failure()
    expected: dict[str, PocketManifestFile] = {}
    paths: list[tuple[Path, PocketManifestFile]] = []
    selected_item: PocketManifestFile | None = None
    for item in config.files:
        if item.relative_path in expected:
            raise _model_failure()
        expected[item.relative_path] = item
        relative = _safe_relative(item.relative_path)
        path = root.joinpath(*relative.parts)
        try:
            resolved = path.resolve(strict=True)
            if resolved != path or not resolved.is_relative_to(root):
                raise OSError
            parent = path.parent
            while parent.is_relative_to(root):
                _validate_root(parent)
                if parent == root:
                    break
                parent = parent.parent
        except (OSError, RuntimeError, ValueError):
            raise _model_failure() from None
        paths.append((path, item))
        if path == selected:
            selected_item = item
    if selected_item is None or selected_item.size > MAX_CONFIG_BYTES:
        raise _model_failure()

    # Prove the bounded directory exactly matches the declaration before hashing.
    _scan_manifest(root, expected)
    config_bytes: bytes | None = None
    for path, item in paths:
        is_selected = path == selected
        data = _verify_source(
            path,
            item.size,
            item.sha256,
            MAX_CONFIG_BYTES if is_selected else MAX_MODEL_FILE_BYTES,
            capture_limit=MAX_CONFIG_BYTES if is_selected else None,
        )
        if is_selected:
            config_bytes = data
    if config_bytes is None:
        raise _model_failure()
    _inspect_yaml(config_bytes)

    sizes: dict[str, int] = {}
    for asset in config.voices:
        prompt = _canonical_absolute(asset.path, voice=True)
        _validate_root(prompt.parent, voice=True)
        if prompt.is_relative_to(root) or prompt.suffix.lower() != ".safetensors":
            raise _voice_failure()
        try:
            prompt_size = prompt.lstat().st_size
        except OSError:
            raise _voice_failure() from None
        asset_bytes = _verify_source(
            prompt,
            prompt_size,
            asset.sha256,
            MAX_VOICE_BYTES,
            voice=True,
            capture_limit=MAX_VOICE_BYTES,
        )
        if asset_bytes is None:
            raise _voice_failure()
        # Every renderable asset is a compiled embedding: WAV prompts are
        # compiled during provisioning, so the render path has one branch.
        _validate_safetensors(asset_bytes)
        sizes[asset.sha256] = prompt_size
    return root, selected, expected, sizes


def preflight_pocket(
    config: PocketEngineConfig,
    voice: VoicePlan | None = None,
    model_revision: str | None = None,
) -> None:
    """Validate every declaration and byte without importing inference dependencies."""
    config = _validate_fields(config)
    if config.package_version != POCKET_PACKAGE_VERSION:
        raise ModelError(ErrorCode.POCKET_VERSION_UNSUPPORTED)
    try:
        installed = importlib.metadata.version("pocket-tts")
    except importlib.metadata.PackageNotFoundError:
        raise ModelError(ErrorCode.POCKET_PACKAGE_MISSING) from None
    if installed != config.package_version:
        raise ModelError(ErrorCode.POCKET_VERSION_UNSUPPORTED)
    if model_revision is not None and (
        type(model_revision) is not str or model_revision != config.model_revision
    ):
        raise _model_failure()
    _manifest(config)
    if voice is None:
        return
    try:
        asset = config.voice_by_sha(voice.content_fingerprint)
    except KeyError:
        raise _voice_failure() from None
    if (
        voice.provenance != asset.provenance
        or voice.license_id != asset.license_id
        or voice.commercial_use_allowed is not asset.commercial_use_allowed
        or config.model_revision not in voice.compatible_model_revisions
    ):
        raise _voice_failure()


def _network_audit(event: str, _args: tuple[object, ...]) -> None:
    # Pocket workers require no sockets. Deny the complete socket audit namespace
    # so connectionless sends and future CPython socket operations fail closed too.
    if event.startswith("socket."):
        raise PermissionError


def _enter_spawned_worker(token: object) -> None:
    """Private scheduler handshake, called before any Pocket/dependency import."""
    global _audit_installed, _worker_marker
    if token is not _WORKER_TOKEN:
        raise RuntimeError
    _worker_marker = token
    if not _audit_installed:
        sys.addaudithook(_network_audit)
        _audit_installed = True


def _copy_snapshot(
    source: Path, destination: Path, item: PocketManifestFile, *, voice: bool = False
) -> None:
    failure = _voice_failure if voice else _model_failure
    descriptor, before = _open_source(
        source,
        item.size,
        item.sha256,
        MAX_VOICE_BYTES
        if voice
        else (
            MAX_CONFIG_BYTES
            if source.suffix.lower() in {".yaml", ".yml"}
            else MAX_MODEL_FILE_BYTES
        ),
        voice=voice,
    )
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    out = -1
    try:
        out = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o400,
        )
        digest = hashlib.sha256()
        copied = 0
        while copied < item.size:
            chunk = os.read(descriptor, min(_HASH_CHUNK_BYTES, item.size - copied))
            if not chunk:
                raise OSError
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(out, view)
                if written <= 0:
                    raise OSError
                view = view[written:]
            copied += len(chunk)
        after = os.fstat(descriptor)
        if (
            os.read(descriptor, 1)
            or copied != item.size
            or digest.hexdigest() != item.sha256
            or (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
        ):
            raise OSError
        os.fsync(out)
    except OSError:
        raise failure() from None
    finally:
        os.close(descriptor)
        if out >= 0:
            os.close(out)


def _make_snapshot(config: PocketEngineConfig) -> tuple[Path, PocketEngineConfig]:
    root, selected, expected, sizes = _manifest(config)
    snapshot = Path(tempfile.mkdtemp(prefix="kenkui-pocket-"))
    snapshot.chmod(0o700)
    model = snapshot / "model"
    voice_directory = snapshot / "voice"
    try:
        model.mkdir(mode=0o700)
        for relative_name, item in expected.items():
            _copy_snapshot(
                root.joinpath(*PurePosixPath(relative_name).parts),
                model.joinpath(*PurePosixPath(relative_name).parts),
                item,
            )
        snapshot_voices: list[VoiceAsset] = []
        for asset in config.voices:
            # Named by digest, not by voice id: two casts can hold different
            # voices whose display names collide, and the digest cannot.
            destination = voice_directory / f"{asset.sha256}.safetensors"
            voice_item = PocketManifestFile(
                destination.name, sizes[asset.sha256], asset.sha256
            )
            _copy_snapshot(Path(asset.path), destination, voice_item, voice=True)
            snapshot_voices.append(replace(asset, path=str(destination)))
        for current, directories, _names in os.walk(snapshot, topdown=False):
            for directory in directories:
                (Path(current) / directory).chmod(0o500)
        snap = PocketEngineConfig(
            model_root=str(model),
            config_path=str(model / selected.relative_to(root)),
            model_revision=config.model_revision,
            package_version=config.package_version,
            files=config.files,
            voices=tuple(snapshot_voices),
            cloning_capable=config.cloning_capable,
            sample_rate_hz=config.sample_rate_hz,
            device=config.device,
            timeout_seconds=config.timeout_seconds,
        )
        return snapshot, snap
    except Exception:
        shutil.rmtree(snapshot, ignore_errors=True)
        raise


def _deny_remote(value: object, allowed: frozenset[Path], root: Path) -> Path:
    """Map a declared relative asset name to a verified file inside the snapshot.

    Config YAML declares assets by relative name: `_inspect_yaml` rejects
    absolute paths, and an absolute path could not be written in advance anyway
    because the snapshot root is a temporary directory created per engine. This
    is the single place that resolves a declared name against that root and
    checks it against the manifest allowlist.
    """
    if type(value) is not str and not isinstance(value, Path):
        raise RuntimeError
    try:
        declared = PurePosixPath(str(value))
        if declared.is_absolute() or any(part in {"..", ""} for part in declared.parts):
            raise OSError
        path = root.joinpath(*declared.parts)
        resolved = path.resolve(strict=True)
        info = resolved.lstat()
        if (
            resolved != path
            or resolved not in allowed
            or not resolved.is_relative_to(root)
            or not _safe_mode(info)
            or info.st_nlink != 1
        ):
            raise OSError
    except (OSError, RuntimeError, ValueError):
        raise RuntimeError from None
    return path


class PocketTTSEngine:
    """Reusable adapter constructible only by the spawned scheduler worker."""

    def __init__(self, config: PocketEngineConfig, *, reusable: bool = False) -> None:
        self._snapshot: Path | None = None
        self._states: dict[str, Any] = {}
        self._patched: list[tuple[Any, Any]] = []
        self._reusable = reusable
        if _worker_marker is not _WORKER_TOKEN:
            raise ModelError(ErrorCode.POCKET_MODEL_LOAD_FAILED)
        try:
            preflight_pocket(config)
            self._snapshot, self._config = _make_snapshot(config)
            os.environ.update(
                {
                    "HF_HUB_OFFLINE": "1",
                    "HF_HUB_DISABLE_TELEMETRY": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                }
            )
            module = importlib.import_module("pocket_tts")
            implementation = importlib.import_module("pocket_tts.models.tts_model")
            # Resolve the root once and build the allowlist from it. The
            # per-component symlink check below compares a candidate against
            # its own resolution, which only means "the final component is not
            # a symlink" when the ancestors are already resolved. On macOS the
            # snapshot lives under /var, a symlink to /private/var, so an
            # unresolved root made every candidate compare unequal.
            root = Path(self._config.model_root).resolve()
            allowed = frozenset(
                root / item.relative_path for item in self._config.files
            )

            def deny(value: object) -> Path:
                return _deny_remote(value, allowed, root)

            self._patched = []
            setattr(implementation, "download_if_necessary", deny)
            # Every pocket-tts module that imported the downloader gets the
            # allowlist, not a hand-picked few. pocket_tts.conditioners.text
            # holds its own reference and loads the sentencepiece tokenizer
            # through it; leaving that one unpatched both bypassed the
            # allowlist and left a relative tokenizer path unresolved.
            for alias_name in [
                name for name in sys.modules if name.startswith("pocket_tts")
            ]:
                alias = sys.modules.get(alias_name)
                if alias is not None and hasattr(alias, "download_if_necessary"):
                    self._patched.append(
                        (alias, getattr(alias, "download_if_necessary"))
                    )
                    setattr(alias, "download_if_necessary", deny)
            if any(
                getattr(sys.modules.get(name), "download_if_necessary", deny)
                is not deny
                for name in sys.modules
                if name.startswith("pocket_tts")
            ):
                raise RuntimeError
            model_type = getattr(module, "TTSModel")
            if not hasattr(model_type, "load_model"):
                raise RuntimeError
            model = model_type.load_model(
                config=Path(self._config.config_path), quantize=False
            )
            if (
                type(model.sample_rate) is not int
                or model.sample_rate != self._config.sample_rate_hz
                or str(model.device) != self._config.device
            ):
                raise RuntimeError
            self._model = model
            self._tensor_type = importlib.import_module("torch").Tensor
        except Exception:
            self.close()
            raise ModelError(ErrorCode.POCKET_MODEL_LOAD_FAILED) from None

    def close(self) -> None:
        # Restore the downloader before dropping the snapshot. The rebinding is
        # process-global, so leaving it in place outlives the engine and, in a
        # single process, would poison provisioning's own fetch.
        for module, original in getattr(self, "_patched", ()):
            with suppress(AttributeError, TypeError):
                setattr(module, "download_if_necessary", original)
        self._patched = []
        # States hold tensors derived from snapshot files that are about to be
        # removed, so they must not outlive it.
        self._states = {}
        snapshot = self._snapshot
        self._snapshot = None
        if snapshot is not None:
            with suppress(OSError):
                for current, directories, files in os.walk(snapshot):
                    Path(current).chmod(0o700)
                    for name in directories:
                        (Path(current) / name).chmod(0o700)
                    for name in files:
                        (Path(current) / name).chmod(0o600)
                shutil.rmtree(snapshot)

    def __del__(self) -> None:
        self.close()

    def _voice_state(self, voice_asset_sha256: str) -> Any:
        """Derive one conditioning state per voice and reuse each for its segments.

        Workers hold a reusable engine and process a batch serially, so deriving
        this per segment was pure waste. One model serves the whole cast: a
        language engine is ~225 MB of weights against ~6.5 MB per embedding, and
        the scheduler rejects any worker reporting more than one initialization.
        The argument is a Path, never a str: get_state_for_audio_prompt calls
        download_if_necessary only on str, so a Path cannot reach the network
        even before _deny_remote intervenes.
        """
        cached = self._states.get(voice_asset_sha256)
        if cached is not None:
            return cached
        try:
            asset = self._config.voice_by_sha(voice_asset_sha256)
            state = self._model.get_state_for_audio_prompt(Path(asset.path))
        except Exception:
            self.close()
            raise VoiceError(ErrorCode.POCKET_VOICE_LOAD_FAILED) from None
        self._states[voice_asset_sha256] = state
        return state

    def synthesize(self, task: SynthesisTask) -> SynthesizedAudio:
        state = self._voice_state(
            task.voice_asset_sha256 or self._config.voices[0].sha256
        )
        try:
            output = self._model.generate_audio(state, task.text)
            audio = tensor_to_pcm(
                output, self._tensor_type, task, self._config.sample_rate_hz
            )
        except RenderError:
            self.close()
            raise
        except Exception:
            self.close()
            raise RenderError(ErrorCode.POCKET_INFERENCE_FAILED) from None
        if not self._reusable:
            self.close()
        return audio


def tensor_to_pcm(
    output: Any, tensor_type: type[Any], task: SynthesisTask, sample_rate_hz: int
) -> SynthesizedAudio:
    """Convert a bounded exact CPU contiguous float32 mono tensor in small chunks."""
    try:
        if (
            type(output) is not tensor_type
            or str(output.dtype) != "torch.float32"
            or type(output.ndim) is not int
            or output.ndim != 1
            or str(output.device) != "cpu"
            or output.is_contiguous() is not True
        ):
            raise ValueError
        samples = output.shape[0]
        if type(samples) is not int or samples <= 0 or samples > MAX_OUTPUT_SAMPLES:
            raise ValueError
        byte_count = samples * 2
        if (
            type(task.max_output_bytes) is not int
            or byte_count > task.max_output_bytes
            or task.channels != 1
            or task.sample_rate_hz != sample_rate_hz
        ):
            raise ValueError
        pcm = bytearray(byte_count)
        for start in range(0, samples, _TENSOR_CHUNK_SAMPLES):
            stop = min(samples, start + _TENSOR_CHUNK_SAMPLES)
            chunk = output[start:stop]
            if type(chunk) is not tensor_type:
                raise ValueError
            values = chunk.tolist()
            if type(values) is not list or len(values) != stop - start:
                raise ValueError
            for offset, value in enumerate(values):
                if (
                    type(value) is not float
                    or not math.isfinite(value)
                    or value < -SAMPLE_TOLERANCE
                    or value > SAMPLE_TOLERANCE
                ):
                    raise ValueError
                clamped = max(-1.0, min(value, 1.0))
                struct.pack_into(
                    "<h", pcm, (start + offset) * 2, round(clamped * 32767.0)
                )
        duration = samples * 1000 // sample_rate_hz
        if duration <= 0:
            raise ValueError
        return SynthesizedAudio(
            task.segment_id,
            task.chapter_id,
            bytes(pcm),
            sample_rate_hz,
            1,
            samples,
            duration,
        )
    except (
        AttributeError,
        IndexError,
        OverflowError,
        TypeError,
        ValueError,
        struct.error,
    ):
        raise RenderError(ErrorCode.INVALID_AUDIO) from None
