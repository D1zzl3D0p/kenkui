"""Private fail-closed production manifest and binding construction."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path, PurePath
from typing import Final, cast

from kenkui._audio.production import FFmpegM4BAssembler
from kenkui._execution.cache import CacheStore
from kenkui._execution.coordinator import ExecutionBindings
from kenkui._execution.process_pool import EngineSpecification
from kenkui._tts.pocket import (
    MAX_CONFIG_BYTES,
    PocketEngineConfig,
    PocketManifestFile,
    preflight_pocket,
)
from kenkui.errors import ErrorCode, ModelError, RenderError, VoiceError
from kenkui.voices.types import Voice

_MANIFEST_ENV: Final = "KENKUI_POCKET_MANIFEST"
MANIFEST_SCHEMA_VERSION: Final = "kenkui-pocket-production-v1"
MAX_PRODUCTION_MANIFEST_BYTES: Final = MAX_CONFIG_BYTES
_SHA256_LENGTH: Final = 64
_MAX_DECLARED_STRING_LENGTH: Final = 4096


def _require_manifest_io(*, condition: bool) -> None:
    if not condition:
        raise OSError


def default_cache_root() -> Path:
    """Return the private versioned cache location for the current operating system."""
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Caches" / "kenkui" / "v1"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg and Path(xdg).is_absolute() else home / ".cache"
    return base / "kenkui" / "v1"


def production_bindings_from_environment(voice_id: str) -> ExecutionBindings:
    """Resolve one explicitly assigned voice from a strict local manifest."""
    value = os.environ.get(_MANIFEST_ENV)
    if value is None:
        raise RenderError(ErrorCode.RENDERER_UNAVAILABLE)
    manifest = _read_manifest(value)
    root = _object(manifest, {"schema_version", "engine", "voices"})
    if root["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ModelError(ErrorCode.POCKET_MODEL_INVALID)
    engine = _object(
        root["engine"],
        {
            "model_root",
            "config_path",
            "model_revision",
            "package_version",
            "files",
            "sample_rate_hz",
            "device",
            "timeout_seconds",
        },
    )
    voices_raw = root["voices"]
    if type(voices_raw) is not dict or not voices_raw:
        raise VoiceError(ErrorCode.VOICE_UNRESOLVED)
    voices = cast("dict[object, object]", voices_raw)
    if any(type(key) is not str or not key for key in voices):
        raise VoiceError(ErrorCode.VOICE_UNRESOLVED)
    selected = voices.get(voice_id)
    if selected is None:
        raise VoiceError(ErrorCode.VOICE_UNRESOLVED)
    voice_data = _object(
        selected,
        {
            "name",
            "enabled",
            "provenance",
            "license_id",
            "commercial_use_allowed",
            "language",
            "content_fingerprint",
            "compatible_model_revisions",
            "voice_prompt_path",
            "voice_prompt_sha256",
            "voice_rights",
        },
        voice=True,
    )
    enabled = _boolean(voice_data["enabled"], voice=True)
    if not enabled:
        raise VoiceError(ErrorCode.VOICE_DISABLED)
    revision = _string(engine["model_revision"])
    compatible = _string_tuple(voice_data["compatible_model_revisions"])
    voice = Voice(
        id=voice_id,
        name=_string(voice_data["name"], voice=True),
        enabled=enabled,
        provenance=_string(voice_data["provenance"], voice=True),
        license_id=_string(voice_data["license_id"], voice=True),
        commercial_use_allowed=_boolean(
            voice_data["commercial_use_allowed"], voice=True
        ),
        language=_string(voice_data["language"], voice=True),
        content_fingerprint=_digest(voice_data["content_fingerprint"], voice=True),
        compatible_model_revisions=compatible,
    )
    if revision not in compatible:
        raise VoiceError(ErrorCode.VOICE_INCOMPATIBLE)
    files_raw = engine["files"]
    if type(files_raw) is not list or not files_raw:
        raise ModelError(ErrorCode.POCKET_MODEL_INVALID)
    files = tuple(_manifest_file(item) for item in cast("list[object]", files_raw))
    config = PocketEngineConfig(
        model_root=_absolute_path(engine["model_root"]),
        config_path=_absolute_path(engine["config_path"]),
        model_revision=revision,
        package_version=_string(engine["package_version"]),
        files=files,
        voice_prompt_path=_absolute_path(voice_data["voice_prompt_path"], voice=True),
        voice_prompt_sha256=_digest(voice_data["voice_prompt_sha256"], voice=True),
        voice_provenance=cast("str", voice.provenance),
        voice_license_id=cast("str", voice.license_id),
        voice_rights=_string(voice_data["voice_rights"], voice=True),
        commercial_use_allowed=cast("bool", voice.commercial_use_allowed),
        sample_rate_hz=_integer(engine["sample_rate_hz"]),
        device=_string(engine["device"]),
        timeout_seconds=_floating(engine["timeout_seconds"]),
    )
    return pocket_production_bindings(config, voice)


def pocket_production_bindings(
    config: PocketEngineConfig,
    voice: Voice,
) -> ExecutionBindings:
    """Bind Pocket, FFmpeg, resolved voice metadata, and the private OS cache."""
    from kenkui._domain.planning import VoicePlan  # noqa: PLC0415

    if type(voice.enabled) is not bool or not isinstance(
        voice.commercial_use_allowed, bool
    ):
        raise VoiceError(ErrorCode.VOICE_PROVENANCE_REQUIRED)
    if not voice.enabled:
        raise VoiceError(ErrorCode.VOICE_DISABLED)
    plan_voice = VoicePlan(
        id=voice.id,
        name=voice.name,
        content_fingerprint=voice.content_fingerprint or "",
        language=voice.language or "",
        provenance=voice.provenance or "",
        license_id=voice.license_id or "",
        commercial_use_allowed=voice.commercial_use_allowed,
        compatible_model_revisions=voice.compatible_model_revisions,
    )
    preflight_pocket(config, plan_voice, config.model_revision)
    return ExecutionBindings(
        EngineSpecification.pocket(config),
        FFmpegM4BAssembler(),
        voice,
        config.model_revision,
        CacheStore(default_cache_root()),
    )


def _read_manifest(value: object) -> object:
    if type(value) is not str or not value or len(value) > _MAX_DECLARED_STRING_LENGTH:
        raise ModelError(ErrorCode.POCKET_MODEL_INVALID)
    descriptor = -1
    try:
        path = Path(value)
        _require_manifest_io(
            condition=path.is_absolute()
            and PurePath(value) == path
            and path.resolve(strict=True) == path
        )
        named = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        opened = os.fstat(descriptor)
        owned = getattr(os, "getuid", lambda: opened.st_uid)() == opened.st_uid
        _require_manifest_io(
            condition=stat.S_ISREG(opened.st_mode)
            and not stat.S_ISLNK(named.st_mode)
            and opened.st_nlink == 1
            and owned
            and not opened.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            and (opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino)
            and 1 <= opened.st_size <= MAX_PRODUCTION_MANIFEST_BYTES
        )
        data = b""
        while len(data) <= MAX_PRODUCTION_MANIFEST_BYTES:
            chunk = os.read(descriptor, min(64 * 1024, opened.st_size - len(data)))
            if not chunk:
                break
            data += chunk
        after = os.fstat(descriptor)
        _require_manifest_io(
            condition=len(data) == opened.st_size
            and _identity(opened) == _identity(after)
        )
        return json.loads(data, object_pairs_hook=_unique_object)
    except (OSError, ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
        raise ModelError(ErrorCode.POCKET_MODEL_INVALID) from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _object(value: object, keys: set[str], *, voice: bool = False) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        if voice:
            raise VoiceError(ErrorCode.VOICE_PROVENANCE_REQUIRED)
        raise ModelError(ErrorCode.POCKET_MODEL_INVALID)
    return cast("dict[str, object]", value)


def _string(value: object, *, voice: bool = False) -> str:
    if (
        type(value) is not str
        or not value.strip()
        or len(value) > _MAX_DECLARED_STRING_LENGTH
    ):
        if voice:
            raise VoiceError(ErrorCode.VOICE_PROVENANCE_REQUIRED)
        raise ModelError(ErrorCode.POCKET_MODEL_INVALID)
    return value


def _boolean(value: object, *, voice: bool = False) -> bool:
    if type(value) is not bool:
        if voice:
            raise VoiceError(ErrorCode.VOICE_PROVENANCE_REQUIRED)
        raise ModelError(ErrorCode.POCKET_MODEL_INVALID)
    return value


def _integer(value: object) -> int:
    if type(value) is not int:
        raise ModelError(ErrorCode.POCKET_MODEL_INVALID)
    return value


def _floating(value: object) -> float:
    if type(value) is not float:
        raise ModelError(ErrorCode.POCKET_MODEL_INVALID)
    return value


def _digest(value: object, *, voice: bool = False) -> str:
    digest = _string(value, voice=voice)
    if len(digest) != _SHA256_LENGTH or any(
        c not in "0123456789abcdef" for c in digest
    ):
        if voice:
            raise VoiceError(ErrorCode.VOICE_PROVENANCE_REQUIRED)
        raise ModelError(ErrorCode.POCKET_MODEL_INVALID)
    return digest


def _string_tuple(value: object) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise VoiceError(ErrorCode.VOICE_PROVENANCE_REQUIRED)
    result = tuple(_string(item, voice=True) for item in cast("list[object]", value))
    if len(set(result)) != len(result):
        raise VoiceError(ErrorCode.VOICE_PROVENANCE_REQUIRED)
    return result


def _absolute_path(value: object, *, voice: bool = False) -> str:
    raw = _string(value, voice=voice)
    path = Path(raw)
    if not path.is_absolute():
        if voice:
            raise VoiceError(ErrorCode.POCKET_VOICE_INVALID)
        raise ModelError(ErrorCode.POCKET_MODEL_INVALID)
    return raw


def _manifest_file(value: object) -> PocketManifestFile:
    item = _object(value, {"relative_path", "size", "sha256"})
    return PocketManifestFile(
        _string(item["relative_path"]),
        _integer(item["size"]),
        _digest(item["sha256"]),
    )
