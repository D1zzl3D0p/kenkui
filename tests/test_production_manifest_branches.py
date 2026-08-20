"""Strict production-manifest validation and local-file security branches."""
# ruff: noqa: D103, PLR0913, PLR0917, SLF001, TC001, TC003

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from kenkui import ErrorCode, ModelError, RenderError, VoiceError
from kenkui._tts import production
from kenkui.errors import KenkuiError

Payload = dict[str, Any]


def _payload(tmp_path: Path) -> Payload:
    root = tmp_path / "model"
    root.mkdir(mode=0o700)
    config = root / "config.yaml"
    config.write_text("model: local\n", encoding="utf-8")
    prompt = tmp_path / "voice.wav"
    prompt.write_bytes(b"fixture")
    return {
        "schema_version": production.MANIFEST_SCHEMA_VERSION,
        "engines": {
            "english": {
                "language": "english",
                "model_root": str(root),
                "config_path": str(config),
                "model_revision": "revision-1",
                "package_version": "2.1.0",
                "files": [
                    {
                        "relative_path": "config.yaml",
                        "size": config.stat().st_size,
                        "sha256": "a" * 64,
                    }
                ],
                "sample_rate_hz": 24000,
                "device": "cpu",
                "timeout_seconds": 30.0,
                "cloning_capable": False,
            }
        },
        "voices": {
            "narrator": {
                "variety": "built-in",
                "state": "loaded",
                "name": "Narrator",
                "enabled": True,
                "provenance": "project fixture",
                "license_id": "CC0-1.0",
                "commercial_use_allowed": True,
                "language": "english",
                "engine_id": "english",
                "compatible_model_revisions": ["revision-1"],
                "asset_path": str(prompt),
                "asset_sha256": "c" * 64,
                "voice_rights": "project-owned",
            }
        },
    }


def _write_manifest(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "production.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)
    return path


def _activate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: object) -> None:
    path = _write_manifest(tmp_path, payload)
    monkeypatch.setenv("KENKUI_POCKET_MANIFEST", str(path))
    production.production_bindings_from_environment("narrator")


def _set_path(payload: Payload, path: tuple[str, ...], value: object) -> None:
    target: Any = payload
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value


@pytest.fixture(autouse=True)
def _avoid_native_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(production, "preflight_pocket", lambda *_: None)


@pytest.mark.parametrize("container", ["root", "engine", "voice"])
@pytest.mark.parametrize("operation", ["missing", "unknown"])
def test_objects_require_exact_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    container: str,
    operation: str,
) -> None:
    payload = _payload(tmp_path)
    if container == "root":
        target = payload
    elif container == "engine":
        target = payload["engines"]["english"]
    else:
        target = payload["voices"]["narrator"]
    assert isinstance(target, dict)
    if operation == "missing":
        # variety and state are validated before the key-set check, because the
        # expected key set depends on them, and they have their own codes and
        # tests. Drop a key whose absence exercises exact-key-set enforcement.
        gated = {"variety", "state"}
        target.pop(next(key for key in target if key not in gated))
    else:
        target["unknown"] = None

    error = VoiceError if container == "voice" else ModelError
    code = (
        ErrorCode.VOICE_PROVENANCE_REQUIRED
        if container == "voice"
        else ErrorCode.POCKET_MODEL_INVALID
    )
    with pytest.raises(error) as caught:
        _activate(tmp_path, monkeypatch, payload)
    assert caught.value.code is code


@pytest.mark.parametrize(
    ("path", "value", "error", "code"),
    [
        (("schema_version",), "wrong", ModelError, ErrorCode.POCKET_MODEL_INVALID),
        (("engines", "english"), [], ModelError, ErrorCode.POCKET_MODEL_INVALID),
        (("voices",), [], VoiceError, ErrorCode.VOICE_UNRESOLVED),
        (("voices",), {}, VoiceError, ErrorCode.VOICE_UNRESOLVED),
        (
            ("engines", "english", "model_revision"),
            None,
            ModelError,
            ErrorCode.POCKET_MODEL_INVALID,
        ),
        (
            ("engines", "english", "model_revision"),
            " ",
            ModelError,
            ErrorCode.POCKET_MODEL_INVALID,
        ),
        (
            ("engines", "english", "package_version"),
            2.1,
            ModelError,
            ErrorCode.POCKET_MODEL_INVALID,
        ),
        (
            ("engines", "english", "files"),
            {},
            ModelError,
            ErrorCode.POCKET_MODEL_INVALID,
        ),
        (
            ("engines", "english", "files"),
            [],
            ModelError,
            ErrorCode.POCKET_MODEL_INVALID,
        ),
        (
            ("engines", "english", "sample_rate_hz"),
            True,
            ModelError,
            ErrorCode.POCKET_MODEL_INVALID,
        ),
        (
            ("engines", "english", "timeout_seconds"),
            30,
            ModelError,
            ErrorCode.POCKET_MODEL_INVALID,
        ),
        (
            ("engines", "english", "device"),
            "",
            ModelError,
            ErrorCode.POCKET_MODEL_INVALID,
        ),
        (
            ("engines", "english", "model_root"),
            "relative",
            ModelError,
            ErrorCode.POCKET_MODEL_INVALID,
        ),
        (("voices", "narrator"), [], VoiceError, ErrorCode.VOICE_PROVENANCE_REQUIRED),
        (
            ("voices", "narrator", "name"),
            1,
            VoiceError,
            ErrorCode.VOICE_PROVENANCE_REQUIRED,
        ),
        (
            ("voices", "narrator", "provenance"),
            " ",
            VoiceError,
            ErrorCode.VOICE_PROVENANCE_REQUIRED,
        ),
        (
            ("voices", "narrator", "enabled"),
            1,
            VoiceError,
            ErrorCode.VOICE_PROVENANCE_REQUIRED,
        ),
        (
            ("voices", "narrator", "commercial_use_allowed"),
            1,
            VoiceError,
            ErrorCode.VOICE_PROVENANCE_REQUIRED,
        ),
        (
            ("voices", "narrator", "asset_sha256"),
            "B" * 64,
            VoiceError,
            ErrorCode.VOICE_PROVENANCE_REQUIRED,
        ),
        (
            ("voices", "narrator", "asset_sha256"),
            "c" * 63,
            VoiceError,
            ErrorCode.VOICE_PROVENANCE_REQUIRED,
        ),
        (
            ("voices", "narrator", "compatible_model_revisions"),
            {},
            VoiceError,
            ErrorCode.VOICE_PROVENANCE_REQUIRED,
        ),
        (
            ("voices", "narrator", "compatible_model_revisions"),
            [],
            VoiceError,
            ErrorCode.VOICE_PROVENANCE_REQUIRED,
        ),
        (
            ("voices", "narrator", "compatible_model_revisions"),
            ["revision-1", "revision-1"],
            VoiceError,
            ErrorCode.VOICE_PROVENANCE_REQUIRED,
        ),
        (
            ("voices", "narrator", "asset_path"),
            "relative.wav",
            VoiceError,
            ErrorCode.POCKET_VOICE_INVALID,
        ),
    ],
)
def test_malformed_declared_fields_fail_with_stable_domain_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[str, ...],
    value: object,
    error: type[KenkuiError],
    code: ErrorCode,
) -> None:
    payload = _payload(tmp_path)
    _set_path(payload, path, value)
    with pytest.raises(error) as caught:
        _activate(tmp_path, monkeypatch, payload)
    assert caught.value.code is code


@pytest.mark.parametrize(
    "item",
    [
        None,
        {},
        {"relative_path": "config.yaml", "size": 1, "sha256": "a" * 64, "extra": 1},
        {"relative_path": 1, "size": 1, "sha256": "a" * 64},
        {"relative_path": "config.yaml", "size": True, "sha256": "a" * 64},
        {"relative_path": "config.yaml", "size": 1, "sha256": "z" * 64},
    ],
)
def test_manifest_file_requires_exact_typed_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    item: object,
) -> None:
    payload = _payload(tmp_path)
    payload["engines"]["english"]["files"] = [item]
    with pytest.raises(ModelError) as caught:
        _activate(tmp_path, monkeypatch, payload)
    assert caught.value.code is ErrorCode.POCKET_MODEL_INVALID


def test_duplicate_json_keys_are_rejected_before_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":"a","schema_version":"b"}', encoding="utf-8")
    path.chmod(0o600)
    monkeypatch.setenv("KENKUI_POCKET_MANIFEST", str(path))
    with pytest.raises(ModelError) as caught:
        production.production_bindings_from_environment("narrator")
    assert caught.value.code is ErrorCode.POCKET_MODEL_INVALID


def test_unknown_incompatible_disabled_and_absent_voice_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("KENKUI_POCKET_MANIFEST", raising=False)
    with pytest.raises(RenderError) as absent:
        production.production_bindings_from_environment("narrator")
    assert absent.value.code is ErrorCode.RENDERER_UNAVAILABLE

    payload = _payload(tmp_path)
    path = _write_manifest(tmp_path, payload)
    monkeypatch.setenv("KENKUI_POCKET_MANIFEST", str(path))
    with pytest.raises(VoiceError) as unknown:
        production.production_bindings_from_environment("unknown")
    assert unknown.value.code is ErrorCode.VOICE_UNRESOLVED

    payload["voices"]["narrator"]["compatible_model_revisions"] = ["other"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(VoiceError) as incompatible:
        production.production_bindings_from_environment("narrator")
    assert incompatible.value.code is ErrorCode.VOICE_INCOMPATIBLE

    payload["voices"]["narrator"]["compatible_model_revisions"] = ["revision-1"]
    payload["voices"]["narrator"]["enabled"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(VoiceError) as disabled:
        production.production_bindings_from_environment("narrator")
    assert disabled.value.code is ErrorCode.VOICE_DISABLED


@pytest.mark.parametrize("bad_data", [b"{", b'"\xff"', b"null"])
def test_json_syntax_encoding_and_non_object_errors_are_masked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_data: bytes,
) -> None:
    path = tmp_path / "bad.json"
    path.write_bytes(bad_data)
    path.chmod(0o600)
    monkeypatch.setenv("KENKUI_POCKET_MANIFEST", str(path))
    with pytest.raises(ModelError) as caught:
        production.production_bindings_from_environment("narrator")
    assert caught.value.code is ErrorCode.POCKET_MODEL_INVALID
    assert str(path) not in str(caught.value)


@pytest.mark.parametrize("value", [None, "", "x" * 4097])
def test_manifest_path_declaration_is_bounded_and_exact(value: object) -> None:
    with pytest.raises(ModelError) as caught:
        production._read_manifest(value)
    assert caught.value.code is ErrorCode.POCKET_MODEL_INVALID


def _assert_read_rejected(path: Path) -> None:
    with pytest.raises(ModelError) as caught:
        production._read_manifest(str(path))
    assert caught.value.code is ErrorCode.POCKET_MODEL_INVALID


def test_manifest_path_must_be_absolute_regular_single_link_and_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    valid = _write_manifest(tmp_path, _payload(tmp_path))

    monkeypatch.chdir(tmp_path)
    _assert_read_rejected(Path("production.json"))

    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(valid)
    _assert_read_rejected(symlink)

    hardlink = tmp_path / "hardlink.json"
    os.link(valid, hardlink)
    _assert_read_rejected(valid)
    hardlink.unlink()

    valid.chmod(0o620)
    _assert_read_rejected(valid)
    valid.chmod(0o600)

    directory = tmp_path / "directory"
    directory.mkdir()
    _assert_read_rejected(directory)

    monkeypatch.setattr(os, "getuid", lambda: valid.stat().st_uid + 1)
    _assert_read_rejected(valid)


def test_manifest_empty_oversize_short_read_and_identity_mutation_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty.json"
    empty.touch(mode=0o600)
    _assert_read_rejected(empty)

    large = tmp_path / "large.json"
    large.write_bytes(b" " * (production.MAX_PRODUCTION_MANIFEST_BYTES + 1))
    large.chmod(0o600)
    _assert_read_rejected(large)

    valid = _write_manifest(tmp_path, _payload(tmp_path))
    real_read: Callable[[int, int], bytes] = os.read
    monkeypatch.setattr(os, "read", lambda _descriptor, _size: b"")
    _assert_read_rejected(valid)
    monkeypatch.setattr(os, "read", real_read)

    mutated = False

    def read_then_mutate(descriptor: int, size: int) -> bytes:
        nonlocal mutated
        data = real_read(descriptor, size)
        if data and not mutated:
            before = valid.stat()
            os.utime(valid, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
            mutated = True
        return data

    monkeypatch.setattr(os, "read", read_then_mutate)
    _assert_read_rejected(valid)
    assert mutated


@pytest.mark.usefixtures("_real_cache_root")
def test_default_cache_roots_cover_darwin_xdg_and_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))

    monkeypatch.setattr(sys, "platform", "darwin")
    assert production.default_cache_root() == home / "Library/Caches/kenkui/v1"

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert production.default_cache_root() == tmp_path / "xdg/kenkui/v1"

    monkeypatch.setenv("XDG_CACHE_HOME", "relative")
    assert production.default_cache_root() == home / ".cache/kenkui/v1"
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert production.default_cache_root() == home / ".cache/kenkui/v1"
