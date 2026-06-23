"""Artifact transport abstractions for optional Modal runtime calls."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlparse


class ArtifactStore(Protocol):
    def put_bytes(self, key: str, data: bytes) -> str: ...

    def get_bytes(self, uri: str) -> bytes: ...

    def put_path(self, key: str, source: str | Path) -> str: ...

    def get_path(self, uri: str, target: str | Path) -> Path: ...


class LocalFilesystemArtifactStore:
    """Filesystem-backed ArtifactStore used for tests and local dry-runs."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, key: str) -> Path:
        relative = Path(key)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe artifact key: {key!r}")
        return self.root / relative

    @staticmethod
    def _path_from_uri(uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            raise ValueError(f"Unsupported artifact URI: {uri!r}")
        return Path(unquote(parsed.path))

    def put_bytes(self, key: str, data: bytes) -> str:
        path = self._path_for_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path.resolve().as_uri()

    def get_bytes(self, uri: str) -> bytes:
        return self._path_from_uri(uri).read_bytes()

    def put_path(self, key: str, source: str | Path) -> str:
        path = self._path_for_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, path)
        return path.resolve().as_uri()

    def get_path(self, uri: str, target: str | Path) -> Path:
        target_path = Path(target)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path_from_uri(uri), target_path)
        return target_path


__all__ = ["ArtifactStore", "LocalFilesystemArtifactStore"]
