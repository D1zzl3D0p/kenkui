from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol


def progress_key(job_id: str) -> str:
    return f"jobs/{job_id}/progress.json"


def nlp_roster_key(job_id: str) -> str:
    return f"jobs/{job_id}/nlp/roster.json"


def attribution_chapters_key(job_id: str) -> str:
    return f"jobs/{job_id}/attribution/chapters.json"


def tts_wav_key(job_id: str, chapter_index: int) -> str:
    return f"jobs/{job_id}/tts/ch_{chapter_index:04d}.wav"


def output_m4b_key(job_id: str) -> str:
    return f"jobs/{job_id}/output.m4b"


class StorageBackend(Protocol):
    def put_json(self, key: str, data: dict) -> None: ...
    def get_json(self, key: str) -> dict: ...
    def put_bytes(self, key: str, data: bytes) -> None: ...
    def get_bytes(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...


class LocalStorageBackend:
    """Writes to a local directory — used in tests and local dev."""

    def __init__(self, root: Path | str):
        self._root = Path(root)

    def _path(self, key: str) -> Path:
        return self._root / key

    def put_json(self, key: str, data: dict) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data), encoding="utf-8")

    def get_json(self, key: str) -> dict:
        return json.loads(self._path(key).read_text(encoding="utf-8"))

    def put_bytes(self, key: str, data: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()


class BotoStorageBackend:
    """Reads/writes to R2 or S3 via boto3.

    Env vars:
      KENKUI_MODAL_BUCKET          — required
      KENKUI_MODAL_BUCKET_ENDPOINT — optional (R2 endpoint)
      AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY — standard boto3
    """

    def __init__(self):
        import boto3  # type: ignore[import]

        bucket = os.environ["KENKUI_MODAL_BUCKET"]
        endpoint = os.environ.get("KENKUI_MODAL_BUCKET_ENDPOINT")
        kwargs: dict = {}
        if endpoint:
            kwargs["endpoint_url"] = endpoint
        self._bucket = bucket
        self._s3 = boto3.client("s3", **kwargs)

    def put_json(self, key: str, data: dict) -> None:
        body = json.dumps(data).encode("utf-8")
        self._s3.put_object(Bucket=self._bucket, Key=key, Body=body, ContentType="application/json")

    def get_json(self, key: str) -> dict:
        obj = self._s3.get_object(Bucket=self._bucket, Key=key)
        return json.loads(obj["Body"].read())

    def put_bytes(self, key: str, data: bytes) -> None:
        self._s3.put_object(Bucket=self._bucket, Key=key, Body=data)

    def get_bytes(self, key: str) -> bytes:
        obj = self._s3.get_object(Bucket=self._bucket, Key=key)
        return obj["Body"].read()

    def exists(self, key: str) -> bool:
        try:
            self._s3.head_object(Bucket=self._bucket, Key=key)
            return True
        except self._s3.exceptions.ClientError:
            return False


def build_storage_backend() -> StorageBackend:
    if os.environ.get("KENKUI_MODAL_BUCKET"):
        return BotoStorageBackend()
    raise RuntimeError("KENKUI_MODAL_BUCKET must be set when Modal execution is enabled.")
