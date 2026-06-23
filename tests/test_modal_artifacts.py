from __future__ import annotations

from kenkui.modal_runtime.artifacts import LocalFilesystemArtifactStore


def test_local_filesystem_artifact_store_round_trips_bytes(tmp_path):
    store = LocalFilesystemArtifactStore(tmp_path)

    uri = store.put_bytes("job-1/book.txt", b"hello")

    assert uri.startswith("file://")
    assert store.get_bytes(uri) == b"hello"


def test_local_filesystem_artifact_store_round_trips_paths(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"wav")
    target = tmp_path / "download.wav"
    store = LocalFilesystemArtifactStore(tmp_path / "artifacts")

    uri = store.put_path("job-1/source.wav", source)
    store.get_path(uri, target)

    assert target.read_bytes() == b"wav"
