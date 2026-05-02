from __future__ import annotations

import json
from pathlib import Path

from kenkui.modal.storage import (
    LocalStorageBackend,
    attribution_chapters_key,
    nlp_roster_key,
    output_m4b_key,
    progress_key,
    tts_wav_key,
)


def test_progress_key():
    assert progress_key("job1") == "jobs/job1/progress.json"


def test_nlp_roster_key():
    assert nlp_roster_key("job1") == "jobs/job1/nlp/roster.json"


def test_attribution_chapters_key():
    assert attribution_chapters_key("job1") == "jobs/job1/attribution/chapters.json"


def test_tts_wav_key():
    assert tts_wav_key("job1", 7) == "jobs/job1/tts/ch_0007.wav"


def test_output_m4b_key():
    assert output_m4b_key("job1") == "jobs/job1/output.m4b"


class TestLocalStorageBackend:
    def test_put_and_get_json(self, tmp_path):
        storage = LocalStorageBackend(tmp_path)
        storage.put_json("jobs/j1/progress.json", {"status": "running"})
        result = storage.get_json("jobs/j1/progress.json")
        assert result == {"status": "running"}

    def test_put_and_get_bytes(self, tmp_path):
        storage = LocalStorageBackend(tmp_path)
        storage.put_bytes("jobs/j1/output.m4b", b"audio-data")
        assert storage.get_bytes("jobs/j1/output.m4b") == b"audio-data"

    def test_exists_true(self, tmp_path):
        storage = LocalStorageBackend(tmp_path)
        storage.put_bytes("jobs/j1/file.bin", b"x")
        assert storage.exists("jobs/j1/file.bin") is True

    def test_exists_false(self, tmp_path):
        storage = LocalStorageBackend(tmp_path)
        assert storage.exists("jobs/j1/missing.json") is False

    def test_put_json_creates_parent_dirs(self, tmp_path):
        storage = LocalStorageBackend(tmp_path)
        storage.put_json("deeply/nested/key.json", {"x": 1})
        assert (tmp_path / "deeply/nested/key.json").exists()
