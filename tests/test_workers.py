"""Tests for kenkui.voice_loader and kenkui.workers helpers."""

from __future__ import annotations

import tempfile
from unittest.mock import patch

from kenkui.models import Chapter
from kenkui.voice_loader import load_voice
from kenkui.workers import DEFAULT_BATCH_SIZE, FIRST_CHAPTER_BATCH_SIZE, get_batch_info


def _make_chapter(paragraphs: list[str], index: int = 0) -> Chapter:
    return Chapter(
        index=index,
        title=f"Chapter {index}",
        paragraphs=paragraphs,
    )


# ---------------------------------------------------------------------------
# load_voice
# ---------------------------------------------------------------------------


class TestLoadVoice:
    def test_builtin_name_returned_unchanged(self):
        assert load_voice("alba") == "alba"
        assert load_voice("cosette") == "cosette"

    def test_existing_local_file_returned_as_str(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF")  # minimal wav header stub
            path = f.name
        result = load_voice(path)
        assert result == path

    def test_nonexistent_path_treated_as_builtin(self):
        # A path that looks like a file but doesn't exist falls through to built-in
        result = load_voice("/nonexistent/path/voice.wav")
        # Should return something (either the path treated as builtin name, or fallback)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_hf_url_triggers_download(self):
        # hf_hub_download is imported inside the function; patch at huggingface_hub
        with patch("huggingface_hub.hf_hub_download", return_value="/tmp/voice.wav") as mock_dl:
            result = load_voice("hf://user/repo/voice.wav")
            assert result == "/tmp/voice.wav"
            mock_dl.assert_called_once()

    def test_hf_url_bad_format_falls_back_to_alba(self):
        with patch("huggingface_hub.hf_hub_download", side_effect=Exception("network")):
            result = load_voice("hf://bad/url")
            assert result == "alba"

    def test_hf_url_parses_repo_and_filename(self):
        """repo_id and filename are correctly split from hf:// URL."""
        captured: dict = {}

        def mock_download(repo_id, filename, **kwargs):
            captured["repo_id"] = repo_id
            captured["filename"] = filename
            return "/tmp/voice.wav"

        with patch("huggingface_hub.hf_hub_download", side_effect=mock_download):
            load_voice("hf://myuser/myrepo/voices/speaker.wav")

        assert captured["repo_id"] == "myuser/myrepo"
        assert captured["filename"] == "voices/speaker.wav"


# ---------------------------------------------------------------------------
# get_batch_info
# ---------------------------------------------------------------------------


class TestGetBatchInfo:
    def test_returns_tuple_of_two_ints(self):
        ch = _make_chapter(["Hello world. This is a test paragraph."])
        count, chars = get_batch_info(ch)
        assert isinstance(count, int)
        assert isinstance(chars, int)
        assert count >= 1
        assert chars > 0

    def test_first_chapter_uses_smaller_batch(self):
        # A long chapter should produce more batches with the smaller first-chapter size
        long_paras = ["Short sentence. " * 5] * 20  # ~1600 chars total
        ch = _make_chapter(long_paras)
        count_first, chars_first = get_batch_info(ch, is_first_chapter=True)
        count_rest, chars_rest = get_batch_info(ch, is_first_chapter=False)
        # Smaller batch size → same or more batches
        assert count_first >= count_rest
        # Total chars should be approximately equal — small differences arise
        # because merging adds spaces between joined paragraphs and different
        # batch-size boundaries produce different merge groupings.
        assert abs(chars_first - chars_rest) < 50

    def test_empty_chapter_returns_zero_batches(self):
        ch = _make_chapter([])
        count, chars = get_batch_info(ch)
        assert count == 0
        assert chars == 0

    def test_batch_size_constants_correct(self):
        assert FIRST_CHAPTER_BATCH_SIZE < DEFAULT_BATCH_SIZE
        assert FIRST_CHAPTER_BATCH_SIZE == 250
        assert DEFAULT_BATCH_SIZE == 800
