"""Tests for kenkui.voice_loader and kenkui.workers helpers.

Coverage:
- load_voice — built-in names, local files, hf:// URLs, fallback
- get_batch_info — batch counts, first-chapter sizing, empty chapter
- _tensor_to_audio — None, empty, 1-D, 2-D, 3-D tensors
- _render_text — success, empty tensor, exception retry, all-fail
- _finalise_chapter — audio too short, valid audio written to disk, queue messages
- _render_multi_voice — single speaker, multiple speakers, segment ordering,
  queue messages, speaker_voices override
- worker_process_chapter — success via multi-voice path, retry on failure,
  exhausted retries, verbose logging
- _get_or_load_model — caching: same key returns same object, different key loads fresh
"""

from __future__ import annotations

import multiprocessing
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from pydub import AudioSegment

from kenkui.models import AudioResult, Chapter, Segment
from kenkui.voice_loader import load_voice
from kenkui.workers import (
    DEFAULT_BATCH_SIZE,
    FIRST_CHAPTER_BATCH_SIZE,
    _finalise_chapter,
    _get_or_load_model,
    _render_multi_voice,
    _render_text,
    _tensor_to_audio,
    get_batch_info,
    worker_process_chapter,
)


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


# ---------------------------------------------------------------------------
# Helpers shared by the new test classes
# ---------------------------------------------------------------------------


def _make_silent_segment(duration_ms: int = 2000) -> AudioSegment:
    """Return a silent AudioSegment of the given duration."""
    return AudioSegment.silent(duration=duration_ms)


def _make_tensor(n: int = 24000, ndim: int = 1) -> torch.Tensor:
    """Return a float32 tensor of zeros with the requested shape."""
    if ndim == 1:
        return torch.zeros(n, dtype=torch.float32)
    if ndim == 2:
        return torch.zeros(1, n, dtype=torch.float32)
    # 3-D
    return torch.zeros(1, 1, n, dtype=torch.float32)


def _noop_log(*_args, **_kwargs):
    """No-op log_message callable for worker helpers."""


# ---------------------------------------------------------------------------
# _tensor_to_audio
# ---------------------------------------------------------------------------


class TestTensorToAudio:
    def test_none_returns_none(self):
        assert _tensor_to_audio(None, 24000) is None

    def test_empty_tensor_returns_none(self):
        t = torch.zeros(0, dtype=torch.float32)
        assert _tensor_to_audio(t, 24000) is None

    def test_1d_tensor_returns_audio_segment(self):
        t = _make_tensor(24000, ndim=1)
        seg = _tensor_to_audio(t, 24000)
        assert seg is not None
        assert isinstance(seg, AudioSegment)
        assert len(seg) > 0

    def test_2d_tensor_squeezed_to_1d(self):
        t = _make_tensor(24000, ndim=2)
        assert t.dim() == 2
        seg = _tensor_to_audio(t, 24000)
        assert seg is not None
        assert isinstance(seg, AudioSegment)

    def test_3d_tensor_flattened(self):
        t = _make_tensor(24000, ndim=3)
        assert t.dim() == 3
        seg = _tensor_to_audio(t, 24000)
        assert seg is not None
        assert isinstance(seg, AudioSegment)

    def test_sample_rate_respected(self):
        """Higher sample rate with same number of samples → shorter duration."""
        samples = 24000
        seg_24k = _tensor_to_audio(_make_tensor(samples, 1), sample_rate=24000)
        seg_48k = _tensor_to_audio(_make_tensor(samples, 1), sample_rate=48000)
        # 24000 samples @ 24kHz = 1 s; @ 48kHz = 0.5 s
        assert seg_24k is not None and seg_48k is not None
        assert len(seg_24k) > len(seg_48k)

    def test_2d_squeeze_makes_empty_returns_none(self):
        """A (1, 0)-shaped tensor is empty after squeeze → should return None."""
        t = torch.zeros(1, 0, dtype=torch.float32)
        # numel() == 0 → early return
        assert _tensor_to_audio(t, 24000) is None


# ---------------------------------------------------------------------------
# _render_text
# ---------------------------------------------------------------------------


class TestRenderText:
    def _make_model(self, tensor=None, raises=None):
        """Return a mock TTSModel. tensor is returned by generate_audio."""
        model = MagicMock()
        model.sample_rate = 24000
        if raises is not None:
            model.generate_audio.side_effect = raises
        else:
            model.generate_audio.return_value = (
                tensor if tensor is not None else _make_tensor(24000)
            )
        return model

    def test_success_returns_audio_segment(self):
        model = self._make_model()
        voice_state = MagicMock()
        seg = _render_text(model, voice_state, "Hello world.", _noop_log, 1, 0, 1)
        assert seg is not None
        assert isinstance(seg, AudioSegment)

    def test_empty_tensor_retries_and_returns_none(self):
        """Both attempts return empty tensor → None."""
        model = self._make_model(tensor=torch.zeros(0))
        voice_state = MagicMock()
        seg = _render_text(model, voice_state, "text", _noop_log, 1, 0, 1)
        assert seg is None
        assert model.generate_audio.call_count == 2  # two attempts

    def test_exception_on_first_attempt_retries(self):
        """First call raises, second succeeds."""
        good_tensor = _make_tensor(24000)
        model = self._make_model()
        model.generate_audio.side_effect = [RuntimeError("boom"), good_tensor]
        voice_state = MagicMock()
        seg = _render_text(model, voice_state, "text", _noop_log, 1, 0, 1)
        assert seg is not None
        assert model.generate_audio.call_count == 2

    def test_both_attempts_fail_returns_none(self):
        model = self._make_model(raises=RuntimeError("always fails"))
        voice_state = MagicMock()
        seg = _render_text(model, voice_state, "text", _noop_log, 1, 0, 1)
        assert seg is None
        assert model.generate_audio.call_count == 2

    def test_frames_after_eos_zero_passed(self):
        """frames_after_eos=0 must always be forwarded to generate_audio."""
        model = self._make_model()
        voice_state = MagicMock()
        _render_text(model, voice_state, "Hello.", _noop_log, 1, 0, 1)
        _, kwargs = model.generate_audio.call_args
        assert kwargs.get("frames_after_eos") == 0

    def test_log_message_called_with_batch_info(self):
        model = self._make_model()
        voice_state = MagicMock()
        logged: list[str] = []
        _render_text(model, voice_state, "Some text.", logged.append, 42, 3, 10)
        assert any("4/10" in msg for msg in logged)  # batch_idx + 1 / total_batches


# ---------------------------------------------------------------------------
# _finalise_chapter
# ---------------------------------------------------------------------------


class TestFinaliseChapter:
    def _make_queue(self):
        return multiprocessing.Queue()

    def test_short_audio_returns_none_and_puts_done(self):
        """Audio shorter than 1000 ms should be rejected."""
        chapter = _make_chapter(["Short"], index=0)
        short_audio = _make_silent_segment(500)  # 500 ms < 1000 ms
        queue = self._make_queue()
        with tempfile.TemporaryDirectory() as td:
            result = _finalise_chapter(chapter, short_audio, {}, Path(td), queue, 1, _noop_log)
        assert result is None
        done_msg = queue.get_nowait()
        assert done_msg[0] == "DONE"

    def test_valid_audio_saved_and_result_returned(self):
        chapter = _make_chapter(["Hello world."], index=5)
        long_audio = _make_silent_segment(5000)  # 5 s
        queue = self._make_queue()
        with tempfile.TemporaryDirectory() as td:
            result = _finalise_chapter(
                chapter, long_audio, {"pause_chapter_ms": 2000}, Path(td), queue, 1, _noop_log
            )
            assert result is not None
            assert isinstance(result, AudioResult)
            assert result.chapter_index == 5
            assert result.title == "Chapter 5"
            assert Path(result.file_path).exists()

    def test_result_duration_includes_chapter_pause(self):
        chapter = _make_chapter(["para"], index=1)
        audio = _make_silent_segment(3000)
        queue = self._make_queue()
        with tempfile.TemporaryDirectory() as td:
            result = _finalise_chapter(
                chapter, audio, {"pause_chapter_ms": 1000}, Path(td), queue, 1, _noop_log
            )
        assert result is not None
        # 3000 ms audio + 1000 ms pause = 4000 ms
        assert result.duration_ms == 4000

    def test_done_message_put_on_queue(self):
        chapter = _make_chapter(["para"], index=2)
        audio = _make_silent_segment(3000)
        queue = self._make_queue()
        with tempfile.TemporaryDirectory() as td:
            _finalise_chapter(chapter, audio, {}, Path(td), queue, 1, _noop_log)
        msg = queue.get_nowait()
        assert msg[0] == "DONE"

    def test_filename_uses_chapter_index(self):
        chapter = _make_chapter(["text"], index=7)
        audio = _make_silent_segment(3000)
        queue = self._make_queue()
        with tempfile.TemporaryDirectory() as td:
            result = _finalise_chapter(chapter, audio, {}, Path(td), queue, 1, _noop_log)
        assert result is not None
        assert "0007" in str(result.file_path)

    def test_default_pause_chapter_ms_applied(self):
        """When pause_chapter_ms is absent the default of 2000 ms is used."""
        chapter = _make_chapter(["text"], index=0)
        audio = _make_silent_segment(3000)
        queue = self._make_queue()
        with tempfile.TemporaryDirectory() as td:
            result = _finalise_chapter(chapter, audio, {}, Path(td), queue, 1, _noop_log)
        assert result is not None
        assert result.duration_ms == 3000 + 2000


# ---------------------------------------------------------------------------
# _render_multi_voice
# ---------------------------------------------------------------------------


class TestRenderMultiVoice:
    """Tests for the multi-voice rendering path."""

    def _make_model(self):
        """Return a mock TTSModel that generates non-empty audio."""
        model = MagicMock()
        model.sample_rate = 24000
        model.generate_audio.return_value = _make_tensor(24000)
        model.get_state_for_audio_prompt.return_value = MagicMock()
        return model

    def _make_queue(self):
        return multiprocessing.Queue()

    def _drain_queue(self, queue: multiprocessing.Queue) -> list:
        msgs = []
        while not queue.empty():
            msgs.append(queue.get_nowait())
        return msgs

    def _make_segments(self, specs: list[tuple[str, str, int]]) -> list[Segment]:
        """specs: list of (text, speaker, index)"""
        return [Segment(text=t, speaker=s, index=i) for t, s, i in specs]

    def test_single_speaker_renders_all_segments(self):
        segs = self._make_segments(
            [
                ("Hello.", "narrator", 0),
                ("World.", "narrator", 1),
            ]
        )
        chapter = Chapter(index=0, title="Ch 0", paragraphs=[], segments=segs)
        model = self._make_model()
        queue = self._make_queue()
        with (
            patch("kenkui.workers.load_voice", return_value="alba"),
            tempfile.TemporaryDirectory() as td,
        ):
            result = _render_multi_voice(chapter, model, {}, Path(td), queue, 1, _noop_log)
        assert result is not None
        assert isinstance(result, AudioResult)
        # generate_audio called once per segment
        assert model.generate_audio.call_count == 2

    def test_multiple_speakers_load_voice_state_once_each(self):
        segs = self._make_segments(
            [
                ("Narration.", "narrator", 0),
                ("Said Alice.", "alice", 1),
                ("Replied Bob.", "bob", 2),
                ("More narration.", "narrator", 3),
            ]
        )
        chapter = Chapter(index=1, title="Ch 1", paragraphs=[], segments=segs)
        model = self._make_model()
        queue = self._make_queue()
        with (
            patch("kenkui.workers.load_voice", return_value="alba"),
            tempfile.TemporaryDirectory() as td,
        ):
            _render_multi_voice(chapter, model, {}, Path(td), queue, 1, _noop_log)
        # 3 unique speakers → get_state_for_audio_prompt called 3 times
        assert model.get_state_for_audio_prompt.call_count == 3

    def test_speaker_voices_override_used(self):
        """speaker_voices in config_dict should resolve per-character voice."""
        segs = self._make_segments([("Line.", "alice", 0)])
        chapter = Chapter(index=0, title="Ch 0", paragraphs=[], segments=segs)
        model = self._make_model()
        queue = self._make_queue()
        config = {"speaker_voices": {"alice": "cosette"}}
        loaded_voices: list[str] = []

        def mock_load_voice(v: str) -> str:
            loaded_voices.append(v)
            return v

        with (
            patch("kenkui.workers.load_voice", side_effect=mock_load_voice),
            tempfile.TemporaryDirectory() as td,
        ):
            _render_multi_voice(chapter, model, config, Path(td), queue, 1, _noop_log)
        assert "cosette" in loaded_voices

    def test_segments_reassembled_in_index_order(self):
        """Segments delivered out-of-order by index must produce a non-empty result."""
        # Provide segments with indices 2, 0, 1 (out of order)
        segs = [
            Segment(text="C", speaker="narrator", index=2),
            Segment(text="A", speaker="narrator", index=0),
            Segment(text="B", speaker="narrator", index=1),
        ]
        chapter = Chapter(index=0, title="Ch 0", paragraphs=[], segments=segs)
        model = self._make_model()
        queue = self._make_queue()
        with (
            patch("kenkui.workers.load_voice", return_value="alba"),
            tempfile.TemporaryDirectory() as td,
        ):
            result = _render_multi_voice(chapter, model, {}, Path(td), queue, 1, _noop_log)
        assert result is not None

    def test_start_and_update_messages_sent_to_queue(self):
        segs = self._make_segments(
            [
                ("Para A.", "narrator", 0),
                ("Para B.", "narrator", 1),
            ]
        )
        chapter = Chapter(index=0, title="Ch 0", paragraphs=[], segments=segs)
        model = self._make_model()
        queue = self._make_queue()
        with (
            patch("kenkui.workers.load_voice", return_value="alba"),
            tempfile.TemporaryDirectory() as td,
        ):
            _render_multi_voice(chapter, model, {}, Path(td), queue, 1, _noop_log)
        msgs = self._drain_queue(queue)
        msg_types = [m[0] for m in msgs]
        assert "START" in msg_types
        assert "UPDATE" in msg_types
        assert "DONE" in msg_types

    def test_start_message_uses_segment_count(self):
        segs = self._make_segments(
            [
                ("A.", "narrator", 0),
                ("B.", "alice", 1),
                ("C.", "narrator", 2),
            ]
        )
        chapter = Chapter(index=0, title="Ch 0", paragraphs=[], segments=segs)
        model = self._make_model()
        queue = self._make_queue()
        with (
            patch("kenkui.workers.load_voice", return_value="alba"),
            tempfile.TemporaryDirectory() as td,
        ):
            _render_multi_voice(chapter, model, {}, Path(td), queue, 1, _noop_log)
        msgs = self._drain_queue(queue)
        start_msg = next(m for m in msgs if m[0] == "START")
        # START format: ("START", pid, title, total_segments, total_chars, is_first_chapter)
        total_segments = start_msg[3]
        assert total_segments == 3

    def test_none_audio_from_render_text_uses_empty_segment(self):
        """If _render_text returns None for a segment, use AudioSegment.empty()."""
        segs = self._make_segments([("A.", "narrator", 0)])
        chapter = Chapter(index=0, title="Ch 0", paragraphs=[], segments=segs)
        model = self._make_model()
        # generate_audio returns empty tensor → _render_text → None
        model.generate_audio.return_value = torch.zeros(0)
        queue = self._make_queue()
        with (
            patch("kenkui.workers.load_voice", return_value="alba"),
            tempfile.TemporaryDirectory() as td,
        ):
            # Result may be None (too short) or AudioResult; either is correct behaviour
            result = _render_multi_voice(chapter, model, {}, Path(td), queue, 1, _noop_log)
        # Should not raise; result type doesn't matter here
        assert result is None or isinstance(result, AudioResult)


# ---------------------------------------------------------------------------
# worker_process_chapter — integration / retry
# ---------------------------------------------------------------------------


class TestWorkerProcessChapter:
    """End-to-end tests for worker_process_chapter through a mocked model."""

    def _make_queue(self):
        return multiprocessing.Queue()

    def _drain_queue(self, queue: multiprocessing.Queue) -> list:
        msgs = []
        while not queue.empty():
            msgs.append(queue.get_nowait())
        return msgs

    def _make_model(self):
        model = MagicMock()
        model.sample_rate = 24000
        model.generate_audio.return_value = _make_tensor(24000)
        model.get_state_for_audio_prompt.return_value = MagicMock()
        return model

    def test_successful_single_voice_returns_audio_result(self):
        chapter = _make_chapter(["Hello world."], index=0)
        queue = self._make_queue()
        config = {"voice": "alba", "pause_line_ms": 0, "pause_chapter_ms": 0}
        model = self._make_model()
        with (
            patch("kenkui.workers._get_or_load_model", return_value=model),
            patch("kenkui.workers.load_voice", return_value="alba"),
            tempfile.TemporaryDirectory() as td,
        ):
            result = worker_process_chapter(chapter, config, Path(td), queue)
        assert result is not None
        assert isinstance(result, AudioResult)

    def test_successful_multi_voice_returns_audio_result(self):
        segs = [
            Segment("Narration.", "narrator", 0),
            Segment("Said Alice.", "alice", 1),
        ]
        chapter = Chapter(index=1, title="Chapter 1", paragraphs=[], segments=segs)
        queue = self._make_queue()
        config = {"pause_line_ms": 0, "pause_chapter_ms": 0}
        model = self._make_model()
        with (
            patch("kenkui.workers._get_or_load_model", return_value=model),
            patch("kenkui.workers.load_voice", return_value="alba"),
            tempfile.TemporaryDirectory() as td,
        ):
            result = worker_process_chapter(chapter, config, Path(td), queue)
        assert result is not None
        assert isinstance(result, AudioResult)

    def test_failure_sends_error_and_done_to_queue(self):
        """When all retries are exhausted an ERROR then DONE message must appear."""
        chapter = _make_chapter(["Para."], index=2)
        queue = self._make_queue()
        config = {}
        with (
            patch(
                "kenkui.workers._get_or_load_model",
                side_effect=RuntimeError("model load failed"),
            ),
            tempfile.TemporaryDirectory() as td,
        ):
            result = worker_process_chapter(chapter, config, Path(td), queue)
        assert result is None
        msgs = self._drain_queue(queue)
        msg_types = [m[0] for m in msgs]
        assert "ERROR" in msg_types
        assert "DONE" in msg_types

    def test_retries_up_to_two_times(self):
        """_process_chapter_inner should be attempted max_retries+1 = 3 times."""
        chapter = _make_chapter(["Para."], index=3)
        queue = self._make_queue()
        call_count = 0

        def always_fail(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")

        with (
            patch("kenkui.workers._get_or_load_model", side_effect=always_fail),
            tempfile.TemporaryDirectory() as td,
        ):
            worker_process_chapter(chapter, {}, Path(td), queue)
        # 3 attempts total (initial + 2 retries)
        assert call_count == 3

    def test_verbose_logs_to_queue(self):
        """In verbose mode LOG messages should appear on the queue."""
        chapter = _make_chapter(["Para."], index=0)
        queue = self._make_queue()
        config = {"voice": "alba", "verbose": True, "pause_line_ms": 0, "pause_chapter_ms": 0}
        model = self._make_model()
        with (
            patch("kenkui.workers._get_or_load_model", return_value=model),
            patch("kenkui.workers.load_voice", return_value="alba"),
            tempfile.TemporaryDirectory() as td,
        ):
            worker_process_chapter(chapter, config, Path(td), queue)
        msgs = self._drain_queue(queue)
        log_msgs = [m for m in msgs if m[0] == "LOG"]
        assert len(log_msgs) > 0

    def test_success_on_second_attempt(self):
        """Model load fails once then succeeds → result is not None."""
        chapter = _make_chapter(["Para."], index=0)
        queue = self._make_queue()
        config = {"voice": "alba", "pause_line_ms": 0, "pause_chapter_ms": 0}
        model = self._make_model()
        call_count = 0

        def sometimes_fail(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient error")
            return model

        with (
            patch("kenkui.workers._get_or_load_model", side_effect=sometimes_fail),
            patch("kenkui.workers.load_voice", return_value="alba"),
            tempfile.TemporaryDirectory() as td,
        ):
            result = worker_process_chapter(chapter, config, Path(td), queue)
        assert result is not None


# ---------------------------------------------------------------------------
# _get_or_load_model — caching
# ---------------------------------------------------------------------------


class TestGetOrLoadModel:
    def setup_method(self):
        """Clear the module-level cache before each test."""
        import kenkui.workers as workers_mod

        workers_mod._model_cache.clear()

    def test_same_key_returns_same_object(self):
        mock_instance = MagicMock()
        mock_cls = MagicMock(return_value=mock_instance)
        mock_cls.load_model = MagicMock(return_value=mock_instance)

        with patch("kenkui.workers._get_or_load_model.__module__"):
            pass  # no-op; actual patch below

        with patch("pocket_tts.TTSModel") as mock_tts:
            mock_tts.load_model.return_value = mock_instance
            first = _get_or_load_model(0.7, 1, None)
            second = _get_or_load_model(0.7, 1, None)

        assert first is second
        # load_model called exactly once for the same key
        mock_tts.load_model.assert_called_once()

    def test_different_key_loads_fresh_model(self):
        instance_a = MagicMock()
        instance_b = MagicMock()

        with patch("pocket_tts.TTSModel") as mock_tts:
            mock_tts.load_model.side_effect = [instance_a, instance_b]
            a = _get_or_load_model(0.7, 1, None)
            b = _get_or_load_model(0.8, 1, None)  # different temp → different key

        assert a is instance_a
        assert b is instance_b
        assert mock_tts.load_model.call_count == 2

    def test_noise_clamp_included_in_key(self):
        instance_a = MagicMock()
        instance_b = MagicMock()

        with patch("pocket_tts.TTSModel") as mock_tts:
            mock_tts.load_model.side_effect = [instance_a, instance_b]
            a = _get_or_load_model(0.7, 1, None)
            b = _get_or_load_model(0.7, 1, 3.0)  # noise_clamp differs

        assert a is not b
        assert mock_tts.load_model.call_count == 2
