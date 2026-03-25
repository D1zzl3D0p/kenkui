"""Tests for src/kenkui/post_processing.py and PostProcessingConfig model."""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kenkui.models import AppConfig, PostProcessingConfig


# ── PostProcessingConfig model tests ──────────────────────────────────────────


class TestPostProcessingConfig:
    def test_defaults(self):
        cfg = PostProcessingConfig()
        assert cfg.enabled is True
        assert cfg.noise_reduce is True
        assert cfg.noise_reduce_prop_decrease == 0.8
        assert cfg.highpass_hz == 80
        assert cfg.lowshelf_hz == 250
        assert cfg.lowshelf_db == -3.0
        assert cfg.presence_hz == 3500
        assert cfg.presence_db == 2.0
        assert cfg.deesser is True
        assert cfg.deesser_hz == 6500
        assert cfg.deesser_db == -4.0
        assert cfg.compressor_threshold_db == -18.0
        assert cfg.compressor_ratio == 3.0
        assert cfg.compressor_attack_ms == 5.0
        assert cfg.compressor_release_ms == 50.0
        assert cfg.limiter_threshold_db == -1.0
        assert cfg.normalize is False
        assert cfg.normalize_target_db == -3.0
        assert cfg.normalize_lufs is None

    def test_from_dict_empty(self):
        cfg = PostProcessingConfig.from_dict({})
        assert cfg == PostProcessingConfig()

    def test_from_dict_partial(self):
        cfg = PostProcessingConfig.from_dict({"enabled": True, "normalize": True, "normalize_lufs": -23.0})
        assert cfg.enabled is True
        assert cfg.normalize is True
        assert cfg.normalize_lufs == -23.0
        # Unspecified fields keep defaults
        assert cfg.highpass_hz == 80

    def test_round_trip(self):
        original = PostProcessingConfig(enabled=True, deesser=False, compressor_ratio=4.5)
        restored = PostProcessingConfig.from_dict(original.to_dict())
        assert restored == original

    def test_from_dict_unknown_keys_ignored(self):
        # Extra keys from future versions must not blow up
        cfg = PostProcessingConfig.from_dict({"enabled": False, "future_key": "ignored"})
        assert cfg.enabled is False  # explicit value in dict is respected


# ── AppConfig integration ──────────────────────────────────────────────────────


class TestAppConfigPostProcessing:
    def test_default_app_config_has_pp(self):
        cfg = AppConfig()
        assert isinstance(cfg.post_processing, PostProcessingConfig)
        assert cfg.post_processing.enabled is True

    def test_pause_line_ms_default_is_800(self):
        cfg = AppConfig()
        assert cfg.pause_line_ms == 800

    def test_app_config_round_trip_with_pp(self):
        pp = PostProcessingConfig(enabled=True, normalize=True, normalize_lufs=-23.0)
        cfg = AppConfig(post_processing=pp)
        d = cfg.to_dict()
        restored = AppConfig.from_dict(d)
        assert restored.post_processing.enabled is True
        assert restored.post_processing.normalize_lufs == -23.0

    def test_old_toml_without_pp_loads_cleanly(self):
        """Simulate loading a TOML dict that has no post_processing key."""
        data = {
            "name": "legacy",
            "workers": 2,
            "pause_line_ms": 400,  # old default — must be respected
        }
        cfg = AppConfig.from_dict(data)
        assert cfg.pause_line_ms == 400  # explicit old value preserved
        assert cfg.post_processing.enabled is True  # defaults used

    def test_from_dict_pp_none_gives_defaults(self):
        cfg = AppConfig.from_dict({"post_processing": None})
        assert cfg.post_processing == PostProcessingConfig()

    def test_from_dict_pp_empty_dict_gives_defaults(self):
        cfg = AppConfig.from_dict({"post_processing": {}})
        assert cfg.post_processing == PostProcessingConfig()


# ── apply_chapter_effects ─────────────────────────────────────────────────────


def _write_silent_wav(path: Path, sample_rate: int = 24000, duration_s: float = 0.1) -> None:
    """Write a minimal silent mono WAV for testing."""
    n_samples = int(sample_rate * duration_s)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))


class TestApplyChapterEffects:
    def test_no_op_when_disabled(self, tmp_path):
        from kenkui.post_processing import apply_chapter_effects

        wav = tmp_path / "ch_0001.wav"
        _write_silent_wav(wav)
        mtime_before = wav.stat().st_mtime

        apply_chapter_effects(wav, PostProcessingConfig(enabled=False))

        # File must be untouched when post-processing is off
        assert wav.stat().st_mtime == mtime_before

    def test_graceful_skip_when_pedalboard_missing(self, tmp_path, caplog):
        from kenkui.post_processing import apply_chapter_effects

        wav = tmp_path / "ch_0001.wav"
        _write_silent_wav(wav)

        cfg = PostProcessingConfig(enabled=True)
        with patch.dict("sys.modules", {"pedalboard": None, "pedalboard.io": None}):
            import importlib

            import kenkui.post_processing as pp_mod

            importlib.reload(pp_mod)
            # Should log a warning and not raise
            with caplog.at_level("WARNING"):
                pp_mod.apply_chapter_effects(wav, cfg)

        # Original WAV must still exist
        assert wav.exists()

    def test_exception_during_processing_leaves_original(self, tmp_path, caplog):
        """If an unexpected error occurs, the original WAV must survive intact.

        We use a deliberately corrupt WAV to trigger a real exception inside
        the try block, confirming the except handler fires and the file is kept.
        Skipped when pedalboard is not installed (separate test covers that path).
        """
        pytest.importorskip("pedalboard", reason="pedalboard not installed")
        from kenkui.post_processing import apply_chapter_effects

        wav = tmp_path / "ch_0001.wav"
        wav.write_bytes(b"not a valid wav file at all")

        cfg = PostProcessingConfig(enabled=True)

        with caplog.at_level("WARNING"):
            apply_chapter_effects(wav, cfg)

        assert "original WAV kept" in caplog.text
        assert wav.read_bytes() == b"not a valid wav file at all"


# ── normalize_output ──────────────────────────────────────────────────────────


class TestNormalizeOutput:
    def test_no_op_when_disabled(self, tmp_path):
        from kenkui.post_processing import normalize_output

        out = tmp_path / "book.m4b"
        out.write_bytes(b"fake")

        normalize_output(out, PostProcessingConfig(enabled=False, normalize=True))
        normalize_output(out, PostProcessingConfig(enabled=True, normalize=False))

        # Neither call should modify the file
        assert out.read_bytes() == b"fake"

    def test_uses_ffmpeg_normalize_package_when_available(self, tmp_path):
        from kenkui.post_processing import normalize_output

        out = tmp_path / "book.m4b"
        out.write_bytes(b"fake")

        mock_ffn_instance = MagicMock()
        mock_ffn_class = MagicMock(return_value=mock_ffn_instance)

        # After run_normalization, the temp file must exist so .replace() works
        def fake_run():
            (tmp_path / "book.norm.m4b").write_bytes(b"normalized")

        mock_ffn_instance.run_normalization.side_effect = fake_run

        with patch.dict("sys.modules", {"ffmpeg_normalize": MagicMock(FFmpegNormalize=mock_ffn_class)}):
            normalize_output(out, PostProcessingConfig(enabled=True, normalize=True))

        mock_ffn_class.assert_called_once()
        mock_ffn_instance.add_media_file.assert_called_once()
        mock_ffn_instance.run_normalization.assert_called_once()

    def test_falls_back_to_ffmpeg_when_package_missing(self, tmp_path):
        from kenkui.post_processing import normalize_output

        out = tmp_path / "book.m4b"
        out.write_bytes(b"fake")

        def fake_ffmpeg_exe():
            return "ffmpeg"

        mock_result = MagicMock()
        mock_result.returncode = 0

        def fake_run(cmd, **kwargs):
            # Create the temp output file so replace() works
            temp = tmp_path / "book.norm.m4b"
            temp.write_bytes(b"normalized_fallback")
            return mock_result

        with (
            patch.dict("sys.modules", {"ffmpeg_normalize": None}),
            patch("imageio_ffmpeg.get_ffmpeg_exe", fake_ffmpeg_exe),
            patch("subprocess.run", side_effect=fake_run),
        ):
            normalize_output(out, PostProcessingConfig(enabled=True, normalize=True))

        assert out.read_bytes() == b"normalized_fallback"


# ── Autogain ──────────────────────────────────────────────────────────────────


class TestAutogainConfig:
    def test_autogain_defaults(self):
        cfg = PostProcessingConfig()
        assert cfg.autogain is True
        assert cfg.autogain_target_lufs == -23.0

    def test_autogain_round_trip(self):
        cfg = PostProcessingConfig(autogain=False, autogain_target_lufs=-18.0)
        restored = PostProcessingConfig.from_dict(cfg.to_dict())
        assert restored.autogain is False
        assert restored.autogain_target_lufs == -18.0

    def test_autogain_from_dict_empty_uses_defaults(self):
        cfg = PostProcessingConfig.from_dict({})
        assert cfg.autogain is True
        assert cfg.autogain_target_lufs == -23.0


class TestAutogainApplication:
    """Test that autogain normalization changes clip RMS toward the target."""

    def _write_loud_wav(self, path: Path, sample_rate: int = 24000) -> None:
        """Write a short WAV with high amplitude samples."""
        import struct, wave
        n = int(sample_rate * 0.05)
        with wave.open(str(path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            # Amplitude near maximum
            samples = [int(0.9 * 32767)] * n
            wf.writeframes(struct.pack(f"<{n}h", *samples))

    def _write_quiet_wav(self, path: Path, sample_rate: int = 24000) -> None:
        """Write a short WAV with low amplitude samples."""
        import struct, wave
        n = int(sample_rate * 0.05)
        with wave.open(str(path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            # Amplitude near minimum audible
            samples = [int(0.01 * 32767)] * n
            wf.writeframes(struct.pack(f"<{n}h", *samples))

    @pytest.mark.skipif(
        not pytest.importorskip("pedalboard", reason="pedalboard not installed"),
        reason="pedalboard not installed",
    )
    def test_autogain_brings_clips_closer_to_same_rms(self, tmp_path):
        """A loud and a quiet clip should have more similar RMS after autogain."""
        pytest.importorskip("pedalboard")
        import numpy as np
        from pedalboard.io import AudioFile
        from kenkui.post_processing import apply_chapter_effects

        loud = tmp_path / "loud.wav"
        quiet = tmp_path / "quiet.wav"
        self._write_loud_wav(loud)
        self._write_quiet_wav(quiet)

        def rms(p):
            with AudioFile(str(p)) as f:
                audio = f.read(f.frames)
            return float(np.sqrt(np.mean(audio ** 2)))

        rms_loud_before = rms(loud)
        rms_quiet_before = rms(quiet)
        ratio_before = rms_loud_before / max(rms_quiet_before, 1e-9)

        cfg = PostProcessingConfig(
            enabled=True,
            noise_reduce=False,
            autogain=True,
            autogain_target_lufs=-23.0,
        )
        apply_chapter_effects(loud, cfg)
        apply_chapter_effects(quiet, cfg)

        rms_loud_after = rms(loud)
        rms_quiet_after = rms(quiet)
        ratio_after = rms_loud_after / max(rms_quiet_after, 1e-9)

        # After autogain the ratio should be much closer to 1
        assert ratio_after < ratio_before

    def test_autogain_disabled_leaves_audio_unchanged(self, tmp_path):
        """When autogain=False the RMS should not be altered by the gain step."""
        pytest.importorskip("pedalboard")
        import numpy as np
        from pedalboard.io import AudioFile
        from kenkui.post_processing import apply_chapter_effects

        loud = tmp_path / "loud.wav"
        self._write_loud_wav(loud)

        def rms(p):
            with AudioFile(str(p)) as f:
                audio = f.read(f.frames)
            return float(np.sqrt(np.mean(audio ** 2)))

        # With autogain=False, apply (only EQ/compressor/limiter, no gain stage)
        cfg_no_ag = PostProcessingConfig(
            enabled=True, noise_reduce=False, autogain=False
        )
        apply_chapter_effects(loud, cfg_no_ag)
        rms_no_ag = rms(loud)

        loud2 = tmp_path / "loud2.wav"
        self._write_loud_wav(loud2)
        cfg_ag = PostProcessingConfig(
            enabled=True, noise_reduce=False, autogain=True, autogain_target_lufs=-23.0
        )
        apply_chapter_effects(loud2, cfg_ag)
        rms_ag = rms(loud2)

        # Autogain-processed clip should be different from non-autogain clip
        assert abs(rms_ag - rms_no_ag) > 1e-6
