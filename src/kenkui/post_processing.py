"""Audio post-processing: noise reduction, EQ, dynamics, loudness normalization.

This module is fully optional — if ``pedalboard`` or ``noisereduce`` are not
installed, the per-chapter effects are silently skipped and the original WAV
is kept intact.  ``ffmpeg-normalize`` package is also optional; when absent,
loudness normalization falls back to ffmpeg's built-in ``loudnorm`` filter
(available via the ``imageio-ffmpeg`` dependency that kenkui already requires).

Install optional deps:
    pip install "kenkui[post-processing]"
  or individually:
    pip install pedalboard noisereduce ffmpeg-normalize
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import PostProcessingConfig

logger = logging.getLogger(__name__)


def apply_chapter_effects(wav_path: Path, cfg: "PostProcessingConfig") -> None:
    """Apply noisereduce + pedalboard effects chain to a chapter WAV file in-place.

    Called from worker subprocesses after each chapter WAV is written.
    Gracefully degrades if libraries are missing. On any per-file error
    the original WAV is left untouched — no audio data is ever lost.
    """
    if not cfg.enabled:
        return

    try:
        import numpy as np
        from pedalboard import (  # type: ignore[import]
            Compressor,
            HighpassFilter,
            Limiter,
            LowShelfFilter,
            Pedalboard,
            PeakFilter,
        )
        from pedalboard.io import AudioFile  # type: ignore[import]
    except ImportError as exc:
        logger.warning(
            "post_processing: pedalboard not installed — chapter effects skipped (%s). "
            "Install with: pip install pedalboard",
            exc,
        )
        return

    try:
        # 1. Read WAV → float32 [channels, samples]
        with AudioFile(str(wav_path)) as f:
            audio = f.read(f.frames)
            sample_rate = f.samplerate

        # 2. Noise reduction (optional, independent of pedalboard)
        if cfg.noise_reduce:
            try:
                import noisereduce as nr  # type: ignore[import]

                # noisereduce expects [samples] (mono) or [samples, channels]
                # pedalboard gives us [channels, samples] — transpose first
                audio_t = audio.T
                if audio_t.ndim == 2 and audio_t.shape[1] == 1:
                    audio_t = audio_t[:, 0]  # squeeze to 1-D for mono
                reduced = nr.reduce_noise(
                    y=audio_t,
                    sr=sample_rate,
                    prop_decrease=cfg.noise_reduce_prop_decrease,
                )
                if reduced.ndim == 1:
                    audio = reduced[np.newaxis, :]  # back to [1, samples]
                else:
                    audio = reduced.T
            except ImportError:
                logger.debug("noisereduce not installed — skipping noise reduction step")

        # 3. Build pedalboard effects chain
        effects = [
            HighpassFilter(cutoff_frequency_hz=float(cfg.highpass_hz)),
            LowShelfFilter(
                cutoff_frequency_hz=float(cfg.lowshelf_hz),
                gain_db=float(cfg.lowshelf_db),
                q=1.0,
            ),
            PeakFilter(
                cutoff_frequency_hz=float(cfg.presence_hz),
                gain_db=float(cfg.presence_db),
                q=1.0,
            ),
        ]
        if cfg.deesser:
            effects.append(
                PeakFilter(
                    cutoff_frequency_hz=float(cfg.deesser_hz),
                    gain_db=float(cfg.deesser_db),
                    q=2.0,
                )
            )
        effects.extend(
            [
                Compressor(
                    threshold_db=float(cfg.compressor_threshold_db),
                    ratio=float(cfg.compressor_ratio),
                    attack_ms=float(cfg.compressor_attack_ms),
                    release_ms=float(cfg.compressor_release_ms),
                ),
                Limiter(threshold_db=float(cfg.limiter_threshold_db)),
            ]
        )

        # 4. Process audio through the chain
        processed = Pedalboard(effects)(audio, sample_rate)

        # 5. Autogain — normalize clip RMS to a common target level
        if cfg.autogain:
            rms = float(np.sqrt(np.mean(processed ** 2)))
            if rms > 1e-9:
                target_linear = 10 ** (cfg.autogain_target_lufs / 20)
                gain_db = float(np.clip(20 * np.log10(target_linear / rms), -24.0, 24.0))
                from pedalboard import Gain  # type: ignore[import]
                processed = Pedalboard([Gain(gain_db=gain_db), Limiter(threshold_db=cfg.limiter_threshold_db)])(
                    processed, sample_rate
                )

        # 6. Write back in-place (same path, same sample rate and channel count)
        with AudioFile(str(wav_path), "w", sample_rate, processed.shape[0]) as f:
            f.write(processed)

        logger.debug("post_processing: applied effects to %s", wav_path.name)

    except Exception as exc:
        logger.warning(
            "post_processing: apply_chapter_effects failed on %s (%s) — original WAV kept",
            wav_path.name,
            exc,
        )


def normalize_output(output_path: Path, cfg: "PostProcessingConfig") -> None:
    """Loudness-normalize the final M4B/MP3 output file.

    Tries the ``ffmpeg-normalize`` Python package first (accurate two-pass
    EBU R128 or peak normalization).  Falls back to ffmpeg's built-in
    ``loudnorm`` filter when the package is absent.

    Chapter markers, cover art, and all other metadata are preserved via
    ``-map_metadata 0 -map_chapters 0`` in both code paths.
    """
    if not cfg.enabled or not cfg.normalize:
        return

    try:
        _normalize_with_package(output_path, cfg)
    except ImportError:
        logger.info(
            "post_processing: ffmpeg-normalize package not installed — "
            "using ffmpeg loudnorm fallback. "
            "For more accurate results: pip install ffmpeg-normalize"
        )
        _normalize_with_ffmpeg(output_path, cfg)


# ── Private helpers ────────────────────────────────────────────────────────────


def _normalize_with_package(output_path: Path, cfg: "PostProcessingConfig") -> None:
    """Two-pass normalization via the ffmpeg-normalize Python package."""
    from ffmpeg_normalize import FFmpegNormalize  # type: ignore[import]

    suffix = output_path.suffix
    codec = "aac" if suffix == ".m4b" else "libmp3lame"
    temp_out = output_path.with_suffix(".norm" + suffix)

    norm_type = "ebu" if cfg.normalize_lufs is not None else "peak"
    target = cfg.normalize_lufs if cfg.normalize_lufs is not None else cfg.normalize_target_db

    ffn = FFmpegNormalize(
        normalization_type=norm_type,
        target_level=float(target),
        audio_codec=codec,
        extra_output_options=["-map_metadata", "0", "-map_chapters", "0"],
        progress=False,
        force_input_reset=True,
    )
    ffn.add_media_file(str(output_path), str(temp_out))
    ffn.run_normalization()
    temp_out.replace(output_path)
    logger.info(
        "post_processing: normalized %s (%s → %.1f)",
        output_path.name,
        norm_type,
        target,
    )


def _normalize_with_ffmpeg(output_path: Path, cfg: "PostProcessingConfig") -> None:
    """Single-pass normalization via ffmpeg loudnorm (bundled with imageio-ffmpeg)."""
    import subprocess

    import imageio_ffmpeg  # already a required dep

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    suffix = output_path.suffix
    codec = "aac" if suffix == ".m4b" else "libmp3lame"
    temp_out = output_path.with_suffix(".norm" + suffix)

    target_i = cfg.normalize_lufs if cfg.normalize_lufs is not None else -23.0
    target_tp = cfg.normalize_target_db

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-v",
        "error",
        "-i",
        str(output_path),
        "-af",
        f"loudnorm=I={target_i}:TP={target_tp}:LRA=11",
        "-c:a",
        codec,
        "-map_metadata",
        "0",
        "-map_chapters",
        "0",
        str(temp_out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg loudnorm failed: {result.stderr}")
    temp_out.replace(output_path)
    logger.info(
        "post_processing: normalized %s (loudnorm I=%.1f TP=%.1f)",
        output_path.name,
        target_i,
        target_tp,
    )
