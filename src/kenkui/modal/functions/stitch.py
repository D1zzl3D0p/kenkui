"""Modal function: audio_stitch (ffmpeg M4B assembly)."""

from __future__ import annotations

import modal  # type: ignore[import]

from ..app import app
from ..images import tts_image
from ..storage import BotoStorageBackend, output_m4b_key, progress_key, tts_wav_key
from ..volumes import VOLUME_MOUNTS


@app.function(
    image=tts_image,
    cpu=2,
    memory=4096,
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("kenkui-secrets")],
    timeout=3600,
)
def audio_stitch(payload: dict) -> dict:
    """Assemble per-chapter WAVs from R2 into a single M4B.

    Idempotent: returns existing output.m4b artifact_key if already present.
    """
    import pathlib
    import subprocess
    import tempfile

    storage = BotoStorageBackend()
    job_id: str = payload["job_id"]
    chapter_count: int = payload["chapter_count"]
    m4b_bitrate: str = payload.get("m4b_bitrate", "96k")
    title: str = payload.get("title", "")
    m4b_key = output_m4b_key(job_id)

    if storage.exists(m4b_key):
        return {"status": "completed", "artifact_key": m4b_key}

    storage.put_json(progress_key(job_id), {"status": "processing", "progress": 0.0})

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = pathlib.Path(tmp_dir)
            wav_paths: list[pathlib.Path] = []

            for idx in range(chapter_count):
                wav_data = storage.get_bytes(tts_wav_key(job_id, idx))
                p = tmp / f"ch_{idx:04d}.wav"
                p.write_bytes(wav_data)
                wav_paths.append(p)
                pct = (idx + 1) / chapter_count * 50.0
                storage.put_json(progress_key(job_id), {
                    "status": "processing",
                    "progress": pct,
                    "current_chapter": f"Downloading ch {idx}",
                })

            import imageio_ffmpeg  # type: ignore[import]

            concat_list = tmp / "concat.txt"
            concat_list.write_text(
                "\n".join(f"file '{p}'" for p in wav_paths), encoding="utf-8"
            )
            out_path = tmp / "output.m4b"
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg, "-y", "-v", "error",
                "-f", "concat", "-safe", "0", "-i", str(concat_list),
                "-c:a", "aac", "-b:a", m4b_bitrate,
                "-metadata", f"title={title}",
                str(out_path),
            ]
            subprocess.run(cmd, check=True)

            m4b_bytes = out_path.read_bytes()
            storage.put_bytes(m4b_key, m4b_bytes)

        storage.put_json(progress_key(job_id), {
            "status": "completed",
            "progress": 100.0,
            "artifact_key": m4b_key,
        })
        return {"status": "completed", "artifact_key": m4b_key}

    except Exception as exc:
        storage.put_json(progress_key(job_id), {"status": "failed", "error": str(exc)})
        raise
