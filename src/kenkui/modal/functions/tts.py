"""Modal function: tts_inference (pocket-tts per-chapter WAV generation)."""

from __future__ import annotations

import modal  # type: ignore[import]

from ..app import app
from ..images import tts_image
from ..storage import BotoStorageBackend, progress_key, tts_wav_key
from ..volumes import VOLUME_MOUNTS


@app.function(
    image=tts_image,
    cpu=4,
    memory=8192,
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("kenkui-secrets")],
    timeout=14400,
)
def tts_inference(payload: dict) -> dict:
    """Render each chapter to WAV and write to R2.

    Idempotent: skips chapters whose WAV already exists in R2.
    """
    storage = BotoStorageBackend()
    job_id: str = payload["job_id"]
    chapters: list = payload["chapters"]
    config: dict = payload["config"]
    voice_manifest: dict = payload["voice_manifest"]

    storage.put_json(progress_key(job_id), {"status": "processing", "progress": 0.0})

    try:
        import pathlib
        import tempfile

        from kenkui.workers import worker_process_chapter  # type: ignore[import]

        total = len(chapters)
        for idx, chapter_dict in enumerate(chapters):
            wav_key = tts_wav_key(job_id, idx)
            if storage.exists(wav_key):
                continue

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                out_path = tmp.name

            worker_process_chapter(
                chapter_dict=chapter_dict,
                output_path=out_path,
                voice_manifest=voice_manifest,
                config=config,
            )
            storage.put_bytes(wav_key, pathlib.Path(out_path).read_bytes())
            pathlib.Path(out_path).unlink(missing_ok=True)

            pct = (idx + 1) / total * 100.0
            storage.put_json(progress_key(job_id), {
                "status": "processing",
                "progress": pct,
                "current_chapter": chapter_dict.get("title", f"Chapter {idx}"),
            })

        storage.put_json(progress_key(job_id), {
            "status": "completed",
            "progress": 100.0,
            "artifact_key": f"jobs/{job_id}/tts/",
        })
        return {"status": "completed"}

    except Exception as exc:
        storage.put_json(progress_key(job_id), {"status": "failed", "error": str(exc)})
        raise
