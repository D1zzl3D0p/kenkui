"""Modal function: quote_attribution (Stage 3-4 NLP)."""

from __future__ import annotations

import modal  # type: ignore[import]

from ..app import app
from ..gpu_tiers import resolve_gpu_tier
from ..images import nlp_image
from ..storage import (
    BotoStorageBackend,
    attribution_chapters_key,
    nlp_roster_key,
    progress_key,
)
from ..volumes import VOLUME_MOUNTS


@app.function(
    image=nlp_image,
    gpu=modal.gpu.Any(),
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("kenkui-secrets")],
    timeout=7200,
)
def quote_attribution(payload: dict) -> dict:
    """Run speaker attribution for each chapter, write chapters.json to R2.

    Idempotent: returns existing chapters.json if already present.
    Reads roster.json from R2 (written by nlp_entity_clustering).
    """
    storage = BotoStorageBackend()
    job_id: str = payload["job_id"]
    chapters_key_val = attribution_chapters_key(job_id)

    if storage.exists(chapters_key_val):
        return {"status": "completed", "artifact_key": chapters_key_val}

    storage.put_json(progress_key(job_id), {"status": "processing", "progress": 0.0})

    try:
        nlp_provider = payload.get("nlp_provider", "ollama")
        nlp_model = payload.get("nlp_model", "llama3.2")
        chapters = payload["chapters"]

        roster_data = storage.get_json(nlp_roster_key(job_id))

        def _on_progress(pct: int, msg: str) -> None:
            storage.put_json(progress_key(job_id), {
                "status": "processing",
                "progress": float(pct),
                "current_chapter": msg,
            })

        import pathlib
        import tempfile

        from kenkui.models import Chapter  # type: ignore[import]
        from kenkui.nlp.models import CharacterRoster  # type: ignore[import]
        from kenkui.services.nlp_service import attribute_only  # type: ignore[import]

        roster = CharacterRoster.model_validate(roster_data)
        chapter_objs = [Chapter.from_dict(ch) for ch in chapters]

        tmp_ebook = pathlib.Path(tempfile.mktemp(suffix=".epub"))
        tmp_ebook.touch()
        try:
            nlp_result = attribute_only(
                roster=roster,
                chapters=chapter_objs,
                ebook_path=str(tmp_ebook),
                nlp_model=nlp_model,
                nlp_provider=nlp_provider,
                progress_callback=_on_progress,
            )
        finally:
            tmp_ebook.unlink(missing_ok=True)

        chapters_data = [ch.to_dict() for ch in nlp_result.chapters]
        storage.put_json(chapters_key_val, {"chapters": chapters_data})
        storage.put_json(progress_key(job_id), {
            "status": "completed",
            "progress": 100.0,
            "artifact_key": chapters_key_val,
        })
        return {"status": "completed", "artifact_key": chapters_key_val}

    except Exception as exc:
        storage.put_json(progress_key(job_id), {"status": "failed", "error": str(exc)})
        raise
