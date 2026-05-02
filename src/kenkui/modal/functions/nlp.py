"""Modal function: nlp_entity_clustering (Stage 1-2 NLP)."""

from __future__ import annotations

import modal  # type: ignore[import]

from ..app import app
from ..images import nlp_image
from ..storage import BotoStorageBackend, nlp_roster_key, progress_key
from ..volumes import VOLUME_MOUNTS


@app.function(
    image=nlp_image,
    gpu="T4",
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("kenkui-secrets")],
    timeout=3600,
)
def nlp_entity_clustering(payload: dict) -> dict:
    """Extract entities, build character roster, write roster.json to R2.

    Idempotent: returns existing roster.json if already present.
    """
    storage = BotoStorageBackend()
    job_id: str = payload["job_id"]
    roster_key_val = nlp_roster_key(job_id)

    if storage.exists(roster_key_val):
        return {"status": "completed", "artifact_key": roster_key_val}

    storage.put_json(progress_key(job_id), {"status": "processing", "progress": 0.0})

    try:
        chapters = payload["chapters"]
        nlp_provider = payload.get("nlp_provider", "ollama")
        nlp_model = payload.get("nlp_model", "llama3.2")

        def _on_progress(pct: int, msg: str) -> None:
            storage.put_json(progress_key(job_id), {
                "status": "processing",
                "progress": float(pct),
                "current_chapter": msg,
            })

        from kenkui.models import AppConfig as _AppConfig, Chapter as _Chapter  # type: ignore[import]
        from kenkui.nlp.providers import get_provider  # type: ignore[import]

        app_cfg = _AppConfig.from_dict({
            "nlp_provider": nlp_provider,
            "nlp_model": nlp_model,
        })
        provider = get_provider(app_cfg)

        chapter_objs = [_Chapter.from_dict(ch) for ch in chapters]
        roster = provider.build_roster(chapter_objs)

        roster_data = roster.model_dump(mode="json")
        storage.put_json(roster_key_val, roster_data)
        storage.put_json(progress_key(job_id), {
            "status": "completed",
            "progress": 100.0,
            "artifact_key": roster_key_val,
        })
        return {"status": "completed", "artifact_key": roster_key_val}

    except Exception as exc:
        storage.put_json(progress_key(job_id), {"status": "failed", "error": str(exc)})
        raise
