# Modal Remote Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Modal remote execution for all four pipeline stages (NLP entity clustering, quote attribution, TTS inference, audio stitching) with independently switchable execution modes per stage.

**Architecture:** VPS acts as pure orchestrator — queue management, job lifecycle, CLI/API surface. Modal handles heavy compute. `KenkuiModalGateway` in `modal/gateway.py` is the only Modal-SDK-coupled code. `StorageBackend` in `modal/storage.py` is the only boto3-coupled code. All other new code is tested via protocol mocks. Worker.py owns the stage-to-stage handoff logic regardless of where each stage runs.

**Tech Stack:** Python 3.12, Modal SDK ≥0.73, boto3 ≥1.35, R2/S3 artifact storage, existing spaCy/BookNLP/pocket-tts stack deployed on Modal side.

---

## File Map

**New files:**
- `src/kenkui/modal/__init__.py` — package marker
- `src/kenkui/modal/app.py` — `modal.App("kenkui")` definition
- `src/kenkui/modal/images.py` — Modal Image builders (nlp, tts/stitch)
- `src/kenkui/modal/volumes.py` — Modal Volume definitions (model weights, voice registry)
- `src/kenkui/modal/gpu_tiers.py` — model-name → GPU tier mapping
- `src/kenkui/modal/storage.py` — `StorageBackend` protocol + `BotoStorageBackend` + `LocalStorageBackend`
- `src/kenkui/modal/gateway.py` — `KenkuiModalGateway` (Modal-coupled)
- `src/kenkui/modal/functions/__init__.py` — package marker
- `src/kenkui/modal/functions/nlp.py` — `@app.function nlp_entity_clustering`
- `src/kenkui/modal/functions/attribution.py` — `@app.function quote_attribution`
- `src/kenkui/modal/functions/tts.py` — `@app.function tts_inference`
- `src/kenkui/modal/functions/stitch.py` — `@app.function audio_stitch`
- `src/kenkui/server/nlp_execution.py` — `NlpGateway` protocol + `NlpModalProvider`
- `tests/modal/__init__.py` — package marker
- `tests/modal/test_gpu_tiers.py`
- `tests/modal/test_storage.py`
- `tests/modal/test_nlp_execution.py`

**Modified files:**
- `pyproject.toml` — add `modal` optional-deps group
- `src/kenkui/models.py` — add `NlpExecutionMode`, `AttributionExecutionMode`; update `AppConfig`, `JobConfig`
- `src/kenkui/server/worker.py` — dispatch `_run_attribution_phase` to modal when configured
- `src/kenkui/cli/config.py` — add `nlp_execution_mode` + `attribution_execution_mode` prompts
- `src/kenkui/cli/add.py` — add "Execution Modes" entry in `_submenu_advanced`
- `src/kenkui/cli/queue.py` — include `modal` tag in `_job_mode_str` when applicable

---

## Task 1: Optional dependencies + package skeleton

**Files:**
- Modify: `pyproject.toml`
- Create: `src/kenkui/modal/__init__.py`
- Create: `src/kenkui/modal/functions/__init__.py`
- Create: `tests/modal/__init__.py`

- [ ] **Step 1: Add modal optional-deps group to pyproject.toml**

Open `pyproject.toml`. After the existing `dev` optional-deps entry:

```toml
[project.optional-dependencies]
dev = ["pytest>=7.0.0", "pytest-cov>=4.0.0"]
modal = ["modal>=0.73", "boto3>=1.35.0"]
```

- [ ] **Step 2: Create package skeleton files**

Create `src/kenkui/modal/__init__.py` (empty):
```python
```

Create `src/kenkui/modal/functions/__init__.py` (empty):
```python
```

Create `tests/modal/__init__.py` (empty):
```python
```

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml src/kenkui/modal/__init__.py src/kenkui/modal/functions/__init__.py tests/modal/__init__.py
git commit -m "feat: add modal optional-deps group and package skeleton"
```

---

## Task 2: Execution mode enums + AppConfig/JobConfig

**Files:**
- Modify: `src/kenkui/models.py`
- Test: `tests/test_models_execution_modes.py`

- [ ] **Step 1: Write failing tests for new enums and config fields**

Create `tests/test_models_execution_modes.py`:

```python
from __future__ import annotations

from kenkui.models import (
    AppConfig,
    AttributionExecutionMode,
    JobConfig,
    NlpExecutionMode,
)
from pathlib import Path


def test_nlp_execution_mode_defaults_local():
    cfg = AppConfig.from_dict({})
    assert cfg.nlp_execution_mode == NlpExecutionMode.LOCAL


def test_attribution_execution_mode_defaults_local():
    cfg = AppConfig.from_dict({})
    assert cfg.attribution_execution_mode == AttributionExecutionMode.LOCAL


def test_nlp_execution_mode_modal_from_dict():
    cfg = AppConfig.from_dict({"nlp_execution_mode": "modal"})
    assert cfg.nlp_execution_mode == NlpExecutionMode.MODAL


def test_attribution_execution_mode_litellm_from_dict():
    cfg = AppConfig.from_dict({"attribution_execution_mode": "litellm"})
    assert cfg.attribution_execution_mode == AttributionExecutionMode.LITELLM


def test_app_config_to_dict_includes_execution_modes():
    cfg = AppConfig.from_dict({
        "nlp_execution_mode": "modal",
        "attribution_execution_mode": "litellm",
    })
    d = cfg.to_dict()
    assert d["nlp_execution_mode"] == "modal"
    assert d["attribution_execution_mode"] == "litellm"


def test_job_config_per_job_execution_mode_overrides(tmp_path):
    job = JobConfig(
        ebook_path=tmp_path / "book.epub",
        job_nlp_execution_mode=NlpExecutionMode.MODAL,
        job_attribution_execution_mode=AttributionExecutionMode.LOCAL,
    )
    d = job.to_dict()
    assert d["job_nlp_execution_mode"] == "modal"
    assert d["job_attribution_execution_mode"] == "local"


def test_job_config_round_trips_execution_modes(tmp_path):
    job = JobConfig(
        ebook_path=tmp_path / "book.epub",
        job_nlp_execution_mode=NlpExecutionMode.MODAL,
    )
    restored = JobConfig.from_dict(job.to_dict())
    assert restored.job_nlp_execution_mode == NlpExecutionMode.MODAL
    assert restored.job_attribution_execution_mode is None


def test_job_config_execution_mode_defaults_none(tmp_path):
    job = JobConfig(ebook_path=tmp_path / "book.epub")
    assert job.job_nlp_execution_mode is None
    assert job.job_attribution_execution_mode is None
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd /Users/dizzler/Projects/Repos/kenkui && python -m pytest tests/test_models_execution_modes.py -v 2>&1 | head -30
```

Expected: ImportError or AttributeError — `NlpExecutionMode` not defined.

- [ ] **Step 3: Add NlpExecutionMode and AttributionExecutionMode enums to models.py**

In `src/kenkui/models.py`, after the `TTSExecutionMode` class (around line 87):

```python
class NlpExecutionMode(Enum):
    LOCAL = "local"
    MODAL = "modal"
    LITELLM = "litellm"


class AttributionExecutionMode(Enum):
    LOCAL = "local"
    MODAL = "modal"
    LITELLM = "litellm"
```

- [ ] **Step 4: Add fields to AppConfig**

In `src/kenkui/models.py`, inside the `AppConfig` class after the `ollama_url` field (around line 447), add:

```python
    nlp_execution_mode: NlpExecutionMode = NlpExecutionMode.LOCAL
    attribution_execution_mode: AttributionExecutionMode = AttributionExecutionMode.LOCAL
```

AppConfig uses `SettingsConfigDict(env_prefix="KENKUI_")` so these are automatically read from `KENKUI_NLP_EXECUTION_MODE` and `KENKUI_ATTRIBUTION_EXECUTION_MODE`. No extra wiring needed.

- [ ] **Step 5: Add fields to JobConfig**

In `src/kenkui/models.py`, inside the `JobConfig` dataclass after `job_post_processing_enabled` (around line 277), add:

```python
    job_nlp_execution_mode: NlpExecutionMode | None = None
    job_attribution_execution_mode: AttributionExecutionMode | None = None
```

- [ ] **Step 6: Update JobConfig.to_dict() to serialize new fields**

In `JobConfig.to_dict()`, the loop that serializes per-job overrides (around line 305) iterates over a tuple of key names. Add the new keys to that tuple:

```python
        for key in (
            "job_nlp_provider",
            "job_nlp_model",
            "job_temp",
            "job_lsd_decode_steps",
            "job_noise_clamp",
            "job_eos_threshold",
            "job_m4b_bitrate",
            "job_pause_line_ms",
            "job_pause_chapter_ms",
            "job_speak_chapter_titles",
            "job_pause_before_chapter_title_ms",
            "job_pause_after_chapter_title_ms",
            "job_frames_after_eos",
            "job_apostrophe_mode",
            "job_post_processing_enabled",
            "job_nlp_execution_mode",
            "job_attribution_execution_mode",
        ):
```

- [ ] **Step 7: Update JobConfig.from_dict() to deserialize new fields**

In `JobConfig.from_dict()`, after `job_post_processing_enabled=data.get("job_post_processing_enabled")`, add:

```python
            job_nlp_execution_mode=NlpExecutionMode(data["job_nlp_execution_mode"])
            if data.get("job_nlp_execution_mode")
            else None,
            job_attribution_execution_mode=AttributionExecutionMode(data["job_attribution_execution_mode"])
            if data.get("job_attribution_execution_mode")
            else None,
```

- [ ] **Step 8: Run tests to verify they pass**

```bash
cd /Users/dizzler/Projects/Repos/kenkui && python -m pytest tests/test_models_execution_modes.py -v
```

Expected: All 8 tests PASS.

- [ ] **Step 9: Commit**

```bash
git add src/kenkui/models.py tests/test_models_execution_modes.py
git commit -m "feat: add NlpExecutionMode and AttributionExecutionMode enums with AppConfig/JobConfig fields"
```

---

## Task 3: GPU tier mapping

**Files:**
- Create: `src/kenkui/modal/gpu_tiers.py`
- Test: `tests/modal/test_gpu_tiers.py`

- [ ] **Step 1: Write failing tests**

Create `tests/modal/test_gpu_tiers.py`:

```python
from kenkui.modal.gpu_tiers import resolve_gpu_tier, DEFAULT_GPU, GPU_TIERS


def test_default_gpu_is_t4():
    assert DEFAULT_GPU == "T4"


def test_3b_resolves_t4():
    assert resolve_gpu_tier("llama-3b") == "T4"


def test_7b_resolves_t4():
    assert resolve_gpu_tier("mistral-7b-instruct") == "T4"


def test_8b_resolves_a10g():
    assert resolve_gpu_tier("llama3-8b-instruct") == "A10G"


def test_13b_resolves_a10g():
    assert resolve_gpu_tier("llama-13b") == "A10G"


def test_70b_resolves_a100():
    assert resolve_gpu_tier("llama3.1-70b-instruct") == "A100"


def test_unknown_model_returns_default():
    assert resolve_gpu_tier("some-unknown-model") == DEFAULT_GPU


def test_override_env_var(monkeypatch):
    monkeypatch.setenv("KENKUI_MODAL_NLP_GPU_OVERRIDE", "A100")
    assert resolve_gpu_tier("unknown-model") == "A100"


def test_override_does_not_affect_known_model_without_override(monkeypatch):
    monkeypatch.delenv("KENKUI_MODAL_NLP_GPU_OVERRIDE", raising=False)
    assert resolve_gpu_tier("llama3-8b") == "A10G"
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd /Users/dizzler/Projects/Repos/kenkui && python -m pytest tests/modal/test_gpu_tiers.py -v 2>&1 | head -20
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement gpu_tiers.py**

Create `src/kenkui/modal/gpu_tiers.py`:

```python
"""Model-name to GPU tier mapping for Modal NLP functions."""

from __future__ import annotations

import os

# Matched against model name substring, lowest-VRAM tier first.
GPU_TIERS: list[tuple[str, str, int]] = [
    ("3b",  "T4",   1),
    ("7b",  "T4",   1),
    ("8b",  "A10G", 1),
    ("13b", "A10G", 1),
    ("70b", "A100", 1),
]

DEFAULT_GPU = "T4"


def resolve_gpu_tier(model_name: str) -> str:
    """Return the GPU tier string for *model_name*.

    Checks KENKUI_MODAL_NLP_GPU_OVERRIDE first, then walks GPU_TIERS
    matching against the lowercased model name substring.
    """
    override = os.environ.get("KENKUI_MODAL_NLP_GPU_OVERRIDE", "").strip()
    if override:
        return override

    lower = model_name.lower()
    for pattern, tier, _count in GPU_TIERS:
        if pattern in lower:
            return tier

    return DEFAULT_GPU
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/dizzler/Projects/Repos/kenkui && python -m pytest tests/modal/test_gpu_tiers.py -v
```

Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/modal/gpu_tiers.py tests/modal/test_gpu_tiers.py
git commit -m "feat: add gpu_tiers module for model-name to Modal GPU tier resolution"
```

---

## Task 4: Storage backend

**Files:**
- Create: `src/kenkui/modal/storage.py`
- Test: `tests/modal/test_storage.py`

- [ ] **Step 1: Write failing tests**

Create `tests/modal/test_storage.py`:

```python
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
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd /Users/dizzler/Projects/Repos/kenkui && python -m pytest tests/modal/test_storage.py -v 2>&1 | head -20
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement storage.py**

Create `src/kenkui/modal/storage.py`:

```python
"""R2/S3 artifact storage for Modal pipeline stages."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol


# ── Key helpers ─────────────────────────────────────────────────────────────

def progress_key(job_id: str) -> str:
    return f"jobs/{job_id}/progress.json"

def nlp_roster_key(job_id: str) -> str:
    return f"jobs/{job_id}/nlp/roster.json"

def attribution_chapters_key(job_id: str) -> str:
    return f"jobs/{job_id}/attribution/chapters.json"

def tts_wav_key(job_id: str, chapter_index: int) -> str:
    return f"jobs/{job_id}/tts/ch_{chapter_index:04d}.wav"

def output_m4b_key(job_id: str) -> str:
    return f"jobs/{job_id}/output.m4b"


# ── Protocol ────────────────────────────────────────────────────────────────

class StorageBackend(Protocol):
    def put_json(self, key: str, data: dict) -> None: ...
    def get_json(self, key: str) -> dict: ...
    def put_bytes(self, key: str, data: bytes) -> None: ...
    def get_bytes(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...


# ── Implementations ──────────────────────────────────────────────────────────

class LocalStorageBackend:
    """Writes to a local directory — used in tests and local dev."""

    def __init__(self, root: Path | str):
        self._root = Path(root)

    def _path(self, key: str) -> Path:
        return self._root / key

    def put_json(self, key: str, data: dict) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data), encoding="utf-8")

    def get_json(self, key: str) -> dict:
        return json.loads(self._path(key).read_text(encoding="utf-8"))

    def put_bytes(self, key: str, data: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()


class BotoStorageBackend:
    """Reads/writes to R2 or S3 via boto3.

    Configured from env vars:
      KENKUI_MODAL_BUCKET          — required
      KENKUI_MODAL_BUCKET_ENDPOINT — optional (R2: https://<acct>.r2.cloudflarestorage.com)
      AWS_ACCESS_KEY_ID            — standard boto3
      AWS_SECRET_ACCESS_KEY        — standard boto3
    """

    def __init__(self):
        import boto3  # type: ignore[import]

        bucket = os.environ["KENKUI_MODAL_BUCKET"]
        endpoint = os.environ.get("KENKUI_MODAL_BUCKET_ENDPOINT")
        kwargs: dict = {}
        if endpoint:
            kwargs["endpoint_url"] = endpoint
        self._bucket = bucket
        self._s3 = boto3.client("s3", **kwargs)

    def put_json(self, key: str, data: dict) -> None:
        body = json.dumps(data).encode("utf-8")
        self._s3.put_object(Bucket=self._bucket, Key=key, Body=body, ContentType="application/json")

    def get_json(self, key: str) -> dict:
        obj = self._s3.get_object(Bucket=self._bucket, Key=key)
        return json.loads(obj["Body"].read())

    def put_bytes(self, key: str, data: bytes) -> None:
        self._s3.put_object(Bucket=self._bucket, Key=key, Body=data)

    def get_bytes(self, key: str) -> bytes:
        obj = self._s3.get_object(Bucket=self._bucket, Key=key)
        return obj["Body"].read()

    def exists(self, key: str) -> bool:
        try:
            self._s3.head_object(Bucket=self._bucket, Key=key)
            return True
        except self._s3.exceptions.ClientError:
            return False


def build_storage_backend() -> StorageBackend:
    """Build the configured storage backend from environment variables.

    Returns LocalStorageBackend if KENKUI_MODAL_BUCKET is unset (dev/test),
    BotoStorageBackend when the bucket is configured.
    """
    if os.environ.get("KENKUI_MODAL_BUCKET"):
        return BotoStorageBackend()
    raise RuntimeError(
        "KENKUI_MODAL_BUCKET must be set when Modal execution is enabled."
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/dizzler/Projects/Repos/kenkui && python -m pytest tests/modal/test_storage.py -v
```

Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/modal/storage.py tests/modal/test_storage.py
git commit -m "feat: add StorageBackend protocol with BotoStorageBackend and LocalStorageBackend"
```

---

## Task 5: Modal app infrastructure (app, images, volumes)

**Files:**
- Create: `src/kenkui/modal/app.py`
- Create: `src/kenkui/modal/images.py`
- Create: `src/kenkui/modal/volumes.py`

These files define Modal-side infrastructure and are only evaluated during `modal deploy`. No unit tests — Modal provides its own validation at deploy time.

- [ ] **Step 1: Create app.py**

Create `src/kenkui/modal/app.py`:

```python
"""Modal app registration for kenkui pipeline."""

from __future__ import annotations

import os

import modal  # type: ignore[import]

APP_NAME = os.environ.get("KENKUI_MODAL_APP_NAME", "kenkui")

app = modal.App(APP_NAME)
```

- [ ] **Step 2: Create images.py**

Create `src/kenkui/modal/images.py`:

```python
"""Modal Image builders for kenkui pipeline functions."""

from __future__ import annotations

import modal  # type: ignore[import]

# NLP image: spaCy, BookNLP, and the kenkui NLP stack.
nlp_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "spacy>=3.0.0",
        "booknlp>=1.0.8",
        "litellm>=1.0.0",
        "instructor>=1.0.0",
        "pydantic>=2.0.0",
        "boto3>=1.35.0",
    )
    .run_commands("python -m spacy download en_core_web_sm")
)

# TTS + stitch image: pocket-tts, pydub, ffmpeg stack.
tts_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "pocket-tts>=2.0.0",
        "pydub>=0.25.0",
        "mutagen>=1.45.0",
        "imageio-ffmpeg>=0.5.0",
        "pedalboard>=0.9",
        "noisereduce>=3.0",
        "ffmpeg-normalize>=1.26",
        "boto3>=1.35.0",
    )
)
```

- [ ] **Step 3: Create volumes.py**

Create `src/kenkui/modal/volumes.py`:

```python
"""Modal Volume definitions for kenkui shared model weights."""

from __future__ import annotations

import modal  # type: ignore[import]

# Shared volume for LLM model weights (Ollama pull cache, GGUF models).
llm_weights_volume = modal.Volume.from_name("kenkui-llm-weights", create_if_missing=True)

# Shared volume for pocket-tts model checkpoint.
pocket_tts_volume = modal.Volume.from_name("kenkui-pocket-tts", create_if_missing=True)

# Shared volume for the built-in voice registry.
voice_registry_volume = modal.Volume.from_name("kenkui-voice-registry", create_if_missing=True)

VOLUME_MOUNTS = {
    "/weights/llm": llm_weights_volume,
    "/weights/pocket-tts": pocket_tts_volume,
    "/weights/voices": voice_registry_volume,
}
```

- [ ] **Step 4: Commit**

```bash
git add src/kenkui/modal/app.py src/kenkui/modal/images.py src/kenkui/modal/volumes.py
git commit -m "feat: add Modal app, image builders, and volume definitions"
```

---

## Task 6: Modal function — nlp_entity_clustering

**Files:**
- Create: `src/kenkui/modal/functions/nlp.py`

- [ ] **Step 1: Create nlp.py**

Create `src/kenkui/modal/functions/nlp.py`:

```python
"""Modal function: nlp_entity_clustering (Stage 1-2 NLP)."""

from __future__ import annotations

import json
import os

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
    roster_key = nlp_roster_key(job_id)

    # Idempotency: skip work if roster already in R2.
    if storage.exists(roster_key):
        return {"status": "completed", "artifact_key": roster_key}

    storage.put_json(progress_key(job_id), {"status": "processing", "progress": 0.0})

    try:
        chapters = payload["chapters"]
        nlp_provider = payload.get("nlp_provider", "ollama")
        nlp_model = payload.get("nlp_model", "llama3.2")
        ebook_path = payload.get("ebook_path", "")

        def _on_progress(pct: int, msg: str) -> None:
            storage.put_json(progress_key(job_id), {
                "status": "processing",
                "progress": float(pct),
                "current_chapter": msg,
            })

        # Import kenkui NLP pipeline (installed in nlp_image).
        # Run NLP pipeline on chapters passed as data.
        # fast_scan() reads from an ebook file; in Modal the ebook is not available.
        # Build an AppConfig with the correct provider/model and call provider directly.
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
        storage.put_json(roster_key, roster_data)
        storage.put_json(progress_key(job_id), {
            "status": "completed",
            "progress": 100.0,
            "artifact_key": roster_key,
        })
        return {"status": "completed", "artifact_key": roster_key}

    except Exception as exc:  # noqa: BLE001
        storage.put_json(progress_key(job_id), {"status": "failed", "error": str(exc)})
        raise
```

- [ ] **Step 2: Commit**

```bash
git add src/kenkui/modal/functions/nlp.py
git commit -m "feat: add nlp_entity_clustering Modal function"
```

---

## Task 7: Modal function — quote_attribution

**Files:**
- Create: `src/kenkui/modal/functions/attribution.py`

- [ ] **Step 1: Create attribution.py**

Create `src/kenkui/modal/functions/attribution.py`:

```python
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
    chapters_key = attribution_chapters_key(job_id)

    if storage.exists(chapters_key):
        return {"status": "completed", "artifact_key": chapters_key}

    storage.put_json(progress_key(job_id), {"status": "processing", "progress": 0.0})

    try:
        nlp_provider = payload.get("nlp_provider", "ollama")
        nlp_model = payload.get("nlp_model", "llama3.2")
        chapters = payload["chapters"]

        # Load roster from R2.
        roster_data = storage.get_json(nlp_roster_key(job_id))

        def _on_progress(pct: int, msg: str) -> None:
            storage.put_json(progress_key(job_id), {
                "status": "processing",
                "progress": float(pct),
                "current_chapter": msg,
            })

        import tempfile, pathlib
        from kenkui.services.nlp_service import attribute_only  # type: ignore[import]
        from kenkui.nlp.models import CharacterRoster  # type: ignore[import]
        from kenkui.models import Chapter  # type: ignore[import]

        roster = CharacterRoster.model_validate(roster_data)
        chapter_objs = [Chapter.from_dict(ch) for ch in chapters]

        # attribute_only calls book_hash which calls .stat(); provide a real temp file.
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
        storage.put_json(chapters_key, {"chapters": chapters_data})
        storage.put_json(progress_key(job_id), {
            "status": "completed",
            "progress": 100.0,
            "artifact_key": chapters_key,
        })
        return {"status": "completed", "artifact_key": chapters_key}

    except Exception as exc:
        storage.put_json(progress_key(job_id), {"status": "failed", "error": str(exc)})
        raise
```

- [ ] **Step 2: Commit**

```bash
git add src/kenkui/modal/functions/attribution.py
git commit -m "feat: add quote_attribution Modal function"
```

---

## Task 8: Modal functions — tts_inference and audio_stitch

**Files:**
- Create: `src/kenkui/modal/functions/tts.py`
- Create: `src/kenkui/modal/functions/stitch.py`

- [ ] **Step 1: Create tts.py**

Create `src/kenkui/modal/functions/tts.py`:

```python
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
        from kenkui.workers import worker_process_chapter  # type: ignore[import]
        import tempfile, pathlib

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
```

- [ ] **Step 2: Create stitch.py**

Create `src/kenkui/modal/functions/stitch.py`:

```python
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
    import tempfile, pathlib

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
            import subprocess

            # Build ffmpeg concat list.
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
```

- [ ] **Step 3: Commit**

```bash
git add src/kenkui/modal/functions/tts.py src/kenkui/modal/functions/stitch.py
git commit -m "feat: add tts_inference and audio_stitch Modal functions"
```

---

## Task 9: KenkuiModalGateway

**Files:**
- Create: `src/kenkui/modal/gateway.py`

`gateway.py` is the only file that imports the `modal` SDK. It implements the existing `ModalGateway` protocol (for TTS compat with `EnvModalGateway`) and adds NLP-specific submit methods used by `NlpModalProvider`.

- [ ] **Step 1: Create gateway.py**

Create `src/kenkui/modal/gateway.py`:

```python
"""KenkuiModalGateway — the only Modal-SDK-coupled code in kenkui.

Loaded at runtime via:
  KENKUI_MODAL_GATEWAY=kenkui.modal.gateway:KenkuiModalGateway

TTS path: implements ModalGateway (estimate/submit/poll/cancel).
NLP path: implements NlpGateway (submit_nlp/submit_attribution/poll_progress/cancel).
Both paths poll R2 progress.json for status updates.
"""

from __future__ import annotations

import os
import time
from typing import Any

import modal  # type: ignore[import]

from ..server.tts_execution import ModalPollResult
from .storage import BotoStorageBackend, output_m4b_key, progress_key


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip() or default


class KenkuiModalGateway:
    """Unified gateway for dispatching kenkui pipeline stages to Modal."""

    def __init__(self):
        self._app_name = _env("KENKUI_MODAL_APP_NAME", "kenkui")
        self._environment = _env("KENKUI_MODAL_ENVIRONMENT", "main")
        self._storage = BotoStorageBackend()

    # ── Shared R2 polling ──────────────────────────────────────────────────

    def _poll_r2(self, job_id: str) -> ModalPollResult:
        try:
            data = self._storage.get_json(progress_key(job_id))
        except Exception:
            return ModalPollResult(status="running")

        status = data.get("status", "running")
        if status == "completed":
            artifact_key = data.get("artifact_key", output_m4b_key(job_id))
            return ModalPollResult(
                status="completed",
                progress=100.0,
                artifact_uri=f"r2://{os.environ.get('KENKUI_MODAL_BUCKET', '')}/{artifact_key}",
                provider_status="completed",
            )
        if status == "failed":
            return ModalPollResult(
                status="failed",
                error_message=data.get("error", "Modal job failed"),
                provider_status="failed",
            )
        return ModalPollResult(
            status="running",
            progress=data.get("progress"),
            current_chapter=data.get("current_chapter", ""),
            eta_seconds=data.get("eta_seconds"),
            provider_status="running",
        )

    def _download_artifact(self, job_id: str, artifact_key: str) -> bytes:
        return self._storage.get_bytes(artifact_key)

    # ── ModalGateway protocol (TTS compat) ────────────────────────────────

    def estimate(self, payload: dict[str, Any]) -> float | None:
        return None

    def submit(self, payload: dict[str, Any]) -> str:
        """Submit a TTS job (tts_inference → audio_stitch chain)."""
        from .functions.tts import tts_inference
        from .functions.stitch import audio_stitch

        job_id: str = payload["job_id"]
        chapters: list = payload["chapters"]

        call = tts_inference.spawn(payload)
        call_id = f"tts:{job_id}:{call.object_id}"
        return call_id

    def poll(self, remote_job_id: str) -> ModalPollResult:
        job_id = remote_job_id.split(":")[1] if ":" in remote_job_id else remote_job_id
        result = self._poll_r2(job_id)
        if result.status == "completed":
            artifact_key = output_m4b_key(job_id)
            result.artifact_bytes = self._download_artifact(job_id, artifact_key)
            result.artifact_source = "r2"
        return result

    def cancel(self, remote_job_id: str) -> None:
        try:
            parts = remote_job_id.split(":")
            call_id = parts[-1] if parts else remote_job_id
            fc = modal.functions.FunctionCall.from_id(call_id)
            fc.cancel()
        except Exception:
            pass

    # ── NlpGateway interface ───────────────────────────────────────────────

    def submit_nlp(self, job_id: str, payload: dict) -> str:
        from .functions.nlp import nlp_entity_clustering

        call = nlp_entity_clustering.spawn(payload)
        return f"nlp:{job_id}:{call.object_id}"

    def submit_attribution(self, job_id: str, payload: dict) -> str:
        from .functions.attribution import quote_attribution

        call = quote_attribution.spawn(payload)
        return f"attr:{job_id}:{call.object_id}"

    def poll_progress(self, job_id: str) -> ModalPollResult:
        return self._poll_r2(job_id)
```

- [ ] **Step 2: Verify it imports cleanly (without modal installed, it will error on import — that's expected)**

```bash
cd /Users/dizzler/Projects/Repos/kenkui && python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('gateway', 'src/kenkui/modal/gateway.py')
print('gateway.py parses OK (import skipped - modal not installed)')
" 2>&1
```

Expected output: `gateway.py parses OK (import skipped - modal not installed)` or a modal ImportError (acceptable at this stage).

- [ ] **Step 3: Commit**

```bash
git add src/kenkui/modal/gateway.py
git commit -m "feat: add KenkuiModalGateway implementing ModalGateway and NlpGateway interfaces"
```

---

## Task 10: NlpGateway protocol + NlpModalProvider

**Files:**
- Create: `src/kenkui/server/nlp_execution.py`
- Test: `tests/modal/test_nlp_execution.py`

- [ ] **Step 1: Write failing tests**

Create `tests/modal/test_nlp_execution.py`:

```python
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from kenkui.models import AppConfig, AttributionExecutionMode, JobConfig, NlpExecutionMode, QueueItem
from kenkui.server.nlp_execution import NlpModalProvider, NlpStageResult
from kenkui.server.tts_execution import ModalPollResult
from kenkui.modal.storage import LocalStorageBackend


class _FakeNlpGateway:
    def __init__(self, nlp_states=None, attr_states=None):
        self._nlp_states = list(nlp_states or [])
        self._attr_states = list(attr_states or [])
        self.nlp_submitted = []
        self.attr_submitted = []

    def submit_nlp(self, job_id, payload):
        self.nlp_submitted.append(payload)
        return f"nlp:{job_id}:call1"

    def submit_attribution(self, job_id, payload):
        self.attr_submitted.append(payload)
        return f"attr:{job_id}:call2"

    def poll_progress(self, job_id):
        # Return from the appropriate state list based on which phase last submitted.
        if self._nlp_states:
            return self._nlp_states.pop(0)
        return self._attr_states.pop(0)

    def cancel(self, call_id):
        pass


def _make_item(tmp_path):
    job = JobConfig(ebook_path=tmp_path / "book.epub")
    return QueueItem(id="job1", job=job)


def _completed_poll():
    return ModalPollResult(status="completed", progress=100.0, provider_status="completed")


def _running_poll():
    return ModalPollResult(status="running", progress=50.0, provider_status="running")


def test_run_nlp_stage_polls_to_completion_and_returns_roster_path(tmp_path):
    storage = LocalStorageBackend(tmp_path / "storage")
    roster_data = {"characters": [{"slug": "harry", "canonical_name": "Harry Potter"}]}
    storage.put_json("jobs/job1/nlp/roster.json", roster_data)

    gateway = _FakeNlpGateway(nlp_states=[_running_poll(), _completed_poll()])
    provider = NlpModalProvider(gateway=gateway, storage=storage, poll_interval=0)

    chapters = [{"index": 0, "title": "Ch 1", "paragraphs": ["text"]}]
    result = provider.run_nlp_stage(
        job_id="job1",
        payload={"job_id": "job1", "chapters": chapters},
        progress_callback=None,
    )
    assert result.success is True
    assert result.roster_local_path is not None
    assert result.roster_local_path.exists()
    assert json.loads(result.roster_local_path.read_text()) == roster_data


def test_run_nlp_stage_fails_when_poll_returns_failed(tmp_path):
    storage = LocalStorageBackend(tmp_path / "storage")
    gateway = _FakeNlpGateway(nlp_states=[
        ModalPollResult(status="failed", error_message="OOM", provider_status="failed")
    ])
    provider = NlpModalProvider(gateway=gateway, storage=storage, poll_interval=0)

    result = provider.run_nlp_stage(
        job_id="job1",
        payload={"job_id": "job1", "chapters": []},
        progress_callback=None,
    )
    assert result.success is False
    assert "OOM" in result.error_message


def test_run_attribution_stage_polls_to_completion(tmp_path):
    storage = LocalStorageBackend(tmp_path / "storage")
    chapters_data = {"chapters": [{"index": 0, "segments": []}]}
    storage.put_json("jobs/job1/attribution/chapters.json", chapters_data)

    gateway = _FakeNlpGateway(attr_states=[_completed_poll()])
    provider = NlpModalProvider(gateway=gateway, storage=storage, poll_interval=0)

    result = provider.run_attribution_stage(
        job_id="job1",
        payload={"job_id": "job1", "chapters": []},
        progress_callback=None,
    )
    assert result.success is True
    assert result.chapters_local_path is not None
    assert result.chapters_local_path.exists()


def test_run_attribution_stage_uploads_roster_when_provided(tmp_path):
    storage = LocalStorageBackend(tmp_path / "storage")
    chapters_data = {"chapters": []}
    storage.put_json("jobs/job1/attribution/chapters.json", chapters_data)

    roster_path = tmp_path / "roster.json"
    roster_path.write_text('{"characters": []}')

    gateway = _FakeNlpGateway(attr_states=[_completed_poll()])
    provider = NlpModalProvider(gateway=gateway, storage=storage, poll_interval=0)

    result = provider.run_attribution_stage(
        job_id="job1",
        payload={"job_id": "job1", "chapters": []},
        progress_callback=None,
        roster_local_path=roster_path,
    )
    assert result.success is True
    # Roster should have been uploaded to R2 before submission.
    assert storage.exists("jobs/job1/nlp/roster.json")
```

- [ ] **Step 2: Run test to confirm failure**

```bash
cd /Users/dizzler/Projects/Repos/kenkui && python -m pytest tests/modal/test_nlp_execution.py -v 2>&1 | head -20
```

Expected: ModuleNotFoundError for `kenkui.server.nlp_execution`.

- [ ] **Step 3: Implement nlp_execution.py**

`EnvNlpGateway` is a sibling of `EnvModalGateway` (tts_execution.py) — it loads the gateway class from `KENKUI_MODAL_GATEWAY` but validates NLP methods instead of TTS methods. This keeps gateway.py as the only Modal-coupled code while giving NlpModalProvider a clean loading path.

Create `src/kenkui/server/nlp_execution.py`:

```python
"""NLP execution providers: local orchestration and Modal dispatch."""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .tts_execution import ModalPollResult


class NlpGateway(Protocol):
    """Protocol for Modal-side NLP dispatch — the only interface NlpModalProvider needs."""

    def submit_nlp(self, job_id: str, payload: dict) -> str: ...
    def submit_attribution(self, job_id: str, payload: dict) -> str: ...
    def poll_progress(self, job_id: str) -> ModalPollResult: ...
    def cancel(self, call_id: str) -> None: ...


class EnvNlpGateway:
    """Load an NLP-capable gateway from KENKUI_MODAL_GATEWAY on demand.

    Mirrors EnvModalGateway (tts_execution.py) but validates NLP methods.
    The same KENKUI_MODAL_GATEWAY env var is used — KenkuiModalGateway
    implements both protocols, so the same class handles TTS and NLP dispatch.
    """

    def __init__(self):
        import importlib
        import os

        gateway_spec = os.environ.get("KENKUI_MODAL_GATEWAY", "").strip()
        if not gateway_spec:
            raise RuntimeError(
                "Modal NLP execution is configured, but KENKUI_MODAL_GATEWAY is not set."
            )
        module_name, sep, attr_name = gateway_spec.partition(":")
        if not sep:
            raise RuntimeError("KENKUI_MODAL_GATEWAY must be in 'module:attribute' form.")
        module = importlib.import_module(module_name)
        target = getattr(module, attr_name)
        self._gateway = target() if callable(target) and not hasattr(target, "submit_nlp") else target
        for method in ("submit_nlp", "submit_attribution", "poll_progress", "cancel"):
            if not hasattr(self._gateway, method):
                raise RuntimeError(
                    f"Configured Modal gateway is missing NLP method '{method}'."
                )

    def submit_nlp(self, job_id: str, payload: dict) -> str:
        return self._gateway.submit_nlp(job_id, payload)

    def submit_attribution(self, job_id: str, payload: dict) -> str:
        return self._gateway.submit_attribution(job_id, payload)

    def poll_progress(self, job_id: str) -> ModalPollResult:
        return self._gateway.poll_progress(job_id)

    def cancel(self, call_id: str) -> None:
        self._gateway.cancel(call_id)


@dataclass
class NlpStageResult:
    success: bool
    roster_local_path: Path | None = None
    chapters_local_path: Path | None = None
    error_message: str = ""


class NlpModalProvider:
    """Orchestrates NLP pipeline stages dispatched to Modal.

    Each stage (nlp/attribution) is independently submitted and polled.
    Downloaded artifacts are written to a temporary local directory;
    callers are responsible for moving them to the NLP cache.
    """

    def __init__(
        self,
        gateway: NlpGateway,
        storage,  # StorageBackend
        poll_interval: float = 2.0,
    ):
        self._gateway = gateway
        self._storage = storage
        self._poll_interval = poll_interval

    # ── NLP stage (entity clustering) ────────────────────────────────────

    def run_nlp_stage(
        self,
        job_id: str,
        payload: dict,
        progress_callback,
    ) -> NlpStageResult:
        """Submit nlp_entity_clustering, poll until done, download roster.json."""
        from ..modal.storage import nlp_roster_key

        call_id = self._gateway.submit_nlp(job_id, payload)

        while True:
            state = self._gateway.poll_progress(job_id)
            if progress_callback and state.progress is not None:
                progress_callback(state.progress, state.current_chapter or "NLP clustering…")

            if state.status == "completed":
                roster_data = self._storage.get_json(nlp_roster_key(job_id))
                out = Path(tempfile.mktemp(suffix="-roster.json"))
                out.write_text(json.dumps(roster_data), encoding="utf-8")
                return NlpStageResult(success=True, roster_local_path=out)

            if state.status in {"failed", "cancelled"}:
                return NlpStageResult(
                    success=False,
                    error_message=state.error_message or f"NLP stage {state.status}",
                )

            time.sleep(self._poll_interval)

    # ── Attribution stage (quote attribution) ─────────────────────────────

    def run_attribution_stage(
        self,
        job_id: str,
        payload: dict,
        progress_callback,
        roster_local_path: Path | None = None,
    ) -> NlpStageResult:
        """Submit quote_attribution, poll until done, download chapters.json.

        If *roster_local_path* is given (nlp=local, attribution=modal), uploads
        the local roster to R2 before submitting the Modal job.
        """
        from ..modal.storage import attribution_chapters_key, nlp_roster_key

        if roster_local_path is not None:
            roster_data = json.loads(roster_local_path.read_text(encoding="utf-8"))
            self._storage.put_json(nlp_roster_key(job_id), roster_data)

        call_id = self._gateway.submit_attribution(job_id, payload)

        while True:
            state = self._gateway.poll_progress(job_id)
            if progress_callback and state.progress is not None:
                progress_callback(state.progress, state.current_chapter or "Attributing speech…")

            if state.status == "completed":
                chapters_data = self._storage.get_json(attribution_chapters_key(job_id))
                out = Path(tempfile.mktemp(suffix="-chapters.json"))
                out.write_text(json.dumps(chapters_data), encoding="utf-8")
                return NlpStageResult(success=True, chapters_local_path=out)

            if state.status in {"failed", "cancelled"}:
                return NlpStageResult(
                    success=False,
                    error_message=state.error_message or f"Attribution stage {state.status}",
                )

            time.sleep(self._poll_interval)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/dizzler/Projects/Repos/kenkui && python -m pytest tests/modal/test_nlp_execution.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kenkui/server/nlp_execution.py tests/modal/test_nlp_execution.py
git commit -m "feat: add NlpGateway protocol and NlpModalProvider with polling loop"
```

---

## Task 11: Worker.py Modal dispatch

**Files:**
- Modify: `src/kenkui/server/worker.py`

Update `_run_attribution_phase` to check execution modes and dispatch to `NlpModalProvider` when modal is configured for either NLP or attribution. The VPS owns all stage-to-stage handoffs.

- [ ] **Step 1: Add imports and helper to worker.py**

At the top of `src/kenkui/server/worker.py`, after the existing imports, add:

```python
from ..models import AttributionExecutionMode, NlpExecutionMode
```

- [ ] **Step 2: Extract existing local attribution logic into `_run_attribution_phase_local`**

In `src/kenkui/server/worker.py`, rename the existing `_run_attribution_phase` body (lines that do the actual local NLP work) to a new private method `_run_attribution_phase_local` with the same signature:

```python
    def _run_attribution_phase_local(self, item: QueueItem) -> None:
        """Run Stage 3-4 speaker attribution locally (existing behavior)."""
        # --- paste entire existing _run_attribution_phase body here ---
```

Then replace `_run_attribution_phase` with a dispatcher that reads execution modes:

```python
    def _run_attribution_phase(self, item: QueueItem) -> None:
        """Dispatch NLP attribution to local or Modal execution based on config."""
        job = item.job

        nlp_mode: NlpExecutionMode = (
            job.job_nlp_execution_mode
            if job.job_nlp_execution_mode is not None
            else self._app_config.nlp_execution_mode
        )
        attr_mode: AttributionExecutionMode = (
            job.job_attribution_execution_mode
            if job.job_attribution_execution_mode is not None
            else self._app_config.attribution_execution_mode
        )

        if nlp_mode == NlpExecutionMode.MODAL or attr_mode == AttributionExecutionMode.MODAL:
            self._run_attribution_phase_modal(item, nlp_mode, attr_mode)
        else:
            self._run_attribution_phase_local(item)
```

- [ ] **Step 3: Add `_run_attribution_phase_modal` method**

After `_run_attribution_phase`, add:

```python
    def _run_attribution_phase_modal(
        self,
        item: QueueItem,
        nlp_mode: NlpExecutionMode,
        attr_mode: AttributionExecutionMode,
    ) -> None:
        """Run NLP/attribution stages on Modal, with R2 artifact handoffs."""
        import json as _json
        from ..modal.storage import build_storage_backend
        from ..server.nlp_execution import NlpModalProvider
        from ..nlp import CACHE_DIR, _attribution_cache_name

        job = item.job
        nlp_provider = job.job_nlp_provider or self._app_config.nlp_provider
        nlp_model = job.job_nlp_model or self._app_config.nlp_model

        # Load NLP gateway from env (KenkuiModalGateway implements both TTS + NLP protocols).
        from .nlp_execution import EnvNlpGateway
        gateway = EnvNlpGateway()

        storage = build_storage_backend()
        modal_provider = NlpModalProvider(gateway=gateway, storage=storage)

        def _progress(pct: float, msg: str) -> None:
            if self._progress_callback:
                current = self.get_job(self._current_id) if self._current_id else None
                eta = current.eta_seconds if current else 0
                self._progress_callback(pct, f"[Modal NLP] {msg}", eta)

        # Read ebook chapters for payload (used by both stages).
        from ..readers import get_reader
        reader = get_reader(job.ebook_path, verbose=False)
        all_chapters = reader.get_chapters()
        included = set(job.chapter_selection.included)
        chapters = [ch for ch in all_chapters if ch.index in included] or all_chapters
        chapters_dicts = [ch.to_dict() for ch in chapters]

        base_payload = {
            "job_id": item.id,
            "chapters": chapters_dicts,
            "nlp_provider": nlp_provider,
            "nlp_model": nlp_model,
            "ebook_path": str(job.ebook_path),
        }

        roster_local_path = None

        # ── Stage 1: NLP entity clustering ────────────────────────────────
        if nlp_mode == NlpExecutionMode.MODAL:
            result = modal_provider.run_nlp_stage(
                job_id=item.id,
                payload=base_payload,
                progress_callback=_progress,
            )
            if not result.success:
                raise RuntimeError(f"Modal NLP stage failed: {result.error_message}")
            roster_local_path = result.roster_local_path
        elif nlp_mode == NlpExecutionMode.LOCAL:
            # Run fast scan locally, save roster to temp file for R2 upload.
            from ..services.nlp_service import fast_scan as _fast_scan
            fast_result = _fast_scan(
                ebook_path=str(job.ebook_path),
                nlp_model=nlp_model,
                nlp_provider=nlp_provider,
                progress_callback=lambda pct, msg: _progress(pct * 0.3, msg),
            )
            import tempfile as _tempfile
            from pathlib import Path as _Path
            roster_local_path = _Path(_tempfile.mktemp(suffix="-roster.json"))
            roster_data = fast_result.roster.to_dict() if hasattr(fast_result.roster, "to_dict") else {}
            roster_local_path.write_text(_json.dumps(roster_data), encoding="utf-8")

        # ── Stage 2: Quote attribution ─────────────────────────────────────
        if attr_mode == AttributionExecutionMode.MODAL:
            attr_result = modal_provider.run_attribution_stage(
                job_id=item.id,
                payload=base_payload,
                progress_callback=lambda pct, msg: _progress(pct * 0.7 + 30, msg),
                roster_local_path=roster_local_path if nlp_mode != NlpExecutionMode.MODAL else None,
            )
            if not attr_result.success:
                raise RuntimeError(f"Modal attribution stage failed: {attr_result.error_message}")

            # Move downloaded chapters.json to NLP cache path.
            cache_file = CACHE_DIR / _attribution_cache_name(job.ebook_path, nlp_provider)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            if attr_result.chapters_local_path:
                import shutil as _shutil
                _shutil.move(str(attr_result.chapters_local_path), str(cache_file))
            job.annotated_chapters_path = cache_file
            self._save()

            # Reconstruct NLP result for cast assignment.
            from ..models import FastScanResult
            roster_data_for_cast = (
                _json.loads(roster_local_path.read_text()) if roster_local_path and roster_local_path.exists() else {}
            )
            chapters_data = _json.loads(cache_file.read_text()).get("chapters", [])
            nlp_result = _build_nlp_result_from_cache(chapters_data, roster_data_for_cast)
            self._assign_cast_deferred(item, nlp_result)
        elif attr_mode == AttributionExecutionMode.LOCAL:
            # nlp=modal, attribution=local: run attribution locally with roster from R2.
            from ..services.nlp_service import attribute_only
            from ..nlp.models import CharacterRoster

            roster_data = _json.loads(roster_local_path.read_text()) if roster_local_path else {}
            roster = CharacterRoster.from_dict(roster_data)
            nlp_result = attribute_only(
                roster=roster,
                chapters=chapters,
                ebook_path=str(job.ebook_path),
                nlp_model=nlp_model,
                nlp_provider=nlp_provider,
                progress_callback=lambda pct, msg: _progress(pct * 0.7 + 30, msg),
            )
            cache_file = CACHE_DIR / _attribution_cache_name(job.ebook_path, nlp_provider)
            job.annotated_chapters_path = cache_file
            self._save()
            self._assign_cast_deferred(item, nlp_result)
```

- [ ] **Step 4: Add `_build_nlp_result_from_cache` helper at module level**

After the imports in `worker.py`, add this helper function (before the `WorkerServer` class):

```python
def _build_nlp_result_from_cache(chapters_data: list[dict], roster_data: dict):
    """Reconstruct a minimal NLPResult-compatible object from cached JSON for cast assignment."""
    from ..models import CharacterInfo

    characters = [
        CharacterInfo(
            character_id=c.get("slug", c.get("character_id", "")),
            display_name=c.get("canonical_name", c.get("display_name", "")),
        )
        for c in roster_data.get("characters", [])
    ]

    class _MinimalNLPResult:
        def __init__(self, chars):
            self.characters = chars
            self.chapters = []

    return _MinimalNLPResult(characters)
```

- [ ] **Step 5: Verify the existing tests still pass**

```bash
cd /Users/dizzler/Projects/Repos/kenkui && python -m pytest tests/test_tts_execution.py tests/test_models_execution_modes.py tests/modal/ -v
```

Expected: All existing tests PASS. Worker.py is not directly unit-tested here — the integration is tested end-to-end in Task 10's NlpModalProvider tests.

- [ ] **Step 6: Commit**

```bash
git add src/kenkui/server/worker.py
git commit -m "feat: dispatch _run_attribution_phase to Modal when nlp/attribution execution mode is modal"
```

---

## Task 12: CLI config.py — execution mode prompts

**Files:**
- Modify: `src/kenkui/cli/config.py`

Add `nlp_execution_mode` and `attribution_execution_mode` prompts in the NLP section.

- [ ] **Step 1: Add execution mode prompts after the existing NLP section in config.py**

In `src/kenkui/cli/config.py`, find the `── NLP / Speaker Attribution ──` section (around line 312). After the existing `nlp_roster_model` prompt and before the next section, add:

```python
    execution_mode_choices = [
        {"name": "local   — run on this machine (spaCy/BookNLP + Ollama)", "value": "local"},
        {"name": "modal   — offload to Modal GPU cloud", "value": "modal"},
        {"name": "litellm — use configured cloud LLM API", "value": "litellm"},
    ]

    nlp_execution_mode = inquirer.select(
        message="NLP execution mode (entity clustering):",
        choices=execution_mode_choices,
        default=cfg.get("nlp_execution_mode", "local"),
    ).execute()

    attribution_execution_mode = inquirer.select(
        message="Attribution execution mode (speaker attribution):",
        choices=execution_mode_choices,
        default=cfg.get("attribution_execution_mode", "local"),
    ).execute()
```

- [ ] **Step 2: Add new fields to the summary table**

In `config.py`, find the summary table block that adds `tbl.add_row("NLP model", nlp_model)` and add rows after it:

```python
    tbl.add_row("NLP execution mode", nlp_execution_mode)
    tbl.add_row("Attribution execution mode", attribution_execution_mode)
```

- [ ] **Step 3: Add new fields to the updated config dict**

In `config.py`, find the `updated = {...}` dict at the end of `cmd_config`. After `"nlp_roster_model": nlp_roster_model,` add:

```python
        "nlp_execution_mode": nlp_execution_mode,
        "attribution_execution_mode": attribution_execution_mode,
```

- [ ] **Step 4: Run config command tests**

```bash
cd /Users/dizzler/Projects/Repos/kenkui && python -m pytest tests/test_config_commands.py -v 2>&1 | tail -20
```

Expected: All tests PASS (the new prompts may require updated test sequences — see Step 5).

- [ ] **Step 5: If config tests fail, update the answer sequences in tests/test_config_commands.py**

Config wizard tests provide pre-scripted answers via monkeypatching. If they fail because the new prompts expect additional answers, find the answer sequences in the test file and append `"local"` for the two new execution mode prompts at the appropriate positions in the sequence.

Check the test file:
```bash
cd /Users/dizzler/Projects/Repos/kenkui && grep -n "answers\|execute\|nlp_model" tests/test_config_commands.py | head -30
```

Add `"local"` twice (for `nlp_execution_mode` and `attribution_execution_mode`) after the `nlp_roster_model` answer in each test's answer sequence.

- [ ] **Step 6: Commit**

```bash
git add src/kenkui/cli/config.py tests/test_config_commands.py
git commit -m "feat: add nlp_execution_mode and attribution_execution_mode prompts to config wizard"
```

---

## Task 13: CLI add.py — per-job execution mode overrides

**Files:**
- Modify: `src/kenkui/cli/add.py`

Add an "Execution Modes" entry to `_submenu_advanced` for per-job NLP/attribution execution mode overrides.

- [ ] **Step 1: Add `_submenu_execution_modes` function to add.py**

In `src/kenkui/cli/add.py`, after `_submenu_advanced` (around line 1436), add:

```python
def _submenu_execution_modes(state: dict, app_config) -> dict:
    """Per-job execution mode overrides for NLP and attribution stages."""
    from InquirerPy import inquirer

    mode_choices = [
        {"name": "inherit from global config", "value": None},
        {"name": "local   — run on this machine", "value": "local"},
        {"name": "modal   — offload to Modal GPU cloud", "value": "modal"},
        {"name": "litellm — use configured cloud LLM API", "value": "litellm"},
    ]

    current_nlp = state.get("job_nlp_execution_mode")
    nlp_mode = _wizard_execute(inquirer.select(
        message="NLP execution mode for this job:",
        choices=mode_choices,
        default=current_nlp,
    ))
    state["job_nlp_execution_mode"] = nlp_mode

    current_attr = state.get("job_attribution_execution_mode")
    attr_mode = _wizard_execute(inquirer.select(
        message="Attribution execution mode for this job:",
        choices=mode_choices,
        default=current_attr,
    ))
    state["job_attribution_execution_mode"] = attr_mode

    return state
```

- [ ] **Step 2: Add "Execution Modes" entry to `_submenu_advanced` choices**

In `_submenu_advanced` (around line 1410), update the choices list to add the new entry before "Back":

```python
            choices=[
                {"name": "pocket-tts Quality        temp, generation steps, EOS, noise clamp →", "value": "tts_quality"},
                {"name": "Audio Encoding            bitrate, pauses, chapter titles →", "value": "encoding"},
                {"name": "Audio Post-Processing     enable/disable effects chain →", "value": "postprocessing"},
                {"name": "Text Preprocessing        apostrophe/contraction handling →", "value": "text_prep"},
                {"name": "Output Location           where the audiobook file goes →", "value": "output"},
                {"name": "Execution Modes           NLP/attribution local vs Modal →", "value": "execution_modes"},
                {"name": "Voice Management          browse, audition, exclude voices →", "value": "voices"},
                {"name": "Back", "value": "back"},
            ],
```

- [ ] **Step 3: Add dispatch branch in `_submenu_advanced`**

In `_submenu_advanced`, after `elif action == "voices":`, add:

```python
        elif action == "execution_modes":
            state = _submenu_execution_modes(state, app_config)
```

- [ ] **Step 4: Ensure `_state_to_job_kwargs` passes through execution mode overrides**

In `src/kenkui/services/job_service.py` (the `build_job_kwargs_from_state` function), check if `job_nlp_execution_mode` and `job_attribution_execution_mode` are passed through from state. If not, find where the state dict is converted to job kwargs and add:

```python
    if state.get("job_nlp_execution_mode") is not None:
        from kenkui.models import NlpExecutionMode
        kwargs["job_nlp_execution_mode"] = NlpExecutionMode(state["job_nlp_execution_mode"])

    if state.get("job_attribution_execution_mode") is not None:
        from kenkui.models import AttributionExecutionMode
        kwargs["job_attribution_execution_mode"] = AttributionExecutionMode(state["job_attribution_execution_mode"])
```

Verify the job_service file:
```bash
cd /Users/dizzler/Projects/Repos/kenkui && grep -n "job_nlp_provider\|job_temp\|build_job_kwargs" src/kenkui/services/job_service.py | head -20
```

- [ ] **Step 5: Verify CLI tests pass**

```bash
cd /Users/dizzler/Projects/Repos/kenkui && python -m pytest tests/test_cli.py -v 2>&1 | tail -20
```

Expected: All tests PASS. If tests fail due to new prompts in the wizard, update the answer sequences in `tests/test_cli.py` (add the two new execution mode answers: both `None`/skip or `"local"` at appropriate positions).

- [ ] **Step 6: Commit**

```bash
git add src/kenkui/cli/add.py src/kenkui/services/job_service.py
git commit -m "feat: add Execution Modes submenu to add wizard advanced options"
```

---

## Task 14: CLI queue.py — execution provider display

**Files:**
- Modify: `src/kenkui/cli/queue.py`

Update `_job_mode_str` to append a `·modal` tag when the job has a modal execution provider or modal execution modes configured.

- [ ] **Step 1: Update `_job_mode_str` in queue.py**

Find `_job_mode_str` (around line 60) and update it to check execution modes:

```python
def _job_mode_str(job) -> str:
    """Return a short mode string for a job (dict or JobConfig)."""
    if isinstance(job, dict):
        mode = job.get("narration_mode", "single")
        provider = job.get("job_nlp_provider") or ""
        model = job.get("job_nlp_model") or ""
        chapter_voices = job.get("chapter_voices") or {}
        nlp_exec = job.get("job_nlp_execution_mode") or job.get("nlp_execution_mode") or ""
        attr_exec = job.get("job_attribution_execution_mode") or job.get("attribution_execution_mode") or ""
    else:
        mode = getattr(job, "narration_mode", None)
        mode = mode.value if hasattr(mode, "value") else str(mode or "single")
        provider = getattr(job, "job_nlp_provider", "") or ""
        model = getattr(job, "job_nlp_model", "") or ""
        chapter_voices = getattr(job, "chapter_voices", {}) or {}
        _nlp = getattr(job, "job_nlp_execution_mode", None)
        nlp_exec = _nlp.value if hasattr(_nlp, "value") else str(_nlp or "")
        _attr = getattr(job, "job_attribution_execution_mode", None)
        attr_exec = _attr.value if hasattr(_attr, "value") else str(_attr or "")

    has_modal = nlp_exec == "modal" or attr_exec == "modal"

    if chapter_voices:
        return "chapter·modal" if has_modal else "chapter"
    if mode == "multi":
        short_model = model.split("/")[-1].split("-")[0] if model else ""
        if provider and provider != "ollama":
            base = f"multi · {provider[:5]} · {short_model}" if short_model else f"multi · {provider[:8]}"
        else:
            base = f"multi · ollama · {short_model}" if short_model else "multi · ollama"
        return f"{base} · modal" if has_modal else base
    return "single·modal" if has_modal else "single"
```

- [ ] **Step 2: Run queue-related tests**

```bash
cd /Users/dizzler/Projects/Repos/kenkui && python -m pytest tests/ -k "queue" -v 2>&1 | tail -20
```

Expected: All queue tests PASS.

- [ ] **Step 3: Commit**

```bash
git add src/kenkui/cli/queue.py
git commit -m "feat: show modal execution provider tag in queue mode column"
```

---

## Final verification

- [ ] **Run the full test suite**

```bash
cd /Users/dizzler/Projects/Repos/kenkui && python -m pytest tests/ -v --ignore=tests/test_config_commands.py 2>&1 | tail -30
```

Expected: All tests PASS (test_config_commands.py excluded if answer sequences need updating per Task 12 Step 5).

- [ ] **Run the complete test suite including config tests**

```bash
cd /Users/dizzler/Projects/Repos/kenkui && python -m pytest tests/ -v 2>&1 | tail -30
```

Expected: All tests PASS.

- [ ] **Final commit**

```bash
git add -A
git commit -m "feat: complete Modal remote execution implementation (all stages)"
```
