"""WorkerServer - Queue and processing logic for the kenkui server."""

from __future__ import annotations

import json
import logging
import threading
import tomllib
import uuid
from collections.abc import Callable

import tomli_w

from ..chapter_filter import FilterOperation
from ..config import CONFIG_DIR
from ..models import AppConfig, CostStatus, JobConfig, JobStatus, QueueItem
from ..parsing import AnnotatedChaptersCacheMissError
from .tts_execution import get_tts_execution_provider

logger = logging.getLogger(__name__)

QUEUE_FILE = CONFIG_DIR / "queue.toml"
_LEGACY_QUEUE_FILE = CONFIG_DIR / "queue.yaml"


def _resolve(job_val, app_val):
    """Return job_val if explicitly set (not None), else fall back to app_val."""
    return job_val if job_val is not None else app_val


def _strip_none(obj: object) -> object:
    """Recursively remove None values — TOML has no null type."""
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(v) for v in obj if v is not None]
    return obj


class WorkerServer:
    """Server managing job queue and audio processing.

    This class combines queue management and processing logic,
    providing a thread-safe interface for the HTTP API.
    """

    def __init__(self):
        self._items: list[QueueItem] = []
        self._current_id: str | None = None
        self._app_config = AppConfig()
        self._lock = threading.RLock()
        self._processing_thread: threading.Thread | None = None
        self._running = False
        self._progress_callback: Callable[[float, str, int], None] | None = None
        self._pause_requested: bool = False
        self._modal_gateway = None
        self._load()

        from ..services.book_cache import BookCache
        from .tasks import TaskRegistry, TaskRunner

        self.book_cache = BookCache()
        self.task_registry = TaskRegistry()
        self.task_runner = TaskRunner(self.task_registry)

    def _load(self):
        # Auto-migrate from legacy queue.yaml if queue.toml does not exist yet.
        if not QUEUE_FILE.exists() and _LEGACY_QUEUE_FILE.exists():
            self._migrate_yaml_to_toml()

        if QUEUE_FILE.exists():
            try:
                data = tomllib.loads(QUEUE_FILE.read_text(encoding="utf-8"))
                if data:
                    self._items = [QueueItem.from_dict(d) for d in data.get("items", [])]
                    self._app_config = AppConfig.from_dict(data.get("app_config", {}))
            except Exception:
                pass
        self._reset_stale_processing()

    def _migrate_yaml_to_toml(self) -> None:
        """Convert queue.yaml → queue.toml and remove the old file."""
        try:
            import yaml  # pyyaml may still be present as a transitive dep

            data = yaml.safe_load(_LEGACY_QUEUE_FILE.read_text())
            if data:
                QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
                QUEUE_FILE.write_bytes(tomli_w.dumps(_strip_none(data)).encode("utf-8"))
            _LEGACY_QUEUE_FILE.unlink(missing_ok=True)
            logger.info("Migrated queue.yaml → queue.toml")
        except Exception as exc:
            logger.warning("Could not migrate queue.yaml: %s — starting fresh", exc)

    def _save(self):
        raw = {
            "items": [item.to_dict() for item in self._items],
            "app_config": self._app_config.to_dict(),
        }
        data: dict = _strip_none(raw)  # type: ignore[assignment]
        QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
        QUEUE_FILE.write_bytes(tomli_w.dumps(data).encode("utf-8"))

    @property
    def app_config(self) -> AppConfig:
        return self._app_config

    @app_config.setter
    def app_config(self, config: AppConfig):
        with self._lock:
            self._app_config = config
            self._save()

    @property
    def current_item(self) -> QueueItem | None:
        with self._lock:
            return next((i for i in self._items if i.status == JobStatus.PROCESSING), None)

    @property
    def pending_items(self) -> list[QueueItem]:
        with self._lock:
            return [i for i in self._items if i.status == JobStatus.PENDING]

    @property
    def completed_items(self) -> list[QueueItem]:
        with self._lock:
            return [i for i in self._items if i.status == JobStatus.COMPLETED]

    @property
    def failed_items(self) -> list[QueueItem]:
        with self._lock:
            return [i for i in self._items if i.status == JobStatus.FAILED]

    @property
    def all_items(self) -> list[QueueItem]:
        with self._lock:
            return list(self._items)

    @property
    def is_running(self) -> bool:
        return self._running

    def add_job(self, job: JobConfig) -> QueueItem:
        with self._lock:
            item = QueueItem(
                id=str(uuid.uuid4())[:8],
                job=job,
                status=JobStatus.PENDING,
                execution_provider=job.tts_execution_mode.value,
            )
            self._items.append(item)
            self._save()
            return item

    def get_job(self, job_id: str) -> QueueItem | None:
        with self._lock:
            return next((i for i in self._items if i.id == job_id), None)

    def remove_job(self, job_id: str) -> bool:
        with self._lock:
            for i, item in enumerate(self._items):
                if item.id == job_id:
                    if item.status == JobStatus.PROCESSING:
                        return False
                    self._items.pop(i)
                    self._save()
                    return True
            return False

    def clear_all_jobs(self):
        with self._lock:
            self._items = []
            self._save()

    def _reset_stale_processing(self):
        """Reset any PROCESSING jobs to PENDING (e.g., from previous session)."""
        with self._lock:
            for item in self._items:
                if item.status == JobStatus.PROCESSING:
                    item.status = JobStatus.PENDING
                    item.progress = 0.0
                    item.current_chapter = ""
                    item.error_message = ""
            if any(i.status == JobStatus.PENDING for i in self._items):
                self._save()

    def get_next_pending(self) -> QueueItem | None:
        with self._lock:
            return next((i for i in self._items if i.status == JobStatus.PENDING), None)

    def start_next_job(self) -> QueueItem | None:
        import time

        with self._lock:
            item = self.get_next_pending()
            if item:
                item.status = JobStatus.PROCESSING
                item.started_at = time.time()
                self._current_id = item.id
                self._save()
            return item

    def update_progress(self, job_id: str, progress: float, current_chapter: str, eta_seconds: int):
        with self._lock:
            for item in self._items:
                if item.id == job_id:
                    item.progress = progress
                    item.current_chapter = current_chapter
                    item.eta_seconds = eta_seconds
                    break
            self._save()

    def complete_job(self, job_id: str, output_path: str = ""):
        import time

        with self._lock:
            for item in self._items:
                if item.id == job_id:
                    item.status = JobStatus.COMPLETED
                    item.progress = 100.0
                    item.current_chapter = ""
                    item.output_path = output_path
                    item.completed_at = time.time()
                    break
            self._save()

    def update_job_metadata(self, job_id: str, **fields) -> None:
        with self._lock:
            item = next((i for i in self._items if i.id == job_id), None)
            if item is None:
                return

            for key, value in fields.items():
                if value is None:
                    continue
                if key == "cost_status" and not isinstance(value, CostStatus):
                    value = CostStatus(str(value))
                setattr(item, key, value)
            self._save()

    def fail_job(self, job_id: str, error: str):
        with self._lock:
            for item in self._items:
                if item.id == job_id:
                    item.status = JobStatus.FAILED
                    item.error_message = error
                    break
            self._save()

    def cancel_current_job(self) -> bool:
        with self._lock:
            if self.current_item:
                self.current_item.status = JobStatus.CANCELLED
                self._save()
                return True
            return False

    def pause_job(self, job_id: str) -> bool:
        with self._lock:
            item = next((i for i in self._items if i.id == job_id), None)
            if item is None or item.status != JobStatus.PROCESSING:
                return False
            if item.job.tts_execution_mode.value == "modal":
                return False
            self._pause_requested = True
            return True

    def resume_job(self, job_id: str) -> bool:
        with self._lock:
            item = next((i for i in self._items if i.id == job_id), None)
            if item is None or item.status != JobStatus.PAUSED:
                return False
            if item.job.tts_execution_mode.value == "modal":
                return False
            item.status = JobStatus.PENDING
            # started_at will be set by start_next_job() when the loop picks it up
            self._pause_requested = False
            self._save()
        self.start_processing()
        return True

    def start_processing(self, progress_callback: Callable[[float, str, int], None] | None = None):
        """Start processing the next job in the queue."""
        if self._running:
            return False

        # Create a wrapper callback that also updates server state
        def progress_wrapper(percent: float, chapter: str, eta: int):
            # Call the external callback if provided
            if progress_callback:
                progress_callback(percent, chapter, eta)
            # Always update server state for API queries
            if self._current_id:
                self.update_progress(self._current_id, percent, chapter, eta)

        self._progress_callback = progress_wrapper
        self._running = True

        self._processing_thread = threading.Thread(target=self._process_loop)
        self._processing_thread.start()
        return True

    def stop_processing(self):
        """Stop the current processing job."""
        current = self.current_item
        if current is not None:
            try:
                provider = get_tts_execution_provider(current, gateway=self._modal_gateway)
                provider.cancel(current)
            except Exception as exc:
                logger.warning("Could not cancel provider job %s: %s", current.id, exc)
        self._running = False
        if self._processing_thread:
            self._processing_thread.join(timeout=5)

    def _process_loop(self):
        """Main processing loop - runs in background thread."""
        while self._running:
            item = self.start_next_job()
            if not item:
                break

            self._process_job(item)

            if not self._running:
                break

    def _process_job(self, item: QueueItem):
        """Process a single job."""
        job = item.job

        try:
            # Pre-TTS phase: for multi-voice jobs, ensure speaker attribution is cached.
            if job.narration_mode.value == "multi" and not (
                job.annotated_chapters_path and job.annotated_chapters_path.exists()
            ):
                self._run_attribution_phase(item)
                # Reload job reference (attribution phase may have updated annotated_chapters_path)
                job = item.job

            cfg = self._build_config(job)
            provider = get_tts_execution_provider(item, gateway=self._modal_gateway)
            self.update_job_metadata(
                item.id,
                execution_provider=job.tts_execution_mode.value,
                provider_status="starting",
            )
            outcome = provider.execute(
                item=item,
                cfg=cfg,
                app_config=self._app_config,
                progress_callback=self._progress_callback,
                metadata_callback=lambda **fields: self.update_job_metadata(item.id, **fields),
                pause_check=lambda: self._pause_requested,
            )

            if outcome.paused is True:
                with self._lock:
                    item = next((i for i in self._items if i.id == self._current_id), None)
                    if item is not None:
                        item.status = JobStatus.PAUSED
                        self._pause_requested = False
                        self._save()
                return

            if outcome.success:
                output_path = outcome.output_path or str(cfg.output_path / f"{job.name}.m4b")
                m4b_file = job.output_path if (job.output_path and job.output_path.suffix) else None
                if m4b_file is None:
                    m4b_file = cfg.output_path / f"{job.name}.m4b"
                self.update_job_metadata(
                    item.id,
                    remote_job_id=outcome.remote_job_id,
                    estimated_cost_usd=outcome.estimated_cost_usd,
                    actual_cost_usd=outcome.actual_cost_usd,
                    cost_status="final" if outcome.actual_cost_usd is not None else (
                        "estimated" if outcome.estimated_cost_usd is not None else "none"
                    ),
                    artifact_uri=outcome.artifact_uri,
                    artifact_source=outcome.artifact_source,
                    provider_status=outcome.provider_status or "completed",
                )

                # Append synthesized credits audio (if enabled and file exists)
                if self._app_config.credits_enabled and m4b_file.exists():
                    self._append_credits(item, m4b_file)

                self.complete_job(item.id, output_path)

                # Post-job cast notification for multi-voice jobs
                if job.narration_mode.value == "multi":
                    logger.info(
                        "Cast saved for job %s — run `kenkui voices cast %s` to review",
                        item.id, item.id,
                    )
            else:
                self.update_job_metadata(
                    item.id,
                    remote_job_id=outcome.remote_job_id,
                    estimated_cost_usd=outcome.estimated_cost_usd,
                    actual_cost_usd=outcome.actual_cost_usd,
                    cost_status="final" if outcome.actual_cost_usd is not None else (
                        "estimated" if outcome.estimated_cost_usd is not None else "none"
                    ),
                    provider_status=outcome.provider_status or "failed",
                )
                self.fail_job(item.id, outcome.error_message or "Conversion failed")

        except AnnotatedChaptersCacheMissError as e:
            logger.error("Job %s failed (cache miss): %s", item.id, e)
            self.fail_job(item.id, f"CACHE_MISS: {e}")
        except Exception as e:
            logger.exception("Job %s failed: %s", item.id, e)
            self.fail_job(item.id, str(e))

    def _run_attribution_phase(self, item: QueueItem) -> None:
        """Run Stage 3-4 speaker attribution and update item.job.annotated_chapters_path.

        Loads the roster from ``roster_cache_path`` if available; falls back to a
        fresh Stage 1-2 scan if not (e.g., job submitted via API without wizard).
        """
        from ..nlp import (
            CACHE_DIR,
            _attribution_cache_name,
            book_hash,
            get_cached_result,
        )
        from ..readers import get_reader
        from ..services.nlp_service import attribute_only

        job = item.job
        book_path = job.ebook_path

        # Resolve provider early so the cache lookup uses the correct namespace.
        _nlp_provider_early = job.job_nlp_provider or self._app_config.nlp_provider

        # Return early if full NLP cache already exists for this provider
        cached = get_cached_result(book_path, provider=_nlp_provider_early)
        if cached is not None:
            h = book_hash(book_path)
            job.annotated_chapters_path = CACHE_DIR / _attribution_cache_name(book_path, _nlp_provider_early)
            return

        _attrib_step: list[int] = [0]

        def _cb(msg: str) -> None:
            if self._progress_callback:
                _attrib_step[0] += 1
                # Each callback call = one chunk processed; cap at 14.9% (TTS starts at 0%)
                # Assume ~50 chunks typical; adjust estimate conservatively
                pct = min(_attrib_step[0] * 0.3, 14.9)
                current_item = self.get_job(self._current_id) if self._current_id else None
                eta = current_item.eta_seconds if current_item is not None else 0
                self._progress_callback(pct, f"[Attribution] {msg}", eta)

        # Load chapters
        _cb("reading ebook…")
        try:
            reader = get_reader(book_path, verbose=False)
            all_chapters = reader.get_chapters()
        except Exception as exc:
            raise RuntimeError(f"Could not read ebook for attribution: {exc}") from exc

        # Filter to selected chapters
        included = set(job.chapter_selection.included)
        if included:
            chapters = [ch for ch in all_chapters if ch.index in included] or all_chapters
        else:
            chapters = all_chapters

        # Per-job overrides take precedence over the global app config.
        nlp_provider = job.job_nlp_provider or self._app_config.nlp_provider
        nlp_model = job.job_nlp_model or self._app_config.nlp_model

        # For cloud providers, validate that credentials are available before starting.
        if nlp_provider != "ollama":
            from ..config import load_provider_credentials
            creds = load_provider_credentials()
            cred = creds.get(nlp_provider)
            if not (cred and cred.api_key):
                import os as _os
                env_var_map = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "google": "GEMINI_API_KEY"}
                env_var = env_var_map.get(nlp_provider, f"{nlp_provider.upper()}_API_KEY")
                if not _os.environ.get(env_var):
                    raise RuntimeError(
                        f"No API key found for provider '{nlp_provider}'. "
                        f"Run `kenkui configure-provider` or set {env_var}."
                    )

        # Get roster — load from roster_cache_path, or re-run fast scan as fallback.
        roster = None
        if job.roster_cache_path and job.roster_cache_path.exists():
            try:
                from ..models import FastScanResult

                data = json.loads(job.roster_cache_path.read_text(encoding="utf-8"))
                roster = FastScanResult.from_dict(data).roster
            except Exception as exc:
                logger.warning(
                    "Could not load roster cache %s: %s — re-scanning", job.roster_cache_path, exc
                )

        if roster is None:
            # Roster cache missing — rebuild using the configured provider (not Ollama-only).
            _cb("rebuilding character roster…")
            from ..services.nlp_service import fast_scan as _svc_fast_scan
            fast_result = _svc_fast_scan(
                ebook_path=str(book_path),
                nlp_model=nlp_model,
                nlp_provider=nlp_provider,
                progress_callback=lambda pct, msg: _cb(msg),
            )
            roster = fast_result.roster

        # Run Stage 3-4 attribution via the configured provider (cloud or ollama).
        nlp_result = attribute_only(
            roster=roster,
            chapters=chapters,
            ebook_path=str(book_path),
            nlp_model=nlp_model,
            nlp_provider=nlp_provider,
            progress_callback=lambda pct, msg: _cb(msg),
        )

        cache_file = CACHE_DIR / _attribution_cache_name(book_path, nlp_provider)
        job.annotated_chapters_path = cache_file
        self._save()

        # Deferred cast assignment: auto-assign voices now that characters are known.
        self._assign_cast_deferred(item, nlp_result)

    def _assign_cast_deferred(self, item: QueueItem, nlp_result) -> None:
        """Auto-assign character voices after NLP attribution completes.

        Skips silently if character voices are already assigned (re-queued job).
        Inherits voices from a linked series manifest first, then auto-assigns
        any remaining characters via suggest_cast().
        """
        import json as _json

        from ..models import FastScanResult
        from ..nlp import CACHE_DIR, book_hash
        from ..series import load_series, match_characters, save_series, update_manifest

        job = item.job

        # Skip if character voices are already assigned (not just NARRATOR)
        if any(k != "NARRATOR" for k in job.speaker_voices):
            return

        narrator_voice = job.speaker_voices.get("NARRATOR", self._app_config.default_voice)

        # Load the roster cache for alias lookup (written by run_fast_scan)
        h = book_hash(job.ebook_path)
        roster_path = CACHE_DIR / f"{h}-roster.json"
        fast_scan_result = None
        if roster_path.exists():
            try:
                data = _json.loads(roster_path.read_text(encoding="utf-8"))
                fast_scan_result = FastScanResult.from_dict(data)
            except Exception as exc:
                logger.warning("Could not load roster for cast assignment: %s", exc)

        # 1. Series voice inheritance
        inherited: dict[str, str] = {}
        pinned: set[str] = set()
        manifest = None
        if job.series_slug and fast_scan_result is not None:
            manifest = load_series(job.series_slug)
            if manifest:
                inherited, pinned = match_characters(
                    nlp_result.characters, fast_scan_result, manifest
                )

        # 2. Auto-assign remaining characters using voice pool template (then suggest_cast fallback)
        from ..services.voice_service import apply_voice_pool_template
        from ..voice_pool import load_voice_pool_template

        template = load_voice_pool_template()

        # Build role lookup from roster if available
        roster_roles: dict[str, str] = {}
        if fast_scan_result is not None:
            for char_rec in getattr(fast_scan_result.roster, "characters", []):
                slug = getattr(char_rec, "slug", None)
                role = getattr(char_rec, "role", "supporting")
                if slug:
                    roster_roles[slug] = role

        unmatched = [c for c in nlp_result.characters if c.character_id not in inherited]
        new_voices = apply_voice_pool_template(
            roster=unmatched,
            template=template,
            series_voices=inherited,
            narrator_voice=narrator_voice,
            excluded_voices=self._app_config.excluded_voices,
            roster_roles=roster_roles or None,
        )

        # 3. Merge: preserved NARRATOR + template-assigned + inherited (inherited wins)
        job.speaker_voices = {
            "NARRATOR": narrator_voice,
            **new_voices,
            **inherited,
        }
        self._save()

        # 4. Update series manifest with the final assignments
        if manifest is not None and fast_scan_result is not None:
            updated = update_manifest(
                manifest,
                nlp_result.characters,
                fast_scan_result,
                job.speaker_voices,
                pinned,
            )
            save_series(updated)

    def _append_credits(self, item: QueueItem, m4b_path: "Path") -> None:
        """Synthesize a credits segment and append it to the m4b without a chapter marker."""
        import subprocess
        import tempfile
        from pathlib import Path as _Path

        job = item.job
        cfg = self._app_config

        # Build credits script
        parts = [f"This audiobook was generated with kenkui."]
        if job.name:
            parts.append(f"{job.name}.")
        if job.narration_mode.value == "multi" and job.speaker_voices:
            cast_lines = []
            for char_id, voice in sorted(job.speaker_voices.items()):
                if char_id == "NARRATOR":
                    continue
                cast_lines.append(f"{char_id.replace('_', ' ').title()}, voiced by {voice}")
            if cast_lines:
                parts.append("Cast: " + "; ".join(cast_lines) + ".")
        if cfg.credits_acknowledgements:
            parts.append(cfg.credits_acknowledgements)
        if cfg.credits_license:
            parts.append(cfg.credits_license)

        credits_text = " ".join(parts)
        narrator_voice = job.speaker_voices.get("NARRATOR", cfg.default_voice) or cfg.default_voice

        try:
            from ..workers import _get_or_load_model, _render_text

            import logging as _logging
            _log = _logging.getLogger(__name__)

            model = _get_or_load_model(
                temp=cfg.temp,
                lsd_decode_steps=cfg.lsd_decode_steps,
                noise_clamp=cfg.noise_clamp,
                eos_threshold=cfg.eos_threshold,
            )
            from ..voice_loader import load_voice

            voice_path = load_voice(narrator_voice)
            voice_state = model.get_state_for_audio_prompt(voice_path)

            credits_seg = _render_text(
                model,
                voice_state,
                credits_text,
                log_message=lambda msg: _log.debug(msg),
                pid=0,
                batch_idx=0,
                total_batches=1,
                frames_after_eos=0,
            )
            if credits_seg is None:
                logger.warning("Credits synthesis returned no audio for job %s", item.id)
                return
        except Exception as exc:
            logger.warning("Credits synthesis failed for job %s: %s", item.id, exc)
            return

        # Append credits audio to m4b via ffmpeg concat (no chapter marker)
        try:
            bitrate = _resolve(job.job_m4b_bitrate, cfg.m4b_bitrate) or "96k"
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = _Path(tmpdir)
                credits_wav = tmp / "credits.wav"
                credits_aac = tmp / "credits.aac"
                orig_aac = tmp / "original.aac"
                combined_aac = tmp / "combined.aac"
                output_m4b = tmp / "output.m4b"

                credits_seg.export(str(credits_wav), format="wav")

                # Encode credits WAV → ADTS AAC
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(credits_wav),
                     "-c:a", "aac", "-b:a", str(bitrate), "-f", "adts",
                     str(credits_aac)],
                    check=True, capture_output=True,
                )

                # Extract original m4b audio → ADTS AAC (no re-encode)
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(m4b_path),
                     "-vn", "-acodec", "copy", "-f", "adts",
                     str(orig_aac)],
                    check=True, capture_output=True,
                )

                # Binary concat ADTS streams
                with open(combined_aac, "wb") as out_f:
                    out_f.write(orig_aac.read_bytes())
                    out_f.write(credits_aac.read_bytes())

                # Re-mux: audio from combined stream, chapters from original m4b
                subprocess.run(
                    ["ffmpeg", "-y",
                     "-i", str(combined_aac),
                     "-i", str(m4b_path),
                     "-map", "0:a",
                     "-map_chapters", "1",
                     "-c:a", "copy",
                     str(output_m4b)],
                    check=True, capture_output=True,
                )

                import shutil
                shutil.move(str(output_m4b), str(m4b_path))
                logger.info("Credits appended to %s", m4b_path.name)
        except Exception as exc:
            logger.warning("Could not append credits to %s: %s", m4b_path.name, exc)

    def _build_config(self, job: JobConfig):
        """Build ProcessingConfig from JobConfig and AppConfig."""
        from ..models import ProcessingConfig

        preset = job.chapter_selection.preset
        if preset.value in ("manual", "custom"):
            # Use explicit index list from the UI checkbox selection
            operations = [
                FilterOperation("index", str(idx)) for idx in job.chapter_selection.included
            ]
            if not operations:
                operations = [FilterOperation("preset", "content-only")]
        else:
            operations = [FilterOperation("preset", preset.value)]

        output_path = job.output_path or job.ebook_path.parent

        from ..models import _normalize_bitrate

        cfg = ProcessingConfig(
            voice=job.voice,
            ebook_path=job.ebook_path,
            output_path=output_path,
            pause_line_ms=_resolve(job.job_pause_line_ms, self._app_config.pause_line_ms),
            pause_chapter_ms=_resolve(job.job_pause_chapter_ms, self._app_config.pause_chapter_ms),
            speak_chapter_titles=_resolve(
                job.job_speak_chapter_titles, self._app_config.speak_chapter_titles
            ),
            pause_before_chapter_title_ms=_resolve(
                job.job_pause_before_chapter_title_ms,
                self._app_config.pause_before_chapter_title_ms,
            ),
            pause_after_chapter_title_ms=_resolve(
                job.job_pause_after_chapter_title_ms, self._app_config.pause_after_chapter_title_ms
            ),
            workers=self._app_config.workers,
            m4b_bitrate=_normalize_bitrate(
                _resolve(job.job_m4b_bitrate, self._app_config.m4b_bitrate)
            ),
            keep_temp=self._app_config.keep_temp,
            debug_html=self._app_config.verbose,
            chapter_filters=operations,
            verbose=self._app_config.verbose,
            temp=_resolve(job.job_temp, self._app_config.temp),
            lsd_decode_steps=_resolve(job.job_lsd_decode_steps, self._app_config.lsd_decode_steps),
            noise_clamp=_resolve(job.job_noise_clamp, self._app_config.noise_clamp),
            eos_threshold=_resolve(job.job_eos_threshold, self._app_config.eos_threshold),
            frames_after_eos=_resolve(job.job_frames_after_eos, self._app_config.frames_after_eos),
            # Multi-voice fields
            speaker_voices=job.speaker_voices,
            annotated_chapters_path=job.annotated_chapters_path,
            _included_indices=job.chapter_selection.included,
            # Chapter-voice mode
            chapter_voices=job.chapter_voices,
            # Audio post-processing
            post_processing=self._app_config.post_processing,
            apostrophe_mode=_resolve(job.job_apostrophe_mode, self._app_config.apostrophe_mode),
        )
        return cfg


_server: WorkerServer | None = None


def get_server() -> WorkerServer:
    """Get the global WorkerServer instance."""
    global _server
    if _server is None:
        _server = WorkerServer()
    return _server


def reset_server():
    """Reset the global WorkerServer instance."""
    global _server
    if _server:
        _server.stop_processing()
    _server = None
