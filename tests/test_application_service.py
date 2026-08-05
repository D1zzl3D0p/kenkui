from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import pytest

from kenkui.config import ProviderCredentials
from kenkui.models import (
    AppConfig,
    ChapterPreset,
    ChapterSelection,
    CharacterInfo,
    JobConfig,
    JobStatus,
    NLPResult,
    NumberNormalizationConfig,
    PostProcessingConfig,
)
from kenkui.models.api import JobCreateRequest, VoiceResponse
from kenkui.progress import ChapterProgress, ProgressEvent
from kenkui.services.application_service import (
    KenkuiService,
    _progress_update_from_args,
    job_create_request_to_config,
)
from kenkui.services.execution_service import ExecutionOutcome
from kenkui.services.job_service import build_processing_config
from kenkui.services.task_service import Task, TaskStatus, TaskType


def test_job_create_request_to_config_preserves_api_fields(tmp_path):
    req = JobCreateRequest(
        ebook_path=str(tmp_path / "book.epub"),
        voice="alba",
        chapter_selection=ChapterSelection(
            preset=ChapterPreset.MANUAL,
            included=[1, 3],
        ),
        output_path=str(tmp_path / "out"),
        narration_mode="multi",
        speaker_voices={"NARRATOR": "alba", "alice": "clara"},
        job_pause_line_ms=1200,
        job_apostrophe_mode="always_remove",
        job_tts_max_tokens_per_chunk=75,
        job_number_normalization=NumberNormalizationConfig(
            phone_numbers_mode="raw",
            identifier_min_digits=10,
        ),
    )

    job = job_create_request_to_config(req)

    assert job.ebook_path == tmp_path / "book.epub"
    assert job.output_path == tmp_path / "out"
    assert job.chapter_selection.included == [1, 3]
    assert job.narration_mode.value == "multi"
    assert job.speaker_voices["alice"] == "clara"
    assert job.job_pause_line_ms == 1200
    assert job.job_apostrophe_mode.value == "always_remove"
    assert job.job_tts_max_tokens_per_chunk == 75
    assert job.job_number_normalization is not None
    assert job.job_number_normalization.phone_numbers_mode.value == "raw"
    assert job.job_number_normalization.identifier_min_digits == 10


def test_build_processing_config_is_shared_job_boundary(tmp_path):
    job = JobConfig(
        ebook_path=tmp_path / "book.epub",
        voice="clara",
        chapter_selection=ChapterSelection(
            preset=ChapterPreset.MANUAL,
            included=[2],
        ),
        output_path=tmp_path / "audio",
        speaker_voices={"NARRATOR": "clara"},
        job_m4b_bitrate="64",
        job_post_processing_enabled=False,
        job_number_normalization=NumberNormalizationConfig(identifiers_mode="raw"),
    )
    app_config = AppConfig(
        default_voice="alba",
        workers=3,
        pause_line_ms=900,
        number_normalization=NumberNormalizationConfig(identifiers_mode="digits"),
    )

    cfg = build_processing_config(job, app_config)

    assert cfg.voice == "clara"
    assert cfg.output_path == tmp_path / "audio"
    assert cfg.workers == 3
    assert cfg.pause_line_ms == 900
    assert cfg.m4b_bitrate == "64k"
    assert cfg.chapter_filters[0].type == "index"
    assert cfg.chapter_filters[0].value == "2"
    assert cfg._included_indices == [2]
    assert cfg.post_processing.enabled is False
    assert cfg.number_normalization.identifiers_mode.value == "raw"


def test_application_service_queue_contract_persists(tmp_path):
    queue_file = tmp_path / "queue.toml"
    service = KenkuiService(queue_file=queue_file, app_config=AppConfig(default_voice="alba"))

    item = service.add_job(JobConfig(ebook_path=Path("book.epub"), voice="clara"))
    queue = service.queue()

    assert queue.pending_count == 1
    assert queue.items[0].id == item.id
    assert queue.items[0].status == "pending"
    assert queue.items[0].job["voice"] == "clara"
    assert queue_file.exists()

    restored = KenkuiService(queue_file=queue_file)
    assert restored.queue().items[0].id == item.id
    assert restored.get_job_item(item.id).status == JobStatus.PENDING


def test_application_service_task_response_preserves_dict_results(tmp_path):
    """Completed analysis tasks must expose dict payloads directly to API clients."""
    service = KenkuiService(queue_file=tmp_path / "queue.toml")
    task = Task(
        task_id="analysis-1",
        type=TaskType.FULL_ANALYSIS,
        status=TaskStatus.COMPLETED,
        progress=100,
        message="Done",
        result={"characters": [{"character_id": "alice", "display_name": "Alice"}]},
    )

    response = service.task_response(task)

    assert response.result == {
        "characters": [{"character_id": "alice", "display_name": "Alice"}]
    }


def test_application_service_list_voices_serializes_phrase_catalog_and_previews(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "kenkui.services.voice_service.list_voices",
        lambda **_kwargs: [
            SimpleNamespace(
                voice_id="alba",
                origin="pocket_tts_builtin",
                gender="Male",
                accent="American",
                dataset=None,
                speaker_id=None,
                description="Male",
                display_label="Alba",
                pool_enabled=True,
                previews=(
                    SimpleNamespace(
                        phrase_id="pride-and-prejudice",
                        audio_url="https://audio.example/alba.mp3",
                        content_type="audio/mpeg",
                        duration_ms=1234,
                        sha256="abc",
                    ),
                ),
                preview_url=None,
            )
        ],
    )
    response = KenkuiService(queue_file=tmp_path / "queue.toml").list_voices()
    payload = response.model_dump(mode="json")

    assert response.default_phrase_id == "pride-and-prejudice"
    assert len(response.phrase_catalog) == 3
    assert payload["voices"][0]["previews"][0]["phrase_id"] == "pride-and-prejudice"
    assert payload["voices"][0]["previews"][0]["audio_url"].startswith("https://")


def test_application_service_get_voice_serializes_previews(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "kenkui.services.voice_service.get_voice",
        lambda _name: SimpleNamespace(
            voice_id="alba",
            origin="pocket_tts_builtin",
            gender="Male",
            accent="American",
            dataset=None,
            speaker_id=None,
            description="Male",
            display_label="Alba",
            pool_enabled=True,
            previews=(
                SimpleNamespace(
                    phrase_id="pride-and-prejudice",
                    audio_url="https://audio.example/alba.mp3",
                    content_type="audio/mpeg",
                    duration_ms=1234,
                    sha256="abc",
                ),
            ),
            preview_url=None,
        ),
    )

    response = KenkuiService(queue_file=tmp_path / "queue.toml").get_voice("alba")

    assert response is not None
    assert response.previews[0].phrase_id == "pride-and-prejudice"
    assert response.previews[0].audio_url == "https://audio.example/alba.mp3"


def test_voice_response_previews_default_is_backward_compatible() -> None:
    voice = VoiceResponse(
        name="alba",
        source="pocket_tts_builtin",
        description="Male",
        display_label="Alba",
        excluded=False,
    )

    assert voice.previews == []


def test_application_service_resets_stale_processing_jobs(tmp_path):
    queue_file = tmp_path / "queue.toml"
    service = KenkuiService(queue_file=queue_file)
    item = service.add_job(JobConfig(ebook_path=Path("book.epub")))
    item.status = JobStatus.PROCESSING
    service._save()

    restored = KenkuiService(queue_file=queue_file)

    assert restored.get_job_item(item.id).status == JobStatus.PENDING


def test_application_service_retries_failed_job(tmp_path, monkeypatch):
    service = KenkuiService(queue_file=tmp_path / "queue.toml")
    item = service.add_job(JobConfig(ebook_path=Path("book.epub")))
    item.status = JobStatus.FAILED
    item.progress = 57.0
    item.current_chapter = "Chapter 7"
    item.eta_seconds = 99
    item.error_message = "render failed"
    item.output_path = "/tmp/failed.m4b"
    item.started_at = 100.0
    item.completed_at = 200.0
    item.provider_status = "failed"
    service._save()
    started = []
    monkeypatch.setattr(service._jobs, "start_processing", lambda: started.append(True) or True)

    assert service.retry_job(item.id)

    assert item.status == JobStatus.PENDING
    assert item.progress == 0.0
    assert item.current_chapter == ""
    assert item.eta_seconds == 0
    assert item.error_message == ""
    assert item.output_path == ""
    assert item.started_at == 0.0
    assert item.completed_at == 0.0
    assert item.provider_status == "retrying"
    assert started == [True]


def test_application_service_patch_config_merges_nested_values(tmp_path):
    service = KenkuiService(
        queue_file=tmp_path / "queue.toml",
        app_config=AppConfig(
            default_voice="alba",
            post_processing=PostProcessingConfig(enabled=True, normalize=False),
        ),
    )

    response = service.patch_config({
        "default_voice": "clara",
        "post_processing": {"normalize": True},
    })

    assert response.config["default_voice"] == "clara"
    assert response.config["post_processing"]["enabled"] is True
    assert response.config["post_processing"]["normalize"] is True


def test_application_service_logs_queue_and_job_lifecycle(tmp_path, monkeypatch, caplog):
    class SuccessfulProvider:
        def execute(
            self,
            *,
            item,
            cfg,
            app_config,
            progress_callback,
            metadata_callback,
            pause_check,
            cancel_check,
        ):
            del item, cfg, app_config, progress_callback, pause_check, cancel_check
            metadata_callback(provider_status="running")
            return ExecutionOutcome(
                success=True,
                output_path=str(tmp_path / "audio" / "storybook.m4b"),
                provider_status="completed",
            )

        def cancel(self, item):
            del item

    service = KenkuiService(queue_file=tmp_path / "queue.toml")
    monkeypatch.setattr(
        "kenkui.services.application_service.get_tts_execution_provider",
        lambda item: SuccessfulProvider(),
    )

    # Queue-lifecycle logs now originate from the extracted kenkui.services
    # submodules (queue_manager / job_executor), so capture the package logger.
    with caplog.at_level(logging.INFO, logger="kenkui.services"):
        item = service.add_job(
            JobConfig(
                ebook_path=tmp_path / "book.epub",
                output_path=tmp_path / "audio",
                name="storybook",
            )
        )
        service._process_job(item)
        other = service.add_job(JobConfig(ebook_path=tmp_path / "second.epub"))
        assert service.cancel_job(other.id)
        assert service.remove_job(other.id)
        service.clear_all_jobs()

    messages = [record.message for record in caplog.records]
    assert any("Queued job_id=" in message for message in messages)
    assert any("Processing job job_id=" in message for message in messages)
    assert any("Job completed job_id=" in message for message in messages)
    assert any("Removed queued job job_id=" in message for message in messages)
    assert any("Cleared queue removed=1" in message for message in messages)


def test_progress_update_accepts_structured_event():
    progress, chapter, eta = _progress_update_from_args(
        ProgressEvent(
            stage="tts_synthesis",
            status="advanced",
            completed_units=25,
            total_units=100,
            unit="chars",
            active_chapters=(ChapterProgress(index=2, title="Chapter 2"),),
        )
    )

    assert progress == 25
    assert chapter == "Chapter 2"
    assert eta == 0


def test_progress_update_accepts_legacy_shapes():
    assert _progress_update_from_args(40, "Chapter 4", 90) == (40.0, "Chapter 4", 90)
    assert _progress_update_from_args(55, "Attributing") == (55.0, "Attributing", 0)
    assert _progress_update_from_args("Preparing") == (0.0, "Preparing", 0)


def test_queue_processing_accepts_legacy_three_argument_progress(tmp_path, monkeypatch):
    class LegacyProgressProvider:
        def execute(
            self,
            *,
            item,
            cfg,
            app_config,
            progress_callback,
            metadata_callback,
            pause_check,
            cancel_check,
        ):
            del item, cfg, app_config, pause_check, cancel_check
            progress_callback(33, "Legacy Chapter", 12)
            metadata_callback(provider_status="completed")
            return ExecutionOutcome(success=True, output_path=str(tmp_path / "book.m4b"))

        def cancel(self, item):
            del item

    service = KenkuiService(queue_file=tmp_path / "queue.toml")
    monkeypatch.setattr(
        "kenkui.services.application_service.get_tts_execution_provider",
        lambda item: LegacyProgressProvider(),
    )
    item = service.add_job(JobConfig(ebook_path=Path("book.epub")))

    service._process_job(item)

    assert item.status == JobStatus.COMPLETED
    assert item.progress == 100.0
    assert item.current_chapter == ""
    assert item.error_message == ""


def test_queue_failure_message_names_synthesis_recovery(tmp_path, monkeypatch):
    class BrokenPipeProvider:
        def execute(
            self,
            *,
            item,
            cfg,
            app_config,
            progress_callback,
            metadata_callback,
            pause_check,
            cancel_check,
        ):
            del item, cfg, app_config, progress_callback, metadata_callback, pause_check, cancel_check
            return ExecutionOutcome(success=False, error_message="[Errno 32] Broken pipe")

        def cancel(self, item):
            del item

    service = KenkuiService(queue_file=tmp_path / "queue.toml")
    monkeypatch.setattr(
        "kenkui.services.application_service.get_tts_execution_provider",
        lambda item: BrokenPipeProvider(),
    )
    item = service.add_job(JobConfig(ebook_path=Path("book.epub")))

    service._process_job(item)

    assert item.status == JobStatus.FAILED
    assert "Synthesis failed" in item.error_message
    assert "lower worker count" in item.error_message


def test_queue_pause_keeps_job_visible_and_blocks_following_jobs(tmp_path, monkeypatch):
    class PausingProvider:
        def __init__(self):
            self.calls: list[str] = []

        def execute(
            self,
            *,
            item,
            cfg,
            app_config,
            progress_callback,
            metadata_callback,
            pause_check,
            cancel_check,
        ):
            del cfg, app_config, progress_callback, metadata_callback, pause_check, cancel_check
            self.calls.append(item.id)
            if len(self.calls) == 1:
                return ExecutionOutcome(success=False, paused=True, provider_status="paused")
            return ExecutionOutcome(success=True, output_path=str(tmp_path / "unused.m4b"))

        def cancel(self, item):
            del item

    service = KenkuiService(queue_file=tmp_path / "queue.toml")
    provider = PausingProvider()
    monkeypatch.setattr(
        "kenkui.services.application_service.get_tts_execution_provider",
        lambda item: provider,
    )
    first = service.add_job(JobConfig(ebook_path=Path("first.epub")))
    second = service.add_job(JobConfig(ebook_path=Path("second.epub")))

    service._running = True
    service._process_loop()

    assert first.status == JobStatus.PAUSED
    assert second.status == JobStatus.PENDING
    assert service.current_item is first
    assert service.status().current_job == first.id
    assert not service.remove_job(first.id)


def test_cancel_running_job_marks_cancelled_without_deleting(tmp_path, monkeypatch):
    started = threading.Event()

    class CancellableProvider:
        def __init__(self):
            self.cancel_calls: list[str] = []

        def execute(
            self,
            *,
            item,
            cfg,
            app_config,
            progress_callback,
            metadata_callback,
            pause_check,
            cancel_check,
        ):
            del cfg, app_config, progress_callback, metadata_callback, pause_check
            started.set()
            while not cancel_check():
                time.sleep(0.01)
            return ExecutionOutcome(success=False, cancelled=True, provider_status="cancelled")

        def cancel(self, item):
            self.cancel_calls.append(item.id)

    service = KenkuiService(queue_file=tmp_path / "queue.toml")
    provider = CancellableProvider()
    monkeypatch.setattr(
        "kenkui.services.application_service.get_tts_execution_provider",
        lambda item: provider,
    )
    item = service.add_job(JobConfig(ebook_path=Path("book.epub")))
    item.status = JobStatus.PROCESSING
    service._save()

    thread = threading.Thread(target=service._process_job, args=(item,), daemon=True)
    thread.start()
    assert started.wait(timeout=1)

    assert service.cancel_job(item.id)
    thread.join(timeout=2)

    assert item.status == JobStatus.CANCELLED
    assert item.provider_status == "cancelled"
    assert provider.cancel_calls == [item.id]
    assert service.current_item is None
    assert service.remove_job(item.id)


def test_http_adapter_uses_service_contract(tmp_path, monkeypatch, caplog):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kenkui.server import api

    service = KenkuiService(queue_file=tmp_path / "http-queue.toml")
    monkeypatch.setattr(api, "get_service", lambda: service)

    with caplog.at_level(logging.INFO, logger="kenkui.server.api"):
        with TestClient(api.create_app()) as client:
            response = client.get("/v1/health", headers={"X-Request-Id": "req-123"})

    assert response.status_code == 200
    body = response.json()
    assert body["api_version"] == "v1"
    assert "local-queue" in body["capabilities"]
    assert "provider-models" in body["capabilities"]
    assert "provider-credentials" in body["capabilities"]
    assert response.headers["X-Request-Id"] == "req-123"
    assert any("Server startup" in record.message for record in caplog.records)
    assert any("HTTP request request_id=req-123 method=GET path=/v1/health status=200" in record.message for record in caplog.records)


def test_http_adapter_exposes_cancel_and_remove_routes(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kenkui.server import api

    service = KenkuiService(queue_file=tmp_path / "http-queue.toml")
    monkeypatch.setattr(api, "get_service", lambda: service)
    item = service.add_job(JobConfig(ebook_path=Path("book.epub")))

    with TestClient(api.create_app()) as client:
        cancel_response = client.post(f"/v1/queue/{item.id}/cancel")
        assert cancel_response.status_code == 200
        assert service.get_job_item(item.id).status == JobStatus.CANCELLED

        remove_response = client.delete(f"/v1/queue/{item.id}")
        assert remove_response.status_code == 200
        assert service.get_job_item(item.id) is None


def test_http_adapter_exposes_retry_route(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kenkui.server import api

    service = KenkuiService(queue_file=tmp_path / "http-queue.toml")
    monkeypatch.setattr(api, "get_service", lambda: service)
    monkeypatch.setattr(service._jobs, "start_processing", lambda: True)
    item = service.add_job(JobConfig(ebook_path=Path("book.epub")))
    item.status = JobStatus.FAILED
    item.error_message = "render failed"
    service._save()

    with TestClient(api.create_app()) as client:
        retry_response = client.post(f"/v1/queue/{item.id}/retry")

    assert retry_response.status_code == 200
    body = retry_response.json()
    assert body["id"] == item.id
    assert body["status"] == "pending"
    assert body["error_message"] == ""


def test_http_adapter_exposes_config_patch_and_provider_credentials(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kenkui.server import api

    stored = {
        "openai": ProviderCredentials(api_key="sk-file-secret", default_model="gpt-4o"),
    }

    def fake_load_provider_credentials():
        return dict(stored)

    def fake_save_provider_credentials(next_credentials):
        stored.clear()
        stored.update(next_credentials)

    service = KenkuiService(queue_file=tmp_path / "http-queue.toml")
    monkeypatch.setattr(api, "get_service", lambda: service)
    monkeypatch.setattr(
        "kenkui.services.application_service.load_provider_credentials",
        fake_load_provider_credentials,
    )
    monkeypatch.setattr(
        "kenkui.services.application_service.save_provider_credentials",
        fake_save_provider_credentials,
    )
    monkeypatch.setattr(
        "kenkui.services.application_service.inject_provider_env_vars",
        lambda credentials: None,
    )

    with TestClient(api.create_app()) as client:
        patch_response = client.patch("/v1/config", json={"default_voice": "clara"})
        assert patch_response.status_code == 200
        assert patch_response.json()["config"]["default_voice"] == "clara"

        list_response = client.get("/v1/provider-credentials")
        assert list_response.status_code == 200
        openai = next(p for p in list_response.json()["providers"] if p["provider"] == "openai")
        assert openai["configured"] is True
        assert openai["default_model"] == "gpt-4o"
        assert "sk-file-secret" not in openai["masked_key_hint"]
        assert openai["masked_key_hint"] == "sk-f...cret"

        update_response = client.put(
            "/v1/provider-credentials/openrouter",
            json={"api_key": "sk-or-secret", "default_model": "openai/gpt-4.1-mini"},
        )
        assert update_response.status_code == 200
        assert update_response.json()["configured"] is True
        assert stored["openrouter"].api_key == "sk-or-secret"

        preserve_response = client.put(
            "/v1/provider-credentials/openrouter",
            json={"default_model": "anthropic/claude-sonnet-4"},
        )
        assert preserve_response.status_code == 200
        assert stored["openrouter"].api_key == "sk-or-secret"
        assert stored["openrouter"].default_model == "anthropic/claude-sonnet-4"

        delete_response = client.delete("/v1/provider-credentials/openrouter")
        assert delete_response.status_code == 200
        assert "openrouter" not in stored


def test_http_analyze_route_forwards_use_cache_false(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kenkui.server import api

    service = KenkuiService(queue_file=tmp_path / "http-queue.toml")
    captured: dict[str, object] = {}

    def fake_analyze_book(ebook_path, **kwargs):
        captured["ebook_path"] = ebook_path
        captured.update(kwargs)
        return service.task_response(
            Task(
                task_id="analysis-1",
                type=TaskType.FULL_ANALYSIS,
            )
        )

    monkeypatch.setattr(service, "analyze_book", fake_analyze_book)
    monkeypatch.setattr(api, "get_service", lambda: service)

    with TestClient(api.create_app()) as client:
        response = client.post(
            "/v1/books/analyze",
            json={"ebook_path": "/books/demo.epub", "use_cache": False},
        )

    assert response.status_code == 202
    assert captured["ebook_path"] == "/books/demo.epub"
    assert captured["use_cache"] is False


def test_http_adapter_exposes_provider_models_and_credential_test(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kenkui.server import api

    service = KenkuiService(queue_file=tmp_path / "http-queue.toml")
    monkeypatch.setattr(api, "get_service", lambda: service)
    monkeypatch.setattr(
        "kenkui.services.application_service._list_provider_models",
        lambda provider: {"provider": provider, "models": ["gpt-4o", "gpt-4o-mini"]},
    )
    monkeypatch.setattr(
        "kenkui.services.application_service.load_provider_credentials",
        lambda: {
            "openai": ProviderCredentials(api_key="sk-file-secret", default_model="gpt-4o"),
        },
    )
    monkeypatch.setattr(
        "kenkui.services.application_service._validate_provider_credentials",
        lambda provider, credentials=None: {"status": "ok", "message": f"{provider} ok"},
    )

    with TestClient(api.create_app()) as client:
        models_response = client.get("/v1/provider-models/openai")
        assert models_response.status_code == 200
        assert models_response.json() == {
            "provider": "openai",
            "models": ["gpt-4o", "gpt-4o-mini"],
        }

        test_response = client.post("/v1/provider-credentials/openai/test")
        assert test_response.status_code == 200
        assert test_response.json()["status"] == "ok"


def test_application_service_lists_analysis_cache_candidates(tmp_path, monkeypatch):
    import kenkui.nlp as nlp_module

    ebook = tmp_path / "book.epub"
    ebook.write_text("fake", encoding="utf-8")
    cache_dir = tmp_path / "nlp_cache"
    cache_dir.mkdir()
    book_hash = "abc123"
    attribution = cache_dir / f"{book_hash}-ollama-llama3_2.json"
    attribution.write_text(
        json.dumps({
            "book_hash": book_hash,
            "provider": "ollama",
            "model": "llama3.2",
            "characters": [
                {"character_id": "alice", "display_name": "Alice", "quote_count": 3, "mention_count": 5, "gender_pronoun": "she"}
            ],
            "chapters": [{"index": 1}],
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(nlp_module, "book_hash", lambda _ebook: book_hash)
    monkeypatch.setattr(nlp_module, "_get_config_dir", lambda: tmp_path)

    service = KenkuiService(queue_file=tmp_path / "queue.toml")
    result = service.analysis_cache_candidates(str(ebook))

    assert result["book_hash"] == book_hash
    assert result["candidates"][0] == {
        "cache_id": attribution.name,
        "step": "attribution",
        "provider": "ollama",
        "model": "llama3.2",
        "method": "",
        "created_at": "",
        "description": "Full attribution cache · ollama · llama3.2",
        "path": str(attribution),
        "character_count": 1,
        "chapter_count": 1,
        "quote_count": 3,
    }


def test_application_service_full_analysis_task_result_shape(tmp_path, monkeypatch):
    import kenkui.nlp as nlp_module
    import kenkui.services.nlp_service as nlp_service

    ebook = tmp_path / "book.epub"
    ebook.write_text("fake", encoding="utf-8")
    annotated = tmp_path / "annotated.json"
    progress: list[tuple[int, str]] = []

    def fake_full_analysis(**kwargs):
        kwargs["extraction_progress_callback"](100, "Extraction complete")
        kwargs["attribution_progress_callback"](100, "Attribution complete")
        return NLPResult(
            characters=[
                CharacterInfo(
                    character_id="alice",
                    display_name="Alice",
                    quote_count=4,
                    mention_count=10,
                    gender_pronoun="she",
                )
            ],
            chapters=[],
            book_hash="bookhash",
        )

    monkeypatch.setattr(nlp_service, "full_analysis", fake_full_analysis)
    monkeypatch.setattr(nlp_module, "get_cached_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(nlp_module, "attribution_cache_path", lambda *args, **kwargs: annotated)
    monkeypatch.setattr(nlp_module, "list_cached_rosters", lambda _ebook: [])

    service = KenkuiService(queue_file=tmp_path / "queue.toml")
    result = service._run_full_analysis(
        ebook_path=str(ebook),
        nlp_provider="openrouter",
        nlp_model="openai/gpt-4.1-mini",
        progress_callback=lambda pct, msg: progress.append((pct, msg)),
    )

    assert result["characters"][0]["character_id"] == "alice"
    assert result["annotated_chapters_path"] == str(annotated)
    assert result["nlp_provider"] == "openrouter"
    assert result["attribution_provider"] == "openrouter"
    assert result["cache_status"] == "miss"
    assert progress[-1] == (100, "Attribution: Attribution complete")


def test_application_service_full_analysis_can_bypass_cache(tmp_path, monkeypatch):
    import kenkui.nlp as nlp_module
    import kenkui.services.nlp_service as nlp_service

    ebook = tmp_path / "book.epub"
    ebook.write_text("fake", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_full_analysis(**kwargs):
        captured.update(kwargs)
        return NLPResult(characters=[], chapters=[], book_hash="bookhash")

    monkeypatch.setattr(nlp_service, "full_analysis", fake_full_analysis)
    monkeypatch.setattr(nlp_module, "get_cached_result", lambda *args, **kwargs: object())
    monkeypatch.setattr(nlp_module, "attribution_cache_path", lambda *args, **kwargs: tmp_path / "annotated.json")
    monkeypatch.setattr(nlp_module, "list_cached_rosters", lambda _ebook: [])

    service = KenkuiService(queue_file=tmp_path / "queue.toml")
    result = service._run_full_analysis(ebook_path=str(ebook), use_cache=False)

    assert captured["use_cache"] is False
    assert result["cache_status"] == "miss"


def test_task_runner_logs_task_lifecycle(caplog):
    from kenkui.services.task_service import TaskRegistry, TaskRunner, TaskType

    registry = TaskRegistry()
    runner = TaskRunner(registry, max_workers=1)

    def _success(*, progress_callback):
        progress_callback(50, "Halfway")
        return {"ok": True}

    with caplog.at_level(logging.INFO, logger="kenkui.services.task_service"):
        task = runner.submit(TaskType.FAST_SCAN, _success)
        runner.shutdown()

    assert task.status.value == "completed"
    messages = [record.message for record in caplog.records]
    assert any(f"task created task_id={task.task_id}" in message for message in messages)
    assert any(f"task completed task_id={task.task_id}" in message for message in messages)


def test_task_runner_logs_failures(caplog):
    from kenkui.services.task_service import TaskRegistry, TaskRunner, TaskType

    registry = TaskRegistry()
    runner = TaskRunner(registry, max_workers=1)

    def _fail(*, progress_callback):
        del progress_callback
        raise RuntimeError("boom")

    with caplog.at_level(logging.INFO, logger="kenkui.services.task_service"):
        task = runner.submit(TaskType.FAST_SCAN, _fail)
        runner.shutdown()

    assert task.status.value == "failed"
    messages = [record.message for record in caplog.records]
    assert any(f"task created task_id={task.task_id}" in message for message in messages)
    assert any(f"task execution failed task_id={task.task_id}" in message for message in messages)
    assert any(f"task failed task_id={task.task_id} error=boom" in message for message in messages)


def test_kenkui_cli_serve_runs_http_server(monkeypatch):
    from kenkui import cli

    calls = []
    monkeypatch.setattr(cli, "run_server", lambda host, port, reload: calls.append((host, port, reload)))
    monkeypatch.setattr("sys.argv", ["kenkui", "serve"])

    cli.main()

    assert calls == [("127.0.0.1", 45365, False)]


def test_register_runtimes_uses_pre_load_config_not_persisted_config(tmp_path, monkeypatch):
    """Regression: register_configured_runtimes must receive the injected/default
    AppConfig, not the config loaded from the persisted queue file.

    Before the fix, QueueManager was constructed first (running load() which
    could overwrite app_config from queue.toml) and THEN register_configured_runtimes
    was called with the post-load config.  A persisted queue.toml with
    modal_enabled=True would therefore cause RuntimeRegistrationError at
    construction even when the caller passed a safe AppConfig().
    """
    import tomli_w

    # Write a persisted queue.toml with modal_enabled=True in app_config.
    queue_file = tmp_path / "queue.toml"
    persisted_config = AppConfig(modal_enabled=True)
    queue_file.write_bytes(
        tomli_w.dumps({"items": [], "app_config": persisted_config.to_dict()}).encode()
    )

    # Monkeypatch register_configured_runtimes and record the argument it receives.
    recorded: list[AppConfig] = []

    def fake_register(cfg: AppConfig) -> None:
        recorded.append(cfg)

    monkeypatch.setattr(
        "kenkui.services.runtime_service.register_configured_runtimes",
        fake_register,
    )
    # Also patch the import inside application_service.__init__ which does a
    # local `from kenkui.services.runtime_service import register_configured_runtimes`.
    import kenkui.services.runtime_service as rts
    monkeypatch.setattr(rts, "register_configured_runtimes", fake_register)

    # Inject a plain AppConfig (modal_enabled=False — the safe, pre-load config).
    injected = AppConfig()
    assert not injected.modal_enabled

    service = KenkuiService(queue_file=queue_file, app_config=injected)

    # register_configured_runtimes should have been called exactly once with the
    # injected config (modal_enabled=False), not the persisted config (modal_enabled=True).
    assert len(recorded) == 1, f"Expected 1 call, got {len(recorded)}"
    assert recorded[0].modal_enabled is False, (
        f"register_configured_runtimes received modal_enabled={recorded[0].modal_enabled!r}; "
        "expected False (the pre-load injected config)"
    )
    # Sanity: the post-load queue config does carry the persisted value.
    assert service.app_config.modal_enabled is True


def test_openapi_export_uses_app_schema(capsys, tmp_path, monkeypatch):
    pytest.importorskip("fastapi")

    from kenkui.server import api
    from kenkui.server.openapi import main

    monkeypatch.setattr(api, "get_service", lambda: KenkuiService(queue_file=tmp_path / "openapi-queue.toml"))
    main()
    body = capsys.readouterr().out

    assert '"openapi"' in body
    assert '"/v1/health"' in body
