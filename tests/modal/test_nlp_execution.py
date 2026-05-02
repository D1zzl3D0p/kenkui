from __future__ import annotations

import json
import tempfile
from pathlib import Path

from kenkui.models import AppConfig, AttributionExecutionMode, JobConfig, NlpExecutionMode
from kenkui.server.nlp_execution import NlpModalProvider, NlpStageResult
from kenkui.server.tts_execution import ModalPollResult
from kenkui.modal.storage import LocalStorageBackend


class _FakeNlpGateway:
    def __init__(self, nlp_states=None, attr_states=None):
        self._nlp_states = list(nlp_states or [])
        self._attr_states = list(attr_states or [])
        self.nlp_submitted = []
        self.attr_submitted = []
        self._last_submit = None

    def submit_nlp(self, job_id, payload):
        self.nlp_submitted.append(payload)
        self._last_submit = "nlp"
        return f"nlp:{job_id}:call1"

    def submit_attribution(self, job_id, payload):
        self.attr_submitted.append(payload)
        self._last_submit = "attr"
        return f"attr:{job_id}:call2"

    def poll_progress(self, job_id):
        if self._last_submit == "nlp" and self._nlp_states:
            return self._nlp_states.pop(0)
        if self._attr_states:
            return self._attr_states.pop(0)
        return self._nlp_states.pop(0)

    def cancel(self, call_id):
        pass


def _make_completed_poll():
    return ModalPollResult(status="completed", progress=100.0)


def _make_running_poll():
    return ModalPollResult(status="running", progress=50.0)


def test_run_nlp_stage_polls_to_completion_and_returns_roster_path(tmp_path):
    storage = LocalStorageBackend(tmp_path / "storage")
    roster_data = {"characters": [{"slug": "harry", "canonical_name": "Harry Potter"}]}
    storage.put_json("jobs/job1/nlp/roster.json", roster_data)

    gateway = _FakeNlpGateway(nlp_states=[_make_running_poll(), _make_completed_poll()])
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
        ModalPollResult(status="failed", error_message="OOM")
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

    gateway = _FakeNlpGateway(attr_states=[_make_completed_poll()])
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

    gateway = _FakeNlpGateway(attr_states=[_make_completed_poll()])
    provider = NlpModalProvider(gateway=gateway, storage=storage, poll_interval=0)

    result = provider.run_attribution_stage(
        job_id="job1",
        payload={"job_id": "job1", "chapters": []},
        progress_callback=None,
        roster_local_path=roster_path,
    )
    assert result.success is True
    assert storage.exists("jobs/job1/nlp/roster.json")
