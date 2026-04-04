"""Tests for new APIClient methods calling the service-layer endpoints."""
from __future__ import annotations
from unittest.mock import patch, MagicMock
import httpx
from kenkui.api_client import APIClient


def _client():
    return APIClient()


def _mock_response(status=200, json_data=None):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    return resp


# --- Books ---

def test_parse_book():
    c = _client()
    with patch.object(c._client, "post",
                      return_value=_mock_response(json_data={"book_hash": "abc", "chapters": []})):
        result = c.parse_book("/path/to/book.epub")
    assert result["book_hash"] == "abc"


def test_filter_chapters():
    c = _client()
    with patch.object(c._client, "post",
                      return_value=_mock_response(json_data={"included_indices": [0, 1], "chapter_count": 2,
                                                              "estimated_word_count": 1000, "chapters": []})):
        result = c.filter_chapters("abc", {"mode": "all"})
    assert result["chapter_count"] == 2


def test_scan_book():
    c = _client()
    with patch.object(c._client, "post",
                      return_value=_mock_response(json_data={"task_id": "t1", "type": "scan",
                                                              "status": "pending", "progress": 0,
                                                              "message": ""})):
        result = c.scan_book("/path/to/book.epub")
    assert result["task_id"] == "t1"


def test_scan_book_with_nlp_model():
    c = _client()
    mock_post = MagicMock(return_value=_mock_response(json_data={"task_id": "t2", "type": "scan",
                                                                  "status": "pending", "progress": 0,
                                                                  "message": ""}))
    with patch.object(c._client, "post", mock_post):
        result = c.scan_book("/path/to/book.epub", nlp_model="en_core_web_trf")
    assert result["task_id"] == "t2"
    call_kwargs = mock_post.call_args
    assert call_kwargs[1]["json"]["nlp_model"] == "en_core_web_trf"


# --- Voices ---

def test_list_voices_no_filters():
    c = _client()
    with patch.object(c._client, "get",
                      return_value=_mock_response(json_data={"voices": [], "total": 0})):
        result = c.list_voices()
    assert result["total"] == 0


def test_list_voices_with_filters():
    c = _client()
    mock_get = MagicMock(return_value=_mock_response(json_data={"voices": [], "total": 0}))
    with patch.object(c._client, "get", mock_get):
        c.list_voices(gender="female", accent="british")
    call_params = mock_get.call_args[1]["params"]
    assert call_params["gender"] == "female"
    assert call_params["accent"] == "british"
    assert "dataset" not in call_params


def test_get_voice():
    c = _client()
    with patch.object(c._client, "get",
                      return_value=_mock_response(json_data={"name": "alba", "source": "piper"})):
        result = c.get_voice("alba")
    assert result["name"] == "alba"


def test_exclude_voice():
    c = _client()
    with patch.object(c._client, "post",
                      return_value=_mock_response(json_data={"name": "alba", "excluded": True})):
        result = c.exclude_voice("alba")
    assert result["excluded"] is True


def test_include_voice():
    c = _client()
    with patch.object(c._client, "delete",
                      return_value=_mock_response(json_data={"name": "alba", "excluded": False})):
        result = c.include_voice("alba")
    assert result["excluded"] is False


def test_audition_voice():
    c = _client()
    with patch.object(c._client, "post",
                      return_value=_mock_response(json_data={"task_id": "aud1", "type": "audition",
                                                              "status": "pending", "progress": 0,
                                                              "message": ""})):
        result = c.audition_voice("alba", text="Hello world")
    assert result["task_id"] == "aud1"


def test_get_audition_wav_url():
    c = _client()
    url = c.get_audition_wav_url("task123")
    assert "task123" in url
    assert url.endswith(".wav")


def test_download_compiled_voices():
    c = _client()
    with patch.object(c._client, "post",
                      return_value=_mock_response(json_data={"task_id": "dl1", "type": "voice_download",
                                                              "status": "pending", "progress": 0,
                                                              "message": ""})):
        result = c.download_compiled_voices()
    assert result["task_id"] == "dl1"


def test_fetch_uncompiled_voices():
    c = _client()
    with patch.object(c._client, "post",
                      return_value=_mock_response(json_data={"task_id": "dl2", "type": "voice_fetch",
                                                              "status": "pending", "progress": 0,
                                                              "message": ""})):
        result = c.fetch_uncompiled_voices(repo_id="user/repo", patterns=["*.pt"])
    assert result["task_id"] == "dl2"


def test_suggest_cast():
    c = _client()
    with patch.object(c._client, "post",
                      return_value=_mock_response(json_data={"speaker_voices": {"char1": "alba"},
                                                              "warnings": []})):
        result = c.suggest_cast(roster=[{"name": "char1"}], excluded_voices=[], default_voice="narrator")
    assert result["speaker_voices"]["char1"] == "alba"


# --- Tasks ---

def test_get_task():
    c = _client()
    with patch.object(c._client, "get",
                      return_value=_mock_response(json_data={"task_id": "t1", "type": "scan",
                                                              "status": "completed", "progress": 100,
                                                              "message": "done"})):
        result = c.get_task("t1")
    assert result["status"] == "completed"


def test_poll_task_completes_immediately():
    c = _client()
    task_data = {"task_id": "t1", "status": "completed", "progress": 100, "message": "done"}
    with patch.object(c, "get_task", return_value=task_data):
        result = c.poll_task("t1", interval=0.01, timeout=5.0)
    assert result["status"] == "completed"


def test_poll_task_timeout():
    import pytest
    c = _client()
    task_data = {"task_id": "t1", "status": "pending", "progress": 0, "message": "waiting"}
    with patch.object(c, "get_task", return_value=task_data):
        with pytest.raises(TimeoutError):
            c.poll_task("t1", interval=0.01, timeout=0.05)


# --- Auth ---

def test_get_hf_status():
    c = _client()
    with patch.object(c._client, "get",
                      return_value=_mock_response(json_data={"authenticated": True, "username": "user",
                                                              "has_pocket_tts_access": True})):
        result = c.get_hf_status()
    assert result["authenticated"] is True


def test_login_hf():
    c = _client()
    with patch.object(c._client, "post",
                      return_value=_mock_response(json_data={"authenticated": True, "username": "user",
                                                              "has_pocket_tts_access": True})):
        result = c.login_hf("my_token")
    assert result["authenticated"] is True


# --- Status ---

def test_get_multivoice_status():
    c = _client()
    with patch.object(c._client, "get",
                      return_value=_mock_response(json_data={"spacy_ok": True, "spacy_model": "en_core_web_trf",
                                                              "ollama_ok": False, "ollama_url": None,
                                                              "message": "Ollama not available"})):
        result = c.get_multivoice_status()
    assert result["spacy_ok"] is True
    assert result["message"] == "Ollama not available"


# --- Series ---

def test_list_series():
    c = _client()
    with patch.object(c._client, "get",
                      return_value=_mock_response(json_data={"series": [], "total": 0})):
        result = c.list_series()
    assert result["total"] == 0


def test_get_series():
    c = _client()
    with patch.object(c._client, "get",
                      return_value=_mock_response(json_data={"slug": "my-series", "name": "My Series",
                                                              "updated_at": "", "characters": []})):
        result = c.get_series("my-series")
    assert result["slug"] == "my-series"


def test_delete_series():
    c = _client()
    with patch.object(c._client, "delete",
                      return_value=_mock_response(json_data={"status": "deleted", "slug": "my-series"})):
        result = c.delete_series("my-series")
    assert result["status"] == "deleted"


# --- Queue Cast ---

def test_get_queue_cast():
    c = _client()
    with patch.object(c._client, "get",
                      return_value=_mock_response(json_data={"job_id": "j1", "book_name": "My Book",
                                                              "narration_mode": "multi", "cast": []})):
        result = c.get_queue_cast("j1")
    assert result["job_id"] == "j1"


# --- Context manager ---

def test_context_manager():
    with APIClient() as c:
        assert c is not None
    # After exiting, _client should be closed (no assertion needed, just no error)
