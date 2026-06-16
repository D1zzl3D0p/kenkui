import json


def test_append_chapter_attribution_writes_jsonl(tmp_path):
    from kenkui.analytics import ChapterAttributionRecord, append_chapter_attribution
    record = ChapterAttributionRecord(
        book_hash="abc123",
        chapter="Chapter 1",
        provider="litellm",
        model="anthropic/claude-haiku-4-5-20251001",
        quotes_total=20,
        quotes_attributed=18,
        quotes_unknown=2,
        retries=1,
        duration_seconds=5.4,
    )
    path = tmp_path / "analytics.jsonl"
    append_chapter_attribution(record, path=path)

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["event"] == "chapter_attributed"
    assert data["chapter"] == "Chapter 1"
    assert data["quotes_total"] == 20
    assert data["quotes_attributed"] == 18
    assert data["quotes_unknown"] == 2
    assert data["retries"] == 1
    assert data["llm_cached_prompt_tokens"] == 0
    assert data["llm_cache_write_tokens"] == 0
    assert data["llm_cost"] == 0.0
    assert "started_at" in data


def test_load_chapter_attributions_returns_records(tmp_path):
    from kenkui.analytics import (
        ChapterAttributionRecord,
        append_chapter_attribution,
        load_chapter_attributions,
    )
    path = tmp_path / "analytics.jsonl"
    r1 = ChapterAttributionRecord(
        book_hash="x", chapter="Ch 1", provider="ollama",
        model="ollama/llama3.2", quotes_total=10,
        quotes_attributed=10, quotes_unknown=0,
        retries=0, duration_seconds=2.0,
    )
    append_chapter_attribution(r1, path=path)
    records = load_chapter_attributions(path=path)
    assert len(records) == 1
    assert records[0].chapter == "Ch 1"


def test_load_chapter_attributions_ignores_stage_records(tmp_path):
    """load_chapter_attributions must only return chapter_attributed events."""
    from kenkui.analytics import (
        ChapterAttributionRecord,
        StageRecord,
        append_chapter_attribution,
        append_record,
        load_chapter_attributions,
    )
    path = tmp_path / "analytics.jsonl"
    stage = StageRecord(
        stage="nlp_extraction", started_at="2026-01-01T00:00:00+00:00",
        duration_seconds=1.0, success=True,
    )
    append_record(stage, path=path)
    ch = ChapterAttributionRecord(
        book_hash="y", chapter="Ch 2", provider="litellm",
        model="anthropic/claude-haiku-4-5-20251001", quotes_total=5,
        quotes_attributed=5, quotes_unknown=0, retries=0, duration_seconds=1.0,
    )
    append_chapter_attribution(ch, path=path)
    records = load_chapter_attributions(path=path)
    assert len(records) == 1
    assert records[0].chapter == "Ch 2"
