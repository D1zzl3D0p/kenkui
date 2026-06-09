"""Tests for public API attribution option pass-through."""

from __future__ import annotations

from unittest.mock import MagicMock


def test_full_analysis_passes_advanced_attribution_options(monkeypatch, tmp_path):
    import kenkui.services.nlp_service as nlp_service
    from kenkui import api
    from kenkui.models import NLPResult

    captured = {}

    def fake_full_analysis(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return NLPResult(characters=[], chapters=[], book_hash="abc123")

    monkeypatch.setattr(nlp_service, "full_analysis", fake_full_analysis)

    result = api.full_analysis(
        tmp_path / "book.epub",
        attribution_max_quotes_per_call=20,
        attribution_review_confidence=True,
        review_model="reviewer",
    )

    assert result.book_hash == "abc123"
    assert captured["kwargs"]["attribution_max_quotes_per_call"] == 20
    assert captured["kwargs"]["attribution_review_confidence"] is True
    assert captured["kwargs"]["review_model"] == "reviewer"


def test_attribute_only_passes_advanced_attribution_options(monkeypatch, tmp_path):
    import kenkui.services.nlp_service as nlp_service
    from kenkui import api
    from kenkui.models import Chapter, NLPResult

    captured = {}

    def fake_attribute_only(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return NLPResult(characters=[], chapters=[], book_hash="abc123")

    monkeypatch.setattr(nlp_service, "attribute_only", fake_attribute_only)

    result = api.attribute_only(
        roster=MagicMock(),
        chapters=[Chapter(index=0, title="Ch", paragraphs=["t"])],
        ebook_path=tmp_path / "book.epub",
        attribution_max_quotes_per_call=12,
        attribution_review_confidence=True,
        review_model="reviewer-small",
    )

    assert result.book_hash == "abc123"
    assert captured["kwargs"]["attribution_max_quotes_per_call"] == 12
    assert captured["kwargs"]["attribution_review_confidence"] is True
    assert captured["kwargs"]["review_model"] == "reviewer-small"
