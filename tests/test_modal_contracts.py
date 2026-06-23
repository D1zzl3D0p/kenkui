from __future__ import annotations

from pathlib import Path

from kenkui.modal_runtime.contracts import (
    RemoteAttributionRequest,
    RemoteAttributionResult,
    RemoteChapterRenderRequest,
    RemoteChapterRenderResult,
    RemoteExtractionRequest,
    RemoteExtractionResult,
    RemoteTTSRequest,
    RemoteTTSResult,
)
from kenkui.models import Chapter
from kenkui.nlp.models import AttributionItem, AttributionResult, CharacterRecord, CharacterRoster


def _chapter() -> Chapter:
    return Chapter(index=1, title="One", paragraphs=["Hello there."])


def test_remote_tts_contract_round_trips_json_safe_payloads(tmp_path):
    request = RemoteTTSRequest(
        job_id="job-1",
        config={"ebook_path": str(tmp_path / "book.epub"), "output_path": str(tmp_path)},
        chapters=[_chapter().to_dict()],
        output_name="book.m4b",
    )
    restored = RemoteTTSRequest.model_validate_json(request.model_dump_json())

    assert restored.job_id == "job-1"
    assert restored.config["ebook_path"] == str(tmp_path / "book.epub")
    assert restored.chapters[0]["title"] == "One"

    result = RemoteTTSResult(success=True, artifact_uri="modal://artifact", output_filename="book.m4b")
    restored_result = RemoteTTSResult.model_validate_json(result.model_dump_json())
    assert restored_result.artifact_uri == "modal://artifact"


def test_remote_chapter_render_contract_round_trips():
    request = RemoteChapterRenderRequest(
        chapter=_chapter().to_dict(),
        config={"voice": "alba"},
        is_first_chapter=True,
    )
    result = RemoteChapterRenderResult(success=True, chapter_index=1, wav_artifact_uri="modal://wav")

    assert RemoteChapterRenderRequest.model_validate(request.model_dump()).is_first_chapter is True
    assert RemoteChapterRenderResult.model_validate(result.model_dump()).wav_artifact_uri == "modal://wav"


def test_remote_nlp_contracts_round_trip_roster_and_attribution():
    roster = CharacterRoster(
        characters=[
            CharacterRecord(slug="alice", canonical_name="Alice", aliases=["Alice"], mention_count=2)
        ]
    )
    extraction_request = RemoteExtractionRequest(
        chapters=[_chapter().to_dict()],
        config={"extraction_tool": "spacy"},
        book_path=str(Path("book.epub")),
        series_roster=roster.model_dump(mode="json"),
    )
    extraction_result = RemoteExtractionResult(roster=roster.model_dump(mode="json"))

    restored_request = RemoteExtractionRequest.model_validate_json(extraction_request.model_dump_json())
    restored_result = RemoteExtractionResult.model_validate_json(extraction_result.model_dump_json())
    assert restored_request.series_roster["characters"][0]["slug"] == "alice"
    assert CharacterRoster.model_validate(restored_result.roster).characters[0].canonical_name == "Alice"

    attribution = AttributionResult(attributions=[AttributionItem(quote_id=1, speaker="alice")])
    attribution_request = RemoteAttributionRequest(
        chapter=_chapter().to_dict(),
        roster=roster.model_dump(mode="json"),
        config={"attribution_tool": "booknlp"},
    )
    attribution_result = RemoteAttributionResult(result=attribution.model_dump(mode="json"))

    assert RemoteAttributionRequest.model_validate_json(attribution_request.model_dump_json()).chapter[
        "index"
    ] == 1
    restored_attribution = RemoteAttributionResult.model_validate_json(
        attribution_result.model_dump_json()
    )
    assert AttributionResult.model_validate(restored_attribution.result).attributions[0].speaker == "alice"
