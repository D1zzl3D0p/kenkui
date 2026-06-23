from __future__ import annotations

from unittest.mock import MagicMock

from kenkui.modal_runtime.nlp import ModalAttributionProvider, ModalExtractionProvider
from kenkui.modal_runtime.tts import ModalTTSProvider
from kenkui.models import AppConfig, Chapter
from kenkui.nlp.models import AttributionItem, AttributionResult, CharacterRecord, CharacterRoster
from kenkui.services.execution_service import ExecutionOutcome


class _Job:
    tts_execution_mode = type("Mode", (), {"value": "modal"})()


class _Item:
    id = "job-1"
    job = _Job()


def test_modal_tts_provider_delegates_to_injected_client():
    outcome = ExecutionOutcome(success=True, output_path="/tmp/book.m4b", provider_status="completed")
    client = MagicMock()
    client.execute.return_value = outcome
    metadata = MagicMock()
    provider = ModalTTSProvider(app_config=AppConfig.from_dict({"modal_enabled": True}), client=client)

    result = provider.execute(
        item=_Item(),
        cfg=object(),
        app_config=object(),
        progress_callback=None,
        metadata_callback=metadata,
        pause_check=None,
        cancel_check=None,
    )

    assert result is outcome
    metadata.assert_any_call(provider_status="queued", execution_provider="modal")
    assert client.execute.call_args.kwargs["runtime_config"].enabled is True


def test_modal_tts_provider_reports_failure_without_client():
    provider = ModalTTSProvider(app_config=AppConfig.from_dict({"modal_enabled": True}))

    result = provider.execute(
        item=_Item(),
        cfg=object(),
        app_config=object(),
        progress_callback=None,
        metadata_callback=None,
        pause_check=None,
        cancel_check=None,
    )

    assert result.success is False
    assert result.provider_status == "failed"
    assert "no remote client" in result.error_message


def test_modal_nlp_providers_delegate_and_emit_progress():
    roster = CharacterRoster(
        characters=[CharacterRecord(slug="alice", canonical_name="Alice", aliases=["Alice"])]
    )
    attribution = AttributionResult(attributions=[AttributionItem(quote_id=1, speaker="alice")])
    client = MagicMock()
    client.build_roster.return_value = roster
    client.attribute_chapter.return_value = attribution
    progress_messages: list[str] = []
    step_messages: list[str] = []
    chapter = Chapter(index=0, title="One", paragraphs=["Hi."])

    extraction = ModalExtractionProvider(config=object(), client=client)
    assert extraction.build_roster(
        [chapter],
        progress_callback=progress_messages.append,
        step_callback=step_messages.append,
    ) is roster

    attribution_provider = ModalAttributionProvider(config=object(), client=client)
    assert attribution_provider.attribute_chapter(
        chapter,
        roster,
        progress_callback=progress_messages.append,
    ) is attribution

    assert "Submitting character discovery to Modal" in progress_messages
    assert "Character discovery complete" in progress_messages
    assert "Submitting quote attribution to Modal" in progress_messages
    assert "Quote attribution complete" in progress_messages
    assert step_messages == ["Extracted characters"]
