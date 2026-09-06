"""A resolved pipeline is an inspectable checkpoint that can resume rendering."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING

import pytest

import kenkui as kk
from kenkui._audio.m4b import FakeArtifactAssembler
from kenkui._execution.coordinator import ExecutionBindings
from kenkui._execution.process_pool import EngineSpecification
from test_epub import make_epub, xhtml

if TYPE_CHECKING:
    from pathlib import Path


class _TwoSpeakers:
    """Attribute two quotes to two characters using the real resolution path."""

    def complete(self, model: str, prompt: str) -> str:
        """Return deterministic roster and attribution responses."""
        assert model == "fake/model"
        if "List the speaking characters" in prompt:
            return json.dumps(
                {
                    "characters": [
                        {"id": name, "name": name.title()} for name in ("alice", "bob")
                    ]
                }
            )
        return json.dumps(
            {
                "attributions": [
                    {"quote_id": 0, "speaker": "alice"},
                    {"quote_id": 1, "speaker": "bob"},
                ]
            }
        )


@pytest.fixture
def checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> kk.Pipeline:
    """Resolve a two-character book against one local fake voice."""
    narrator = kk.Voice(
        id="narrator",
        name="Narrator",
        enabled=True,
        provenance="fixture",
        license_id="CC0-1.0",
        commercial_use_allowed=True,
        language="en",
        state="loaded",
        content_fingerprint="a" * 64,
        compatible_model_revisions=("fake-v1",),
    )
    bindings = ExecutionBindings(
        EngineSpecification.fake(), FakeArtifactAssembler(), narrator, "fake-v1"
    )
    monkeypatch.setattr(
        "kenkui._resolution._execution_bindings", lambda _voice_id, **_cast: bindings
    )
    monkeypatch.setattr("kenkui._resolution._attribution_client", _TwoSpeakers)
    monkeypatch.setattr("kenkui.voices.provision.list_voices", lambda: (narrator,))
    path = make_epub(
        tmp_path / "book.epub",
        chapters={"one": xhtml('<p>"Hello," said Alice. "Goodbye," said Bob.</p>')},
        spine=("one",),
    )
    return (
        kk.epub(path)
        .infer_characters("fake/model")
        .attribute_quotes("fake/model")
        .assign_voices(narrator="narrator")
        .resolve()
    )


def test_inspection_exposes_an_immutable_cast_checkpoint(
    checkpoint: kk.Pipeline,
) -> None:
    """Source, roster, offsets, and voice-sharing details can be read together."""
    assert kk.epub(checkpoint.source.path).inspect().casting is None
    inspection = checkpoint.inspect()
    casting = inspection.casting
    assert casting is not None
    assert {character.id for character in casting.characters} == {"alice", "bob"}
    assert dict(casting.assignments) == {"alice": "narrator", "bob": "narrator"}
    assert casting.collisions
    assert {span.character_id for span in casting.spans} >= {"alice", "bob"}
    assert checkpoint.inspect() is inspection
    with pytest.raises(FrozenInstanceError):
        casting.assignments = ()  # type: ignore[misc]


def test_render_operations_preserve_the_checkpoint(
    checkpoint: kk.Pipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resuming an inspected cast must not repeat model or binding resolution."""

    def unexpected_work(*_args: object, **_kwargs: object) -> None:
        pytest.fail("a render-only operation repeated resolution")

    monkeypatch.setattr("kenkui._resolution._attribution_client", unexpected_work)
    monkeypatch.setattr("kenkui._resolution._execution_bindings", unexpected_work)
    resumed = (
        checkpoint.pronounce()
        .pauses(paragraph_ms=100)
        .metadata(title="Reviewed cast")
        .tts()
    )
    assert resumed.inspect() is checkpoint.inspect()
    assert resumed.resolve() is resumed
    events: list[kk.ExecutionEvent] = []
    result = resumed.write(tmp_path / "resumed.m4b", on_event=events.append, workers=1)
    cast_event = next(event for event in events if isinstance(event, kk.CastResolved))
    assert cast_event.assignments == (("alice", "narrator"), ("bob", "narrator"))
    assert result.output.is_file()


def test_source_changes_require_explicit_reresolution(
    checkpoint: kk.Pipeline, tmp_path: Path
) -> None:
    """The reviewed cast stays inspectable but cannot be paired with new text."""
    original = checkpoint.inspect()
    make_epub(
        checkpoint.source.path,
        chapters={"one": xhtml('<p>"New words," said Alice. "Changed," said Bob.</p>')},
        spine=("one",),
    )
    assert checkpoint.inspect() is original
    output = tmp_path / "stale.m4b"
    with pytest.raises(kk.SourceError) as caught:
        checkpoint.tts().write(output, workers=1)
    assert caught.value.code is kk.ErrorCode.SOURCE_CHANGED
    assert not output.exists()
    assert not list(tmp_path.glob(".kenkui-*"))
    refreshed = checkpoint.resolve()
    assert refreshed is not checkpoint
    assert "New words" in refreshed.inspect().chapters[0].text
    assert "New words" not in original.chapters[0].text


def test_source_change_during_planning_cannot_bypass_checkpoint_validation(
    checkpoint: kk.Pipeline, tmp_path: Path
) -> None:
    """Compare the render's private snapshot, not an earlier stat or hash."""

    def change_source(event: kk.ExecutionEvent) -> None:
        if isinstance(event, kk.StageStarted) and event.stage == "planning":
            make_epub(
                checkpoint.source.path,
                chapters={"one": xhtml("<p>Different book.</p>")},
                spine=("one",),
            )

    output = tmp_path / "changed.m4b"
    with pytest.raises(kk.SourceError) as caught:
        checkpoint.tts().write(output, on_event=change_source, workers=1)
    assert caught.value.code is kk.ErrorCode.SOURCE_CHANGED
    assert not output.exists()


def test_selection_invalidates_the_cast_checkpoint(checkpoint: kk.Pipeline) -> None:
    """Selection changes the attribution input even when it selects one chapter."""
    selected = checkpoint.select_chapters(checkpoint.inspect().chapters[0].id)
    assert selected.inspect().casting is None
    assert checkpoint.inspect().casting is not None
