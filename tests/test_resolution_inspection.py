"""A resolved pipeline is an inspectable checkpoint that can resume rendering."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
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

    def __init__(self) -> None:
        """Track model work and allow a reviewed character ID in replies."""
        self.calls: list[str] = []
        self.first_speaker = "alice"

    def complete(self, model: str, prompt: str) -> str:
        """Return deterministic roster and attribution responses."""
        assert model == "fake/model"
        self.calls.append(prompt)
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
                    {"quote_id": 0, "speaker": self.first_speaker},
                    {"quote_id": 1, "speaker": "bob"},
                ]
            }
        )


@pytest.fixture
def review_client() -> _TwoSpeakers:
    """Provide a deterministic model boundary with inspectable calls."""
    return _TwoSpeakers()


@pytest.fixture
def unresolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, review_client: _TwoSpeakers
) -> kk.Pipeline:
    """Configure a two-character book against one local fake voice."""
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
    monkeypatch.setattr("kenkui._resolution._attribution_client", lambda: review_client)
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
    )


@pytest.fixture
def checkpoint(unresolved: kk.Pipeline) -> kk.Pipeline:
    """Materialize a cast ready for inspection and rendering."""
    return unresolved.resolve()


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


def test_character_checkpoint_stops_before_voices_or_attribution(
    unresolved: kk.Pipeline,
    review_client: _TwoSpeakers,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discovery is a useful checkpoint even without any configured voice."""

    def unexpected_binding(*_args: object, **_kwargs: object) -> None:
        pytest.fail("character discovery attempted voice binding")

    monkeypatch.setattr("kenkui._resolution._execution_bindings", unexpected_binding)
    discovery = kk.epub(unresolved.source.path).infer_characters("fake/model")
    characters = discovery.resolve(until="characters")
    assert isinstance(characters, kk.Pipeline)
    inspection = characters.inspect()
    assert inspection.roster is not None
    assert inspection.casting is None
    assert len(review_client.calls) == 1
    assert "List the speaking characters" in review_client.calls[0]
    assert characters.resolve(until="characters").inspect() is inspection
    assert len(review_client.calls) == 1
    assert discovery.inspect().roster is None


def test_reviewed_roster_resumes_attribution_and_rendering(
    unresolved: kk.Pipeline, review_client: _TwoSpeakers, tmp_path: Path
) -> None:
    """Edited IDs, names, aliases, narrator identity, and genders reach attribution."""
    discovery = kk.epub(unresolved.source.path).infer_characters("fake/model")
    checkpoint = discovery.resolve(until="characters")
    roster = checkpoint.inspect().roster
    assert roster is not None
    edited = replace(
        roster,
        characters=(
            replace(
                roster.characters[0],
                id="lead",
                display_name="Alice Example",
                aliases=("Alice", "Al"),
                gender="feminine",
            ),
            roster.characters[1],
        ),
        narrator_id="lead",
    )
    reviewed = checkpoint.with_characters(edited)
    review_client.calls.clear()
    review_client.first_speaker = "lead"
    attributed = (
        reviewed.attribute_quotes("fake/model")
        .assign_voices(narrator="narrator")
        .resolve()
    )
    assert len(review_client.calls) == 1
    prompt = review_client.calls[0]
    assert "List the speaking characters" not in prompt
    assert "Alice Example" in prompt
    assert '"Al"' in prompt
    assert "[narrates this book]" in prompt
    casting = attributed.inspect().casting
    assert casting is not None
    assert dict(casting.assignments) == {"lead": "narrator", "bob": "narrator"}
    assert next(c for c in casting.characters if c.id == "lead").gender == "feminine"
    assert checkpoint.inspect().roster == roster
    assert reviewed.inspect().roster != roster
    assert attributed.tts().write(tmp_path / "reviewed.m4b", workers=1).output.is_file()
    assert len(review_client.calls) == 1


@pytest.mark.parametrize(
    "invalid", ["duplicate", "narrator", "chapter", "gender", "id", "name", "alias"]
)
def test_invalid_roster_edits_fail_without_more_model_work(
    unresolved: kk.Pipeline, review_client: _TwoSpeakers, invalid: str
) -> None:
    """Refuse edits that would create ambiguous or unreachable character identities."""
    checkpoint = unresolved.resolve(until="characters")
    roster = checkpoint.inspect().roster
    assert roster is not None
    first = roster.characters[0]
    variants = {
        "duplicate": replace(roster, characters=(first, first)),
        "narrator": replace(roster, narrator_id="absent"),
        "chapter": replace(
            roster, characters=(replace(first, chapter_ids=("absent",)),)
        ),
        "gender": replace(roster, characters=(replace(first, gender="invalid"),)),
        "id": replace(roster, characters=(replace(first, id="Not a stable ID"),)),
        "name": replace(roster, characters=(replace(first, display_name=" "),)),
        "alias": replace(roster, characters=(replace(first, aliases=(" ",)),)),
    }
    with pytest.raises(kk.ValidationError) as caught:
        checkpoint.with_characters(variants[invalid])
    assert caught.value.code is kk.ErrorCode.INVALID_ROSTER
    assert len(review_client.calls) == 1
    assert checkpoint.inspect().roster == roster


def test_an_empty_reviewed_roster_keeps_all_speech_narrated(
    unresolved: kk.Pipeline, review_client: _TwoSpeakers
) -> None:
    """Removing false positives does not discard speech or rerun discovery."""
    checkpoint = unresolved.resolve(until="characters")
    reviewed = checkpoint.with_characters(kk.CharacterRoster(()))
    review_client.calls.clear()
    casting = reviewed.resolve().inspect().casting
    assert casting is not None
    assert casting.assignments == ()
    assert all(span.character_id is None for span in casting.spans)
    assert review_client.calls == []


def test_review_cannot_silently_follow_a_changed_source(
    unresolved: kk.Pipeline, review_client: _TwoSpeakers
) -> None:
    """Changing input requires rediscovery before a reviewed roster can be reused."""
    checkpoint = unresolved.resolve(until="characters")
    roster = checkpoint.inspect().roster
    assert roster is not None
    reviewed = checkpoint.with_characters(roster)
    make_epub(
        unresolved.source.path,
        chapters={"one": xhtml('<p>"New text," said Alice.</p>')},
        spine=("one",),
    )
    review_client.calls.clear()
    with pytest.raises(kk.SourceError) as caught:
        reviewed.resolve()
    assert caught.value.code is kk.ErrorCode.SOURCE_CHANGED
    assert review_client.calls == []
    refreshed = reviewed.resolve(until="characters")
    assert "New text" in refreshed.inspect().chapters[0].text
    assert "New text" not in reviewed.inspect().chapters[0].text
    assert len(review_client.calls) == 1


def test_selection_invalidates_the_roster_checkpoint(unresolved: kk.Pipeline) -> None:
    """A roster's chapter placement must not leak into a different selection."""
    checkpoint = unresolved.resolve(until="characters")
    selected = checkpoint.select_chapters(checkpoint.inspect().chapters[0].id)
    assert selected.inspect().roster is None


def test_review_requires_a_character_checkpoint(unresolved: kk.Pipeline) -> None:
    """A reviewed roster must be tied to a source snapshot."""
    with pytest.raises(kk.ValidationError) as caught:
        unresolved.with_characters(kk.CharacterRoster(()))
    assert caught.value.code is kk.ErrorCode.ROSTER_UNAVAILABLE


def test_character_resolution_honours_precancellation(
    unresolved: kk.Pipeline, review_client: _TwoSpeakers
) -> None:
    """A cancelled discovery request incurs no model work."""
    token = kk.CancellationToken()
    token.cancel()
    with pytest.raises(kk.CancelledError):
        unresolved.resolve(until="characters", cancel=token)
    assert review_client.calls == []


def test_unknown_resolution_stages_fail_at_the_api_boundary(
    unresolved: kk.Pipeline,
) -> None:
    """A misspelled checkpoint must not silently run the full conversion."""
    with pytest.raises(kk.ValidationError) as caught:
        unresolved.resolve(until="charactres")  # type: ignore[arg-type]
    assert caught.value.code is kk.ErrorCode.INVALID_RESOLUTION_STAGE
