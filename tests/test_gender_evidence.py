"""Direct model gender evidence survives validation, aggregation, and reuse."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

import kenkui as kk
from kenkui._characters import _with_current_roster, resolve_attribution, store
from kenkui._characters.attribution import attribute_chapter
from kenkui._characters.gender_evidence import apply, merge
from kenkui._characters.models import CharacterRoster
from kenkui._domain.casting import CharacterProfile

if TYPE_CHECKING:
    from pathlib import Path


def _character(gender: str | None = None) -> CharacterProfile:
    return CharacterProfile("guard", "Guard", gender, 0, ())


class Client:
    """Return independently specified attribution and gender fields."""

    def __init__(self, genders: object) -> None:
        """Keep the response field, including deliberately malformed values."""
        self.genders = genders

    def complete(self, model: str, prompt: str) -> str:
        """No gender is encoded in the role ID or an adjacent speech tag."""
        assert model
        assert "speaker_genders" in prompt
        return json.dumps(
            {
                "attributions": [{"quote_id": 0, "speaker": "guard"}],
                "speaker_genders": self.genders,
            }
        )


def _record(genders: object, *, reviewed: bool = False) -> store.AttributionRecord:
    text = 'The guard checked his list. "Next applicant."'
    inspection = kk.BookInspection(
        kk.BookMetadata("Test", "Test", cover_available=False),
        (kk.ChapterInspection("ch1", 0, "One", len(text), text),),
    )
    return resolve_attribution(
        inspection,
        "e" * 64,
        "fake/model",
        client=Client(genders),
        roster=CharacterRoster((_character("feminine" if reviewed else None),)),
        reviewed=reviewed,
    )


def test_direct_gender_survives_storage_without_a_pronoun_tag() -> None:
    """The model can connect a possessive to a speaker the regex cannot."""
    record = _record({"guard": "masculine"})
    assert record.characters[0].gender == "masculine"
    assert record.gender_evidence == (("guard", "masculine"),)
    assert store.read_attribution(record.attribution_id) == record


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {"guard": "male"},
        {"guard": True},
        {"guard": {"gender": "masculine"}},
        {"unknown": "masculine"},
        {"someone-else": "masculine"},
    ],
)
def test_invalid_or_unattributed_gender_is_ignored(value: object) -> None:
    """Optional evidence cannot invent a character or accept arbitrary values."""
    record = _record(value)
    assert record.characters[0].gender is None
    assert record.gender_evidence == ()


def test_reviewed_gender_wins_over_model_evidence() -> None:
    """An explicit user correction remains authoritative."""
    assert (
        _record({"guard": "masculine"}, reviewed=True).characters[0].gender
        == "feminine"
    )


def test_conflicting_chapters_abstain(caplog: pytest.LogCaptureFixture) -> None:
    """No majority hides an identity that may combine different people."""
    evidence = merge(
        (
            ("guard", "masculine"),
            ("guard", "feminine"),
            ("guard", "masculine"),
            ("child", "feminine"),
        )
    )
    assert evidence == (("child", "feminine"),)
    assert "attribution_gender_ambiguous" in caplog.text


def test_direct_evidence_corrects_a_roster_and_preserves_other_fields() -> None:
    """Contextual attribution can correct proximity-based roster gender."""
    character = replace(
        _character("feminine"), spoken_characters=42, aliases=("Sentinel",)
    )
    assert apply((character,), (("guard", "masculine"),)) == (
        replace(character, gender="masculine"),
    )


def test_alias_disagreement_does_not_pick_dictionary_order() -> None:
    """Two keys resolving to the same person must agree."""
    record = _record({"guard": "masculine", "Guard": "feminine"})
    assert record.gender_evidence == ()
    assert record.characters[0].gender is None


def test_old_store_migrates_without_losing_attribution(tmp_path: Path) -> None:
    """An existing row has no direct evidence until a new model pass supplies it."""
    path = tmp_path / "old.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            "CREATE TABLE attributions(attribution_id TEXT PRIMARY KEY,"
            "book_id TEXT,model_id TEXT,prompt_version TEXT,params_json TEXT);"
            "INSERT INTO attributions VALUES('old','book','model','v6','{}');"
        )
    old = store.read_attribution("old", path)
    assert old is not None
    assert old.gender_evidence == ()
    record = replace(_record({"guard": "masculine"}), attribution_id="new")
    store.write_attribution(record, path)
    assert store.read_attribution("new", path) == record


def test_offline_roster_refresh_retains_direct_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reusing a cast must not overwrite a model correction with proximity votes."""
    record = _record({"guard": "masculine"})
    monkeypatch.setattr(
        "kenkui._characters.spacy_roster.pipeline_for", lambda _: "fake"
    )
    monkeypatch.setattr(
        "kenkui._characters.spacy_roster.infer_roster",
        lambda *_args, **_kwargs: ((_character("feminine"),), None),
    )
    text = 'The guard checked his list. "Next applicant."'
    inspection = kk.BookInspection(
        kk.BookMetadata("Test", "Test", cover_available=False),
        (kk.ChapterInspection("ch1", 0, "One", len(text), text),),
    )
    refreshed = _with_current_roster(record, inspection, "spacy")
    assert refreshed.characters[0].gender == "masculine"
    assert refreshed.gender_evidence == record.gender_evidence


def test_nonexistent_quote_cannot_introduce_gender_evidence() -> None:
    """Only IDs belonging to this chapter can establish an attributed speaker."""

    class ExtraneousClient:
        def complete(self, model: str, prompt: str) -> str:
            """Return an out-of-range assignment with otherwise valid gender."""
            assert model
            assert prompt
            return json.dumps(
                {
                    "attributions": [{"quote_id": 99, "speaker": "guard"}],
                    "speaker_genders": {"guard": "masculine"},
                }
            )

    text = '"Next applicant."'
    chapter = kk.ChapterInspection("ch1", 0, "One", len(text), text)
    spans, coverage, evidence = attribute_chapter(
        chapter, (_character(),), "fake/model", client=ExtraneousClient()
    )
    assert evidence == ()
    assert coverage.answered == 0
    assert coverage.dropped == 1
    assert all(span.character_id is None for span in spans)
