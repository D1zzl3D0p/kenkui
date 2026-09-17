"""Explicit role descriptions survive attribution and constrain voice selection."""

from __future__ import annotations

import json

import pytest

import kenkui as kk
from kenkui._characters import resolve_attribution
from kenkui._characters.models import CharacterRoster
from kenkui._characters.prompts import role_gender
from kenkui._domain.casting import CastingRequest, CharacterProfile, solve
from kenkui.voices.types import PerceivedGender, Voice


@pytest.mark.parametrize(
    ("role", "gender"),
    [
        ("male-proctor", "masculine"),
        ("female-proctor", "feminine"),
        ("first-male-guard", "masculine"),
        ("second-female-examiner", "feminine"),
        ("old-woman", "feminine"),
        ("boy", "masculine"),
        ("proctor", None),
        ("woman-with-a-boy", None),
        ("female-proctor-assistant", None),
        ("male-female-guard", None),
        ("male-female", None),
        ("femalevolent", None),
    ],
)
def test_role_gender(role: str, gender: str | None) -> None:
    """Only an explicit gender modifying the role itself is evidence."""
    assert role_gender(role) == gender


def test_two_proctors_keep_distinct_genders_and_voices() -> None:
    """An action-tag scene retains model-resolved identities through casting."""
    text = (
        'The proctor checked his list. "Next applicant." '
        'He pointed to the female proctor. "Follow her." '
        'The second proctor opened her ledger. "Your name, please."'
    )
    chapter = kk.ChapterInspection("ch1", 0, "One", len(text), text)
    inspection = kk.BookInspection(
        kk.BookMetadata("Examination", "Test", cover_available=False), (chapter,)
    )

    class Client:
        def complete(self, model: str, prompt: str) -> str:
            """Supply resolved roles without pretending to test model reasoning."""
            assert model
            assert "male-proctor" in prompt
            assert "female-proctor" in prompt
            return json.dumps(
                {
                    "attributions": [
                        {"quote_id": 0, "speaker": "male-proctor"},
                        {"quote_id": 1, "speaker": "male-proctor"},
                        {"quote_id": 2, "speaker": "female-proctor"},
                    ]
                }
            )

    record = resolve_attribution(
        inspection,
        "b" * 64,
        "fake/model",
        client=Client(),
        roster=CharacterRoster(
            (CharacterProfile("applicant", "Applicant", None, 0, ()),)
        ),
    )
    assert {c.id: c.gender for c in record.characters} == {
        "role:male-proctor@ch1": "masculine",
        "role:female-proctor@ch1": "feminine",
    }
    voice_traits: tuple[tuple[str, PerceivedGender], ...] = (
        ("male", "masculine"),
        ("female", "feminine"),
    )
    voices = tuple(
        Voice(
            id=name,
            name=name,
            enabled=True,
            provenance="test",
            license_id="CC0",
            commercial_use_allowed=True,
            language="english",
            state="loaded",
            perceived_gender=gender,
        )
        for name, gender in voice_traits
    )
    cast = solve(
        CastingRequest(
            record.characters, voices, {}, "narrator", "narrator", "gendered"
        )
    )
    assert dict(cast.assignments) == {
        "role:male-proctor@ch1": "male",
        "role:female-proctor@ch1": "female",
    }
