"""A catalog voice's terms follow from the dataset its recording came from."""
# ruff: noqa: D103

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kenkui.voices import datasets, registry


@pytest.mark.parametrize(
    ("origin_url", "dataset"),
    [
        ("hf://kyutai/tts-voices/vctk/p228_023_enhanced.wav", "vctk"),
        ("hf://kyutai/tts-voices/ears/p010/freeform_speech_01_enhanced.wav", "ears"),
        ("hf://kyutai/tts-voices/expresso/ex04-ex02_confused_001.wav", "expresso"),
        (
            "hf://kyutai/pocket-tts/common_voice_it_36520747-enhanced-v2.mp3@64ab",
            "common-voice",
        ),
        ("hf://kyutai/tts-voices/voice-zero/bill_boerst.wav", "kyutai-donation"),
        ("hf://kyutai/pocket-tts/de-DE-juergen.mp3@64ab", "kyutai-donation"),
    ],
)
def test_origin_names_its_dataset(origin_url: str, dataset: str) -> None:
    assert datasets.dataset_for_origin(origin_url) == dataset


def test_an_unrecognized_origin_gets_the_conservative_terms() -> None:
    """A recording from outside kyutai's catalog is unreviewed until reviewed."""
    assert datasets.dataset_for_origin("hf://someone/else/new.wav") == "unreviewed"
    rights = datasets.rights_for("unreviewed")
    assert rights.license_id == "unreviewed"
    assert rights.commercial_use_allowed is False


def test_every_catalog_voice_has_exactly_its_datasets_terms() -> None:
    for entry in registry.CATALOG.values():
        rights = datasets.rights_for(entry.dataset)
        assert (entry.license_id, entry.voice_rights) == (
            rights.license_id,
            rights.voice_rights,
        )
        assert entry.commercial_use_allowed is False


def test_voices_from_one_dataset_share_one_rights_statement() -> None:
    """EARS voices once said both "research-only/noncommercial" and "or"."""
    by_dataset: dict[str, set[str]] = {}
    for entry in registry.CATALOG.values():
        by_dataset.setdefault(entry.dataset, set()).add(entry.voice_rights)
    assert all(len(statements) == 1 for statements in by_dataset.values())


def test_the_bundled_pack_agrees_with_the_derived_licenses() -> None:
    """Drift guard: the published dataset and Kenkui's table must not diverge."""
    payload = json.loads(
        Path(registry.__file__).with_name("pack.json").read_text(encoding="utf-8")
    )
    for voice in payload["voices"]:
        rights = datasets.rights_for(datasets.dataset_for_pack(voice["dataset"]))
        assert voice["license_id"] == rights.license_id, voice["voice_id"]


def test_an_unknown_pack_dataset_is_unreviewed() -> None:
    assert datasets.dataset_for_pack("Some New Corpus") == "unreviewed"


def test_a_catalog_voice_exposes_its_rights_statement() -> None:
    voice = registry.catalog_voice("jean")
    assert voice.voice_rights is not None
    assert "EARS" in voice.voice_rights
