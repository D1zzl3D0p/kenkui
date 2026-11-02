"""Model compatibility must preserve contractions and canonical source text."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from kenkui._execution.cache import CacheStore
from kenkui._execution.checkpoints import chapter_key
from kenkui._execution.process_pool import EngineSpecification
from kenkui._tts import pocket
from kenkui._tts.pocket import _WORKER_TOKEN
from test_cache import _material
from test_pocket_tts import FakeModel, FakeTensor, _fixture, _installed, _task

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("It didn\u2019t fall.", "It didn't fall."),
        ("They couldn\u2019t sell.", "They couldn't sell."),
        ("They hadn\u2019t sold.", "They hadn't sold."),
        ("You can\u2019t buy it.", "You can't buy it."),
        ("You can\u2018t buy it.", "You can't buy it."),
        ("The investor\u2019s capital.", "The investor's capital."),
        (
            "José\u2019s and O\u2019Connor\u2019s shares.",
            "José's and O'Connor's shares.",
        ),
        (
            "\u2018They didn\u2019t sell,\u2019 she said.",
            "\u2018They didn't sell,\u2019 she said.",
        ),
        ("They didn't sell.", "They didn't sell."),
        (
            "Investors\u2019 capital, \u2018quoted words\u2019, and \u2019tis.",
            "Investors\u2019 capital, \u2018quoted words\u2019, and \u2019tis.",
        ),
        (
            "Math x\u2032y and x\u2033, 5\u2019 6\u201d, and year \u201991.",
            "Math x\u2032y and x\u2033, 5\u2019 6\u201d, and year \u201991.",
        ),
    ],
)
def test_model_prompt_preserves_words_and_leaves_source_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, expected: str
) -> None:
    """Only model input changes; every source word remains canonical."""
    config, _ = _fixture(tmp_path)
    _installed(monkeypatch)
    modules = {
        "pocket_tts": SimpleNamespace(TTSModel=FakeModel),
        "pocket_tts.models.tts_model": SimpleNamespace(
            download_if_necessary=lambda _value: None
        ),
        "torch": SimpleNamespace(Tensor=FakeTensor),
    }
    monkeypatch.setattr(
        "kenkui._tts.pocket.importlib.import_module", modules.__getitem__
    )
    monkeypatch.setattr(pocket, "_worker_marker", _WORKER_TOKEN)
    FakeModel.fail = None
    task = replace(_task(), text=source)

    pocket.PocketTTSEngine(config).synthesize(task)

    assert FakeModel.text == expected
    assert task.text == source


def test_prompt_recipe_invalidates_segment_and_chapter_audio_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recipe change cannot reuse either tier of pre-normalization audio."""
    config, _ = _fixture(tmp_path / "assets")
    _, plan, segment, task, _ = _material(tmp_path / "book")
    engine = EngineSpecification.pocket(config)
    store = CacheStore(tmp_path / "cache")
    original_segment = store.key_for(plan, segment, task, engine)
    original_chapter = chapter_key(plan, segment.chapter_id, engine)

    monkeypatch.setattr(
        pocket, "PROMPT_NORMALIZATION_RECIPE", "legacy-unnormalized", raising=False
    )

    assert store.key_for(plan, segment, task, engine) != original_segment
    assert chapter_key(plan, segment.chapter_id, engine) != original_chapter
