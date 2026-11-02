"""The adapter must never send an oversized text chunk into Pocket inference."""

from __future__ import annotations

import importlib
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import pytest
import sentencepiece as spm
from pocket_tts.models import tts_model

from kenkui._domain.planning import CastPlan
from kenkui._execution.cache import CacheStore
from kenkui._execution.checkpoints import chapter_key
from kenkui._execution.process_pool import EngineSpecification
from kenkui._tts import pocket
from kenkui._tts.token_budget import bounded_sentences
from kenkui.errors import ErrorCode, RenderError
from test_cache import _material
from test_pocket_tts import FakeModel, FakeTensor, _fixture, _installed, _task


@pytest.fixture
def tokenizer(tmp_path: Path) -> Any:
    """Train a real tiny tokenizer without downloading model weights."""
    corpus = tmp_path / "tokenizer-corpus.txt"
    corpus.write_text(
        "Investors must recognize that the junk bond mania has distorted the prices of speculative securities and undermined the discipline required to preserve capital.\n"
        * 8
    )
    prefix = str(tmp_path / "tokenizer")
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=prefix,
        model_type="char",
        vocab_size=128,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    processor = spm.SentencePieceProcessor(model_file=prefix + ".model")

    class Tokenizer:
        sp = processor

        def __call__(self, text: str) -> SimpleNamespace:
            return SimpleNamespace(
                tokens=[SimpleNamespace(tolist=lambda: self.sp.encode(text))]
            )

    return Tokenizer()


def _run_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tokenizer: Any, text: str
) -> list[str]:
    chunks = []

    class CheckedModel(FakeModel):
        def generate_audio(self, _state: object, source: str) -> FakeTensor:
            actual = tts_model.split_into_best_sentences(
                tokenizer, source, 50, False, False
            )
            for chunk in actual:
                prepared, _ = tts_model.prepare_text_prompt(chunk, False, False)
                assert len(tokenizer.sp.encode(chunk)) <= 50  # noqa: PLR2004
                assert len(tokenizer.sp.encode(prepared.strip())) <= 50  # noqa: PLR2004
            chunks.extend(actual)
            return FakeTensor([0.0] * 24)

    config, _ = _fixture(tmp_path)
    _installed(monkeypatch)
    modules = {
        "pocket_tts": SimpleNamespace(TTSModel=CheckedModel),
        "pocket_tts.models.tts_model": tts_model,
        "torch": SimpleNamespace(Tensor=FakeTensor),
    }
    original_import = importlib.import_module
    monkeypatch.setattr(
        "kenkui._tts.pocket.importlib.import_module",
        lambda name: modules.get(name) or original_import(name),
    )
    monkeypatch.setattr(pocket, "_worker_marker", pocket._WORKER_TOKEN)  # noqa: SLF001
    audio = pocket.PocketTTSEngine(config).synthesize(replace(_task(), text=text))
    assert len(audio.pcm_s16le) == 48  # noqa: PLR2004 - 24 mono int16 samples
    return chunks


def test_unpunctuated_clause_never_exceeds_actual_model_token_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tokenizer: Any
) -> None:
    """Long clauses must reach inference only as bounded complete-word chunks."""
    text = "Investors must recognize that the junk bond mania has distorted the prices of speculative securities and undermined the discipline required to preserve capital."
    chunks = _run_adapter(tmp_path, monkeypatch, tokenizer, text)
    assert " ".join(chunks) == text


def test_chunk_splitting_keeps_every_number_and_word_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tokenizer: Any
) -> None:
    """Token-bound splitting must retain ordered words and figures exactly."""
    text = "Investors paid one hundred and twenty five dollars for 125 securities and preserved capital."
    chunks = _run_adapter(tmp_path, monkeypatch, tokenizer, text)
    assert " ".join(chunks) == text


def test_splitter_is_restored_after_engine_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tokenizer: Any
) -> None:
    """The worker adapter cannot leave a modified splitter after cleanup."""
    original = tts_model.split_into_best_sentences
    _run_adapter(tmp_path, monkeypatch, tokenizer, "Investors preserve capital.")
    assert tts_model.split_into_best_sentences is original


def test_prompt_preparation_cannot_push_an_indivisible_word_over_budget(
    tokenizer: Any,
) -> None:
    """The final period counts even when an unpunctuated word fits exactly."""
    with pytest.raises(ValueError, match=r"^$"):
        bounded_sentences(
            tokenizer,
            "capital",
            8,
            False,
            False,
            prepare=tts_model.prepare_text_prompt,
        )


def test_indivisible_overlong_word_fails_without_speaking_fragments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokenizer: Any,
) -> None:
    """An unsafe single word cannot become separately pronounced fragments."""
    original = tts_model.split_into_best_sentences
    with pytest.raises(RenderError) as caught:
        _run_adapter(tmp_path, monkeypatch, tokenizer, "capital" * 12)
    assert caught.value.code is ErrorCode.POCKET_INFERENCE_FAILED
    assert tts_model.split_into_best_sentences is original


def test_raw_prompt_is_bounded_even_if_capitalization_uses_fewer_tokens() -> None:
    """Raw short-inference text can cost more than the prepared frame prompt."""
    text = "Investors preserve capital. portfolio prices depend on market conditions."

    class CaseSensitiveTokenizer:
        def __call__(self, text: str) -> SimpleNamespace:
            value = text
            count = len(value.split()) * 5
            if "Investors" in value and "portfolio" in value:
                count = 60
            elif value == "portfolio prices depend on market conditions.":
                count = 51
            elif value == "Portfolio prices depend on market conditions.":
                count = 49
            return SimpleNamespace(tokens=[SimpleNamespace(tolist=lambda: [1] * count)])

    result = bounded_sentences(
        CaseSensitiveTokenizer(),
        text,
        50,
        False,
        False,
        prepare=tts_model.prepare_text_prompt,
    )
    assert len(result) > 2  # noqa: PLR2004 - sentence plus bounded clause pieces
    assert " ".join(result) == text


def test_token_recipe_invalidates_segment_and_chapter_audio_caches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audio made by the old unsafe splitter cannot bypass the new boundary."""
    _, plan, segment, task, _ = _material(tmp_path / "book")
    config, _ = _fixture(tmp_path / "engine")
    narrator = replace(plan.cast.narrator, content_fingerprint=config.voices[0].sha256)
    plan = replace(plan, cast=CastPlan.single(narrator))
    engine = EngineSpecification.pocket(config)
    store = CacheStore(tmp_path / "cache")
    segment_before = store.key_for(plan, segment, task, engine)
    chapter_before = chapter_key(plan, segment.chapter_id, engine)
    monkeypatch.setattr(pocket, "PROMPT_CHUNKING_RECIPE", "prior-unbounded-splitter")
    assert store.key_for(plan, segment, task, engine) != segment_before
    assert chapter_key(plan, segment.chapter_id, engine) != chapter_before
