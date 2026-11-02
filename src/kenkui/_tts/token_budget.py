"""Hard token bounds for the pinned Pocket-TTS text-generation boundary."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class TokenList(Protocol):
    """Token IDs supplied by Pocket's tensor-backed tokenizer."""

    def tolist(self) -> list[int]:
        """Return exact IDs for one encoded prompt."""
        ...


class EncodedText(Protocol):
    """The token sequence consumed by the pinned speech engine."""

    @property
    def tokens(self) -> Sequence[TokenList]:
        """Return the batch containing one encoded prompt."""
        ...


class Tokenizer(Protocol):
    """Pocket tokenizer boundary needed for exact budget checks."""

    def __call__(self, text: str) -> EncodedText:
        """Encode a text prompt with the engine's actual tokenizer."""
        ...


MAX_PROMPT_TOKENS = 50
PROMPT_CHUNKING_RECIPE = "raw-and-prepared-sentencepiece-50-v2"


def bounded_sentences(  # noqa: PLR0913 - exact pinned upstream callback signature
    tokenizer: Tokenizer,
    text_to_generate: str,
    max_tokens: int,
    pad_with_spaces_for_short_inputs: bool,  # noqa: FBT001 - upstream callback
    remove_semicolons: bool,  # noqa: FBT001 - upstream callback
    *,
    prepare: Callable[[str, bool, bool], tuple[str, int]],
) -> list[str]:
    """Pack exact text slices at word boundaries after measuring model prompts.

    The upstream punctuation splitter only warns for an oversized clause. Here
    sentence and clause edges are preferred, but whitespace is a hard fallback.
    Bound both the raw chunk consumed by short inference and its re-prepared
    form, because capitalization, punctuation and short-input padding can
    change token counts in either direction. Never split
    an indivisible word into separately pronounced fragments: fail closed.
    """
    budget = min(max_tokens, MAX_PROMPT_TOKENS)
    if budget < 1:
        raise ValueError
    text = prepare(
        text_to_generate, pad_with_spaces_for_short_inputs, remove_semicolons
    )[0].strip()
    words = list(re.finditer(r"\S+", text))
    chunks = []
    start = 0
    while start < len(words):
        end = start
        sentences = []
        clauses = []
        while end < len(words):
            candidate = text[words[start].start() : words[end].end()]
            prompt = prepare(
                candidate, pad_with_spaces_for_short_inputs, remove_semicolons
            )[0].strip()
            raw_count = len(tokenizer(candidate).tokens[0].tolist())
            prepared_count = len(tokenizer(prompt).tokens[0].tolist())
            if max(raw_count, prepared_count) > budget:
                break
            end += 1
            last = words[end - 1].group().rstrip("\"'”\u2019)]}")
            if last.endswith((".", "!", "?")):
                sentences.append(end)
            elif last.endswith((",", ";", ":")):
                clauses.append(end)
        if end == start:
            raise ValueError
        boundary = end
        if end < len(words):
            boundary = (sentences or clauses or [end])[-1]
        chunks.append(text[words[start].start() : words[boundary - 1].end()])
        start = boundary
    return chunks
