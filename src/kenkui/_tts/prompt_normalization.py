"""Narrow typography compatibility for model prompts, not canonical book text."""

from typing import Final

PROMPT_NORMALIZATION_RECIPE: Final = "word-internal-curly-apostrophes-v1"


def normalize_prompt_apostrophes(text: str) -> str:
    """Use the trained apostrophe token only between alphabetic characters.

    Pocket's tokenizer uses byte fallbacks for curly apostrophes, which can
    change the spoken meaning of contractions. Keep every letter and all other
    punctuation exactly as supplied, including quotation marks and math primes.
    """
    return "".join(
        "'"
        if character in ("\u2018", "\u2019")
        and 0 < index < len(text) - 1
        and text[index - 1].isalpha()
        and text[index + 1].isalpha()
        else character
        for index, character in enumerate(text)
    )
