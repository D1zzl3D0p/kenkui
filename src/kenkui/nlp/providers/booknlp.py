"""BookNLP-backed NLP adapters.

BookNLPExtractionAdapter  — character roster extraction via BookNLP.
BookNLPAttributionAdapter — quote speaker attribution via BookNLP.
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from kenkui.models import Chapter
from kenkui.nlp.booknlp_roster import build_roster_from_booknlp
from kenkui.nlp.models import AttributionItem, AttributionResult, CharacterRoster, slugify

if TYPE_CHECKING:
    from kenkui.nlp_config import NLPConfig

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _run_booknlp(text: str, model_size: str, output_dir: str | Path) -> dict:
    """Run BookNLP on *text* and return the parsed .book JSON dict.

    Applies the transformers BertEmbeddings compat patch before running.
    Suppresses BookNLP stderr.

    Raises:
        ImportError: if the ``booknlp`` package is not installed.
        Exception:   if BookNLP processing fails for any other reason.
    """
    from kenkui.nlp.booknlp_roster import _apply_booknlp_transformers_compat

    try:
        from booknlp.booknlp import BookNLP  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "booknlp package is not installed. "
            "Install it with: pip install booknlp"
        )

    _apply_booknlp_transformers_compat()

    output_dir = Path(output_dir)
    input_file = output_dir / "book.txt"
    input_file.write_text(text, encoding="utf-8")

    model_params = {"pipeline": "entity,quote,coref", "model": model_size}
    with contextlib.redirect_stderr(io.StringIO()):
        bnlp = BookNLP("en", model_params)
        bnlp.process(str(input_file), str(output_dir), "kenkui")

    book_file = output_dir / "kenkui.book"
    return json.loads(book_file.read_text(encoding="utf-8"))


def _name_to_slug(name: str, roster: CharacterRoster) -> str:
    """Map a character name to its slug via case-insensitive roster lookup.

    Priority:
    1. Match against canonical_name (case-insensitive).
    2. Match against any alias (case-insensitive).
    3. Fall back to slugify(name).
    """
    name_lower = name.lower()
    for char in roster.characters:
        if char.canonical_name.lower() == name_lower:
            return char.slug
    for char in roster.characters:
        for alias in char.aliases:
            if alias.lower() == name_lower:
                return char.slug
    return slugify(name)


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class BookNLPExtractionAdapter:
    """Extracts character roster using BookNLP's literary NER + coreference."""

    def __init__(self, config: NLPConfig) -> None:
        self._config = config

    def build_roster(
        self,
        chapters: list[Chapter],
        series_roster: CharacterRoster | None = None,
        progress_callback: Callable[[str], None] | None = None,
        step_callback: Callable[[str], None] | None = None,
        book_path: Path | None = None,
    ) -> CharacterRoster:
        # Concatenate all chapter paragraphs
        chapter_texts = ["\n\n".join(ch.paragraphs) for ch in chapters]
        text = "\n\n".join(chapter_texts)

        if progress_callback:
            progress_callback("Extracting characters via BookNLP")

        result = build_roster_from_booknlp(text, model_size=self._config.extraction_model)

        if result is None:
            return CharacterRoster(characters=[])

        return result.roster


class BookNLPAttributionAdapter:
    """Attributes quote speakers using BookNLP's neural quote detection."""

    def __init__(self, config: NLPConfig) -> None:
        self._config = config

    def attribute_chapter(
        self,
        chapter: Chapter,
        roster: CharacterRoster,
        progress_callback: Callable[[str], None] | None = None,
    ) -> AttributionResult:
        text = "\n\n".join(chapter.paragraphs)

        if progress_callback:
            progress_callback("Attributing chapter via BookNLP")

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                data = _run_booknlp(text, self._config.attribution_model, tmpdir)

                # Build char_id → slug mapping from characters list
                char_id_to_slug: dict[int, str] = {}
                for char_data in data.get("characters", []):
                    char_id = char_data.get("id")
                    if char_id is None:
                        continue
                    proper_entries: list[dict] = (
                        char_data.get("mentions", {}).get("proper", [])
                    )
                    if proper_entries:
                        first_name = proper_entries[0].get("n", "")
                        if first_name:
                            char_id_to_slug[char_id] = _name_to_slug(first_name, roster)

                # Build attribution items from quotes list
                attributions: list[AttributionItem] = []
                for idx, quote in enumerate(data.get("quotes", []), start=1):
                    char_id = quote.get("char_id", -1)
                    if char_id == -1 or char_id not in char_id_to_slug:
                        speaker = "Unknown"
                    else:
                        speaker = char_id_to_slug[char_id]

                    attributions.append(AttributionItem(
                        quote_id=idx,
                        speaker=speaker,
                        confidence=5,
                    ))

                return AttributionResult(attributions=attributions)

        except ImportError:
            _logger.info(
                "BookNLPAttributionAdapter: booknlp not installed; returning empty attribution"
            )
            return AttributionResult(attributions=[])
        except Exception as exc:
            _logger.warning(
                "BookNLPAttributionAdapter: BookNLP attribution failed (%s); returning empty",
                exc,
            )
            return AttributionResult(attributions=[])
