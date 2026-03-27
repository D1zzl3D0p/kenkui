"""Stage 2a: BookNLP-powered character roster extraction.

BookNLP uses a literary-fiction-specific NER model and neural coreference
resolution to cluster all name forms of the same character into one group.
This produces significantly more complete and accurate rosters than generic
NER models (e.g. spaCy en_core_web_sm) on narrative prose.

When ``booknlp`` is not installed this module returns ``None`` from its
public function, allowing the caller to fall through to the LLM or
heuristic tiers without failing.

Public API
----------
build_roster_from_booknlp(text, model_size) → CharacterRoster | None
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import CharacterRoster

logger = logging.getLogger(__name__)


@dataclass
class BookNLPRosterData:
    """Output of ``build_roster_from_booknlp`` — roster plus raw common-noun phrases."""
    roster: "CharacterRoster"
    common_phrases: list[str]


_MIN_COMMON_FREQ = 5


def _apply_booknlp_transformers_compat() -> bool:
    """Patch BertEmbeddings so BookNLP checkpoints saved with older transformers load cleanly.

    transformers >=4.35 changed ``BertEmbeddings.position_ids`` from
    ``persistent=True`` to ``persistent=False``, so it no longer appears in
    ``state_dict()``.  BookNLP's bundled model was saved when it was persistent,
    so loading it raises "Unexpected key(s) in state_dict: bert.embeddings.position_ids".

    Fix: wrap ``BertEmbeddings.__init__`` to re-register ``position_ids`` as a
    persistent buffer.  The patch is applied at most once (guarded by a class
    attribute) and is scoped to the transformers BertEmbeddings class only.
    """
    try:
        import torch  # noqa: F401 — needed for tensor clone
        from transformers.models.bert.modeling_bert import BertEmbeddings

        if getattr(BertEmbeddings, "_kenkui_position_ids_patched", False):
            return True  # Already applied

        _orig_init = BertEmbeddings.__init__

        def _patched_init(self, config):
            _orig_init(self, config)
            if hasattr(self, "position_ids"):
                # Re-register as persistent so the saved key is accepted on load
                self.register_buffer("position_ids", self.position_ids.clone(), persistent=True)

        BertEmbeddings.__init__ = _patched_init
        BertEmbeddings._kenkui_position_ids_patched = True
        logger.debug("Applied BertEmbeddings position_ids compatibility patch for BookNLP")
        return True
    except Exception as exc:
        logger.debug("BertEmbeddings compat patch not applied: %s", exc)
        return False


# BookNLP pipeline: entity + quote + coreference.
# "quote" must be included because coref shares its BERT pass and depends on it.
# The .quotes output file is generated but intentionally ignored — the LLM
# handles speaker attribution at Stage 4.
_PIPELINE = "entity,quote,coref"
_BOOK_ID = "kenkui"


def build_roster_from_booknlp(
    text: str,
    model_size: str = "small",
) -> "BookNLPRosterData | None":
    """Run BookNLP on *text* and return a ``CharacterRoster``.

    Uses only the ``entity,coref`` pipeline components — quote attribution
    is deliberately skipped because the LLM (Stage 4) handles that.

    Returns ``None`` when the ``booknlp`` package is not installed so the
    caller can fall through to the next tier without raising.

    Args:
        text:       Full book text as a single string.
        model_size: BookNLP model size — ``"small"`` (CPU-friendly) or
                    ``"big"`` (more accurate, requires GPU).

    Returns:
        ``CharacterRoster`` or ``None``.
    """
    try:
        from booknlp.booknlp import BookNLP  # type: ignore[import]
    except ImportError:
        logger.info("build_roster_from_booknlp: booknlp not installed; skipping")
        return None

    from .entities import _cluster_by_heuristic
    from .models import AliasGroup, CharacterRoster

    logger.info(
        "build_roster_from_booknlp: running BookNLP (%s model, %d chars of text)",
        model_size, len(text),
    )

    _apply_booknlp_transformers_compat()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "book.txt"
            input_file.write_text(text, encoding="utf-8")

            model_params = {"pipeline": _PIPELINE, "model": model_size}
            with contextlib.redirect_stderr(io.StringIO()):
                bnlp = BookNLP("en", model_params)
                bnlp.process(str(input_file), tmpdir, _BOOK_ID)

            book_file = Path(tmpdir) / f"{_BOOK_ID}.book"
            data = json.loads(book_file.read_text(encoding="utf-8"))

    except Exception as exc:
        err_str = str(exc)
        if "position_ids" in err_str or "state_dict" in err_str:
            logger.warning(
                "build_roster_from_booknlp: BookNLP is incompatible with the "
                "installed transformers version (state_dict mismatch: %s). "
                "Falling back to spaCy + LLM tier.",
                err_str.split("\n")[0],
            )
        else:
            logger.warning("build_roster_from_booknlp: BookNLP run failed (%s)", exc)
        return None

    # ---- Parse .book JSON → AliasGroup list --------------------------------
    name_to_gender: dict[str, str] = {}
    all_aliases: list[str] = []
    char_count = 0

    for char_data in data.get("characters", []):
        mentions: dict = char_data.get("mentions", {})
        proper_entries: list[dict] = mentions.get("proper", [])

        if not proper_entries:
            continue

        aliases = [entry["n"] for entry in proper_entries if entry.get("n")]
        if not aliases:
            continue

        # Read BookNLP's neural gender prediction.
        raw_gender = char_data.get("g", {}).get("argmax", "") or ""
        if raw_gender in ("he/him", "she/her", "they/them"):
            gender = raw_gender
        elif raw_gender in ("he", "him", "his"):
            gender = "he/him"
        elif raw_gender in ("she", "her", "hers"):
            gender = "she/her"
        elif raw_gender in ("they", "them", "their"):
            gender = "they/them"
        else:
            gender = ""

        for name in aliases:
            name_to_gender[name] = gender
        all_aliases.extend(aliases)
        char_count += 1

    # Collect high-frequency common-noun mentions (epithet candidates).
    common_phrases: list[str] = []
    seen_phrases: set[str] = set()
    for char_data in data.get("characters", []):
        for entry in char_data.get("mentions", {}).get("common", []):
            phrase = (entry.get("n") or "").strip()
            count = entry.get("c", 0)
            if phrase and count >= _MIN_COMMON_FREQ and phrase.lower() not in seen_phrases:
                common_phrases.append(phrase)
                seen_phrases.add(phrase.lower())

    logger.info(
        "build_roster_from_booknlp: %d BookNLP character entries, %d alias forms, "
        "%d common phrases",
        char_count, len(all_aliases), len(common_phrases),
    )

    if not all_aliases:
        logger.warning("build_roster_from_booknlp: no proper-name characters found")
        return BookNLPRosterData(roster=CharacterRoster(characters=[]), common_phrases=[])

    groups = _cluster_by_heuristic(all_aliases)

    # Annotate each group with BookNLP's gender.
    for group in groups:
        group.gender = name_to_gender.get(group.canonical, "")
        if not group.gender:
            for alias in group.aliases:
                if name_to_gender.get(alias):
                    group.gender = name_to_gender[alias]
                    break

    logger.info(
        "build_roster_from_booknlp: %d canonical characters after clustering",
        len(groups),
    )
    return BookNLPRosterData(
        roster=CharacterRoster(characters=groups),
        common_phrases=common_phrases,
    )
