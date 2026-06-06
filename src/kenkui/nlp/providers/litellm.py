"""LiteLLM-backed NLP adapters for cloud providers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel, ValidationError

from kenkui.config import inject_provider_env_vars, load_provider_credentials
from kenkui.models import Chapter
from kenkui.nlp import _count_mentions, book_hash, cache_roster, get_cached_roster
from kenkui.nlp.models import (
    AttributionItem,
    AttributionResult,
    AttributionResultWire,
    CharacterRoster,
    Quote,
    attribution_wire_to_full,
    slugify,
)

if TYPE_CHECKING:
    from kenkui.nlp_config import NLPConfig

_logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


def _litellm_model(provider: str, model: str) -> str:
    """Return the runtime LiteLLM model id while preserving stored model ids."""
    provider = provider.lower()
    if provider == "litellm":
        return model
    if provider == "openrouter":
        return model if model.startswith("openrouter/") else f"openrouter/{model}"
    if provider == "google":
        return model if model.startswith(("gemini/", "vertex_ai/")) else f"gemini/{model}"
    if provider in {"anthropic", "openai"}:
        return model
    return model


def _message_content(response: object) -> str:
    try:
        choices = response.choices  # type: ignore[attr-defined]
        message = choices[0].message
        content = getattr(message, "content", None)
    except Exception:
        try:
            content = response["choices"][0]["message"]["content"]  # type: ignore[index]
        except Exception:
            content = None
    return content if isinstance(content, str) else ""


class LiteLLMClient:
    """Structured JSON client using ``litellm.completion``."""

    def __init__(self, provider: str, model: str) -> None:
        inject_provider_env_vars(load_provider_credentials())
        self.provider = provider
        self.model = model
        self.runtime_model = _litellm_model(provider, model)

    def generate(self, prompt: str, schema: type[T]) -> T:
        import litellm

        json_schema = schema.model_json_schema()
        _logger.debug(
            "LiteLLMClient.generate: provider=%s model=%s schema=%s prompt_chars=%d schema_defs=%d",
            self.provider,
            self.runtime_model,
            schema.__name__,
            len(prompt),
            len(json_schema.get("$defs", {})) if isinstance(json_schema, dict) else 0,
        )
        try:
            response = litellm.completion(
                model=self.runtime_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": json_schema,
                        "strict": True,
                    },
                },
                temperature=0,
            )
        except Exception as exc:
            _logger.warning(
                "LiteLLMClient.generate: provider=%s model=%s schema=%s completion failed (%s)",
                self.provider,
                self.runtime_model,
                schema.__name__,
                exc,
            )
            raise
        raw = _message_content(response)
        if not raw:
            _logger.warning(
                "LiteLLMClient.generate: provider=%s model=%s schema=%s returned empty content",
                self.provider,
                self.runtime_model,
                schema.__name__,
            )
            raise ConnectionError(
                f"LiteLLM model '{self.runtime_model}' returned empty content"
            )
        try:
            return schema.model_validate_json(raw)
        except ValidationError as exc:
            _logger.warning(
                "LiteLLMClient.generate: provider=%s model=%s schema=%s validation failed response_chars=%d errors=%d",
                self.provider,
                self.runtime_model,
                schema.__name__,
                len(raw),
                len(exc.errors()),
            )
            raise


class LiteLLMExtractionAdapter:
    """Extract character rosters with a LiteLLM-compatible model."""

    def __init__(self, config: NLPConfig) -> None:
        self._config = config
        self._provider = config.extraction_tool.value

    def build_roster(
        self,
        chapters: list[Chapter],
        series_roster: CharacterRoster | None = None,
        progress_callback: Callable[[str], None] | None = None,
        step_callback: Callable[[str], None] | None = None,
        book_path: Path | None = None,
    ) -> CharacterRoster:
        from kenkui.models import CharacterInfo, FastScanResult
        from kenkui.nlp import _load_spacy_model
        from kenkui.nlp.entities import build_roster_with_llm

        if book_path is None:
            raise ValueError("LiteLLMExtractionAdapter.build_roster() requires book_path")

        cached = get_cached_roster(book_path, provider=self._provider)
        if cached is not None:
            return cached.roster

        _cb: Callable[[str], None] = progress_callback or (lambda _: None)
        llm = LiteLLMClient(self._provider, self._config.extraction_model)

        if self._config.discovery_method in {"auto", "spacy"}:
            _cb("Loading spaCy language model...")
            nlp = _load_spacy_model()
        else:
            nlp = None

        _cb("Building character roster...")
        full_text = "\n\n".join("\n\n".join(ch.paragraphs) for ch in chapters)
        roster = build_roster_with_llm(
            full_text,
            nlp,
            llm,
            method=self._config.discovery_method,
            step_callback=step_callback,
        )

        mention_counts = _count_mentions(roster, full_text)
        characters = [
            CharacterInfo(
                character_id=group.canonical_name,
                display_name=group.canonical_name,
                mention_count=mention_counts.get(group.canonical_name, 0),
                gender_pronoun=group.gender,
            )
            for group in roster.characters
        ]
        characters.sort(key=lambda c: c.mention_count, reverse=True)

        cache_roster(
            FastScanResult(roster=roster, characters=characters, book_hash=book_hash(book_path)),
            book_path,
            method=self._config.discovery_method,
            provider=self._provider,
            model=self._config.extraction_model,
        )
        return roster


class LiteLLMAttributionAdapter:
    """Attribute quote speakers with a LiteLLM-compatible model."""

    def __init__(self, config: NLPConfig) -> None:
        self._config = config
        self._provider = config.attribution_tool.value

    def attribute_chapter(
        self,
        chapter: Chapter,
        roster: CharacterRoster,
        progress_callback: Callable[[str], None] | None = None,
    ) -> AttributionResult:
        from kenkui.nlp.annotator import (
            _build_alias_to_slug,
            _build_attribution_dynamic_block,
            _build_attribution_static_block,
            annotate_chapter,
        )
        from kenkui.nlp.chunker import chunk_paragraphs
        from kenkui.nlp.quotes import extract_quotes, strip_scare_quotes

        if progress_callback:
            progress_callback("Attributing chapter via LiteLLM")

        clean_paragraphs = strip_scare_quotes(chapter.paragraphs)
        quotes = extract_quotes(clean_paragraphs)
        if not quotes:
            return AttributionResult(attributions=[])

        chapter_label = getattr(chapter, "title", None) or getattr(chapter, "index", "?")

        italic_quotes = [q for q in quotes if q.kind == "italic"]
        dialogue_quotes = [q for q in quotes if q.kind != "italic"]
        all_attributions: dict[int, AttributionItem] = {
            q.id: AttributionItem(quote_id=q.id, speaker="NARRATOR", emotion="neutral", confidence=3)
            for q in italic_quotes
        }
        if not dialogue_quotes:
            return AttributionResult(attributions=list(all_attributions.values()))

        alias_to_slug = _build_alias_to_slug(roster)
        slug_to_pronoun = {c.slug: c.gender for c in roster.characters}
        static_block = _build_attribution_static_block(roster)
        llm = LiteLLMClient(self._provider, self._config.attribution_model)

        def _attribute_single(paras: list[str], dq: list[Quote]) -> dict[int, AttributionItem]:
            annotated_text = annotate_chapter(paras, dq, alias_to_slug, slug_to_pronoun)
            prompt = f"{static_block}\n\n{_build_attribution_dynamic_block(annotated_text)}"
            result = llm.generate(prompt, AttributionResultWire)
            return {item.quote_id: item for item in attribution_wire_to_full(result).attributions}

        try:
            attributed = _attribute_single(clean_paragraphs, dialogue_quotes)
            returned = len(attributed)
            expected = len(dialogue_quotes)
            if returned < expected:
                _logger.warning(
                    "LiteLLMAttributionAdapter: chapter %r — LLM returned %d/%d dialogue quotes (missing %d)",
                    chapter_label,
                    returned,
                    expected,
                    expected - returned,
                )
            all_attributions.update(attributed)
        except Exception as exc:
            _logger.warning(
                "LiteLLMAttributionAdapter: chapter %r attribution failed (%s); chunking",
                chapter_label,
                exc,
            )
            chunks = chunk_paragraphs(clean_paragraphs, dialogue_quotes)
            for chunk_idx, chunk in enumerate(chunks, start=1):
                chunk_start = chunk.para_indices[0]
                chunk_end = chunk.para_indices[-1] + 1
                chunk_paras = clean_paragraphs[chunk_start:chunk_end]
                chunk_char_start = sum(len(p) + 2 for p in clean_paragraphs[:chunk_start])
                chunk_quote_ids = set(chunk.quote_ids)
                chunk_quotes = [
                    Quote(
                        id=q.id,
                        text=q.text,
                        para_index=q.para_index - chunk_start,
                        char_offset=q.char_offset - chunk_char_start,
                        kind=q.kind,
                    )
                    for q in dialogue_quotes
                    if q.id in chunk_quote_ids
                ]
                if not chunk_quotes:
                    continue
                try:
                    attributed = _attribute_single(chunk_paras, chunk_quotes)
                    returned = len(attributed)
                    expected = len(chunk_quotes)
                    if returned < expected:
                        _logger.warning(
                            "LiteLLMAttributionAdapter: chapter %r chunk %d/%d — "
                            "LLM returned %d/%d quotes (missing %d)",
                            chapter_label,
                            chunk_idx,
                            len(chunks),
                            returned,
                            expected,
                            expected - returned,
                        )
                    all_attributions.update(attributed)
                except Exception as chunk_exc:
                    _logger.warning(
                        "LiteLLMAttributionAdapter: chapter %r chunk %d/%d failed (%s); using Unknown",
                        chapter_label,
                        chunk_idx,
                        len(chunks),
                        chunk_exc,
                    )
                    for q in chunk_quotes:
                        all_attributions.setdefault(
                            q.id,
                            AttributionItem(
                                quote_id=q.id,
                                speaker="Unknown",
                                emotion="neutral",
                                confidence=1,
                            ),
                        )

        for q in quotes:
            all_attributions.setdefault(
                q.id,
                AttributionItem(quote_id=q.id, speaker="Unknown", emotion="neutral", confidence=1),
            )

        known_slugs = {c.slug for c in roster.characters}
        known_names = {c.canonical_name: c.slug for c in roster.characters}
        known_names.update({alias: c.slug for c in roster.characters for alias in c.aliases})
        for item in all_attributions.values():
            if item.speaker not in {"NARRATOR", "Unknown"} and item.speaker not in known_slugs:
                item.speaker = known_names.get(item.speaker, slugify(item.speaker))

        return AttributionResult(attributions=list(all_attributions.values()))


__all__ = [
    "LiteLLMAttributionAdapter",
    "LiteLLMClient",
    "LiteLLMExtractionAdapter",
    "_litellm_model",
]
