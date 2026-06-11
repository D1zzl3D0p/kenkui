"""LiteLLM-backed NLP adapters for cloud providers."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

from kenkui.analytics import ChapterAttributionRecord, append_chapter_attribution
from kenkui.config import inject_provider_env_vars, load_provider_credentials
from kenkui.models import Chapter
from kenkui.nlp import _count_mentions, book_hash, cache_roster, get_cached_roster
from kenkui.nlp.llm import _is_eof_truncation, _try_recover_truncated_json
from kenkui.nlp.models import (
    AttributionItem,
    AttributionResult,
    AttributionResultConfidenceWire,
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
_DEFAULT_REMOTE_CONTEXT_TOKENS = 32768
_DEFAULT_REMOTE_OUTPUT_TOKENS = 8192
_MIN_REMOTE_OUTPUT_TOKENS = 256
_REMOTE_CONTEXT_SAFETY_TOKENS = 128
_ATTRIBUTION_RESPONSE_TOKENS_PER_QUOTE = 8
_REASONING_ATTRIBUTION_MIN_OUTPUT_TOKENS = 2048
_MAX_RETRIES = 2
_MISSING_RETRY_BATCH_SIZE = 4
_PROVIDER_LIST_RE = re.compile(r"(?:^|\n)\s*Provider List:\s*https://docs\.litellm\.ai/docs/providers\s*", re.I)
_QUOTE_TAG_RE = re.compile(r"\[QUOTE:(\d+)([^\]]*)\]")
_QUOTE_ATTR_RE = re.compile(r'(hint|guess|pronoun)="([^"]+)"')
_OPENROUTER_REASONING_EFFORT_ENV = "KENKUI_NLP_OPENROUTER_REASONING_EFFORT"
_OPENROUTER_REASONING_EFFORT_DEFAULT = "minimal"
_OPENROUTER_REASONING_EFFORTS = {"minimal", "low", "medium", "high"}


def _run_coroutine_sync(coro):
    """Run an async LiteLLM call from the synchronous library API."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


def _openrouter_extra_body(provider: str) -> dict[str, object]:
    if provider.lower() != "openrouter":
        return {}
    return {
        "provider": {
            "require_parameters": True,
        },
    }


def _is_openrouter_provider(provider: str) -> bool:
    return provider.lower() == "openrouter"


def _is_reasoning_model(model: str) -> bool:
    model_norm = model.lower()
    if model_norm.startswith("openrouter/"):
        model_norm = model_norm.removeprefix("openrouter/")
    return model_norm.startswith("openai/gpt-5")


def _openrouter_reasoning_effort() -> str:
    raw = os.environ.get(
        _OPENROUTER_REASONING_EFFORT_ENV,
        _OPENROUTER_REASONING_EFFORT_DEFAULT,
    ).strip().lower()
    if raw in _OPENROUTER_REASONING_EFFORTS:
        return raw
    _logger.warning(
        "Ignoring invalid %s=%r; expected one of %s",
        _OPENROUTER_REASONING_EFFORT_ENV,
        raw,
        ", ".join(sorted(_OPENROUTER_REASONING_EFFORTS)),
    )
    return _OPENROUTER_REASONING_EFFORT_DEFAULT


def _openrouter_reasoning_body(model: str) -> dict[str, object]:
    if not _is_reasoning_model(model):
        return {}
    return {
        "reasoning": {
            "effort": _openrouter_reasoning_effort(),
            "exclude": True,
        },
    }


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


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        _logger.warning("Ignoring invalid %s=%r; expected integer", name, raw)
        return default
    return value if value > 0 else default


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _remote_context_tokens(model: str) -> int:
    configured = os.environ.get("KENKUI_NLP_REMOTE_CONTEXT_TOKENS")
    if configured is not None:
        return _env_int("KENKUI_NLP_REMOTE_CONTEXT_TOKENS", _DEFAULT_REMOTE_CONTEXT_TOKENS)
    try:
        import litellm.utils as _lu
        info = _lu.get_model_info(model)
        tokens = info.get("max_input_tokens") or info.get("max_tokens")
        if isinstance(tokens, int) and tokens > 0:
            return tokens
    except Exception:
        pass
    model_norm = model.lower()
    if "microsoft/phi-4" in model_norm or model_norm.endswith("phi-4"):
        return 16384
    return _DEFAULT_REMOTE_CONTEXT_TOKENS


def _remote_output_tokens() -> int:
    return _env_int("KENKUI_NLP_REMOTE_OUTPUT_TOKENS", _DEFAULT_REMOTE_OUTPUT_TOKENS)


def _remote_attribution_call_budget(prompt: str, quote_count: int, model: str) -> tuple[int, int, int]:
    prompt_tokens = _estimate_tokens(prompt)
    context_tokens = _remote_context_tokens(model)
    output_limit = _remote_output_tokens()
    min_output_tokens = (
        _REASONING_ATTRIBUTION_MIN_OUTPUT_TOKENS
        if _is_reasoning_model(model)
        else _MIN_REMOTE_OUTPUT_TOKENS
    )
    expected_response_tokens = max(min_output_tokens, quote_count * _ATTRIBUTION_RESPONSE_TOKENS_PER_QUOTE)
    available_output_tokens = context_tokens - prompt_tokens - _REMOTE_CONTEXT_SAFETY_TOKENS
    max_tokens = min(output_limit, expected_response_tokens, available_output_tokens)
    return prompt_tokens, context_tokens, max(1, max_tokens)


def _strip_provider_list(text: str) -> str:
    return _PROVIDER_LIST_RE.sub("\n", text).strip()


def _sanitize_litellm_error(exc: Exception) -> str:
    message = _strip_provider_list(str(exc))
    return message or exc.__class__.__name__


def _is_openrouter_parameter_routing_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "openrouter" in message
        and "no endpoints found" in message
        and "requested parameters" in message
    )


def _strip_code_fence(raw: str) -> str:
    text = raw.strip()
    if not text.startswith("```"):
        return text
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.I | re.S)
    return match.group(1).strip() if match else text


def _extract_json_object(raw: str) -> str:
    """Return the first balanced top-level JSON object, ignoring wrapper text."""
    text = _strip_provider_list(_strip_code_fence(raw))
    start = text.find("{")
    if start < 0:
        return text

    depth = 0
    in_string = False
    escaped = False
    for idx, char in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    return text[start:]


def _validate_litellm_json(raw: str, schema: type[T]) -> T:
    """Validate LiteLLM JSON, tolerating common model/provider wrappers."""
    try:
        return schema.model_validate_json(raw)
    except ValidationError as original_exc:
        normalized = _extract_json_object(raw)
        if normalized != raw:
            try:
                return schema.model_validate_json(normalized)
            except ValidationError:
                pass
        if _is_eof_truncation(original_exc):
            recovered = _try_recover_truncated_json(normalized, schema)
            if recovered is not None:
                _logger.warning(
                    "LiteLLMClient.generate: recovered partial %s response from truncated JSON",
                    schema.__name__,
                )
                return recovered
        raise original_exc


def _normalize_strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a provider-safe strict JSON schema without changing Pydantic models."""
    normalized = deepcopy(schema)

    def visit(node: object) -> None:
        if isinstance(node, list):
            for item in node:
                visit(item)
            return
        if not isinstance(node, dict):
            return

        node.pop("default", None)
        properties = node.get("properties")
        node_type = node.get("type")
        if node_type == "object" or isinstance(properties, dict):
            node["additionalProperties"] = False
            if isinstance(properties, dict):
                node["required"] = list(properties)

        for value in node.values():
            visit(value)

    visit(normalized)
    return normalized


def _schema_for_provider(provider: str, schema: type[BaseModel]) -> dict[str, Any]:
    json_schema = schema.model_json_schema()
    if _is_openrouter_provider(provider):
        return _normalize_strict_json_schema(json_schema)
    return json_schema


def _litellm_completion_kwargs(
    *,
    provider: str,
    runtime_model: str,
    prompt: str,
    schema_name: str,
    json_schema: dict[str, Any],
    max_tokens: int | None,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "model": runtime_model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": json_schema,
                "strict": True,
            },
        },
    }
    if _is_openrouter_provider(provider):
        extra_body = _openrouter_extra_body(provider)
        extra_body.update(_openrouter_reasoning_body(runtime_model))
        kwargs["extra_body"] = extra_body
    else:
        kwargs["temperature"] = 0
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return kwargs


def _get_attr_or_key(value: object, key: str) -> object:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _first_choice(response: object) -> object | None:
    choices = _get_attr_or_key(response, "choices")
    if isinstance(choices, (list, tuple)) and choices:
        return choices[0]
    return None


def _message_content(response: object) -> str:
    choice = _first_choice(response)
    message = _get_attr_or_key(choice, "message") if choice is not None else None
    content = _get_attr_or_key(message, "content")
    if isinstance(content, list):
        text_parts = [
            item.get("text")
            for item in content
            if isinstance(item, dict)
            and item.get("type") in {None, "text", "output_text"}
            and isinstance(item.get("text"), str)
        ]
        content = "".join(text_parts)
    return content if isinstance(content, str) else ""


def _response_diagnostics(response: object) -> dict[str, object]:
    choice = _first_choice(response)
    message = _get_attr_or_key(choice, "message") if choice is not None else None
    usage = _get_attr_or_key(response, "usage")
    if not isinstance(usage, dict):
        usage = {
            key: _get_attr_or_key(usage, key)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if _get_attr_or_key(usage, key) is not None
        }
    message_keys: list[str] = []
    if isinstance(message, dict):
        message_keys = sorted(str(key) for key in message)
    elif message is not None:
        try:
            message_keys = sorted(str(key) for key in vars(message))
        except TypeError:
            message_keys = []
    return {
        "finish_reason": _get_attr_or_key(choice, "finish_reason"),
        "native_finish_reason": _get_attr_or_key(choice, "native_finish_reason"),
        "refusal": _get_attr_or_key(message, "refusal"),
        "usage": usage,
        "message_keys": message_keys,
    }


class LiteLLMClient:
    """Structured JSON client using ``litellm.completion``."""

    def __init__(self, provider: str, model: str) -> None:
        inject_provider_env_vars(load_provider_credentials())
        self.provider = provider
        self.model = model
        self.runtime_model = _litellm_model(provider, model)

    def _completion_kwargs(
        self,
        prompt: str,
        schema: type[BaseModel],
        *,
        max_tokens: int | None = None,
    ) -> dict[str, object]:
        json_schema = _schema_for_provider(self.provider, schema)
        _logger.debug(
            "LiteLLMClient.generate: provider=%s model=%s schema=%s prompt_chars=%d max_tokens=%s schema_defs=%d",
            self.provider,
            self.runtime_model,
            schema.__name__,
            len(prompt),
            max_tokens,
            len(json_schema.get("$defs", {})) if isinstance(json_schema, dict) else 0,
        )
        return _litellm_completion_kwargs(
            provider=self.provider,
            runtime_model=self.runtime_model,
            prompt=prompt,
            schema_name=schema.__name__,
            json_schema=json_schema,
            max_tokens=max_tokens,
        )

    def generate(self, prompt: str, schema: type[T], *, max_tokens: int | None = None) -> T:
        return _run_coroutine_sync(self.generate_async(prompt, schema, max_tokens=max_tokens))

    async def generate_async(
        self,
        prompt: str,
        schema: type[T],
        *,
        max_tokens: int | None = None,
    ) -> T:
        import litellm

        last_exc: Exception | None = None
        kwargs = self._completion_kwargs(prompt, schema, max_tokens=max_tokens)
        for attempt in range(_MAX_RETRIES + 1):
            try:
                if hasattr(litellm, "acompletion"):
                    response = await litellm.acompletion(**kwargs)
                else:
                    response = await asyncio.to_thread(litellm.completion, **kwargs)
            except Exception as exc:
                last_exc = exc
                _logger.debug(
                    "LiteLLMClient.generate: provider=%s model=%s schema=%s max_tokens=%s "
                    "completion failed raw_error=%s",
                    self.provider,
                    self.runtime_model,
                    schema.__name__,
                    max_tokens,
                    exc,
                )
                if _is_openrouter_parameter_routing_error(exc):
                    _logger.warning(
                        "LiteLLMClient.generate: provider=%s model=%s schema=%s max_tokens=%s "
                        "strict OpenRouter parameter routing failed (%s)",
                        self.provider,
                        self.runtime_model,
                        schema.__name__,
                        max_tokens,
                        _sanitize_litellm_error(exc),
                    )
                else:
                    _logger.warning(
                        "LiteLLMClient.generate: provider=%s model=%s schema=%s max_tokens=%s "
                        "completion failed (%s)",
                        self.provider,
                        self.runtime_model,
                        schema.__name__,
                        max_tokens,
                        _sanitize_litellm_error(exc),
                    )
                raise
            raw = _message_content(response)
            if not raw:
                last_exc = ConnectionError(
                    f"LiteLLM model '{self.runtime_model}' returned empty content"
                )
                diagnostics = _response_diagnostics(response)
                _logger.warning(
                    "LiteLLMClient.generate: provider=%s model=%s schema=%s returned empty content "
                    "finish_reason=%r native_finish_reason=%r refusal=%r usage=%r message_keys=%r",
                    self.provider,
                    self.runtime_model,
                    schema.__name__,
                    diagnostics["finish_reason"],
                    diagnostics["native_finish_reason"],
                    diagnostics["refusal"],
                    diagnostics["usage"],
                    diagnostics["message_keys"],
                )
                if attempt < _MAX_RETRIES:
                    continue
                raise last_exc
            try:
                return _validate_litellm_json(raw, schema)
            except ValidationError as exc:
                last_exc = exc
                _logger.debug(
                    "LiteLLMClient.generate: provider=%s model=%s schema=%s attempt=%d "
                    "validation failed raw_response=%r",
                    self.provider,
                    self.runtime_model,
                    schema.__name__,
                    attempt + 1,
                    raw,
                )
                if attempt < _MAX_RETRIES:
                    _logger.debug(
                        "LiteLLMClient.generate: provider=%s model=%s schema=%s "
                        "validation failed response_chars=%d errors=%d; retrying",
                        self.provider,
                        self.runtime_model,
                        schema.__name__,
                        len(raw),
                        len(exc.errors()),
                    )
                    continue
                _logger.warning(
                    "LiteLLMClient.generate: provider=%s model=%s schema=%s "
                    "validation failed after %d attempt(s) response_chars=%d errors=%d",
                    self.provider,
                    self.runtime_model,
                    schema.__name__,
                    attempt + 1,
                    len(raw),
                    len(exc.errors()),
                )
                raise

        raise last_exc  # type: ignore[misc]


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
        from kenkui.nlp.entities import build_roster_from_chapters_with_llm

        if book_path is None:
            raise ValueError("LiteLLMExtractionAdapter.build_roster() requires book_path")

        cached = get_cached_roster(
            book_path,
            provider=self._provider,
            model=self._config.extraction_model,
        )
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
        roster = build_roster_from_chapters_with_llm(
            chapters,
            nlp,
            llm,
            method=self._config.discovery_method,
            step_callback=step_callback,
            book_path=book_path,
            provider=self._provider,
            model=self._config.extraction_model,
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


def _attribution_schema(review_confidence: bool) -> type[AttributionResultWire | AttributionResultConfidenceWire]:
    return AttributionResultConfidenceWire if review_confidence else AttributionResultWire


def _quote_metadata(annotated_text: str) -> dict[int, dict[str, str]]:
    metadata: dict[int, dict[str, str]] = {}
    for match in _QUOTE_TAG_RE.finditer(annotated_text):
        attrs = {key: value for key, value in _QUOTE_ATTR_RE.findall(match.group(2))}
        metadata[int(match.group(1))] = attrs
    return metadata


def _chapter_char_offset(paragraphs: list[str], para_index: int) -> int:
    return sum(len(p) + 2 for p in paragraphs[:para_index])


def _remap_quote_to_window(q: Quote, paragraphs: list[str], window_start: int) -> Quote:
    return Quote(
        id=q.id,
        text=q.text,
        para_index=q.para_index - window_start,
        char_offset=q.char_offset - _chapter_char_offset(paragraphs, window_start),
        kind=q.kind,
    )


def _quote_window(paragraphs: list[str], quotes: list[Quote]) -> tuple[list[str], list[Quote]]:
    if not quotes:
        return [], []
    start = max(0, min(q.para_index for q in quotes) - 1)
    end = min(len(paragraphs), max(q.para_index for q in quotes) + 2)
    return paragraphs[start:end], [_remap_quote_to_window(q, paragraphs, start) for q in quotes]


def _quote_capped_windows(
    paragraphs: list[str],
    quotes: list[Quote],
    max_quotes: int,
) -> list[tuple[list[str], list[Quote]]]:
    if max_quotes <= 0 or len(quotes) <= max_quotes:
        return [(paragraphs, quotes)]
    windows = []
    for index in range(0, len(quotes), max_quotes):
        windows.append(_quote_window(paragraphs, quotes[index:index + max_quotes]))
    return windows


def _roster_indexes(roster: CharacterRoster) -> tuple[dict[str, str], dict[str, set[str]], dict[str, str]]:
    slug_to_pronoun = {c.slug: c.gender for c in roster.characters}
    slug_to_aliases = {
        c.slug: {c.canonical_name, *c.aliases, c.slug.replace("_", " ")}
        for c in roster.characters
    }
    alias_to_slug = {
        alias.lower(): slug
        for slug, aliases in slug_to_aliases.items()
        for alias in aliases
        if alias
    }
    return slug_to_pronoun, slug_to_aliases, alias_to_slug


def _context_mentions_speaker(
    paragraphs: list[str],
    quote: Quote,
    speaker: str,
    slug_to_aliases: dict[str, set[str]],
) -> bool:
    aliases = slug_to_aliases.get(speaker, set())
    if not aliases:
        return False
    start = max(0, quote.para_index - 1)
    end = min(len(paragraphs), quote.para_index + 2)
    context = "\n".join(paragraphs[start:end]).lower()
    return any(alias.lower() in context for alias in aliases if alias)


def _suspicious_attributions(
    paragraphs: list[str],
    quotes: list[Quote],
    attributed: dict[int, AttributionItem],
    metadata: dict[int, dict[str, str]],
    roster: CharacterRoster,
) -> list[Quote]:
    slug_to_pronoun, slug_to_aliases, _alias_to_slug = _roster_indexes(roster)
    suspicious: list[Quote] = []
    recent_speakers: list[str] = []
    for q in quotes:
        item = attributed.get(q.id)
        if item is None:
            suspicious.append(q)
            continue
        attrs = metadata.get(q.id, {})
        speaker = item.speaker
        reason = False
        if speaker == "Unknown":
            reason = True
        elif speaker not in {"NARRATOR", "Unknown"}:
            if attrs.get("hint") and attrs["hint"] != speaker:
                reason = True
            if attrs.get("guess") and attrs["guess"] != speaker:
                reason = True
            pronoun = attrs.get("pronoun")
            speaker_pronoun = slug_to_pronoun.get(speaker, "")
            if pronoun and speaker_pronoun and pronoun != speaker_pronoun:
                reason = True
            if not _context_mentions_speaker(paragraphs, q, speaker, slug_to_aliases):
                previous = recent_speakers[-2:]
                if speaker not in previous:
                    reason = True
        if reason:
            suspicious.append(q)
        if speaker not in {"NARRATOR", "Unknown"}:
            recent_speakers.append(speaker)
    return suspicious


def _hint_correct_attributions(
    attributed: dict[int, AttributionItem],
    metadata: dict[int, dict[str, str]],
    known_slugs: set[str],
) -> None:
    for quote_id, attrs in metadata.items():
        hint = attrs.get("hint")
        item = attributed.get(quote_id)
        if item and hint in known_slugs and item.speaker not in {hint, "NARRATOR"}:
            item.speaker = hint
            item.confidence = max(item.confidence, 4)


def _apply_local_attribution_heuristics(
    attributed: dict[int, AttributionItem],
    metadata: dict[int, dict[str, str]],
    roster: CharacterRoster,
) -> None:
    slug_to_pronoun = {c.slug: c.gender for c in roster.characters}
    known_slugs = set(slug_to_pronoun)
    _hint_correct_attributions(attributed, metadata, known_slugs)
    for quote_id, attrs in metadata.items():
        item = attributed.get(quote_id)
        if not item or item.speaker in {"NARRATOR", "Unknown"}:
            continue
        pronoun = attrs.get("pronoun")
        speaker_pronoun = slug_to_pronoun.get(item.speaker, "")
        if pronoun and speaker_pronoun and pronoun != speaker_pronoun:
            item.speaker = "Unknown"
            item.confidence = 1


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
        return _run_coroutine_sync(
            self.attribute_chapter_async(chapter, roster, progress_callback=progress_callback)
        )

    async def attribute_chapter_async(
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
        schema = _attribution_schema(self._config.attribution_review_confidence)
        max_quotes_per_call = self._config.attribution_max_quotes_per_call
        known_slugs = {c.slug for c in roster.characters}
        known_names = {c.canonical_name: c.slug for c in roster.characters}
        known_names.update({alias: c.slug for c in roster.characters for alias in c.aliases})
        full_metadata = _quote_metadata(
            annotate_chapter(clean_paragraphs, dialogue_quotes, alias_to_slug, slug_to_pronoun)
        )

        def _prompt_for(paras: list[str], dq: list[Quote]) -> str:
            annotated_text = annotate_chapter(paras, dq, alias_to_slug, slug_to_pronoun)
            return f"{static_block}\n\n{_build_attribution_dynamic_block(annotated_text)}"

        async def _attribute_single(
            paras: list[str],
            dq: list[Quote],
            *,
            client: LiteLLMClient = llm,
            prompt: str | None = None,
        ) -> dict[int, AttributionItem]:
            if prompt is None:
                prompt = _prompt_for(paras, dq)
            else:
                prompt = f"{static_block}\n\n{prompt}"
            if not dq:
                return {}
            prompt_tokens, context_tokens, max_tokens = _remote_attribution_call_budget(
                prompt,
                len(dq),
                client.runtime_model,
            )
            _logger.debug(
                "LiteLLMAttributionAdapter: chapter %r call budget prompt~%d context=%d max_tokens=%d quotes=%d",
                chapter_label,
                prompt_tokens,
                context_tokens,
                max_tokens,
                len(dq),
            )
            result = await client.generate_async(prompt, schema, max_tokens=max_tokens)
            return {item.quote_id: item for item in attribution_wire_to_full(result).attributions}

        def _normalize_attributed(attributed: dict[int, AttributionItem]) -> None:
            for item in attributed.values():
                if item.speaker not in {"NARRATOR", "Unknown"} and item.speaker not in known_slugs:
                    item.speaker = known_names.get(item.speaker, slugify(item.speaker))
                if item.speaker not in {"NARRATOR", "Unknown"} and item.speaker not in known_slugs:
                    _logger.warning(
                        "LiteLLMAttributionAdapter: speaker %r not in roster; using Unknown",
                        item.speaker,
                    )
                    item.speaker = "Unknown"

        async def _resolve_suspicious(
            paras: list[str],
            dq: list[Quote],
            attributed: dict[int, AttributionItem],
        ) -> None:
            review_model = (self._config.review_model or "").strip()
            if not review_model:
                _apply_local_attribution_heuristics(attributed, full_metadata, roster)
                return
            suspicious = _suspicious_attributions(paras, dq, attributed, full_metadata, roster)
            if not suspicious:
                return
            resolver = LiteLLMClient(self._provider, review_model)
            for batch in _quote_capped_windows(paras, suspicious, max_quotes_per_call or _MISSING_RETRY_BATCH_SIZE):
                batch_paras, batch_quotes = batch
                annotated = annotate_chapter(batch_paras, batch_quotes, alias_to_slug, slug_to_pronoun)
                base_lines = [
                    f"- q={q.id} base={attributed.get(q.id).speaker if attributed.get(q.id) else 'Missing'} "
                    f"metadata={full_metadata.get(q.id, {})}"
                    for q in batch_quotes
                ]
                prompt = (
                    "Resolve suspicious speaker attributions.\n"
                    "Use the local context, base label, hint/guess/pronoun metadata, and roster.\n"
                    "Return only known roster slugs, NARRATOR, or Unknown. Prefer hint= unless local evidence "
                    "clearly supports another speaker. Use pronoun= only to reject incompatible speakers.\n\n"
                    f"BASE LABELS:\n{chr(10).join(base_lines)}\n\n"
                    f"LOCAL CONTEXT:\n{annotated}"
                )
                try:
                    resolved = await _attribute_single(
                        batch_paras,
                        batch_quotes,
                        client=resolver,
                        prompt=prompt,
                    )
                except Exception as resolve_exc:
                    _logger.warning(
                        "LiteLLMAttributionAdapter: chapter %r resolver failed (%s); keeping base labels",
                        chapter_label,
                        _sanitize_litellm_error(resolve_exc),
                    )
                    continue
                _normalize_attributed(resolved)
                for qid, item in resolved.items():
                    if item.speaker in known_slugs or item.speaker in {"NARRATOR", "Unknown"}:
                        attributed[qid] = item
                    else:
                        _logger.warning(
                            "LiteLLMAttributionAdapter: resolver returned unknown speaker %r for quote %d; keeping base",
                            item.speaker,
                            qid,
                        )

        async def _retry_missing_quotes(
            paras: list[str],
            chunk_quotes: list[Quote],
            attributed: dict[int, AttributionItem],
            *,
            chunk_idx: int,
            chunk_total: int,
        ) -> None:
            missing = [q for q in chunk_quotes if q.id not in attributed]
            if not missing:
                return
            _logger.info(
                "LiteLLMAttributionAdapter: chapter %r chunk %d/%d — retrying %d missing quote(s)",
                chapter_label,
                chunk_idx,
                chunk_total,
                len(missing),
            )
            for batch in _quote_capped_windows(paras, missing, max_quotes_per_call or _MISSING_RETRY_BATCH_SIZE):
                retry_paras, retry_quotes = batch
                try:
                    attributed.update(await _attribute_single(retry_paras, retry_quotes))
                except Exception as retry_exc:
                    retry_ids = [q.id for q in retry_quotes]
                    _logger.debug(
                        "LiteLLMAttributionAdapter: chapter %r chunk %d/%d quote ids %s retry raw error: %s",
                        chapter_label,
                        chunk_idx,
                        chunk_total,
                        retry_ids,
                        retry_exc,
                    )
                    _logger.warning(
                        "LiteLLMAttributionAdapter: chapter %r chunk %d/%d quote ids %s retry failed (%s)",
                        chapter_label,
                        chunk_idx,
                        chunk_total,
                        retry_ids,
                        _sanitize_litellm_error(retry_exc),
                    )

            still_missing = [q for q in chunk_quotes if q.id not in attributed]
            if still_missing:
                _logger.warning(
                    "LiteLLMAttributionAdapter: chapter %r chunk %d/%d — defaulting %d missing quote(s) to Unknown",
                    chapter_label,
                    chunk_idx,
                    chunk_total,
                    len(still_missing),
                )
                for q in still_missing:
                    attributed[q.id] = AttributionItem(
                        quote_id=q.id,
                        speaker="Unknown",
                        emotion="neutral",
                        confidence=1,
                    )

        async def _attribute_chunks(reason: str) -> None:
            _logger.warning(
                "LiteLLMAttributionAdapter: chapter %r %s; chunking",
                chapter_label,
                reason,
            )
            chunks = chunk_paragraphs(clean_paragraphs, dialogue_quotes)
            for chunk_idx, chunk in enumerate(chunks, start=1):
                chunk_start = chunk.para_indices[0]
                chunk_end = chunk.para_indices[-1] + 1
                chunk_paras = clean_paragraphs[chunk_start:chunk_end]
                chunk_char_start = sum(len(p) + 2 for p in clean_paragraphs[:chunk_start])
                chunk_quote_ids = set(chunk.quote_ids)
                chunk_quotes_all = [
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
                if not chunk_quotes_all:
                    continue
                for capped_idx, (call_paras, chunk_quotes) in enumerate(
                    _quote_capped_windows(chunk_paras, chunk_quotes_all, max_quotes_per_call),
                    start=1,
                ):
                    try:
                        attributed = await _attribute_single(call_paras, chunk_quotes)
                        returned = len(attributed)
                        expected = len(chunk_quotes)
                        if returned < expected:
                            _logger.info(
                                "LiteLLMAttributionAdapter: chapter %r chunk %d.%d/%d — "
                                "LLM returned %d/%d quotes (missing %d)",
                                chapter_label,
                                chunk_idx,
                                capped_idx,
                                len(chunks),
                                returned,
                                expected,
                                expected - returned,
                            )
                            await _retry_missing_quotes(
                                call_paras,
                                chunk_quotes,
                                attributed,
                                chunk_idx=chunk_idx,
                                chunk_total=len(chunks),
                            )
                        _normalize_attributed(attributed)
                        await _resolve_suspicious(call_paras, chunk_quotes, attributed)
                        all_attributions.update(attributed)
                    except Exception as chunk_exc:
                        _logger.debug(
                            "LiteLLMAttributionAdapter: chapter %r chunk %d/%d raw error: %s",
                            chapter_label,
                            chunk_idx,
                            len(chunks),
                            chunk_exc,
                        )
                        _logger.warning(
                            "LiteLLMAttributionAdapter: chapter %r chunk %d/%d failed (%s); using Unknown",
                            chapter_label,
                            chunk_idx,
                            len(chunks),
                            _sanitize_litellm_error(chunk_exc),
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

        full_prompt = _prompt_for(clean_paragraphs, dialogue_quotes)
        full_prompt_tokens, full_context_tokens, full_max_tokens = _remote_attribution_call_budget(
            full_prompt,
            len(dialogue_quotes),
            llm.runtime_model,
        )
        if max_quotes_per_call and len(dialogue_quotes) > max_quotes_per_call:
            await _attribute_chunks(
                f"exceeds quote-count cap ({len(dialogue_quotes)} > {max_quotes_per_call})"
            )
        elif full_prompt_tokens + full_max_tokens + _REMOTE_CONTEXT_SAFETY_TOKENS > full_context_tokens:
            await _attribute_chunks(
                f"exceeds token estimate (prompt~{full_prompt_tokens} tok + "
                f"max_tokens={full_max_tokens} vs context {full_context_tokens})"
            )
        else:
            try:
                attributed = await _attribute_single(clean_paragraphs, dialogue_quotes)
                returned = len(attributed)
                expected = len(dialogue_quotes)
                if returned < expected:
                    _logger.info(
                        "LiteLLMAttributionAdapter: chapter %r — LLM returned %d/%d dialogue quotes (missing %d)",
                        chapter_label,
                        returned,
                        expected,
                        expected - returned,
                    )
                    await _retry_missing_quotes(
                        clean_paragraphs,
                        dialogue_quotes,
                        attributed,
                        chunk_idx=1,
                        chunk_total=1,
                    )
                _normalize_attributed(attributed)
                await _resolve_suspicious(clean_paragraphs, dialogue_quotes, attributed)
                all_attributions.update(attributed)
            except Exception as exc:
                _logger.debug(
                    "LiteLLMAttributionAdapter: chapter %r attribution raw error: %s",
                    chapter_label,
                    exc,
                )
                await _attribute_chunks(f"attribution failed ({_sanitize_litellm_error(exc)})")

        for q in quotes:
            all_attributions.setdefault(
                q.id,
                AttributionItem(quote_id=q.id, speaker="Unknown", emotion="neutral", confidence=1),
            )

        _normalize_attributed(all_attributions)

        append_chapter_attribution(ChapterAttributionRecord(
            book_hash="",
            chapter=str(chapter_label),
            provider=self._provider,
            model=self._config.attribution_model,
            review_model=(self._config.review_model or "").strip(),
            quotes_total=len(quotes),
            quotes_attributed=sum(
                1 for i in all_attributions.values() if i.speaker not in ("Unknown", "NARRATOR")
            ),
            quotes_unknown=sum(
                1 for i in all_attributions.values() if i.speaker == "Unknown"
            ),
            retries=0,
            duration_seconds=0.0,
        ))

        return AttributionResult(attributions=list(all_attributions.values()))


__all__ = [
    "LiteLLMAttributionAdapter",
    "LiteLLMClient",
    "LiteLLMExtractionAdapter",
    "_litellm_model",
    "_remote_attribution_call_budget",
    "_remote_context_tokens",
]
