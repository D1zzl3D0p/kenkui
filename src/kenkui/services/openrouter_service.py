"""OpenRouter model catalogue helpers."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

_OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
_STRICT_JSON_SCHEMA_PARAMETERS = (
    "response_format",
    "structured_outputs",
    "max_tokens",
)


@dataclass(frozen=True)
class OpenRouterModel:
    """A model entry returned by the OpenRouter models API."""

    id: str
    name: str
    description: str = ""
    context_length: int | None = None
    supported_parameters: tuple[str, ...] = field(default_factory=tuple)
    canonical_slug: str = ""

    @property
    def supports_strict_json_schema(self) -> bool:
        """Return True when OpenRouter advertises JSON schema enforcement."""
        return all(
            parameter in self.supported_parameters
            for parameter in _STRICT_JSON_SCHEMA_PARAMETERS
        )


def _coerce_openrouter_model(raw: dict[str, Any]) -> OpenRouterModel:
    supported_parameters = raw.get("supported_parameters")
    if not isinstance(supported_parameters, list):
        supported_parameters = []

    context_length = raw.get("context_length")
    if not isinstance(context_length, int):
        context_length = None

    return OpenRouterModel(
        id=str(raw.get("id", "")),
        name=str(raw.get("name", "")),
        description=str(raw.get("description", "")),
        context_length=context_length,
        supported_parameters=tuple(str(item) for item in supported_parameters),
        canonical_slug=str(raw.get("canonical_slug", "")),
    )


def _matches_search(model: OpenRouterModel, search: str) -> bool:
    needle = search.strip().lower()
    if not needle:
        return True
    return (
        needle in model.id.lower()
        or needle in model.name.lower()
        or needle in model.description.lower()
    )


def search_openrouter_models(
    search: str = "",
    *,
    require_strict_json_schema: bool = False,
    output_modalities: tuple[str, ...] = ("text",),
    limit: int | None = None,
    timeout: float = 15,
) -> list[OpenRouterModel]:
    """Return OpenRouter catalogue models matching the supplied filters.

    ``require_strict_json_schema`` maps to OpenRouter's
    ``supported_parameters`` catalogue filter, then verifies the returned
    metadata locally so callers only see models compatible with the strict
    structured-output requests used by the library.
    """
    query: dict[str, str] = {}
    if output_modalities:
        query["output_modalities"] = ",".join(output_modalities)
    if require_strict_json_schema:
        query["supported_parameters"] = ",".join(_STRICT_JSON_SCHEMA_PARAMETERS)

    url = _OPENROUTER_MODELS_URL
    if query:
        url = f"{url}?{urllib.parse.urlencode(query)}"

    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))

    data = payload.get("data", []) if isinstance(payload, dict) else []
    models = [
        _coerce_openrouter_model(item)
        for item in data
        if isinstance(item, dict)
    ]
    if require_strict_json_schema:
        models = [model for model in models if model.supports_strict_json_schema]
    if search.strip():
        models = [model for model in models if _matches_search(model, search)]
    if limit is not None:
        models = models[: max(0, limit)]
    return models


__all__ = [
    "OpenRouterModel",
    "search_openrouter_models",
]
