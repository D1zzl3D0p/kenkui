"""Model responses that outlive the run that paid for them.

A book's attribution is stored only once every chapter has an answer, so a
provider failure part-way through a long book used to throw away every
chapter that had already succeeded. Each validated response is kept on its
own, keyed by the exact prompt, and a later run asks the model only about the
chapters that are still missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kenkui._characters import store
from kenkui._characters.llm import complete_json

if TYPE_CHECKING:
    from collections.abc import Mapping

    from kenkui._characters.llm import Client
    from kenkui.cancellation import CancellationToken


def complete_json_checkpointed(
    model_id: str,
    prompt: str,
    schema: Mapping[str, type],
    *,
    client: Client | None = None,
    cancel: CancellationToken | None = None,
) -> dict[str, Any]:
    """Return a stored response for this exact prompt, else ask and store it.

    Only a response that validated is stored, so a failure is retried next
    time rather than remembered. Raises ``ModelError`` exactly as
    ``complete_json`` does.
    """
    key = store.response_key(model_id, prompt, schema)
    stored = store.read_response(key)
    if stored is not None:
        return stored
    payload = complete_json(model_id, prompt, schema, client=client, cancel=cancel)
    store.write_response(key, model_id, payload)
    return payload
