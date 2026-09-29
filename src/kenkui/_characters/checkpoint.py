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
from kenkui._characters.llm import _validated, complete_json

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from kenkui._characters.llm import Client
    from kenkui.cancellation import CancellationToken


def complete_json_checkpointed(  # noqa: PLR0913 - shared completion options
    model_id: str,
    prompt: str,
    schema: Mapping[str, type],
    *,
    client: Client | None = None,
    cancel: CancellationToken | None = None,
    validate: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Return a stored response for this exact prompt, else ask and store it.

    Both cached and fresh responses pass the optional semantic validator.
    Fresh invalid responses share complete_json's bounded retry budget; only
    an accepted response is stored. Raises ``ModelError`` as complete_json does.
    """
    if cancel is not None:
        cancel.raise_if_cancelled()
    key = store.response_key(model_id, prompt, schema)
    stored = store.read_response(key)
    if stored is not None:
        try:
            payload = _validated(stored, schema, validate)
        except ValueError:
            # Older checkpoints may have passed only structural validation.
            # Keep good responses, but never let a bad cache hit skip repair.
            pass
        else:
            if cancel is not None:
                cancel.raise_if_cancelled()
            return payload
    payload = complete_json(
        model_id, prompt, schema, client=client, cancel=cancel, validate=validate
    )
    store.write_response(key, model_id, payload)
    return payload
