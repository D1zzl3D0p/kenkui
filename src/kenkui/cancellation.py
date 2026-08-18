"""Cooperative execution cancellation."""

from threading import Event

from .errors import CancelledError, ErrorCode


class CancellationToken:
    """Thread-safe cooperative cancellation state shared with one execution."""

    __slots__ = ("_event",)

    def __init__(self) -> None:
        """Create a token in the active state."""
        self._event = Event()

    @property
    def cancelled(self) -> bool:
        """Whether cancellation has been requested."""
        return self._event.is_set()

    def cancel(self) -> None:
        """Request cancellation idempotently."""
        self._event.set()

    def raise_if_cancelled(self) -> None:
        """Raise the stable public cancellation failure when requested."""
        if self.cancelled:
            raise CancelledError(ErrorCode.CANCELLED)
