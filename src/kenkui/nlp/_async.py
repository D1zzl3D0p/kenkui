"""Shared asyncio helpers for the kenkui NLP package.

This module centralises the single ``run_coroutine_sync`` bridge that lets the
synchronous public NLP API drive ``async`` provider internals, plus the
background event loop used by the thread-free :class:`NLPJob` execution model.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, TypeVar

_T = TypeVar("_T")


def run_coroutine_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run *coro* to completion from synchronous code and return its result.

    When no event loop is running in the current thread the coroutine is driven
    directly with :func:`asyncio.run`.  When called from *inside* a running loop
    (which forbids nested ``asyncio.run``), the coroutine is handed to a
    short-lived worker thread that owns its own loop, and the caller blocks on
    the result.  This is the single shared implementation imported by
    ``pipeline``, ``entities`` and the ``providers.litellm`` provider.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


class _BackgroundLoop:
    """A lazily-started daemon event loop for asyncio-native background jobs.

    A single shared loop runs on one daemon thread for the whole process; jobs
    are scheduled onto it with :meth:`submit`, which returns a
    :class:`concurrent.futures.Future` usable from synchronous callers.  This is
    the minimal thread bridge required because the public NLP job API is sync
    while the work it wraps is CPU/IO-blocking: asyncio cannot execute blocking
    provider calls without an offload thread, but centralising on one loop keeps
    cancellation first-class (``future.cancel()`` -> task cancellation).
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._init_lock = threading.Lock()  # guards one-time loop creation only

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        loop = self._loop
        if loop is not None:
            return loop
        with self._init_lock:
            if self._loop is None:
                new_loop = asyncio.new_event_loop()
                thread = threading.Thread(
                    target=new_loop.run_forever,
                    name="kenkui-nlp-jobs",
                    daemon=True,
                )
                thread.start()
                self._loop = new_loop
            return self._loop

    def submit(self, coro: Coroutine[Any, Any, _T]) -> Future[_T]:
        """Schedule *coro* on the background loop and return its Future."""
        loop = self._ensure_loop()
        return asyncio.run_coroutine_threadsafe(coro, loop)


# Process-wide singleton — jobs are rare and share one loop/thread.
JOB_LOOP = _BackgroundLoop()
