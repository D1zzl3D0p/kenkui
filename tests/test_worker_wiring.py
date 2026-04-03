"""Tests that WorkerServer wires up BookCache, TaskRegistry, and TaskRunner."""
from unittest.mock import patch
import pytest

def test_worker_server_has_book_cache():
    from kenkui.server.worker import WorkerServer
    from kenkui.services.book_cache import BookCache
    # Patch _load to avoid touching real queue file
    with patch.object(WorkerServer, "_load"):
        server = WorkerServer()
    assert isinstance(server.book_cache, BookCache)

def test_worker_server_has_task_registry():
    from kenkui.server.worker import WorkerServer
    from kenkui.server.tasks import TaskRegistry
    with patch.object(WorkerServer, "_load"):
        server = WorkerServer()
    assert isinstance(server.task_registry, TaskRegistry)

def test_worker_server_has_task_runner():
    from kenkui.server.worker import WorkerServer
    from kenkui.server.tasks import TaskRunner
    with patch.object(WorkerServer, "_load"):
        server = WorkerServer()
    assert isinstance(server.task_runner, TaskRunner)

def test_task_runner_uses_task_registry():
    from kenkui.server.worker import WorkerServer
    with patch.object(WorkerServer, "_load"):
        server = WorkerServer()
    # Runner should be wired to the same registry instance
    assert server.task_runner._registry is server.task_registry

def test_get_server_exposes_book_cache():
    from kenkui.server.worker import get_server, reset_server
    from kenkui.services.book_cache import BookCache
    reset_server()
    try:
        server = get_server()
        assert isinstance(server.book_cache, BookCache)
    finally:
        reset_server()
