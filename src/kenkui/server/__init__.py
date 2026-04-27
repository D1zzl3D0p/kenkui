"""KenkuI Worker Server - Network API service for audiobook processing."""

from .worker import WorkerServer


def create_app():
    from .api import create_app as _create_app

    return _create_app()


def run_server(*args, **kwargs):
    from .server import run_server as _run_server

    return _run_server(*args, **kwargs)


__all__ = ["WorkerServer", "create_app", "run_server"]
