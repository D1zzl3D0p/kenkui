"""CLI entry point for the optional kenkui HTTP API server."""

from __future__ import annotations

import argparse
import logging

from kenkui.log import setup_logging

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 45365
logger = logging.getLogger(__name__)


def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, reload: bool = False) -> None:
    log_path = setup_logging("server", level=logging.INFO)
    logger.info(
        "Starting HTTP server host=%s port=%s reload=%s log_file=%s",
        host,
        port,
        reload,
        log_path or "",
    )
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Install kenkui[server] to run the HTTP API server.") from exc

    logger.info("Using FastAPI app factory kenkui.server.api:create_app")
    uvicorn.run("kenkui.server.api:create_app", host=host, port=port, reload=reload, factory=True)
    logger.info("HTTP server stopped host=%s port=%s", host, port)


def main() -> None:
    parser = argparse.ArgumentParser(description="kenkui HTTP API server")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    run_server(args.host, args.port, args.reload)


if __name__ == "__main__":
    main()
