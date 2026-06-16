"""Command line entry point for kenkui."""

from __future__ import annotations

import argparse

from kenkui.server.server import DEFAULT_HOST, DEFAULT_PORT, run_server


def main() -> None:
    parser = argparse.ArgumentParser(prog="kenkui")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Run the kenkui HTTP API server")
    serve.add_argument("--host", default=DEFAULT_HOST)
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve.add_argument("--reload", action="store_true")

    args = parser.parse_args()
    if args.command == "serve":
        run_server(args.host, args.port, args.reload)
