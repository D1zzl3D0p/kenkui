"""Server startup CLI for kenkui worker server."""

import argparse

import uvicorn

DEFAULT_PORT = 45365
DEFAULT_HOST = "127.0.0.1"


def run_server(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, reload: bool = False
):
    """Run the kenkui worker server."""
    uvicorn.run(
        "kenkui.server.api:app",
        host=host,
        port=port,
        reload=reload,
    )


def main():
    """Main entry point for the server CLI."""
    parser = argparse.ArgumentParser(
        description="KenkuI Worker Server - Network API service for audiobook processing"
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Host to bind to (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to bind to (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development only)",
    )

    args = parser.parse_args()

    print(f"Starting kenkui worker server on {args.host}:{args.port}")
    run_server(args.host, args.port, args.reload)


if __name__ == "__main__":
    main()
