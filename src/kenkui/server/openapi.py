"""Export the kenkui HTTP API OpenAPI schema as JSON."""

from __future__ import annotations

import json

from kenkui.server.api import create_app


def main() -> None:
    print(json.dumps(create_app().openapi(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
