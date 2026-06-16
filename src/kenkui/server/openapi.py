"""Export the kenkui HTTP API OpenAPI schema as JSON."""

from __future__ import annotations

import json
import sys

from kenkui.server.api import create_app


def main() -> None:
    sys.stdout.write(json.dumps(create_app().openapi(), indent=2, sort_keys=True))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
