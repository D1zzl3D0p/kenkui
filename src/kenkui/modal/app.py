from __future__ import annotations

import os

import modal  # type: ignore[import]

APP_NAME = os.environ.get("KENKUI_MODAL_APP_NAME", "kenkui")

app = modal.App(APP_NAME)
