"""Rendering never imports provisioning, and voice types stay a leaf."""
# ruff: noqa: D103, S603

from __future__ import annotations

import subprocess
import sys

_RENDER_ONLY = (
    "import sys",
    "import kenkui._tts.production",
    "import kenkui._tts.pocket",
    "import kenkui._execution.coordinator",
    "print('kenkui.voices.provision' in sys.modules)",
)


def _run(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        check=False,
        text=True,
    )


def test_package_imports_cleanly_in_a_fresh_interpreter() -> None:
    completed = _run("import kenkui")
    assert completed.returncode == 0, completed.stderr


def test_render_modules_do_not_import_provisioning() -> None:
    completed = _run("\n".join(_RENDER_ONLY))
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False"
