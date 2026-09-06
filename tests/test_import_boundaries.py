"""Rendering never depends on provisioning, and the package has no cycles."""
# ruff: noqa: D103

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

_SOURCE = Path(__file__).parent.parent / "src" / "kenkui"
_RENDER_PACKAGES = ("_tts", "_execution", "_audio", "_domain", "_epub")
_FORBIDDEN = "kenkui.voices.provision"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


def test_package_imports_cleanly_in_a_fresh_interpreter() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "import kenkui"],
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_render_modules_never_import_provisioning() -> None:
    """A static check.

    Importing any kenkui submodule executes the root __init__, which exports the
    whole public API, so sys.modules always contains provision. What must hold is
    that no rendering module references it.
    """
    offenders = {
        str(path.relative_to(_SOURCE)): sorted(
            name for name in _imported_modules(path) if _FORBIDDEN in name
        )
        for package in _RENDER_PACKAGES
        for path in (_SOURCE / package).rglob("*.py")
    }
    assert not {key: value for key, value in offenders.items() if value}


def test_manifest_is_the_only_voices_module_reaching_into_tts() -> None:
    """voices.manifest imports production for the cache root; provision must not."""
    imported = _imported_modules(_SOURCE / "voices" / "provision.py")
    assert not [name for name in imported if name.startswith("kenkui._tts")]


def test_series_decisions_do_not_depend_on_effectful_modules() -> None:
    """Records and continuity rules must remain usable without shell services."""
    forbidden = (
        "kenkui._characters.store",
        "kenkui._characters.llm",
        "kenkui.pipeline",
        "kenkui._tts",
        "kenkui._execution",
        "kenkui.voices.provision",
    )
    for name in ("models", "series", "continuity"):
        imported = _imported_modules(_SOURCE / "_characters" / f"{name}.py")
        assert not {module for module in imported if module.startswith(forbidden)}
