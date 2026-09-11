"""Rendering never depends on provisioning, and the package has no cycles."""
# ruff: noqa: D103

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

_SOURCE = Path(__file__).parent.parent / "src" / "kenkui"
_RENDER_PACKAGES = ("_tts", "_execution", "_audio", "_domain", "_epub")
_FORBIDDEN = "kenkui.voices.provision"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    package = ".".join(("kenkui", *path.relative_to(_SOURCE).parts[:-1]))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            relative = "." * node.level + (node.module or "")
            base = (
                importlib.util.resolve_name(relative, package)
                if node.level
                else node.module
            )
            if base is None:
                continue
            found.add(base)
            found.update(f"{base}.{alias.name}" for alias in node.names)
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


def test_domain_never_imports_characters() -> None:
    """Pure partitioning may feed character logic, never depend on it."""
    domain_root = _SOURCE / "_domain"
    offenders = {
        str(path.relative_to(_SOURCE)): sorted(
            module
            for module in _imported_modules(path)
            if module == "kenkui._characters"
            or module.startswith("kenkui._characters.")
        )
        for path in domain_root.rglob("*.py")
    }
    assert not {key: value for key, value in offenders.items() if value}


def test_planning_never_imports_structural_discovery() -> None:
    """Planning consumes grid gaps and cannot rescan canonical block/line text."""
    imported = _imported_modules(_SOURCE / "_domain" / "planning.py")
    forbidden = {
        "kenkui._domain.structure.block_ranges",
        "kenkui._domain.structure.line_ranges",
    }
    assert imported.isdisjoint(forbidden)


def test_planning_contains_no_legacy_chunker_symbols() -> None:
    """No dormant function, ladder, or tuning constant survives the migration."""
    path = _SOURCE / "_domain" / "planning.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    names.update(
        target.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            (*node.targets,) if isinstance(node, ast.Assign) else (node.target,)
        )
        if isinstance(target, ast.Name)
    )
    forbidden = {
        "_chunk_span",
        "_break_offset",
        "_separator_free_end",
        "_BREAK_TIERS",
        "_CLEAN_BREAK_TIERS",
        "MIN_BREAK_FILL",
        "MAX_SEPARATOR_FREE_CHARACTERS",
        "POCKET_SEPARATORS",
        "_structural_gaps",
        "_pause_pieces",
        "_fragments",
        "CHUNKING_SCHEMA_VERSION",
        "STRUCTURAL_CHUNKING_SCHEMA_VERSION",
        "STRUCTURE_SCHEMA_VERSION",
    }
    assert names.isdisjoint(forbidden)
