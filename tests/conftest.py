"""Pytest configuration and shared fixtures."""

# Ensure the src directory is in the path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Provide a minimal tomli_w stub when the real package is not installed.
# This allows kenkui.config (and tests that patch it) to be imported in
# environments that only have the stdlib (e.g. CI / Miniforge without uv venv).
try:
    import tomli_w  # noqa: F401
except ModuleNotFoundError:
    import types

    _stub = types.ModuleType("tomli_w")

    def _dumps(data: dict, *, multiline_strings: bool = False) -> str:  # type: ignore[misc]
        raise NotImplementedError("tomli_w stub: dumps not available")

    def _dump(data: dict, fp, *, multiline_strings: bool = False) -> None:  # type: ignore[misc]
        raise NotImplementedError("tomli_w stub: dump not available")

    _stub.dumps = _dumps  # type: ignore[attr-defined]
    _stub.dump = _dump  # type: ignore[attr-defined]
    sys.modules["tomli_w"] = _stub
