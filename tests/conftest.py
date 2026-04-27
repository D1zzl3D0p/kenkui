"""Pytest configuration and shared fixtures."""

# Ensure the src directory is in the path
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Provide minimal stubs for optional heavy dependencies that may not be
# installed in the test environment (e.g. litellm, instructor).
try:
    import litellm  # noqa: F401
except ModuleNotFoundError:
    _litellm_stub = types.ModuleType("litellm")

    def _litellm_completion(*args, **kwargs):  # type: ignore[misc]
        raise NotImplementedError("litellm stub: completion not available")

    _litellm_stub.completion = _litellm_completion  # type: ignore[attr-defined]
    sys.modules["litellm"] = _litellm_stub

try:
    import instructor  # noqa: F401
except ModuleNotFoundError:
    _instructor_stub = types.ModuleType("instructor")

    class _ModeMeta(type):
        def __getattr__(cls, name):
            return name

    class _Mode(metaclass=_ModeMeta):
        pass

    _instructor_stub.Mode = _Mode  # type: ignore[attr-defined]
    _instructor_stub.from_litellm = lambda *a, **kw: None  # type: ignore[attr-defined]
    sys.modules["instructor"] = _instructor_stub

    # instructor.exceptions submodule is imported directly in tests
    _exc_stub = types.ModuleType("instructor.exceptions")

    class IncompleteOutputException(Exception):
        def __init__(self, *args, last_completion=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.last_completion = last_completion

    _exc_stub.IncompleteOutputException = IncompleteOutputException  # type: ignore[attr-defined]
    sys.modules["instructor.exceptions"] = _exc_stub

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
