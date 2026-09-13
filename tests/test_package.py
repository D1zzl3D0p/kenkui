"""Package-level smoke tests."""

import importlib.metadata

import kenkui


def test_package_exports_version() -> None:
    """``__version__`` agrees with the distribution metadata from pyproject."""
    assert kenkui.__version__ == importlib.metadata.version("kenkui")


def test_package_exports_script_types() -> None:
    """The review model is available from the intentional public namespace."""
    assert {"Script", "ScriptRow"} <= set(kenkui.__all__)
    assert kenkui.Script.__module__ == "kenkui.script"
    assert kenkui.ScriptRow.__module__ == "kenkui.script"
