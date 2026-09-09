"""Package-level smoke tests."""

import kenkui


def test_package_exports_version() -> None:
    """The installed package exposes its distribution version."""
    assert kenkui.__version__ == "0.1.0"


def test_package_exports_script_types() -> None:
    """The review model is available from the intentional public namespace."""
    assert {"Script", "ScriptRow"} <= set(kenkui.__all__)
    assert kenkui.Script.__module__ == "kenkui.script"
    assert kenkui.ScriptRow.__module__ == "kenkui.script"
