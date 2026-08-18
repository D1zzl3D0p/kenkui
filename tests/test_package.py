"""Package-level smoke tests."""

import kenkui


def test_package_exports_version() -> None:
    """The installed package exposes its distribution version."""
    assert kenkui.__version__ == "0.1.0"
