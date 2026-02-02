"""Pytest configuration and shared fixtures."""

import pytest
from pathlib import Path

# Ensure the src directory is in the path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
