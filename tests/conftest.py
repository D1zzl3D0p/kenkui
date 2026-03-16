"""Pytest configuration and shared fixtures."""

# Ensure the src directory is in the path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
