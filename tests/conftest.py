"""
tests/conftest.py
==================
Shared pytest fixtures and configuration for the whole test suite.
"""

import sys
from pathlib import Path

# Ensure 'd:/invoice agent' is on sys.path so `app.*` imports work
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
