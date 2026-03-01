"""
conftest.py
===========
Shared pytest fixtures for CartComplete tests.
"""

import sys
from pathlib import Path

import pytest

# Ensure the project root is on sys.path so `from src.…` works
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
