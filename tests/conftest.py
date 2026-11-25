"""Shared test setup: make the repository root importable (so `from src import ...` works)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def rng():
    """A seeded random generator, so every test is reproducible."""
    return np.random.default_rng(12345)
