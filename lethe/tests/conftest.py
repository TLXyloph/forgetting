from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))  # makes `tests_support` importable

from tests_support import Feeder  # noqa: E402


@pytest.fixture
def feeder():
    f = Feeder()
    return f.state, f
