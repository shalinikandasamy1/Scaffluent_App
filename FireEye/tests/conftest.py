"""Shared pytest fixtures for FireEye tests."""

import sys
from pathlib import Path

import pytest

# Ensure FireEye package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.schemas import BoundingBox, Detection


@pytest.fixture
def make_detection():
    """Factory fixture to create Detection objects."""
    def _make(label: str, conf: float = 0.8,
              x1: float = 0, y1: float = 0,
              x2: float = 100, y2: float = 100) -> Detection:
        return Detection(
            label=label, confidence=conf,
            bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        )
    return _make
