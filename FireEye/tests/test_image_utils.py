"""Tests for image encoding and preprocessing utilities."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.image_utils import (
    MIME_TYPES,
    encode_bytes_to_data_uri,
    encode_image_to_data_uri,
    resize_if_needed,
)

TEST_IMAGE = Path(__file__).resolve().parent.parent / "test_data" / "safe" / "01_extinguisher.jpg"


class TestEncoding:
    def test_encode_image_to_data_uri(self):
        uri = encode_image_to_data_uri(str(TEST_IMAGE))
        assert uri.startswith("data:image/jpeg;base64,")
        assert len(uri) > 100

    def test_encode_bytes_to_data_uri(self):
        uri = encode_bytes_to_data_uri(b"\x89PNG\r\n", mime="image/png")
        assert uri.startswith("data:image/png;base64,")

    def test_mime_types_coverage(self):
        assert ".jpg" in MIME_TYPES
        assert ".jpeg" in MIME_TYPES
        assert ".png" in MIME_TYPES
        assert ".webp" in MIME_TYPES


class TestResize:
    def test_resize_large_image(self):
        img = np.zeros((3000, 2000, 3), dtype=np.uint8)
        resized = resize_if_needed(img, max_dim=1280)
        assert max(resized.shape[:2]) <= 1280
        assert resized.shape[2] == 3

    def test_no_resize_small_image(self):
        img = np.zeros((640, 480, 3), dtype=np.uint8)
        result = resize_if_needed(img, max_dim=1280)
        assert result.shape == img.shape

    def test_resize_preserves_aspect_ratio(self):
        img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        resized = resize_if_needed(img, max_dim=1000)
        h, w = resized.shape[:2]
        assert w == 1000
        assert abs(h / w - 0.5) < 0.01
