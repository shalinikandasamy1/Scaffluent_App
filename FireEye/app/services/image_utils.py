"""Image encoding and preprocessing helpers."""

import base64
from pathlib import Path

import cv2
import numpy as np

MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def encode_image_to_data_uri(image_path: str | Path) -> str:
    """Read an image file and return a base64 data URI for LLM API calls."""
    path = Path(image_path)
    mime = MIME_TYPES.get(path.suffix.lower(), "image/png")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def encode_bytes_to_data_uri(data: bytes, mime: str = "image/jpeg") -> str:
    """Encode raw bytes to a base64 data URI."""
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def load_image(image_path: str | Path) -> np.ndarray:
    """Load an image from disk as a BGR numpy array (OpenCV format)."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return img


def save_image(image: np.ndarray, output_path: str | Path) -> Path:
    """Save a numpy array image to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return path


def resize_if_needed(image: np.ndarray, max_dim: int = 1280) -> np.ndarray:
    """Resize image so its largest dimension is at most max_dim."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
