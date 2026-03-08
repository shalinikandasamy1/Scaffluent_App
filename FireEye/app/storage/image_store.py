"""Simple filesystem-based image storage."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from app.config import settings


def _input_dir() -> Path:
    d = settings.images_input_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def _output_dir() -> Path:
    d = settings.images_output_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def store_input_image(image_id: UUID, filename: str, data: bytes) -> Path:
    """Persist an uploaded image to the input directory. Returns the saved path."""
    ext = Path(filename).suffix or ".jpg"
    dest = _input_dir() / f"{image_id}{ext}"
    dest.write_bytes(data)
    return dest


def get_input_image_path(image_id: UUID) -> Path | None:
    """Look up a stored input image by ID (checks common extensions)."""
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        candidate = _input_dir() / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def store_output_image(image_id: UUID, suffix: str, data: bytes) -> Path:
    """Save a pipeline output image (e.g. annotated result)."""
    dest = _output_dir() / f"{image_id}_{suffix}.jpg"
    dest.write_bytes(data)
    return dest


def get_annotated_path(image_id: UUID) -> Path:
    """Return the path where the YOLO-annotated image should be written."""
    return _output_dir() / f"{image_id}_annotated.jpg"


def get_output_image_path(image_id: UUID, suffix: str) -> Path | None:
    """Look up a stored output image."""
    dest = _output_dir() / f"{image_id}_{suffix}.jpg"
    return dest if dest.exists() else None


def cleanup_image(image_id: UUID) -> None:
    """Remove all stored files for a given image ID."""
    for directory in (_input_dir(), _output_dir()):
        for f in directory.glob(f"{image_id}*"):
            f.unlink(missing_ok=True)
