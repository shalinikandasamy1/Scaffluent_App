"""Stage 1: YOLO object detection.

Runs a YOLO model on the input image and returns a list of detections
with bounding boxes, confidence scores, and class labels.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ultralytics import YOLO

from app.config import settings
from app.models.schemas import BoundingBox, Detection

logger = logging.getLogger(__name__)

_model: YOLO | None = None


def _get_model() -> YOLO:
    """Lazy-load the YOLO model (weights auto-download on first run)."""
    global _model
    if _model is None:
        logger.info("Loading YOLO model: %s", settings.yolo_model_name)
        _model = YOLO(settings.yolo_model_name)
        device = settings.yolo_device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        _model.to(device)
        logger.info("YOLO using device: %s", device)
    return _model


def _extract_detections(results) -> list[Detection]:
    """Extract structured detections from YOLO results."""
    detections: list[Detection] = []
    for result in results:
        for i in range(len(result.boxes)):
            x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()
            confidence = result.boxes.conf[i].item()
            class_id = int(result.boxes.cls[i].item())
            class_name = result.names[class_id]
            detections.append(
                Detection(
                    label=class_name,
                    confidence=round(confidence, 4),
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )
    return detections


def detect(image_path: str | Path) -> list[Detection]:
    """Run YOLO inference on an image and return structured detections."""
    model = _get_model()
    results = model(str(image_path), conf=settings.yolo_confidence_threshold)
    detections = _extract_detections(results)
    logger.info("YOLO detected %d objects in %s", len(detections), image_path)
    return detections


def detect_and_annotate(image_path: str | Path, output_path: str | Path) -> list[Detection]:
    """Run YOLO, save an annotated image, and return detections."""
    model = _get_model()
    results = model(str(image_path), conf=settings.yolo_confidence_threshold)
    detections = _extract_detections(results)
    for result in results:
        result.save(filename=str(output_path))
    logger.info("YOLO annotated image saved to %s", output_path)
    return detections
