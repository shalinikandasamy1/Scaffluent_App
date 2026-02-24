"""Stage 2: Risk classification.

Takes YOLO detections and uses heuristics + an LLM call to classify
the overall scene risk as safe or potentially dangerous.
"""

from __future__ import annotations

import logging

from app.models.schemas import Detection, RiskClassification, RiskLevel
from app.services import openrouter_client
from app.services.image_utils import encode_image_to_data_uri

logger = logging.getLogger(__name__)

# Objects that inherently raise the risk level when detected
HAZARDOUS_LABELS = {
    "fire", "flame", "smoke", "gas cylinder", "gas tank",
    "lighter", "match", "candle", "torch", "welding",
    "fuel", "gasoline", "kerosene", "propane",
}

FLAMMABLE_LABELS = {
    "wood", "cardboard", "paper", "fabric", "cloth",
    "plastic", "foam", "hay", "straw",
}

IGNITION_LABELS = {
    "fire", "flame", "lighter", "match", "candle",
    "torch", "welding", "spark", "heater", "stove",
}

# JSON schema for the LLM risk classification response
_RISK_SCHEMA = {
    "name": "risk_classification",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "risk_level": {
                "type": "string",
                "enum": ["safe", "low", "medium", "high", "critical"],
            },
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["risk_level", "confidence", "reason"],
        "additionalProperties": False,
    },
}


def classify_from_detections(detections: list[Detection]) -> RiskClassification:
    """Quick heuristic-based risk classification from YOLO detections."""
    labels = {d.label.lower() for d in detections}

    has_ignition = bool(labels & IGNITION_LABELS)
    has_flammable = bool(labels & FLAMMABLE_LABELS)
    has_hazardous = bool(labels & HAZARDOUS_LABELS)

    if has_ignition and has_flammable:
        return RiskClassification(
            risk_level=RiskLevel.high,
            confidence=0.8,
            reason="Both ignition source and flammable material detected in scene.",
        )
    if has_hazardous:
        return RiskClassification(
            risk_level=RiskLevel.medium,
            confidence=0.7,
            reason=f"Hazardous object(s) detected: {labels & HAZARDOUS_LABELS}",
        )
    if has_flammable:
        return RiskClassification(
            risk_level=RiskLevel.low,
            confidence=0.6,
            reason="Flammable materials present but no ignition source detected.",
        )
    return RiskClassification(
        risk_level=RiskLevel.safe,
        confidence=0.9,
        reason="No obvious fire hazards detected by object detection.",
    )


def classify_with_llm(image_path: str, detections: list[Detection]) -> RiskClassification:
    """Use the LLM to perform a more nuanced risk classification."""
    detection_summary = "\n".join(
        f"- {d.label} (confidence {d.confidence:.0%})" for d in detections
    )
    data_uri = encode_image_to_data_uri(image_path)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a fire safety risk classifier. Given an image and a list of "
                "detected objects, classify the overall fire risk level as one of: "
                "safe, low, medium, high, critical. Provide your confidence (0-1) and "
                "a brief reason."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Detected objects:\n{detection_summary}\n\nClassify the fire risk.",
                },
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        },
    ]

    result = openrouter_client.chat_completion_json(
        messages, json_schema=_RISK_SCHEMA
    )

    logger.info("LLM risk classification: %s", result)
    return RiskClassification(
        risk_level=RiskLevel(result["risk_level"]),
        confidence=result["confidence"],
        reason=result["reason"],
    )
