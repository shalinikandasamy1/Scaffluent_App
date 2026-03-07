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

# Objects that inherently raise the risk level when detected.
# Aligned with our 12-class fine-tuned YOLO model:
#   fire(0), smoke(1), fire_extinguisher(2), gas_cylinder(3), scaffold_net(4),
#   exit_sign(5), hard_hat(6), safety_vest(7), welding_sparks(8),
#   hose_reel(9), person(10), tarpaulin(11)
HAZARDOUS_LABELS = {
    "fire", "smoke", "gas_cylinder", "welding_sparks",
}

FLAMMABLE_LABELS = {
    "scaffold_net", "tarpaulin",
}

IGNITION_LABELS = {
    "fire", "welding_sparks",
}

# Safety equipment — presence mitigates risk
SAFETY_LABELS = {
    "fire_extinguisher", "hose_reel", "exit_sign",
    "hard_hat", "safety_vest",
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
    has_safety = bool(labels & SAFETY_LABELS)

    if has_ignition and has_flammable:
        level = RiskLevel.high
        if has_safety:
            level = RiskLevel.medium
            reason = (
                "Ignition source near flammable material, but safety equipment present. "
                f"Hazards: {labels & IGNITION_LABELS | labels & FLAMMABLE_LABELS}, "
                f"Safety: {labels & SAFETY_LABELS}"
            )
        else:
            reason = "Both ignition source and flammable material detected without visible safety equipment."
        return RiskClassification(risk_level=level, confidence=0.8, reason=reason)
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
                "You are a fire safety risk classifier for Hong Kong construction sites.\n"
                "Open flames are NORMAL in many work contexts (welding, cutting torches). "
                "The presence of a flame alone does NOT indicate critical risk.\n\n"
                "Classify the fire SPREAD risk using this scale:\n"
                "  safe     — No fire, no active ignition source present.\n"
                "  low      — Controlled or contained flame (welding arc, torch) in a "
                "clear area with no flammable materials within reach. Normal hot work.\n"
                "  medium   — Controlled flame with flammable materials at a safe but noteworthy "
                "distance, OR small uncontrolled flame with no immediate spread path.\n"
                "  high     — Large or actively spreading flame, OR a flame in close proximity "
                "to significant flammable materials that could ignite via radiant heat or embers.\n"
                "  critical — Active fire spread already occurring, OR immediate multi-vector "
                "cascade risk (gas cylinders adjacent to flame, ember storm reaching scaffold nets).\n\n"
                "HK CONSTRUCTION SITE FIRE SAFETY RULES (FSD Circular Letter 2/2008):\n"
                "- Combustibles must be >= 6m from hot work (welding/cutting)\n"
                "- Scaffold nets and tarpaulins must be fire-retardant certified\n"
                "- Gas cylinders (acetylene, oxygen) near ignition = ESCALATE risk\n"
                "- Fire extinguishers, hose reels = MITIGATING factors (lower risk if present)\n"
                "- Absence of PPE (hard hats, safety vests) near hot work = procedural concern\n\n"
                "Base your classification on: flame size/control, proximity of flammable materials "
                "(especially scaffold nets, tarpaulins), gas cylinder proximity, safety equipment "
                "presence, and ember/spark travel. Provide confidence (0-1) and concise reason."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Detected objects:\n{detection_summary}\n\n"
                        "Classify the fire spread risk for this scene."
                    ),
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
