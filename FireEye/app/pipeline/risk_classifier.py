"""Stage 2: Risk classification.

Takes YOLO detections and uses heuristics + an LLM call to classify
the overall scene risk as safe or potentially dangerous.
"""

from __future__ import annotations

import logging

from app.config import settings
from app.models.schemas import Detection, RiskClassification, RiskLevel
from app.pipeline.spatial import format_spatial_summary
from app.services import openrouter_client
from app.services.image_utils import encode_image_to_data_uri
from app.services.prompt_loader import get_system_prompt, get_user_template

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

# Common construction-site fire accidents (FYP Ground Truth booklet)
# mapped to our 12-class YOLO detection targets and coverage status.
COMMON_ACCIDENTS = [
    {
        "cause": "Sparks from hot works igniting combustibles",
        "detectable_by": ["welding_sparks", "fire", "scaffold_net", "tarpaulin"],
        "coverage": "full",
    },
    {
        "cause": "Electrical overload / short circuit",
        "detectable_by": [],
        "coverage": "none",  # future: add electrical_panel class
    },
    {
        "cause": "Improper flammable material storage",
        "detectable_by": ["gas_cylinder"],
        "coverage": "partial",
    },
    {
        "cause": "Unextinguished cigarette butts",
        "detectable_by": [],
        "coverage": "none",  # future: add cigarette class
    },
    {
        "cause": "Lack of fire protection / maintenance",
        "detectable_by": ["fire_extinguisher", "hose_reel", "exit_sign"],
        "coverage": "full",
    },
    {
        "cause": "Non-fire-retardant facade coverings",
        "detectable_by": ["scaffold_net", "tarpaulin"],
        "coverage": "partial",  # can detect presence but not material type
    },
    {
        "cause": "Incomplete fire service installations",
        "detectable_by": ["hose_reel", "fire_extinguisher"],
        "coverage": "partial",
    },
]

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


def _filter_fire_on_extinguisher(detections: list[Detection]) -> list[Detection]:
    """Remove low-confidence 'fire' detections that overlap with fire_extinguisher.

    YOLO sometimes misclassifies the red body of a fire extinguisher as 'fire'.
    If a fire bbox overlaps >50% with any fire_extinguisher bbox and has low
    confidence, suppress it.
    """
    extinguishers = [d for d in detections if d.label == "fire_extinguisher"]
    if not extinguishers:
        return detections

    filtered = []
    for det in detections:
        if det.label == "fire" and det.confidence < 0.4:
            # Check IoU with any extinguisher
            suppress = False
            for ext in extinguishers:
                ix1 = max(det.bbox.x1, ext.bbox.x1)
                iy1 = max(det.bbox.y1, ext.bbox.y1)
                ix2 = min(det.bbox.x2, ext.bbox.x2)
                iy2 = min(det.bbox.y2, ext.bbox.y2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                fire_area = (det.bbox.x2 - det.bbox.x1) * (det.bbox.y2 - det.bbox.y1)
                if fire_area > 0 and inter / fire_area > 0.5:
                    suppress = True
                    break
            if not suppress:
                filtered.append(det)
        else:
            filtered.append(det)
    return filtered


def classify_from_detections(detections: list[Detection]) -> RiskClassification:
    """Heuristic risk classification aligned with HK regulatory criteria.

    Acts as a deterministic fallback when the LLM is unavailable.
    Uses spatial proximity data when available.
    """
    from app.pipeline.spatial import compute_distances

    detections = _filter_fire_on_extinguisher(detections)
    labels = {d.label.lower() for d in detections}
    has_ignition = bool(labels & IGNITION_LABELS)
    has_flammable = bool(labels & FLAMMABLE_LABELS)
    has_hazardous = bool(labels & HAZARDOUS_LABELS)
    has_safety = bool(labels & SAFETY_LABELS)

    # Check for gas cylinder near ignition — CRITICAL per FSD CL 2/2008
    if has_ignition and "gas_cylinder" in labels:
        distances = compute_distances(detections)
        gas_near_fire = any(
            d["safety_concern"] and "gas_cylinder" in d["obj_a"] + d["obj_b"]
            for d in distances
        )
        if gas_near_fire:
            return RiskClassification(
                risk_level=RiskLevel.critical,
                confidence=0.9,
                reason="Gas cylinder detected near ignition source — explosion risk per FSD CL 2/2008.",
            )

    # Ignition + flammable (scaffold net / tarpaulin)
    if has_ignition and has_flammable:
        # Check if safety equipment mitigates
        if has_safety and "fire_extinguisher" in labels:
            return RiskClassification(
                risk_level=RiskLevel.medium,
                confidence=0.7,
                reason="Ignition near flammable material, but fire extinguisher present.",
            )
        return RiskClassification(
            risk_level=RiskLevel.high,
            confidence=0.8,
            reason="Ignition source near flammable material (scaffold net/tarpaulin) without visible fire extinguisher.",
        )

    # Fire/smoke detected but no flammable nearby
    if "fire" in labels:
        if has_safety:
            return RiskClassification(
                risk_level=RiskLevel.medium,
                confidence=0.7,
                reason="Fire detected but safety equipment visible.",
            )
        return RiskClassification(
            risk_level=RiskLevel.high,
            confidence=0.8,
            reason="Fire detected without visible safety equipment.",
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

    if has_safety and not has_hazardous:
        return RiskClassification(
            risk_level=RiskLevel.safe,
            confidence=0.9,
            reason="Safety equipment present, no hazards detected. Site appears compliant.",
        )

    return RiskClassification(
        risk_level=RiskLevel.safe,
        confidence=0.8,
        reason="No fire hazards detected by object detection.",
    )


def classify_with_llm(image_path: str, detections: list[Detection]) -> RiskClassification:
    """Use the LLM to perform a more nuanced risk classification."""
    detection_summary = "\n".join(
        f"- {d.label} (confidence {d.confidence:.0%})" for d in detections
    )
    spatial_summary = format_spatial_summary(detections)
    data_uri = encode_image_to_data_uri(image_path)

    user_text = get_user_template("risk_classifier").format(
        detection_summary=detection_summary,
        spatial_summary=spatial_summary,
    )

    messages = [
        {
            "role": "system",
            "content": get_system_prompt("risk_classifier", settings.local_llm_model if settings.llm_backend == "local" else ""),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
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
