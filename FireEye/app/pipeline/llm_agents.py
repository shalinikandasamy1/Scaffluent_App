"""Stage 3: LLM agents for present assessment and future prediction.

Two agents analyze scenes that pass the risk classifier:
  1. Present Agent  — describes current hazards and spatial relationships.
  2. Future Agent   — predicts branching danger scenarios (TVA-style).
"""

from __future__ import annotations

import logging

from app.models.schemas import (
    Detection,
    FuturePrediction,
    FutureScenario,
    PresentAssessment,
    RiskClassification,
    RiskLevel,
)
from app.services import openrouter_client
from app.services.image_utils import encode_image_to_data_uri

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON schemas for structured LLM responses
# ---------------------------------------------------------------------------

_PRESENT_SCHEMA = {
    "name": "present_assessment",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "hazards": {"type": "array", "items": {"type": "string"}},
            "distances": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "hazards", "distances"],
        "additionalProperties": False,
    },
}

_FUTURE_SCHEMA = {
    "name": "future_prediction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "scenarios": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "scenario": {"type": "string"},
                        "likelihood": {"type": "string"},
                        "severity": {"type": "string"},
                        "time_horizon": {"type": "string"},
                    },
                    "required": ["scenario", "likelihood", "severity", "time_horizon"],
                    "additionalProperties": False,
                },
            },
            "overall_risk": {
                "type": "string",
                "enum": ["safe", "low", "medium", "high", "critical"],
            },
            "recommendation": {"type": "string"},
        },
        "required": ["scenarios", "overall_risk", "recommendation"],
        "additionalProperties": False,
    },
}


# ---------------------------------------------------------------------------
# Present Agent
# ---------------------------------------------------------------------------

def assess_present(
    image_path: str,
    detections: list[Detection],
    risk: RiskClassification,
) -> PresentAssessment:
    """Assess the current situation: what hazards exist and how they relate spatially."""
    detection_summary = "\n".join(
        f"- {d.label} (conf {d.confidence:.0%}) at bbox "
        f"[{d.bbox.x1:.0f}, {d.bbox.y1:.0f}, {d.bbox.x2:.0f}, {d.bbox.y2:.0f}]"
        for d in detections
    )
    data_uri = encode_image_to_data_uri(image_path)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a fire safety assessment agent (the 'Present Agent'). "
                "Analyze the image of a construction site or indoor scene. "
                "Describe the current situation: list specific hazards, note the "
                "spatial relationships between ignition sources and flammable materials, "
                "and estimate distances where possible. Be precise and factual."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Object detections:\n{detection_summary}\n\n"
                        f"Preliminary risk level: {risk.risk_level.value} "
                        f"({risk.reason})\n\n"
                        "Provide your present-state assessment."
                    ),
                },
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        },
    ]

    result = openrouter_client.chat_completion_json(
        messages, json_schema=_PRESENT_SCHEMA
    )
    logger.info("Present assessment: %s", result.get("summary", "")[:120])
    return PresentAssessment(**result)


# ---------------------------------------------------------------------------
# Future Agent
# ---------------------------------------------------------------------------

def predict_future(
    image_path: str,
    detections: list[Detection],
    risk: RiskClassification,
    present: PresentAssessment,
) -> FuturePrediction:
    """Predict branching future scenarios — what could go wrong and how likely."""
    data_uri = encode_image_to_data_uri(image_path)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a predictive fire safety agent (the 'Future Agent'). "
                "Think like the TVA from Marvel: analyze forking timelines of what "
                "could go wrong. Consider:\n"
                "- Could objects be displaced (kicked, blown, dropped) closer to ignition sources?\n"
                "- Could sparks, embers, or heat radiation reach flammable materials?\n"
                "- Could environmental changes (wind, vibration, structural failure) worsen the scene?\n"
                "- What is the time horizon for each scenario?\n\n"
                "Use chain-of-thought reasoning. Output structured scenarios with "
                "likelihood, severity, and time horizon for each."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Present assessment:\n"
                        f"  Summary: {present.summary}\n"
                        f"  Hazards: {', '.join(present.hazards)}\n"
                        f"  Distances: {', '.join(present.distances)}\n\n"
                        f"Risk level: {risk.risk_level.value}\n\n"
                        "Predict dangerous future scenarios."
                    ),
                },
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        },
    ]

    result = openrouter_client.chat_completion_json(
        messages, json_schema=_FUTURE_SCHEMA
    )
    logger.info("Future prediction: %d scenarios", len(result.get("scenarios", [])))
    return FuturePrediction(
        scenarios=[FutureScenario(**s) for s in result["scenarios"]],
        overall_risk=RiskLevel(result["overall_risk"]),
        recommendation=result["recommendation"],
    )
