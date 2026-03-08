"""Stage 3: LLM agents for present assessment and future prediction.

Two agents analyze scenes that pass the risk classifier:
  1. Present Agent  — describes current hazards and spatial relationships.
  2. Future Agent   — predicts branching danger scenarios (TVA-style).
"""

from __future__ import annotations

import logging

from app.models.schemas import (
    ComplianceFlag,
    Detection,
    FuturePrediction,
    FutureScenario,
    PresentAssessment,
    RiskClassification,
    RiskLevel,
)
from app.config import settings
from app.pipeline.spatial import format_spatial_summary
from app.services import openrouter_client
from app.services.image_utils import encode_image_to_data_uri
from app.services.prompt_loader import get_system_prompt, get_user_template

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
            "compliance_flags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["present", "absent", "unclear"],
                        },
                        "note": {"type": "string"},
                    },
                    "required": ["item", "status", "note"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["summary", "hazards", "distances", "compliance_flags"],
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
    spatial_summary = format_spatial_summary(detections)
    data_uri = encode_image_to_data_uri(image_path)

    user_text = get_user_template("present_agent").format(
        detection_summary=detection_summary,
        spatial_summary=spatial_summary,
        risk_level=risk.risk_level.value,
        risk_reason=risk.reason,
    )

    messages = [
        {
            "role": "system",
            "content": get_system_prompt("present_agent", settings.local_llm_model if settings.llm_backend == "local" else ""),
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
        messages, json_schema=_PRESENT_SCHEMA
    )
    logger.info("Present assessment: %s", result.get("summary", "")[:120])
    flags = [ComplianceFlag(**f) for f in result.get("compliance_flags", [])]
    return PresentAssessment(
        summary=result["summary"],
        hazards=result["hazards"],
        distances=result["distances"],
        compliance_flags=flags,
    )


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

    user_text = get_user_template("future_agent").format(
        present_summary=present.summary,
        present_hazards=", ".join(present.hazards),
        present_distances=", ".join(present.distances),
        risk_level=risk.risk_level.value,
    )

    messages = [
        {
            "role": "system",
            "content": get_system_prompt("future_agent", settings.local_llm_model if settings.llm_backend == "local" else ""),
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
        messages, json_schema=_FUTURE_SCHEMA
    )
    logger.info("Future prediction: %d scenarios", len(result.get("scenarios", [])))
    return FuturePrediction(
        scenarios=[FutureScenario(**s) for s in result["scenarios"]],
        overall_risk=RiskLevel(result["overall_risk"]),
        recommendation=result["recommendation"],
    )
