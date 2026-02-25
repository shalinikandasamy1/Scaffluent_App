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
                "You are a fire scene observer (the 'Present Agent'). "
                "Describe the CURRENT state of the scene objectively — what is present, "
                "where it is, and what physical relationships exist. Do not editorialize.\n\n"
                "For each ignition source: state whether it appears controlled (torch, candle, "
                "welding arc) or uncontrolled (freely burning fire).\n"
                "For each flammable material: note its distance from the nearest ignition source.\n"
                "Note any environmental spread factors: wind indicators, embers in flight, "
                "enclosure, structural elements.\n"
                "Be concise, precise, and factual. Avoid alarm language."
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
                        "Describe the current scene state."
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
                "You are a fire spread risk analyst (the 'Future Agent'). "
                "Assess how this fire situation is LIKELY to evolve, anchored in "
                "realistic probabilities — not worst-case brainstorming.\n\n"
                "Core principle: a controlled, isolated flame in a clear space poses "
                "LOW spread risk even though fires can theoretically spread under "
                "exotic circumstances. Construction and industrial work routinely "
                "involves open flames; that alone is not a crisis.\n\n"
                "Assign likelihood honestly:\n"
                "  unlikely — requires a specific accident (displacement, strong wind gust)\n"
                "  possible — plausible given normal activity, but not actively occurring\n"
                "  likely   — a natural continuation of what is already visible in the scene\n"
                "  certain  — already happening or physically inevitable\n\n"
                "Calibrate overall_risk by what the scene ACTUALLY shows:\n"
                "  low      — Controlled, isolated flame; no spread pathway visible.\n"
                "  medium   — Controlled flame with flammable materials visible in the same "
                "frame; spread needs a specific trigger (displacement, prolonged exposure).\n"
                "  high     — ANY of: (a) large uncontrolled fire regardless of visible fuel "
                "targets, (b) active ember/spark dispersal visible in the scene, "
                "(c) controlled flame immediately adjacent to significant fuel.\n"
                "  critical — ANY of: (a) at least one scenario that is 'certain' with "
                "severity 'high' or 'critical' (e.g. explosive container touching flame), "
                "(b) multiple simultaneous 'likely' high-severity pathways, "
                "(c) fire already spreading.\n\n"
                "Important: active ember or spark dispersal visible in the image is itself a "
                "HIGH spread signal — embers travel beyond the visible frame and can land on "
                "unknown materials. Do not rate ember-producing fires as 'low'.\n\n"
                "For each scenario state the physical mechanism (direct contact, radiant heat, "
                "ember travel, displacement) and what trigger is needed. "
                "Do NOT list every conceivable bad outcome — focus on what the scene "
                "actually indicates will happen versus what would require an unlikely accident."
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
                        f"  Ignition sources / flammables: {', '.join(present.hazards)}\n"
                        f"  Distances: {', '.join(present.distances)}\n\n"
                        f"Current risk level: {risk.risk_level.value}\n\n"
                        "Assess how this fire situation is likely to evolve."
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
