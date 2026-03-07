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
    data_uri = encode_image_to_data_uri(image_path)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a fire scene observer (the 'Present Agent') specialising in "
                "Hong Kong construction site fire safety.\n"
                "Describe the CURRENT state of the scene objectively — what is present, "
                "where it is, and what physical relationships exist. Do not editorialize.\n\n"
                "For each ignition source: state whether it appears controlled (welding arc, "
                "cutting torch) or uncontrolled (freely burning fire).\n"
                "For each flammable material: note its distance from the nearest ignition source.\n"
                "Note any environmental spread factors: wind indicators, embers in flight, "
                "enclosure, structural elements.\n\n"
                "REGULATORY CHECKLIST (note presence/absence of each):\n"
                "- Fire extinguishers (required on each floor and near each container)\n"
                "- Hose reels / water supply points\n"
                "- Exit signs (must be clear, marked, illuminated)\n"
                "- PPE compliance (hard hats, safety vests on workers)\n"
                "- Hot work screening (welding area should be screened off)\n"
                "- Combustible clearance (>= 6m from hot work per FSD CL 2/2008)\n"
                "- Gas cylinder storage (acetylene/oxygen within exempted quantities)\n"
                "- Scaffold net / tarpaulin condition (fire-retardant certification)\n\n"
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

    messages = [
        {
            "role": "system",
            "content": (
                "You are a fire spread risk analyst (the 'Future Agent') specialising in "
                "Hong Kong construction site fire safety.\n"
                "Assess how this fire situation is LIKELY to evolve, anchored in "
                "realistic probabilities — not worst-case brainstorming.\n\n"
                "Core principle: a controlled, isolated flame in a clear space poses "
                "LOW spread risk even though fires can theoretically spread under "
                "exotic circumstances. Construction work routinely involves open flames; "
                "that alone is not a crisis.\n\n"
                "Assign likelihood honestly:\n"
                "  unlikely — requires a specific accident (displacement, strong wind gust)\n"
                "  possible — plausible given normal activity, but not actively occurring\n"
                "  likely   — a natural continuation of what is already visible in the scene\n"
                "  certain  — already happening or physically inevitable\n\n"
                "Calibrate overall_risk by what the scene ACTUALLY shows:\n"
                "  low      — Controlled, isolated flame; no spread pathway visible.\n"
                "  medium   — Controlled flame with flammable materials visible in the same "
                "frame; spread needs a specific trigger.\n"
                "  high     — ANY of: (a) large uncontrolled fire, (b) active ember/spark "
                "dispersal, (c) flame immediately adjacent to scaffold nets or tarpaulins.\n"
                "  critical — ANY of: (a) gas cylinders adjacent to flame, "
                "(b) multiple simultaneous 'likely' high-severity pathways, "
                "(c) fire already spreading across multiple materials.\n\n"
                "HK-SPECIFIC ESCALATION FACTORS:\n"
                "- Scaffold nets/tarpaulins catch fire rapidly and spread vertically — "
                "high-rise facade fires are a known HK risk pattern\n"
                "- Gas cylinders (acetylene, LPG) near fire = explosion risk\n"
                "- Buildings >30m require water relaying systems; absence is critical\n"
                "- Non-fire-retardant scaffold nets accelerate vertical fire spread\n\n"
                "HK-SPECIFIC MITIGATION FACTORS:\n"
                "- Visible fire extinguishers reduce immediate spread risk\n"
                "- Hose reels indicate firefighting water availability\n"
                "- Proper PPE suggests trained workers who can respond\n\n"
                "For each scenario state the physical mechanism and what trigger is needed. "
                "Focus on what the scene actually indicates, not every conceivable bad outcome."
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
