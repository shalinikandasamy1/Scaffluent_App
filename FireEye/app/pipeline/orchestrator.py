"""Pipeline orchestrator: Stage 1 → Stage 2 → Stage 3.

Coordinates the three-stage analysis pipeline:
  1. YOLO detection
  2. Risk classification (heuristic + optional LLM)
  3. LLM agents (present + future) — only if risk is above safe
"""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID

from app.models.schemas import AnalysisResult
from app.pipeline import llm_agents, risk_classifier, yolo_detector
from app.services.audit import AuditRecord, write_audit
from app.storage import image_store

logger = logging.getLogger(__name__)


def analyze_image(image_id: UUID) -> AnalysisResult:
    """Run the full FireEye pipeline on a stored image."""
    image_path = image_store.get_input_image_path(image_id)
    if image_path is None:
        raise FileNotFoundError(f"No stored image for id {image_id}")

    audit = AuditRecord(image_id, image_path.name)

    try:
        # ------------------------------------------------------------------
        # Stage 1: YOLO object detection
        # ------------------------------------------------------------------
        logger.info("[Stage 1] Running YOLO on %s", image_path.name)
        audit.start_stage("yolo_detection")
        output_dir = image_store._output_dir()
        annotated_path = output_dir / f"{image_id}_annotated.jpg"
        detections = yolo_detector.detect_and_annotate(image_path, annotated_path)
        audit.end_stage()
        audit.detection_count = len(detections)

        # ------------------------------------------------------------------
        # Stage 2: Risk classification  (short-circuited to LLM for now)
        # ------------------------------------------------------------------
        logger.info("[Stage 2] Classifying risk via LLM (%d detections)", len(detections))
        audit.start_stage("risk_classification")
        risk = risk_classifier.classify_with_llm(str(image_path), detections)
        audit.end_stage()
        audit.risk_level = risk.risk_level.value

        result = AnalysisResult(
            image_id=image_id,
            detections=detections,
            risk_classification=risk,
            annotated_image_path=str(annotated_path),
        )

        # ------------------------------------------------------------------
        # Stage 3: LLM agents — always run for now
        # ------------------------------------------------------------------
        logger.info("[Stage 3] Running LLM agents (risk=%s)", risk.risk_level.value)

        audit.start_stage("present_assessment")
        present = llm_agents.assess_present(str(image_path), detections, risk)
        audit.end_stage()
        result.present_assessment = present
        audit.compliance_score = present.compliance_score

        audit.start_stage("future_prediction")
        future = llm_agents.predict_future(
            str(image_path), detections, risk, present
        )
        audit.end_stage()
        result.future_prediction = future

    except Exception as e:
        audit.end_stage()
        audit.error = str(e)
        write_audit(audit)
        raise

    write_audit(audit)
    return result
