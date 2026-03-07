"""Audit logging for FireEye analysis runs.

Records each analysis with model versions, timings, and results
to a JSONL file for accountability and debugging.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from app.config import settings

logger = logging.getLogger(__name__)

_AUDIT_DIR = settings.base_dir / "audit_logs"


class AuditRecord:
    """Collects timing and metadata for a single analysis run."""

    def __init__(self, image_id: UUID, image_name: str = ""):
        self.image_id = str(image_id)
        self.image_name = image_name
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.yolo_model = settings.yolo_model_name
        self.yolo_confidence = settings.yolo_confidence_threshold
        self.llm_model = settings.llm_model
        self.llm_temperature = settings.llm_temperature
        self._stage_times: dict[str, float] = {}
        self._stage_start: float | None = None
        self._current_stage: str | None = None
        self.detection_count: int = 0
        self.risk_level: str = ""
        self.compliance_score: float | None = None
        self.error: str | None = None

    def start_stage(self, name: str) -> None:
        self._current_stage = name
        self._stage_start = time.monotonic()

    def end_stage(self) -> None:
        if self._current_stage and self._stage_start is not None:
            elapsed = time.monotonic() - self._stage_start
            self._stage_times[self._current_stage] = round(elapsed, 3)
        self._current_stage = None
        self._stage_start = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "image_name": self.image_name,
            "started_at": self.started_at,
            "yolo_model": self.yolo_model,
            "yolo_confidence_threshold": self.yolo_confidence,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "detection_count": self.detection_count,
            "risk_level": self.risk_level,
            "compliance_score": self.compliance_score,
            "stage_times_s": self._stage_times,
            "total_time_s": round(sum(self._stage_times.values()), 3),
            "error": self.error,
        }


def write_audit(record: AuditRecord) -> None:
    """Append an audit record to the daily JSONL file."""
    _AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    audit_file = _AUDIT_DIR / f"audit_{date_str}.jsonl"
    line = json.dumps(record.to_dict(), ensure_ascii=False)
    with open(audit_file, "a") as f:
        f.write(line + "\n")
    logger.info("Audit record written: %s", audit_file.name)
