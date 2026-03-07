"""Pydantic models for API request/response payloads."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SourceType(str, Enum):
    cctv = "cctv"
    mobile = "mobile"


class RiskLevel(str, Enum):
    safe = "safe"
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


# ---------------------------------------------------------------------------
# YOLO detection
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    label: str
    confidence: float
    bbox: BoundingBox


# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------

class RiskClassification(BaseModel):
    risk_level: RiskLevel
    confidence: float
    reason: str = ""


# ---------------------------------------------------------------------------
# LLM agent outputs
# ---------------------------------------------------------------------------

class ComplianceFlag(BaseModel):
    item: str
    status: str  # "present" | "absent" | "unclear"
    note: str = ""


class PresentAssessment(BaseModel):
    summary: str
    hazards: list[str] = Field(default_factory=list)
    distances: list[str] = Field(default_factory=list)
    compliance_flags: list[ComplianceFlag] = Field(default_factory=list)

    @computed_field
    @property
    def compliance_score(self) -> float:
        """Compute a 0-1 compliance score from flags.

        present=1.0, unclear=0.5, absent=0.0.
        Returns 1.0 if no flags (nothing to violate).
        """
        if not self.compliance_flags:
            return 1.0
        scores = {"present": 1.0, "unclear": 0.5, "absent": 0.0}
        total = sum(scores.get(f.status, 0.5) for f in self.compliance_flags)
        return round(total / len(self.compliance_flags), 2)

    @computed_field
    @property
    def compliance_issues(self) -> list[str]:
        """Return human-readable list of non-compliant items."""
        return [
            f"{f.item}: {f.note}" if f.note else f.item
            for f in self.compliance_flags
            if f.status in ("absent", "unclear")
        ]


class FutureScenario(BaseModel):
    scenario: str
    likelihood: str
    severity: str
    time_horizon: str


class FuturePrediction(BaseModel):
    scenarios: list[FutureScenario] = Field(default_factory=list)
    overall_risk: RiskLevel = RiskLevel.safe
    recommendation: str = ""


# ---------------------------------------------------------------------------
# Image ingestion
# ---------------------------------------------------------------------------

class ImageIngestRequest(BaseModel):
    source_type: SourceType
    source_id: str = Field(description="Camera ID or inspector badge number")
    location: str = ""
    notes: str = ""


class ImageMetadata(BaseModel):
    image_id: UUID = Field(default_factory=uuid4)
    filename: str
    source_type: SourceType
    source_id: str
    location: str = ""
    notes: str = ""
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Analysis result
# ---------------------------------------------------------------------------

class AnalysisResult(BaseModel):
    image_id: UUID
    detections: list[Detection] = Field(default_factory=list)
    risk_classification: RiskClassification | None = None
    present_assessment: PresentAssessment | None = None
    future_prediction: FuturePrediction | None = None
    annotated_image_path: str | None = None
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# API responses
# ---------------------------------------------------------------------------

class IngestResponse(BaseModel):
    image_id: UUID
    message: str = "Image ingested successfully"


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
