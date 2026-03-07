"""Unit tests for heuristic risk classifier and spatial reasoning.

These tests run without any API calls or GPU — pure logic testing.
Run with: python -m pytest tests/test_heuristic.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.schemas import BoundingBox, Detection, RiskLevel
from app.pipeline.risk_classifier import classify_from_detections
from app.pipeline.spatial import compute_distances, estimate_scale, format_spatial_summary


def _det(label: str, conf: float = 0.8,
         x1: float = 0, y1: float = 0,
         x2: float = 100, y2: float = 100) -> Detection:
    """Helper to create a Detection."""
    return Detection(
        label=label, confidence=conf,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


# =====================================================================
# Heuristic classifier tests
# =====================================================================

class TestHeuristicClassifier:
    def test_no_detections_is_safe(self):
        result = classify_from_detections([])
        assert result.risk_level == RiskLevel.safe

    def test_only_safety_equipment_is_safe(self):
        detections = [
            _det("fire_extinguisher"),
            _det("hard_hat"),
            _det("safety_vest"),
        ]
        result = classify_from_detections(detections)
        assert result.risk_level == RiskLevel.safe

    def test_fire_without_safety_is_high(self):
        detections = [_det("fire")]
        result = classify_from_detections(detections)
        assert result.risk_level == RiskLevel.high

    def test_fire_with_safety_is_medium(self):
        detections = [
            _det("fire"),
            _det("fire_extinguisher"),
        ]
        result = classify_from_detections(detections)
        assert result.risk_level == RiskLevel.medium

    def test_ignition_near_flammable_without_extinguisher_is_high(self):
        detections = [
            _det("welding_sparks", x1=0, y1=0, x2=50, y2=50),
            _det("scaffold_net", x1=60, y1=0, x2=200, y2=200),
        ]
        result = classify_from_detections(detections)
        assert result.risk_level == RiskLevel.high

    def test_ignition_near_flammable_with_extinguisher_is_medium(self):
        detections = [
            _det("welding_sparks"),
            _det("tarpaulin"),
            _det("fire_extinguisher"),
        ]
        result = classify_from_detections(detections)
        assert result.risk_level == RiskLevel.medium

    def test_gas_cylinder_near_ignition_is_critical(self):
        # Place gas cylinder close to fire
        detections = [
            _det("fire", x1=100, y1=100, x2=200, y2=200),
            _det("gas_cylinder", x1=150, y1=150, x2=250, y2=300),
        ]
        result = classify_from_detections(detections)
        assert result.risk_level == RiskLevel.critical

    def test_flammable_without_ignition_is_low(self):
        detections = [_det("scaffold_net")]
        result = classify_from_detections(detections)
        assert result.risk_level == RiskLevel.low

    def test_smoke_only_is_medium(self):
        detections = [_det("smoke")]
        result = classify_from_detections(detections)
        assert result.risk_level == RiskLevel.medium

    def test_person_only_is_safe(self):
        detections = [_det("person")]
        result = classify_from_detections(detections)
        assert result.risk_level == RiskLevel.safe


# =====================================================================
# Spatial reasoning tests
# =====================================================================

class TestSpatialReasoning:
    def test_compute_distances_empty(self):
        assert compute_distances([]) == []

    def test_compute_distances_single(self):
        assert compute_distances([_det("fire")]) == []

    def test_compute_distances_pair(self):
        d1 = _det("fire", x1=0, y1=0, x2=100, y2=100)  # center: 50,50
        d2 = _det("person", x1=200, y1=200, x2=300, y2=300)  # center: 250,250
        distances = compute_distances([d1, d2])
        assert len(distances) == 1
        assert distances[0]["safety_concern"] is True  # fire <-> person
        expected_dist = ((250 - 50) ** 2 + (250 - 50) ** 2) ** 0.5
        assert abs(distances[0]["distance_px"] - expected_dist) < 1.0

    def test_safety_concern_flagging(self):
        d1 = _det("welding_sparks")
        d2 = _det("gas_cylinder")
        distances = compute_distances([d1, d2])
        assert distances[0]["safety_concern"] is True

    def test_no_safety_concern(self):
        d1 = _det("hard_hat")
        d2 = _det("safety_vest")
        distances = compute_distances([d1, d2])
        assert distances[0]["safety_concern"] is False

    def test_estimate_scale_with_person(self):
        # Person detection with height of 170 pixels → ~1.7m → 100 px/m
        person = _det("person", x1=100, y1=100, x2=200, y2=270)
        scale = estimate_scale([person])
        assert scale is not None
        assert abs(scale - 100.0) < 1.0  # 170px / 1.7m = 100 px/m

    def test_estimate_scale_no_reference(self):
        fire = _det("fire")
        assert estimate_scale([fire]) is None

    def test_format_spatial_summary_empty(self):
        assert format_spatial_summary([]) == "No objects detected."

    def test_format_spatial_summary_with_concerns(self):
        detections = [
            _det("fire", x1=0, y1=0, x2=100, y2=100),
            _det("scaffold_net", x1=50, y1=50, x2=200, y2=200),
        ]
        summary = format_spatial_summary(detections)
        assert "PROXIMITY CONCERNS" in summary
        assert "fire" in summary
        assert "scaffold_net" in summary


# =====================================================================
# Compliance score tests
# =====================================================================

class TestComplianceScore:
    def test_empty_flags_score_is_one(self):
        from app.models.schemas import PresentAssessment
        pa = PresentAssessment(summary="test")
        assert pa.compliance_score == 1.0

    def test_all_present_score_is_one(self):
        from app.models.schemas import ComplianceFlag, PresentAssessment
        flags = [
            ComplianceFlag(item="extinguisher", status="present"),
            ComplianceFlag(item="exit_sign", status="present"),
        ]
        pa = PresentAssessment(summary="test", compliance_flags=flags)
        assert pa.compliance_score == 1.0

    def test_all_absent_score_is_zero(self):
        from app.models.schemas import ComplianceFlag, PresentAssessment
        flags = [
            ComplianceFlag(item="extinguisher", status="absent"),
            ComplianceFlag(item="exit_sign", status="absent"),
        ]
        pa = PresentAssessment(summary="test", compliance_flags=flags)
        assert pa.compliance_score == 0.0

    def test_mixed_score(self):
        from app.models.schemas import ComplianceFlag, PresentAssessment
        flags = [
            ComplianceFlag(item="extinguisher", status="present"),   # 1.0
            ComplianceFlag(item="exit_sign", status="absent"),        # 0.0
            ComplianceFlag(item="PPE", status="unclear"),             # 0.5
        ]
        pa = PresentAssessment(summary="test", compliance_flags=flags)
        assert pa.compliance_score == 0.5  # (1.0 + 0.0 + 0.5) / 3

    def test_compliance_issues_lists_problems(self):
        from app.models.schemas import ComplianceFlag, PresentAssessment
        flags = [
            ComplianceFlag(item="extinguisher", status="present", note="ok"),
            ComplianceFlag(item="exit_sign", status="absent", note="missing"),
            ComplianceFlag(item="PPE", status="unclear", note="can't tell"),
        ]
        pa = PresentAssessment(summary="test", compliance_flags=flags)
        issues = pa.compliance_issues
        assert len(issues) == 2
        assert "exit_sign: missing" in issues
        assert "PPE: can't tell" in issues
