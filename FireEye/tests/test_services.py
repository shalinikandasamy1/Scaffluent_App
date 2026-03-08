"""Unit tests for FireEye service modules.

Tests prompt loading, audit recording, and image utilities
without requiring API keys or network access.
"""

import json
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestPromptLoader:
    def test_load_risk_classifier(self):
        from app.services.prompt_loader import load_prompt
        p = load_prompt("risk_classifier")
        assert "system" in p
        assert "user_template" in p
        assert "fire safety" in p["system"].lower()
        assert "{detection_summary}" in p["user_template"]

    def test_load_present_agent(self):
        from app.services.prompt_loader import load_prompt
        p = load_prompt("present_agent")
        assert "Present Agent" in p["system"]
        assert "{risk_level}" in p["user_template"]

    def test_load_future_agent(self):
        from app.services.prompt_loader import load_prompt
        p = load_prompt("future_agent")
        assert "Future Agent" in p["system"]
        assert "{present_summary}" in p["user_template"]

    def test_missing_prompt_returns_empty(self):
        from app.services.prompt_loader import load_prompt
        p = load_prompt("nonexistent_prompt_xyz")
        assert p["system"] == ""
        assert p["user_template"] == ""

    def test_reload_clears_cache(self):
        from app.services.prompt_loader import load_prompt, reload_prompts
        # Load once (cached)
        load_prompt("risk_classifier")
        # Reload should not raise
        reload_prompts()
        # Load again should work
        p = load_prompt("risk_classifier")
        assert len(p["system"]) > 0


class TestAuditRecord:
    def test_create_record(self):
        from app.services.audit import AuditRecord
        ar = AuditRecord(uuid4(), "test.jpg")
        assert ar.image_name == "test.jpg"
        assert ar.detection_count == 0
        assert ar.error is None

    def test_stage_timing(self):
        import time
        from app.services.audit import AuditRecord
        ar = AuditRecord(uuid4(), "test.jpg")
        ar.start_stage("yolo")
        time.sleep(0.01)
        ar.end_stage()
        d = ar.to_dict()
        assert "yolo" in d["stage_times_s"]
        assert d["stage_times_s"]["yolo"] > 0

    def test_to_dict_fields(self):
        from app.services.audit import AuditRecord
        ar = AuditRecord(uuid4(), "test.jpg")
        ar.detection_count = 5
        ar.risk_level = "high"
        ar.compliance_score = 0.6
        d = ar.to_dict()
        assert d["detection_count"] == 5
        assert d["risk_level"] == "high"
        assert d["compliance_score"] == 0.6
        assert "started_at" in d
        assert "yolo_model" in d
        assert "llm_model" in d

    def test_write_audit(self):
        from app.services.audit import AuditRecord, write_audit, _AUDIT_DIR
        ar = AuditRecord(uuid4(), "test_write.jpg")
        ar.risk_level = "safe"
        # Write to the audit directory
        write_audit(ar)
        # Check a JSONL file was created
        audit_files = list(_AUDIT_DIR.glob("audit_*.jsonl"))
        assert len(audit_files) > 0
        # Read the last line
        with open(audit_files[-1]) as f:
            lines = f.readlines()
        last = json.loads(lines[-1])
        assert last["image_name"] == "test_write.jpg"
        assert last["risk_level"] == "safe"


class TestModelAdapters:
    def test_default_adapter_no_model(self):
        from app.services.prompt_loader import get_system_prompt
        prompt = get_system_prompt("risk_classifier")
        assert "fire safety" in prompt.lower()
        # No model name = no adapter appended
        assert "MUST respond with ONLY" not in prompt

    def test_adapter_for_qwen(self):
        from app.services.prompt_loader import get_system_prompt
        prompt = get_system_prompt("risk_classifier", "qwen2.5vl:7b")
        assert "fire safety" in prompt.lower()
        # Should have adapter JSON instruction appended
        assert "JSON" in prompt

    def test_adapter_for_unknown_model_uses_default(self):
        from app.services.prompt_loader import get_system_prompt
        prompt = get_system_prompt("risk_classifier", "some-unknown-model")
        assert "fire safety" in prompt.lower()
        # Default adapter has a JSON instruction
        assert "JSON" in prompt

    def test_adapter_prefix_match(self):
        from app.services.prompt_loader import get_system_prompt
        # "gemma3:12b" should match "gemma3" adapter prefix
        prompt = get_system_prompt("risk_classifier", "gemma3:12b")
        assert "single JSON object" in prompt


class TestCommonAccidents:
    def test_common_accidents_structure(self):
        from app.pipeline.risk_classifier import COMMON_ACCIDENTS
        assert len(COMMON_ACCIDENTS) == 7
        for accident in COMMON_ACCIDENTS:
            assert "cause" in accident
            assert "detectable_by" in accident
            assert "coverage" in accident
            assert accident["coverage"] in ("full", "partial", "none")
