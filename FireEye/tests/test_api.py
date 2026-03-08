"""Smoke tests for FastAPI endpoints (no GPU/API keys needed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestHealthEndpoints:
    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_health_config(self):
        r = client.get("/health/config")
        assert r.status_code == 200
        data = r.json()
        assert "yolo_model" in data
        assert "llm_model" in data
        assert "api_key_set" in data


class TestIngestEndpoint:
    def test_missing_file_returns_422(self):
        r = client.post("/ingest/")
        assert r.status_code == 422  # missing required file field

    def test_upload_returns_image_id(self):
        # Minimal valid PNG (1x1 pixel)
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
            b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
            b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        r = client.post(
            "/ingest/",
            files={"file": ("test.png", png_bytes, "image/png")},
        )
        assert r.status_code == 200
        data = r.json()
        assert "image_id" in data
