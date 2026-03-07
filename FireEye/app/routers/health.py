"""Health check endpoint."""

from fastapi import APIRouter

from app.config import settings
from app.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse()


@router.get("/health/config")
async def health_config() -> dict:
    """Return current pipeline configuration (no secrets)."""
    return {
        "yolo_model": settings.yolo_model_name,
        "yolo_confidence_threshold": settings.yolo_confidence_threshold,
        "yolo_device": settings.yolo_device,
        "llm_model": settings.llm_model,
        "llm_temperature": settings.llm_temperature,
        "api_key_set": bool(settings.openrouter_api_key),
    }
