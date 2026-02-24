"""FastAPI application entry point for FireEye."""

import logging

from fastapi import FastAPI

from app.config import settings
from app.routers import analysis, health, ingest

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
)

app = FastAPI(
    title="FireEye",
    description="Predictive Fire Safety System — YOLO detection, risk classification, and LLM-powered hazard analysis.",
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(analysis.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=settings.debug)
