"""Analysis endpoints — trigger the pipeline and retrieve results."""

from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.models.schemas import AnalysisResult
from app.pipeline import orchestrator
from app.storage import image_store

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post("/{image_id}", response_model=AnalysisResult)
async def run_analysis(image_id: UUID) -> AnalysisResult:
    """Trigger the full FireEye pipeline on a previously ingested image."""
    try:
        result = orchestrator.analyze_image(image_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found")
    return result


@router.get("/{image_id}/annotated")
async def get_annotated_image(image_id: UUID):
    """Download the YOLO-annotated output image."""
    path = image_store.get_output_image_path(image_id, "annotated")
    if path is None:
        raise HTTPException(status_code=404, detail="Annotated image not found")
    return FileResponse(path, media_type="image/jpeg")
