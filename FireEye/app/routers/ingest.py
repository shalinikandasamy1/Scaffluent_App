"""Image ingestion endpoints — accepts uploads from CCTV feeds and mobile devices."""

from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models.schemas import IngestResponse, SourceType
from app.storage import image_store

router = APIRouter(prefix="/ingest", tags=["ingest"])

# Maximum upload size: 20 MB
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@router.post("/", response_model=IngestResponse)
async def ingest_image(
    file: UploadFile = File(...),
    source_type: SourceType = Form(SourceType.mobile),
    source_id: str = Form("unknown"),
    location: str = Form(""),
    notes: str = Form(""),
) -> IngestResponse:
    """Upload an image for analysis.

    Accepts multipart form data with the image file and metadata fields.
    """
    image_id = uuid4()
    data = await file.read()

    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(data)} bytes exceeds {MAX_UPLOAD_BYTES} byte limit",
        )

    image_store.store_input_image(image_id, file.filename or "upload.jpg", data)

    # Metadata could be persisted to a database in production;
    # for now we just return the assigned ID.
    return IngestResponse(image_id=image_id)
