"""Observatory BDLISA router — serves pre-extracted BDLISA NV3 entity polygons."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/observatory/bdlisa", tags=["observatory-bdlisa"])

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "bdlisa"


@router.get("/entity")
def bdlisa_entity(
    code: str = Query(..., min_length=3, max_length=20, pattern=r"^[A-Za-z0-9]+$"),
):
    """Return the BDLISA NV3 entity polygon by code (e.g. 113AF05)."""
    path = _DATA_DIR / f"{code.upper()}.json"
    if not path.is_file():
        raise HTTPException(404, f"No geometry found for entity {code.upper()}")
    return FileResponse(path, media_type="application/json")
