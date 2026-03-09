"""Datasets API router.

CRUD operations for prepared datasets, CSV upload, DB import, preview, and profiling.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import settings
from api.database import get_brgm_db
from api.serializers import clean_nans
from api.schemas.datasets import (
    DatasetCreateRequest,
    DatasetDetail,
    DatasetPreview,
    DatasetProfile,
    DatasetSummary,
    ImportDBRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])


def _get_registry():
    """Lazy-load DatasetRegistry."""
    from dashboard.utils.dataset_registry import DatasetRegistry

    datasets_dir = Path(settings.data_dir) / "prepared"
    return DatasetRegistry(datasets_dir)


def _find_dataset(registry, dataset_id: str):
    """Find a dataset by directory name (id)."""
    datasets = registry.scan_datasets()
    for ds in datasets:
        if ds.path.name == dataset_id:
            return ds
    return None


def _ds_to_summary(ds) -> DatasetSummary:
    """Convert a DatasetRegistry entry to a DatasetSummary."""
    return DatasetSummary(
        id=ds.path.name,
        name=ds.name,
        source_file=ds.source_file or "",
        target_variable=ds.target_column or "",
        covariates=ds.covariate_columns or [],
        n_rows=ds.n_rows,
        date_range=list(ds.date_range) if ds.date_range else [],
        created_at=ds.creation_date or "",
        station_column=ds.station_column,
        stations=ds.stations or [],
    )


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@router.get("/", response_model=list[DatasetSummary])
async def list_datasets():
    """List all prepared datasets."""
    registry = _get_registry()
    return [_ds_to_summary(ds) for ds in registry.scan_datasets()]


@router.post("/", response_model=DatasetSummary, status_code=201)
async def create_dataset(
    file: UploadFile = File(...),
    name: str = Query(...),
    target_column: str = Query(...),
    covariate_columns: str = Query(""),  # comma-separated
    station_column: Optional[str] = Query(None),
    stations: str = Query(""),  # comma-separated
):
    """Upload a CSV file and register it as a prepared dataset."""
    import pandas as pd

    registry = _get_registry()

    # Read CSV into DataFrame
    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        df = pd.read_csv(tmp_path, index_col=0, parse_dates=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {exc}")

    cov_cols = [c.strip() for c in covariate_columns.split(",") if c.strip()]
    station_list = [s.strip() for s in stations.split(",") if s.strip()]

    try:
        registry.save_dataset(
            name=name,
            df=df,
            source_file=file.filename or "upload.csv",
            station_column=station_column,
            stations=station_list,
            target_column=target_column,
            covariate_columns=cov_cols,
            preprocessing_config={},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save dataset: {exc}")

    return DatasetSummary(
        id=name,
        name=name,
        source_file=file.filename or "upload.csv",
        target_variable=target_column,
        covariates=cov_cols,
        n_rows=len(df),
        date_range=[str(df.index.min()), str(df.index.max())],
        created_at="",
        station_column=station_column,
        stations=station_list,
    )


@router.get("/{dataset_id}", response_model=DatasetDetail)
async def get_dataset(dataset_id: str):
    """Get full details for a dataset."""
    registry = _get_registry()
    ds = _find_dataset(registry, dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    summary = _ds_to_summary(ds)
    return DatasetDetail(
        **summary.model_dump(),
        preprocessing=getattr(ds, "preprocessing", None) or {},
    )


@router.delete("/{dataset_id}", status_code=204)
async def delete_dataset(dataset_id: str):
    """Delete a prepared dataset."""
    registry = _get_registry()
    ds = _find_dataset(registry, dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        registry.delete_dataset(ds)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {exc}")


@router.post("/import-db", response_model=DatasetSummary, status_code=201)
async def import_from_db(
    req: ImportDBRequest,
    brgm_db: AsyncSession = Depends(get_brgm_db),
):
    """Import data from the BRGM PostgreSQL gold schema."""
    from dashboard.utils.postgres_connector import (
        create_connection,
        fetch_data,
    )

    # Build synchronous engine for the postgres_connector functions
    engine = create_connection(
        host=settings.brgm_db_host,
        port=settings.brgm_db_port,
        database=settings.brgm_db_name,
        user=settings.brgm_db_user,
        password=settings.brgm_db_password,
    )

    try:
        df = fetch_data(
            engine=engine,
            table_name=req.table_name,
            columns=req.columns,
            schema=req.schema_name,
            date_column=req.date_column,
            start_date=req.start_date,
            end_date=req.end_date,
            filters=req.filters,
            limit=req.limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database query failed: {exc}")
    finally:
        engine.dispose()

    if df.empty:
        raise HTTPException(status_code=404, detail="Query returned no rows")

    # Save as prepared dataset
    registry = _get_registry()
    dataset_name = req.dataset_name or f"{req.schema_name}_{req.table_name}"

    # Determine station column from filters
    station_col = None
    station_list: list[str] = []
    if req.filters:
        for col, vals in req.filters.items():
            if isinstance(vals, list) and vals:
                station_col = col
                station_list = vals
                break

    # Use the date column as index if available
    import pandas as pd

    if req.date_column and req.date_column in df.columns:
        df[req.date_column] = pd.to_datetime(df[req.date_column])
        df = df.set_index(req.date_column).sort_index()

    # Determine target and covariates from non-metadata columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Exclude station column from numeric analysis
    if station_col and station_col in numeric_cols:
        numeric_cols.remove(station_col)

    target_col = numeric_cols[0] if numeric_cols else ""
    cov_cols = numeric_cols[1:] if len(numeric_cols) > 1 else []

    registry.save_dataset(
        name=dataset_name,
        df=df,
        source_file=f"db://{req.schema_name}.{req.table_name}",
        station_column=station_col,
        stations=station_list,
        target_column=target_col,
        covariate_columns=cov_cols,
        preprocessing_config={"source": "brgm-postgres", "schema": req.schema_name},
    )

    return DatasetSummary(
        id=dataset_name,
        name=dataset_name,
        source_file=f"db://{req.schema_name}.{req.table_name}",
        target_variable=target_col,
        covariates=cov_cols,
        n_rows=len(df),
        date_range=[str(df.index.min()), str(df.index.max())] if len(df) > 0 else [],
        created_at="",
        station_column=station_col,
        stations=station_list,
    )


@router.get("/{dataset_id}/preview", response_model=DatasetPreview)
async def preview_dataset(dataset_id: str, n: int = Query(50, ge=1, le=1000)):
    """Preview the first N rows of a dataset."""
    registry = _get_registry()
    ds = _find_dataset(registry, dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        df, _ = registry.load_dataset(ds)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset data file not found")

    preview_df = df.head(n).reset_index()
    rows = clean_nans(preview_df.to_dict(orient="records"))

    return DatasetPreview(
        columns=list(preview_df.columns),
        rows=rows,
        total_rows=len(df),
    )


@router.get("/{dataset_id}/profile", response_model=DatasetProfile)
async def profile_dataset(dataset_id: str):
    """Compute statistical profile of a dataset."""
    registry = _get_registry()
    ds = _find_dataset(registry, dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        df, _ = registry.load_dataset(ds)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset data file not found")

    # Basic profiling without heavy dependencies
    columns_profile = {}
    for col in df.columns:
        col_stats: dict = {}
        if df[col].dtype.kind in ("f", "i", "u"):
            desc = df[col].describe()
            col_stats = clean_nans(desc.to_dict())
        else:
            col_stats = {
                "count": int(df[col].count()),
                "unique": int(df[col].nunique()),
                "top": str(df[col].mode().iloc[0]) if len(df[col].mode()) > 0 else None,
            }
        columns_profile[col] = col_stats

    return DatasetProfile(
        columns=columns_profile,
        shape=list(df.shape),
        dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
        missing={col: int(df[col].isna().sum()) for col in df.columns},
    )
