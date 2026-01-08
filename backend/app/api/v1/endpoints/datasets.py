"""
Datasets endpoints - CRUD operations for prepared datasets.

Reuses logic from dashboard/utils/dataset_registry.py
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from app.schemas import (
    DatasetCreate,
    DatasetInfo,
    DatasetListResponse,
    DatasetStatistics,
)
from app.config import get_settings

import sys
from pathlib import Path

# Add legacy dashboard to path
sys.path.insert(0, str(Path(__file__).parents[5]))

router = APIRouter()


@router.get("", response_model=DatasetListResponse)
async def list_datasets():
    """
    List all prepared datasets.
    """
    try:
        from core.dataset_registry import get_dataset_registry
        
        registry = get_dataset_registry()
        datasets = registry.scan_datasets()
        
        dataset_list = []
        for ds in datasets:
            dataset_list.append(DatasetInfo(
                id=ds.name,  # Use name as ID for now
                name=ds.name,
                source_type="csv" if ds.source_file else "database",
                source_file=ds.source_file,
                date_column=ds.date_column,
                target_column=ds.target_column,
                covariate_columns=ds.covariate_columns or [],
                station_column=ds.station_column,
                stations=None,  # Would need to load data to get this
                date_range=ds.date_range,
                row_count=0,  # Would need to load data
                preprocessing=ds.preprocessing,
                created_at=ds.created_at if hasattr(ds, 'created_at') else None,
            ))
        
        return DatasetListResponse(
            datasets=dataset_list,
            total=len(dataset_list),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}", response_model=DatasetInfo)
async def get_dataset(dataset_id: str):
    """
    Get details for a specific dataset.
    """
    try:
        from core.dataset_registry import get_dataset_registry
        
        registry = get_dataset_registry()
        datasets = registry.scan_datasets()
        
        for ds in datasets:
            if ds.name == dataset_id:
                return DatasetInfo(
                    id=ds.name,
                    name=ds.name,
                    source_type="csv" if ds.source_file else "database",
                    source_file=ds.source_file,
                    date_column=ds.date_column,
                    target_column=ds.target_column,
                    covariate_columns=ds.covariate_columns or [],
                    station_column=ds.station_column,
                    date_range=ds.date_range,
                    row_count=0,
                    preprocessing=ds.preprocessing,
                    created_at=ds.created_at if hasattr(ds, 'created_at') else None,
                )
        
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=DatasetInfo)
async def create_dataset(request: DatasetCreate):
    """
    Create a new prepared dataset from a data source.
    """
    try:
        from datetime import datetime
        import json
        import pandas as pd
        
        settings = get_settings()
        
        # Create dataset directory
        dataset_dir = settings.datasets_dir / request.name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Load source data
        if request.source_type == "csv" and request.source_file:
            source_path = Path(request.source_file)
            if not source_path.exists():
                source_path = settings.uploads_dir / request.source_file
            df = pd.read_csv(source_path)
        else:
            # For database, we need connection info - simplified for now
            raise HTTPException(
                status_code=400,
                detail="Database source requires connection via /sources/db/query first"
            )
        
        # Apply preprocessing
        # 1. Parse dates
        if request.date_column:
            df[request.date_column] = pd.to_datetime(df[request.date_column])
            df = df.sort_values(request.date_column)
        
        # 2. Handle missing values
        if request.preprocessing.fill_method == "interpolate":
            df = df.interpolate(method='linear')
        elif request.preprocessing.fill_method == "ffill":
            df = df.fillna(method='ffill')
        elif request.preprocessing.fill_method == "bfill":
            df = df.fillna(method='bfill')
        elif request.preprocessing.fill_method == "drop":
            df = df.dropna()
        
        # Get date range
        date_min = df[request.date_column].min()
        date_max = df[request.date_column].max()
        date_range = (str(date_min.date()), str(date_max.date()))
        
        # Get stations if applicable
        stations = None
        if request.station_column and request.station_column in df.columns:
            stations = df[request.station_column].unique().tolist()
        
        # Save processed data
        processed_path = dataset_dir / "data.parquet"
        df.to_parquet(processed_path, index=False)
        
        # Save config
        config = {
            "name": request.name,
            "source_type": request.source_type,
            "source_file": request.source_file,
            "date_column": request.date_column,
            "target_column": request.target_column,
            "covariate_columns": request.covariate_columns or [],
            "station_column": request.station_column,
            "preprocessing": request.preprocessing.model_dump(),
            "date_range": date_range,
            "row_count": len(df),
            "created_at": datetime.now().isoformat(),
        }
        
        config_path = dataset_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        return DatasetInfo(
            id=request.name,
            name=request.name,
            source_type=request.source_type,
            source_file=request.source_file,
            date_column=request.date_column,
            target_column=request.target_column,
            covariate_columns=request.covariate_columns or [],
            station_column=request.station_column,
            stations=stations,
            date_range=date_range,
            row_count=len(df),
            preprocessing=request.preprocessing,
            created_at=datetime.now(),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """
    Delete a dataset.
    """
    try:
        import shutil
        
        settings = get_settings()
        dataset_dir = settings.datasets_dir / dataset_id
        
        if not dataset_dir.exists():
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
        
        # Delete the entire dataset directory
        shutil.rmtree(dataset_dir)
        
        return {"message": f"Dataset '{dataset_id}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/preview")
async def preview_dataset(
    dataset_id: str,
    limit: int = Query(100, le=1000),
):
    """
    Get a preview of dataset data.
    """
    try:
        from core.dataset_registry import get_dataset_registry
        
        registry = get_dataset_registry()
        datasets = registry.scan_datasets()
        
        for ds in datasets:
            if ds.name == dataset_id:
                df, config = registry.load_dataset(ds)
                
                preview_df = df.head(limit)
                
                return {
                    "columns": df.columns.tolist(),
                    "data": preview_df.to_dict(orient="records"),
                    "total_rows": len(df),
                    "preview_rows": len(preview_df),
                }
        
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/statistics", response_model=DatasetStatistics)
async def get_dataset_statistics(dataset_id: str):
    """
    Get statistics for a dataset.
    """
    try:
        from core.dataset_registry import get_dataset_registry
        from core.statistics import compute_statistics
        
        registry = get_dataset_registry()
        datasets = registry.scan_datasets()
        
        for ds in datasets:
            if ds.name == dataset_id:
                df, config = registry.load_dataset(ds)
                
                # Compute basic stats
                target_col = ds.target_column
                target_stats = {
                    "mean": float(df[target_col].mean()),
                    "std": float(df[target_col].std()),
                    "min": float(df[target_col].min()),
                    "max": float(df[target_col].max()),
                    "median": float(df[target_col].median()),
                }
                
                missing = {col: int(df[col].isna().sum()) for col in df.columns}
                
                stations = None
                if ds.station_column and ds.station_column in df.columns:
                    stations = df[ds.station_column].unique().tolist()
                
                return DatasetStatistics(
                    dataset_id=dataset_id,
                    row_count=len(df),
                    date_range=ds.date_range,
                    target_stats=target_stats,
                    missing_values=missing,
                    stations=stations,
                )
        
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
