"""
Models endpoints - List and manage trained models.

Reuses logic from dashboard/utils/model_registry.py
"""

from typing import Optional, Literal
from fastapi import APIRouter, HTTPException, Query

from app.schemas import (
    ModelInfo,
    ModelListResponse,
    ModelMetrics,
)
from app.config import get_settings

import sys
from pathlib import Path

# Add legacy dashboard to path
sys.path.insert(0, str(Path(__file__).parents[5]))

router = APIRouter()


@router.get("", response_model=ModelListResponse)
async def list_models(
    model_type: Optional[Literal["single", "global"]] = None,
    model_name: Optional[str] = None,
    station: Optional[str] = None,
):
    """
    List all trained models with optional filtering.
    """
    try:
        from core.model_registry import get_registry
        
        settings = get_settings()
        registry = get_registry(settings.checkpoints_dir.parent)
        
        if station:
            models = registry.get_models_for_station(
                station_id=station,
                model_type=model_type,
                model_name=model_name,
            )
        else:
            models = registry.list_all_models(
                model_type=model_type,
                model_name=model_name,
            )
        
        model_list = []
        for m in models:
            model_list.append(ModelInfo(
                model_id=m.model_id,
                model_name=m.model_name,
                model_type=m.model_type,
                stations=m.stations,
                primary_station=m.primary_station,
                dataset_id=m.dataset_id if hasattr(m, 'dataset_id') else None,
                data_source=m.data_source if hasattr(m, 'data_source') else None,
                hyperparams=m.hyperparams or {},
                preprocessing_config=m.preprocessing_config if hasattr(m, 'preprocessing_config') else None,
                input_chunk_length=m.hyperparams.get("input_chunk_length", 30) if m.hyperparams else 30,
                output_chunk_length=m.hyperparams.get("output_chunk_length", 7) if m.hyperparams else 7,
                metrics=m.metrics if hasattr(m, 'metrics') else None,
                created_at=m.created_at,
                path=m.path,
            ))
        
        return ModelListResponse(
            models=model_list,
            total=len(model_list),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    """
    Get details for a specific model.
    """
    try:
        from core.model_registry import get_registry
        
        settings = get_settings()
        registry = get_registry(settings.checkpoints_dir.parent)
        
        model = registry.get_model(model_id)
        
        if not model:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        
        return ModelInfo(
            model_id=model.model_id,
            model_name=model.model_name,
            model_type=model.model_type,
            stations=model.stations,
            primary_station=model.primary_station,
            dataset_id=model.dataset_id if hasattr(model, 'dataset_id') else None,
            data_source=model.data_source if hasattr(model, 'data_source') else None,
            hyperparams=model.hyperparams or {},
            preprocessing_config=model.preprocessing_config if hasattr(model, 'preprocessing_config') else None,
            input_chunk_length=model.hyperparams.get("input_chunk_length", 30) if model.hyperparams else 30,
            output_chunk_length=model.hyperparams.get("output_chunk_length", 7) if model.hyperparams else 7,
            metrics=model.metrics if hasattr(model, 'metrics') else None,
            created_at=model.created_at,
            path=model.path,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """
    Delete a trained model.
    """
    try:
        from core.model_registry import get_registry
        
        settings = get_settings()
        registry = get_registry(settings.checkpoints_dir.parent)
        
        success = registry.delete_model(model_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        
        return {"message": f"Model '{model_id}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_id}/metrics", response_model=ModelMetrics)
async def get_model_metrics(model_id: str):
    """
    Get evaluation metrics for a model.
    """
    try:
        from core.model_registry import get_registry
        
        settings = get_settings()
        registry = get_registry(settings.checkpoints_dir.parent)
        
        model = registry.get_model(model_id)
        
        if not model:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        
        metrics = model.metrics if hasattr(model, 'metrics') and model.metrics else {}
        
        return ModelMetrics(
            model_id=model_id,
            station=model.primary_station or model.stations[0] if model.stations else "unknown",
            mae=metrics.get("mae"),
            rmse=metrics.get("rmse"),
            mape=metrics.get("mape"),
            smape=metrics.get("smape"),
            r2=metrics.get("r2"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stations")
async def list_stations():
    """
    List all stations that have trained models.
    """
    try:
        from core.model_registry import get_registry
        
        settings = get_settings()
        registry = get_registry(settings.checkpoints_dir.parent)
        
        stations = registry.get_all_stations()
        
        return {"stations": stations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
