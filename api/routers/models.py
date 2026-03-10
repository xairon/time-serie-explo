"""Models API router.

List available architectures, list/get/delete trained models, download archives.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from api.config import settings
from api.serializers import clean_nans
from api.schemas.models import AvailableModel, ModelDetail, ModelSummary

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/models", tags=["models"])


def _get_model_registry():
    """Lazy-load ModelRegistry."""
    from dashboard.utils.model_registry import ModelRegistry

    return ModelRegistry(checkpoints_dir=Path(settings.checkpoints_dir))


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@router.get("/available", response_model=list[AvailableModel])
async def list_available_models():
    """List available model architectures with categories and default hyperparams."""
    from dashboard.utils.model_factory import ModelFactory

    # Try to load model configs for default hyperparams
    try:
        from dashboard.models_config import ALL_MODELS, MODEL_CATEGORIES
    except ImportError:
        ALL_MODELS = {}
        MODEL_CATEGORIES = {}

    # Build category lookup
    category_lookup = {}
    for cat, model_list in MODEL_CATEGORIES.items() if isinstance(MODEL_CATEGORIES, dict) else []:
        for model_name in model_list:
            category_lookup[model_name] = cat

    results = []
    for name in ModelFactory.get_available_models():
        is_torch = ModelFactory.is_torch_model(name)
        category = category_lookup.get(name, "Deep Learning" if is_torch else "Baselines")

        # Get default hyperparams from model config
        model_config = ALL_MODELS.get(name, {}) if isinstance(ALL_MODELS, dict) else {}
        default_hp = {}
        if isinstance(model_config, dict):
            for k, v in model_config.items():
                if k not in ("description", "category", "name"):
                    # For tuple ranges (min, max, default), use default
                    if isinstance(v, (list, tuple)) and len(v) == 3:
                        default_hp[k] = v[2]
                    else:
                        default_hp[k] = v

        results.append(
            AvailableModel(
                name=name,
                is_torch=is_torch,
                description=f"{'Deep Learning' if is_torch else 'Global Baseline'} model",
                category=category,
                default_hyperparams=default_hp,
            )
        )
    return results


@router.get("", response_model=list[ModelSummary])
async def list_models(
    model_type: Optional[str] = Query(None),
    model_name: Optional[str] = Query(None),
):
    """List all trained models from MLflow."""
    registry = _get_model_registry()
    entries = registry.list_all_models(model_type=model_type, model_name=model_name)
    return [
        ModelSummary(
            model_id=e.model_id,
            model_name=e.model_name,
            model_type=e.model_type,
            stations=e.stations,
            primary_station=e.primary_station,
            created_at=e.created_at,
            metrics=clean_nans(e.metrics),
            data_source=e.data_source,
        )
        for e in entries
    ]


@router.get("/{model_id}", response_model=ModelDetail)
async def get_model(model_id: str):
    """Get full details of a trained model."""
    registry = _get_model_registry()
    entry = registry.get_model(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return ModelDetail(
        model_id=entry.model_id,
        model_name=entry.model_name,
        model_type=entry.model_type,
        stations=entry.stations,
        primary_station=entry.primary_station,
        created_at=entry.created_at,
        metrics=clean_nans(entry.metrics),
        data_source=entry.data_source,
        run_id=entry.run_id,
        hyperparams=clean_nans(entry.hyperparams),
        preprocessing_config=clean_nans(entry.preprocessing_config),
        display_name=entry.display_name,
    )


@router.delete("/{model_id}", status_code=204)
async def delete_model(model_id: str):
    """Delete a trained model (marks the MLflow run as deleted)."""
    registry = _get_model_registry()
    entry = registry.get_model(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Model not found")
    if not registry.delete_model(model_id):
        raise HTTPException(status_code=500, detail="Failed to delete model")


@router.get("/{model_id}/test-info")
async def get_model_test_info(model_id: str):
    """Return test set dates and model chunk lengths for the sliding window UI."""
    registry = _get_model_registry()
    entry = registry.get_model(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Model not found")

    test_df = registry.load_data(entry, "test")
    if test_df is None:
        raise HTTPException(status_code=404, detail="Test data not found")

    model = registry.load_model(entry)
    config = registry.load_model_config(entry)
    input_chunk = getattr(model, "input_chunk_length", 30)
    output_chunk = getattr(model, "output_chunk_length", 7)

    test_dates = [d.isoformat() for d in test_df.index]
    test_len = len(test_dates)

    # Valid slider range
    valid_start = input_chunk
    valid_end = max(valid_start, test_len - output_chunk)

    # Target column from config or first column
    target_col = ""
    if config:
        target_col = config.get("columns", {}).get("target", "")
    if not target_col and len(test_df.columns) > 0:
        target_col = test_df.columns[0]

    return {
        "test_dates": test_dates,
        "test_length": test_len,
        "input_chunk_length": input_chunk,
        "output_chunk_length": output_chunk,
        "valid_start_idx": valid_start,
        "valid_end_idx": valid_end,
        "target_column": target_col,
    }


@router.get("/{model_id}/download")
async def download_model(model_id: str):
    """Download model artifacts as a ZIP archive."""
    registry = _get_model_registry()
    entry = registry.get_model(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Model not found")

    # Download artifacts to a temporary directory and create ZIP
    try:
        import mlflow

        local_dir = mlflow.artifacts.download_artifacts(
            run_id=entry.run_id,
            artifact_path="model",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to download artifacts: {exc}")

    from dashboard.utils.export import create_model_archive

    archive_bytes = create_model_archive(Path(local_dir))
    if archive_bytes is None:
        raise HTTPException(status_code=500, detail="Failed to create archive")

    filename = f"{entry.model_name}_{entry.model_id[:8]}.zip"
    return Response(
        content=archive_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
