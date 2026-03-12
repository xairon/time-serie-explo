"""Pumping detection API router with SSE streaming."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.config import settings
from api.serializers import clean_nans
from api.task_manager import TaskStatus, task_manager
from api.schemas.pumping_detection import PumpingDetectionRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/pumping-detection", tags=["pumping-detection"])


def _run_pipeline_thread(task_id: str, req: PumpingDetectionRequest) -> None:
    """Background thread running the pumping detection pipeline."""
    task = task_manager.get(task_id)
    if task is None:
        return

    with task.lock:
        task.status = TaskStatus.RUNNING

    metrics_file = Path(settings.results_dir) / f"pd_metrics_{task_id}.json"
    task.metrics_file = str(metrics_file)

    def emit(event_type: str, data: dict):
        """Write SSE event to metrics file (overwrite with latest state)."""
        try:
            with open(metrics_file, "w") as f:
                json.dump({"event": event_type, **clean_nans(data)}, f)
        except Exception as e:
            logger.error(f"Failed to write metrics: {e}")

    try:
        import pandas as pd
        from dashboard.utils.dataset_registry import DatasetRegistry
        from dashboard.utils.pumping_detection.pipeline import PumpingDetectionPipeline

        # Load dataset (follows api/routers/datasets.py pattern)
        registry = DatasetRegistry(datasets_dir=Path(settings.data_dir) / "prepared")
        datasets = registry.scan_datasets()
        ds = next((d for d in datasets if d.path.name == req.dataset_id), None)
        if ds is None:
            raise ValueError(f"Dataset not found: {req.dataset_id}")

        df, config = registry.load_dataset(ds)
        target_col = ds.target_column or df.columns[0]
        covariate_cols = ds.covariate_columns or []
        piezo = df[target_col]
        precip = df[covariate_cols[0]] if len(covariate_cols) > 0 else pd.Series(dtype=float)
        etp = df[covariate_cols[1]] if len(covariate_cols) > 1 else pd.Series(dtype=float)

        # Run pipeline
        pipeline_config = req.config.model_dump()
        pipeline = PumpingDetectionPipeline(config=pipeline_config, emit=emit)
        result = pipeline.run(piezo, precip, etp, stop_event=task.stop_event)

        with task.lock:
            task.result = clean_nans(result)
            task.status = TaskStatus.COMPLETED

        emit("done", {"status": "completed"})

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        with task.lock:
            task.error = str(e)
            task.status = TaskStatus.FAILED
        emit("error", {"stage": "pipeline", "error_message": str(e), "recoverable": False})


@router.post("/analyze")
def start_analysis(req: PumpingDetectionRequest):
    """Start a pumping detection analysis."""
    task = task_manager.create("pumping_detection", config=req.model_dump())
    thread = threading.Thread(target=_run_pipeline_thread, args=(task.task_id, req), daemon=True)
    task.thread = thread
    thread.start()
    return {"task_id": task.task_id}


@router.get("/{task_id}/stream")
async def stream_progress(task_id: str):
    """SSE stream of analysis progress."""
    task = task_manager.get(task_id)
    if task is None:
        raise HTTPException(404, "Task not found")

    from sse_starlette.sse import EventSourceResponse

    async def event_generator():
        metrics_file = Path(task.metrics_file) if task.metrics_file else None
        terminal_states = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}

        while True:
            if metrics_file and metrics_file.exists():
                try:
                    with open(metrics_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    yield {"event": "progress", "data": json.dumps(clean_nans(data))}
                except (json.JSONDecodeError, OSError):
                    pass

            current_status = task.status
            if current_status in terminal_states:
                final = {
                    "status": current_status.value,
                    "error": task.error,
                }
                if task.result:
                    final["result"] = task.result
                yield {"event": "done", "data": json.dumps(clean_nans(final))}
                return

            await asyncio.sleep(1.0)

    return EventSourceResponse(event_generator())


@router.get("/{task_id}/results")
def get_results(task_id: str):
    """Get full results after completion."""
    task = task_manager.get(task_id)
    if task is None:
        raise HTTPException(404, "Task not found")
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(400, f"Task status: {task.status.value}")
    return task.result


@router.get("/{task_id}/layer/{layer_name}")
def get_layer_result(task_id: str, layer_name: str):
    """Get partial result for a specific layer (enables progressive rendering)."""
    task = task_manager.get(task_id)
    if task is None:
        raise HTTPException(404, "Task not found")
    valid_layers = {"pastas", "changepoints", "clean_periods", "ml_xai", "embeddings", "fusion"}
    if layer_name not in valid_layers:
        raise HTTPException(400, f"Invalid layer: {layer_name}. Valid: {valid_layers}")
    if not task.result or layer_name not in task.result:
        raise HTTPException(404, f"Layer '{layer_name}' not yet available")
    return task.result[layer_name]


@router.post("/{task_id}/cancel")
def cancel_analysis(task_id: str):
    """Cancel a running analysis."""
    if not task_manager.cancel(task_id):
        raise HTTPException(404, "Task not found or already finished")
    task = task_manager.get(task_id)
    return {"status": "cancelled", "partial_results": task.result if task else None}


@router.get("/bnpe-context")
def get_bnpe_context(lat: float, lon: float, radius_km: float = 5):
    """Fetch nearby BNPE declared pumping facilities."""
    from dashboard.utils.pumping_detection.bnpe_client import BNPEClient
    client = BNPEClient()
    return client.fetch_nearby(lat=lat, lon=lon, radius_km=radius_km)
