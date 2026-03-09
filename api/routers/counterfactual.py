"""Counterfactual API router.

PhysCF gradient-based, Optuna black-box, COMET baseline generation.
SSE streaming for long-running tasks. IPS reference and Pastas validation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from api.config import settings
from api.serializers import clean_nans
from api.task_manager import TaskStatus, task_manager
from api.schemas.counterfactual import (
    CFGenerateRequest,
    CFResult,
    IPSReferenceRequest,
    PastasValidateRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/counterfactual", tags=["counterfactual"])


def _run_cf_thread(task_id: str, method: str, req: CFGenerateRequest) -> None:
    """Background thread for counterfactual generation."""
    task = task_manager.get(task_id)
    if task is None:
        return

    with task.lock:
        task.status = TaskStatus.RUNNING

    metrics_file = Path(settings.results_dir) / f"cf_metrics_{task_id}.json"
    task.metrics_file = str(metrics_file)

    try:
        import torch
        from dashboard.utils.model_registry import ModelRegistry

        registry = ModelRegistry(checkpoints_dir=Path(settings.checkpoints_dir))
        entry = registry.get_model(req.model_id)
        if entry is None:
            raise FileNotFoundError(f"Model not found: {req.model_id}")

        model = registry.load_model(entry)
        scalers = registry.load_scalers(entry)
        config = registry.load_model_config(entry)

        # Load IPS reference from model artifacts
        ips_ref = {}
        try:
            import mlflow

            local_path = mlflow.artifacts.download_artifacts(
                run_id=entry.run_id,
                artifact_path="model/ips_reference.json",
            )
            with open(local_path, "r", encoding="utf-8") as f:
                ips_ref = json.load(f)
        except Exception as exc:
            logger.warning("Could not load IPS reference: %s", exc)

        # Prepare model adapter for CF generation
        from dashboard.utils.counterfactual.darts_adapter import DartsModelAdapter

        # Get data for CF context
        train_df = registry.load_data(entry, "train")
        test_df = registry.load_data(entry, "test")
        if train_df is None or test_df is None:
            raise FileNotFoundError("Model data splits not found")

        columns = config.get("columns", {})
        target_col = columns.get("target", "")

        # Build adapter and tensors (simplified -- actual implementation
        # depends on specific data preparation in the CF page)
        # Here we provide the dispatch to the correct CF function

        if method == "physcf":
            from dashboard.utils.counterfactual.physcf_optim import generate_counterfactual

            # NOTE: Actual tensor preparation requires detailed data pipeline.
            # This is a dispatch stub -- the full tensor prep is complex and
            # will be refined when the CF page is migrated.
            result = {"method": "physcf", "status": "stub", "message": "Full CF pipeline requires tensor preparation"}

        elif method == "optuna":
            from dashboard.utils.counterfactual.optuna_optim import generate_counterfactual_optuna

            result = {"method": "optuna", "status": "stub", "message": "Full CF pipeline requires tensor preparation"}

        elif method == "comet":
            from dashboard.utils.counterfactual.comet_hydro import generate_counterfactual_comet

            result = {"method": "comet", "status": "stub", "message": "Full CF pipeline requires tensor preparation"}

        else:
            raise ValueError(f"Unknown CF method: {method}")

        with task.lock:
            task.status = TaskStatus.COMPLETED
            task.result = clean_nans(result)

    except Exception as exc:
        logger.exception("CF task %s failed", task_id)
        with task.lock:
            task.status = TaskStatus.FAILED
            task.error = str(exc)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@router.post("/generate", response_model=CFResult, status_code=202)
async def generate_physcf(req: CFGenerateRequest):
    """Start PhysCF gradient-based counterfactual generation (background task)."""
    task = task_manager.create(task_type="counterfactual", config={"method": "physcf", **req.model_dump()})
    thread = threading.Thread(
        target=_run_cf_thread, args=(task.task_id, "physcf", req), daemon=True, name=f"cf-physcf-{task.task_id}"
    )
    task.thread = thread
    thread.start()
    return CFResult(task_id=task.task_id, status=task.status.value)


@router.post("/generate-optuna", response_model=CFResult, status_code=202)
async def generate_optuna(req: CFGenerateRequest):
    """Start Optuna black-box counterfactual generation (background task)."""
    task = task_manager.create(task_type="counterfactual", config={"method": "optuna", **req.model_dump()})
    thread = threading.Thread(
        target=_run_cf_thread, args=(task.task_id, "optuna", req), daemon=True, name=f"cf-optuna-{task.task_id}"
    )
    task.thread = thread
    thread.start()
    return CFResult(task_id=task.task_id, status=task.status.value)


@router.post("/generate-comet", response_model=CFResult, status_code=202)
async def generate_comet(req: CFGenerateRequest):
    """Start COMET-Hydro counterfactual generation (background task)."""
    task = task_manager.create(task_type="counterfactual", config={"method": "comet", **req.model_dump()})
    thread = threading.Thread(
        target=_run_cf_thread, args=(task.task_id, "comet", req), daemon=True, name=f"cf-comet-{task.task_id}"
    )
    task.thread = thread
    thread.start()
    return CFResult(task_id=task.task_id, status=task.status.value)


@router.get("/{task_id}/stream")
async def stream_cf_progress(task_id: str):
    """SSE stream for counterfactual generation progress."""
    from sse_starlette.sse import EventSourceResponse

    task = task_manager.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_generator():
        metrics_file = Path(task.metrics_file) if task.metrics_file else None
        terminal_states = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}

        while True:
            current_status = task.status

            if metrics_file and metrics_file.exists():
                try:
                    with open(metrics_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    yield {"event": "progress", "data": json.dumps(clean_nans(data))}
                except (json.JSONDecodeError, OSError):
                    pass

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


@router.get("/ips-reference")
async def ips_reference(
    model_id: str = Query(...),
    window: int = Query(1, ge=1, le=12),
    aquifer_type: str = Query(None),
):
    """Get IPS reference statistics for a trained model."""
    from dashboard.utils.model_registry import ModelRegistry

    registry = ModelRegistry(checkpoints_dir=Path(settings.checkpoints_dir))
    entry = registry.get_model(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Model not found")

    # Try to load pre-computed IPS reference from artifacts
    try:
        import mlflow

        local_path = mlflow.artifacts.download_artifacts(
            run_id=entry.run_id,
            artifact_path="model/ips_reference.json",
        )
        with open(local_path, "r", encoding="utf-8") as f:
            ips_data = json.load(f)
    except Exception:
        raise HTTPException(status_code=404, detail="IPS reference not found for this model")

    # Return requested window
    all_refs = ips_data.get("ref_stats_all", {})
    ref_for_window = all_refs.get(str(window), ips_data.get("ref_stats", {}))

    result = {
        "model_id": model_id,
        "window": window,
        "ref_stats": ref_for_window,
        "mu_target": ips_data.get("mu_target"),
        "sigma_target": ips_data.get("sigma_target"),
        "n_years": ips_data.get("n_years"),
        "validation": ips_data.get("validation"),
    }

    if aquifer_type:
        from dashboard.utils.counterfactual.ips import get_aquifer_ips_info

        result["aquifer_info"] = get_aquifer_ips_info(aquifer_type)

    return clean_nans(result)


@router.post("/pastas-validate")
async def pastas_validate(req: PastasValidateRequest):
    """Run Pastas dual validation on a counterfactual result."""
    from dashboard.utils.model_registry import ModelRegistry

    registry = ModelRegistry(checkpoints_dir=Path(settings.checkpoints_dir))
    entry = registry.get_model(req.model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Model not found")

    # Get CF task result
    cf_task = task_manager.get(req.cf_task_id)
    if cf_task is None:
        raise HTTPException(status_code=404, detail="Counterfactual task not found")
    if cf_task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"CF task status: {cf_task.status.value}, must be COMPLETED")
    if cf_task.result is None:
        raise HTTPException(status_code=400, detail="CF task has no result")

    # NOTE: Full Pastas validation requires the PastasWrapper to be fitted
    # and CF stress tensors to be available. This is a dispatch stub that
    # validates the pipeline is reachable.
    try:
        from dashboard.utils.counterfactual.pastas_validation import PastasWrapper, PASTAS_AVAILABLE

        if not PASTAS_AVAILABLE:
            raise HTTPException(status_code=501, detail="Pastas is not installed")

        return clean_nans({
            "model_id": req.model_id,
            "cf_task_id": req.cf_task_id,
            "gamma": req.gamma,
            "status": "stub",
            "message": "Full Pastas validation requires fitted PastasWrapper and CF stress data",
        })

    except ImportError as exc:
        raise HTTPException(status_code=501, detail=f"Pastas not available: {exc}")
