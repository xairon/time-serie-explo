"""Training API router.

Start training in background, SSE stream of metrics, cancel, and history.
"""

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
from api.schemas.training import TrainingRequest, TrainingResult, TrainingStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/training", tags=["training"])


def _run_training_thread(task_id: str, req: TrainingRequest) -> None:
    """Background thread that runs the training pipeline."""
    task = task_manager.get(task_id)
    if task is None:
        return

    with task.lock:
        task.status = TaskStatus.RUNNING

    metrics_file = Path(settings.results_dir) / f"metrics_{task_id}.json"
    task.metrics_file = str(metrics_file)

    try:
        # Lazy imports to avoid loading torch/darts at startup
        from dashboard.utils.dataset_registry import DatasetRegistry
        from dashboard.utils.preprocessing import prepare_dataframe_for_darts
        from dashboard.utils.training import run_training_pipeline

        # Load dataset
        datasets_dir = Path(settings.data_dir) / "prepared"
        registry = DatasetRegistry(datasets_dir)
        datasets = registry.scan_datasets()

        ds = None
        for d in datasets:
            if d.path.name == req.dataset_id:
                ds = d
                break

        if ds is None:
            raise FileNotFoundError(f"Dataset not found: {req.dataset_id}")

        df, config = registry.load_dataset(ds)

        target_col = ds.target_column
        cov_cols = ds.covariate_columns if req.use_covariates else None

        # Prepare Darts series
        fill_method = ds.preprocessing.get("fill_method", "Interpolation linéaire")
        full_series, covariates = prepare_dataframe_for_darts(
            df, target_col=target_col, covariate_cols=cov_cols, freq="D", fill_method=fill_method
        )

        # Split
        n = len(full_series)
        n_train = int(n * req.train_ratio)
        n_val = int(n * req.val_ratio)

        train = full_series[:n_train]
        val = full_series[n_train : n_train + n_val]
        test = full_series[n_train + n_val :]

        train_cov = covariates[:n_train] if covariates else None
        val_cov = covariates[n_train : n_train + n_val] if covariates else None
        test_cov = covariates[n_train + n_val :] if covariates else None
        full_cov = covariates if covariates else None

        # Check cancellation before training
        if task.stop_event.is_set():
            return

        hyperparams = req.hyperparams.copy()
        if req.n_epochs is not None:
            hyperparams.setdefault("n_epochs", req.n_epochs)

        results = run_training_pipeline(
            model_name=req.model_name,
            hyperparams=hyperparams,
            train=train,
            val=val,
            test=test,
            train_cov=train_cov,
            val_cov=val_cov,
            test_cov=test_cov,
            full_cov=full_cov,
            use_covariates=req.use_covariates,
            station_name=req.station_name,
            verbose=False,
            early_stopping_patience=req.early_stopping_patience,
            metrics_file=metrics_file,
            n_epochs=hyperparams.get("n_epochs"),
            dataset_name=req.dataset_id,
            column_mapping={
                "target_var": target_col,
                "covariate_vars": cov_cols or [],
            },
        )

        with task.lock:
            task.status = TaskStatus.COMPLETED
            task.result = clean_nans({
                "metrics": results.get("metrics"),
                "metrics_sliding": results.get("metrics_sliding"),
                "model_name": results.get("model_name"),
                "station": results.get("station"),
            })

        # Mark metrics file as completed
        if metrics_file.exists():
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["status"] = "completed"
                with open(metrics_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception:
                pass

    except Exception as exc:
        logger.exception("Training task %s failed", task_id)
        with task.lock:
            task.status = TaskStatus.FAILED
            task.error = str(exc)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@router.post("/start", response_model=TrainingStatus, status_code=202)
async def start_training(req: TrainingRequest):
    """Start a training job in a background thread. Returns task_id."""
    task = task_manager.create(task_type="training", config=req.model_dump())

    thread = threading.Thread(
        target=_run_training_thread,
        args=(task.task_id, req),
        daemon=True,
        name=f"training-{task.task_id}",
    )
    task.thread = thread
    thread.start()

    return TrainingStatus(
        task_id=task.task_id,
        status=task.status.value,
        task_type=task.task_type,
        config=task.config,
        created_at=task.created_at,
    )


@router.get("/{task_id}/stream")
async def stream_training_metrics(task_id: str):
    """SSE stream of training metrics (reads MetricsFileCallback JSON)."""
    from sse_starlette.sse import EventSourceResponse

    task = task_manager.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_generator():
        metrics_file = Path(task.metrics_file) if task.metrics_file else None
        last_epoch = -1
        terminal_states = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}

        while True:
            # Check task state
            current_status = task.status

            if metrics_file and metrics_file.exists():
                try:
                    with open(metrics_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    current_epoch = data.get("current_epoch", 0)
                    if current_epoch > last_epoch:
                        last_epoch = current_epoch
                        yield {"event": "metrics", "data": json.dumps(clean_nans(data))}
                except (json.JSONDecodeError, OSError):
                    pass

            if current_status in terminal_states:
                # Final event
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


@router.post("/{task_id}/cancel")
async def cancel_training(task_id: str):
    """Cancel a running training task."""
    if not task_manager.cancel(task_id):
        raise HTTPException(status_code=404, detail="Task not found or already finished")
    return {"status": "cancelled", "task_id": task_id}


@router.get("/history", response_model=list[TrainingStatus])
async def training_history():
    """List all training tasks."""
    tasks = task_manager.list_tasks(task_type="training")
    return [
        TrainingStatus(
            task_id=t.task_id,
            status=t.status.value,
            task_type=t.task_type,
            config=t.config,
            error=t.error,
            created_at=t.created_at,
        )
        for t in tasks
    ]
