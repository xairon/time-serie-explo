"""Training API router.

Start training in background, SSE stream of metrics, cancel, and history.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from api.auth.deps import get_current_user
from api.config import settings
from api.models_db import User
from api.serializers import clean_nans
from api.task_manager import TaskStatus, task_manager
from api.schemas.training import PresetInfo, TrainingRequest, TrainingResult, TrainingStatus
from dashboard.utils.training_presets import PRESETS, apply_preset_to_request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/training", tags=["training"])


def _run_training_thread(task_id: str, req: TrainingRequest, owner_id: str | None = None) -> None:
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
        fill_method = ds.preprocessing.get("fill_method", "Linear interpolation")

        from dashboard.utils.model_factory import ModelFactory
        is_torch = ModelFactory.is_torch_model(req.model_name)
        preprocessing_config = ds.preprocessing if ds.preprocessing else {}
        normalization = preprocessing_config.get("normalization", "Standardization (z-score)")
        do_norm = is_torch and normalization and normalization != "None"

        # Multi-station (global) datasets store one row per (station, date) -> the date
        # index has duplicates. Build ONE series per station, split + scale each one
        # independently, and pass lists to the (global-aware) training pipeline.
        station_col = getattr(ds, "station_column", None)
        # Global multi-station path only when there are genuinely several stations.
        # A single-station dataset that merely carries a station id must use the
        # single-series path, otherwise its covariates would be silently dropped.
        is_multi = (
            bool(station_col)
            and station_col in df.columns
            and df[station_col].nunique() > 1
        )

        target_preprocessor = None
        cov_preprocessor = None
        all_stations = None
        train_cov = val_cov = test_cov = full_cov = None

        if is_multi:
            from dashboard.utils.preprocessing import TimeSeriesPreprocessor
            ic = int(req.hyperparams.get("input_chunk_length", 30))
            oc = int(req.hyperparams.get("output_chunk_length", 7))
            min_len = ic + oc + 1

            train, val, test, all_stations = [], [], [], []
            target_preprocessor = {} if do_norm else None
            skipped = 0
            # Global multi-station: univariate target per station (calendar via add_encoders).
            for stn, sub in df.groupby(station_col, sort=False):
                sub = sub.drop(columns=[station_col])
                series, _ = prepare_dataframe_for_darts(
                    sub, target_col=target_col, covariate_cols=None, freq="D", fill_method=fill_method
                )
                n = len(series)
                n_tr = int(n * req.train_ratio)
                n_v = int(n * req.val_ratio)
                tr = series[:n_tr]; v = series[n_tr:n_tr + n_v]; te = series[n_tr + n_v:]
                # Each split must hold at least one (input+output) window for a torch model.
                if is_torch and min(len(tr), len(v), len(te)) < min_len:
                    skipped += 1
                    continue
                if do_norm:
                    pp = TimeSeriesPreprocessor({"normalization": normalization, "fill_method": "None"})
                    tr = pp.fit_transform(tr); v = pp.transform(v); te = pp.transform(te)
                    target_preprocessor[str(stn)] = pp
                train.append(tr); val.append(v); test.append(te); all_stations.append(str(stn))
            if not train:
                raise ValueError("Aucune station n'a assez de données pour ce horizon (input+output).")
            logger.info(f"Global training: {len(all_stations)} stations retenues, {skipped} écartées (trop courtes).")
        else:
            full_series, covariates = prepare_dataframe_for_darts(
                df, target_col=target_col, covariate_cols=cov_cols, freq="D", fill_method=fill_method
            )
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
        # Top-level fields take priority over hyperparams dict
        if req.n_epochs is not None:
            hyperparams["n_epochs"] = req.n_epochs
        if req.loss_function:
            hyperparams["loss_fn"] = req.loss_function

        # Handle early stopping (only for torch models)
        es_patience = req.early_stopping_patience if (req.early_stopping and is_torch) else None

        # Normalize data for DL models — single-series path (multi-station handled above)
        if not is_multi and do_norm:
            from dashboard.utils.preprocessing import TimeSeriesPreprocessor
            preproc_cfg = {"normalization": normalization, "fill_method": "None"}
            target_preprocessor = TimeSeriesPreprocessor(preproc_cfg)
            train = target_preprocessor.fit_transform(train)
            val = target_preprocessor.transform(val)
            test = target_preprocessor.transform(test)
            if train_cov is not None:
                cov_preprocessor = TimeSeriesPreprocessor(preproc_cfg)
                train_cov = cov_preprocessor.fit_transform(train_cov)
                val_cov = cov_preprocessor.transform(val_cov) if val_cov else None
                test_cov = cov_preprocessor.transform(test_cov) if test_cov else None
                full_cov = cov_preprocessor.transform(full_cov) if full_cov else None

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
            station_name=req.station_name or (getattr(ds, "stations", None) or ["default"])[0],
            verbose=False,
            early_stopping_patience=es_patience,
            metrics_file=metrics_file,
            n_epochs=hyperparams.get("n_epochs"),
            dataset_name=req.dataset_id,
            target_preprocessor=target_preprocessor,
            cov_preprocessor=cov_preprocessor,
            all_stations=all_stations,
            preprocessing_config=preprocessing_config,
            column_mapping={
                "target_var": target_col,
                "covariate_vars": cov_cols or [],
            },
            owner_id=owner_id,
            stop_event=task.stop_event,
        )

        # If the task was cancelled while the fit was running, honour that terminal
        # state instead of overwriting it with COMPLETED (the fit is stopped early).
        if task.stop_event.is_set():
            with task.lock:
                task.status = TaskStatus.CANCELLED
            return

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
        # A cancellation that surfaces as an exception (e.g. the fit aborted mid-step)
        # is a CANCELLED task, not a FAILED one.
        if task.stop_event.is_set():
            logger.info("Training task %s cancelled during execution", task_id)
            with task.lock:
                task.status = TaskStatus.CANCELLED
        else:
            logger.exception("Training task %s failed", task_id)
            with task.lock:
                task.status = TaskStatus.FAILED
                task.error = str(exc)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@router.get("/presets", response_model=list[PresetInfo])
def list_presets():
    """Return the catalogue of opinionated training presets."""
    return [
        PresetInfo(
            id=p["id"],
            label=p["label"],
            description=p["description"],
            target_domain=p["target_domain"],
            model_name=p["model_name"],
            horizon_days=p["horizon_days"],
            input_chunk_length=p["hyperparams"].get("input_chunk_length"),
            n_epochs=p["n_epochs"],
        )
        for p in PRESETS
    ]


@router.post("/start", response_model=TrainingStatus, status_code=202)
async def start_training(req: TrainingRequest, current: User = Depends(get_current_user)):
    """Start a training job in a background thread. Returns task_id."""
    from api.auth.ownership import assert_owns_dataset
    assert_owns_dataset(current, req.dataset_id)

    from api.task_manager import TaskStatus

    # A task counts as "active" (holding the single shared GPU) if it is still
    # PENDING/RUNNING, OR if it was just cancelled but its worker thread is still
    # winding down the fit — starting a second job then would put two jobs on the GPU.
    active = [
        t for t in task_manager.list_tasks(task_type="training")
        if t.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
        or (t.thread is not None and t.thread.is_alive())
    ]
    if active:
        from api.auth.ownership import is_owner_or_admin
        # Single shared GPU: only one training at a time. Avoid disclosing another
        # user's task id to callers who do not own the running task.
        if is_owner_or_admin(current, active[0].owner_id):
            detail = (
                f"Un entraînement est déjà en cours (tâche {active[0].task_id}). "
                "Attendez sa fin ou annulez-le avant d'en démarrer un autre."
            )
        else:
            detail = "Un entraînement est déjà en cours sur le serveur. Réessayez plus tard."
        raise HTTPException(status_code=409, detail=detail)

    # Apply preset if requested. Fills model_name + hyperparams + n_epochs
    # unless the user already supplied them on the request.
    if req.preset_id:
        try:
            merged = apply_preset_to_request(req.model_dump(), req.preset_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        req = TrainingRequest(**merged)

    if not req.model_name:
        raise HTTPException(
            400, "model_name est requis (fournir directement ou via preset_id)"
        )

    task = task_manager.create(
        task_type="training", config=req.model_dump(), owner_id=str(current.id)
    )

    thread = threading.Thread(
        target=_run_training_thread,
        args=(task.task_id, req, str(current.id)),
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
async def stream_training_metrics(
    task_id: str, current: User = Depends(get_current_user)
):
    """SSE stream of training metrics (reads MetricsFileCallback JSON)."""
    from sse_starlette.sse import EventSourceResponse
    from api.auth.ownership import assert_owner_or_admin

    task = task_manager.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Tâche introuvable")
    assert_owner_or_admin(current, task.owner_id)

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
                        # Map callback format to frontend format
                        train_losses = data.get("train_losses", [])
                        val_losses = data.get("val_losses", [])
                        best_val = min((v for v in val_losses if v is not None), default=None)
                        if best_val is None:
                            best_val = min((v for v in train_losses if v is not None), default=None)
                        payload = {
                            "current_epoch": current_epoch,
                            "total_epochs": data.get("total_epochs", 0),
                            "train_loss": train_losses[-1] if train_losses else None,
                            "val_loss": val_losses[-1] if val_losses else None,
                            "best_val_loss": best_val,
                            "train_losses": train_losses,
                            "val_losses": val_losses,
                            "status": data.get("status"),
                            "elapsed_seconds": data.get("elapsed_seconds"),
                            "eta_seconds": data.get("eta_seconds"),
                        }
                        yield {"event": "metrics", "data": json.dumps(clean_nans(payload))}
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
async def cancel_training(task_id: str, current: User = Depends(get_current_user)):
    """Cancel a running training task."""
    from api.auth.ownership import assert_owner_or_admin

    task = task_manager.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Tâche introuvable ou déjà terminée")
    assert_owner_or_admin(current, task.owner_id)
    if not task_manager.cancel(task_id):
        raise HTTPException(status_code=404, detail="Tâche introuvable ou déjà terminée")
    return {"status": "cancelled", "task_id": task_id}


@router.get("/{task_id}/status", response_model=TrainingResult)
async def training_status(task_id: str, current: User = Depends(get_current_user)):
    """Get the current status and result of a training task."""
    from api.auth.ownership import assert_owner_or_admin

    task = task_manager.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Tâche introuvable")
    assert_owner_or_admin(current, task.owner_id)

    result = TrainingResult(
        task_id=task.task_id,
        status=task.status.value,
        error=task.error,
    )

    if task.result:
        result.metrics = task.result.get("metrics")
        result.metrics_sliding = task.result.get("metrics_sliding")
        result.model_name = task.result.get("model_name")
        result.station = task.result.get("station")

    return result


@router.get("/history", response_model=list[TrainingStatus])
async def training_history(current: User = Depends(get_current_user)):
    """List the caller's training tasks (admins see all)."""
    from api.auth.ownership import is_owner_or_admin

    tasks = [
        t
        for t in task_manager.list_tasks(task_type="training")
        if is_owner_or_admin(current, t.owner_id)
    ]
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
