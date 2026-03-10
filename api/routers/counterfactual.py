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
    """Background thread for counterfactual generation.

    Follows the same pipeline as the Streamlit page:
    1. Load model, scalers, config, data from registry
    2. Extract scaler params (mu_target, sigma_target, covariate params)
    3. Build PhysCF scaler dict {col: {mean, std}}
    4. Build DartsModelAdapter with input/output chunk lengths
    5. Extract lookback tensors (h_obs, s_obs_phys/s_obs_norm, months)
    6. Compute target bounds from IPS class + ref_stats
    7. Call the appropriate CF generation function with correct tensor args
    """
    task = task_manager.get(task_id)
    if task is None:
        return

    with task.lock:
        task.status = TaskStatus.RUNNING

    metrics_file = Path(settings.results_dir) / f"cf_metrics_{task_id}.json"
    task.metrics_file = str(metrics_file)

    try:
        import numpy as np
        import pandas as pd
        import torch
        from dashboard.utils.model_registry import ModelRegistry
        from dashboard.utils.counterfactual.darts_adapter import DartsModelAdapter
        from dashboard.utils.counterfactual.ips import (
            IPS_CLASSES,
            extract_scaler_params,
            compute_ips_reference_n,
        )
        from dashboard.utils.counterfactual.perturbation import PerturbationLayer
        from dashboard.utils.preprocessing import detect_columns_from_config, build_complete_dataframe

        registry = ModelRegistry(checkpoints_dir=Path(settings.checkpoints_dir))
        entry = registry.get_model(req.model_id)
        if entry is None:
            raise FileNotFoundError(f"Model not found: {req.model_id}")

        darts_model = registry.load_model(entry)
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

        # Load data splits
        data_dict = {}
        for split in ["train", "val", "test", "train_cov", "val_cov", "test_cov"]:
            df_split = registry.load_data(entry, split)
            if df_split is not None:
                data_dict[split] = df_split

        if "train" not in data_dict or "test" not in data_dict:
            raise FileNotFoundError("Model data splits (train/test) not found")

        # Detect columns from config
        target_col, covariate_cols = detect_columns_from_config(config, data_dict)

        # Build full DataFrame
        df_full, covariate_cols = build_complete_dataframe(data_dict, target_col, covariate_cols)
        if df_full is None or len(df_full) == 0:
            raise ValueError("Could not reconstruct data from model artifacts")

        # Get model dimensions
        L_model = getattr(darts_model, "input_chunk_length", 365)
        H_model = getattr(darts_model, "output_chunk_length", 90)

        # Extract real scaler parameters
        mu_target, sigma_target, cov_params = extract_scaler_params(scalers)
        if mu_target is None or sigma_target is None:
            # Fallback: compute from training data
            if target_col in data_dict["train"].columns:
                mu_target = float(data_dict["train"][target_col].mean())
                sigma_target = float(data_dict["train"][target_col].std())
            else:
                mu_target, sigma_target = 0.0, 1.0

        # Build PhysCF scaler dict {col: {mean, std}} - matching Streamlit's _build_physcf_scaler
        physcf_scaler = {}
        physcf_scaler[target_col] = {"mean": mu_target, "std": sigma_target}
        for col in covariate_cols:
            if col in cov_params:
                physcf_scaler[col] = cov_params[col]
        # Map covariate columns to canonical PerturbationLayer.STRESS_COLUMNS names
        for c in covariate_cols:
            cl = c.lower()
            if ("precip" in cl or "rain" in cl) and c in physcf_scaler and "precip" not in physcf_scaler:
                physcf_scaler["precip"] = physcf_scaler[c]
            elif "temp" in cl and c in physcf_scaler and "temp" not in physcf_scaler:
                physcf_scaler["temp"] = physcf_scaler[c]
            elif ("evap" in cl or "etp" in cl) and c in physcf_scaler and "evap" not in physcf_scaler:
                physcf_scaler["evap"] = physcf_scaler[c]

        # Use test set for CF context - pick a window in the middle of the test set
        test_df = data_dict["test"]
        test_len = len(test_df)
        valid_end = test_len - H_model
        if valid_end <= 0:
            raise ValueError(f"Test set too short ({test_len}) for horizon H={H_model}")

        if req.start_idx is not None and 0 <= req.start_idx <= valid_end:
            start_idx = req.start_idx
        else:
            start_idx = min(valid_end // 2, valid_end)

        # Compute lookback position in full df
        window_pred_start = test_df.index[start_idx]
        try:
            full_pred_loc = df_full.index.get_loc(window_pred_start)
        except KeyError:
            full_pred_loc = df_full.index.searchsorted(window_pred_start)
        if isinstance(full_pred_loc, slice):
            full_pred_loc = full_pred_loc.start

        context_start_loc = max(0, full_pred_loc - L_model)

        # Extract lookback and horizon data
        lookback_df = df_full.iloc[context_start_loc: context_start_loc + L_model]
        horizon_df = df_full.iloc[context_start_loc + L_model: context_start_loc + L_model + H_model]
        horizon_dates = horizon_df.index

        # Build tensors (matching Streamlit pipeline exactly)
        h_obs_norm = lookback_df[target_col].values.astype(np.float32)
        s_obs_norm = (
            lookback_df[covariate_cols].values.astype(np.float32)
            if covariate_cols
            else np.zeros((L_model, 1), dtype=np.float32)
        )
        months_arr = lookback_df.index.month.values.astype(np.int64)

        # Denormalize covariates to physical space
        s_obs_phys = s_obs_norm.copy()
        for j, col in enumerate(covariate_cols):
            if col in physcf_scaler:
                mu_c = physcf_scaler[col]["mean"]
                sigma_c = physcf_scaler[col]["std"]
                if sigma_c > 0:
                    s_obs_phys[:, j] = s_obs_norm[:, j] * sigma_c + mu_c

        # Load/compute IPS reference stats for target bounds
        ref_stats = {}
        all_ref_stats = {}
        if ips_ref and "ref_stats_all" in ips_ref:
            for w_str, month_dict in ips_ref["ref_stats_all"].items():
                all_ref_stats[int(w_str)] = {int(m): tuple(v) for m, v in month_dict.items()}
        elif ips_ref and "ref_stats" in ips_ref:
            all_ref_stats[1] = {int(k): tuple(v) for k, v in ips_ref["ref_stats"].items()}

        if 1 in all_ref_stats:
            ref_stats = all_ref_stats[1]
        else:
            # Compute from raw data
            gwl_all_raw = df_full[target_col].values * sigma_target + mu_target
            gwl_series = pd.Series(gwl_all_raw, index=df_full.index)
            ref_stats = compute_ips_reference_n(gwl_series, window=1, aggregate_to_monthly=True)

        # Compute target bounds from IPS class
        target_ips_key = req.target_ips_class
        if target_ips_key not in IPS_CLASSES:
            # Try mapping common names
            ips_name_map = {
                "very_low": "very_low", "tres_bas": "very_low",
                "low": "low", "bas": "low",
                "moderately_low": "moderately_low", "moderement_bas": "moderately_low",
                "normal": "normal",
                "moderately_high": "moderately_high", "moderement_haut": "moderately_high",
                "high": "high", "haut": "high",
                "very_high": "very_high", "tres_haut": "very_high",
            }
            target_ips_key = ips_name_map.get(target_ips_key, "normal")

        z_min, z_max = IPS_CLASSES[target_ips_key]
        z_min_c = max(z_min, -5.0)
        z_max_c = min(z_max, 5.0)

        horizon_months = horizon_df.index.month.values if len(horizon_df) >= H_model else np.full(H_model, 6)
        lower_norm_arr = np.zeros(H_model, dtype=np.float32)
        upper_norm_arr = np.zeros(H_model, dtype=np.float32)

        for t in range(min(H_model, len(horizon_months))):
            m = int(horizon_months[t])
            mu_m, sigma_m = ref_stats.get(m, (mu_target, sigma_target))
            lower_raw = mu_m + z_min_c * sigma_m if sigma_m > 0 else mu_m
            upper_raw = mu_m + z_max_c * sigma_m if sigma_m > 0 else mu_m
            if sigma_target > 0:
                lower_norm_arr[t] = (lower_raw - mu_target) / sigma_target
                upper_norm_arr[t] = (upper_raw - mu_target) / sigma_target
            else:
                lower_norm_arr[t], upper_norm_arr[t] = z_min_c, z_max_c

        # Convert to tensors
        h_t = torch.tensor(h_obs_norm, dtype=torch.float32)
        months_t = torch.tensor(months_arr, dtype=torch.long)
        lower_norm = torch.tensor(lower_norm_arr, dtype=torch.float32)
        upper_norm = torch.tensor(upper_norm_arr, dtype=torch.float32)
        target_bounds = (lower_norm, upper_norm)

        # Create model adapter
        adapter = DartsModelAdapter(darts_model, L_model, H_model)

        device = req.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Write progress to metrics file
        def write_progress(step: int, total: int, msg: str = ""):
            try:
                with open(metrics_file, "w", encoding="utf-8") as f:
                    json.dump({"step": step, "total": total, "message": msg, "method": method}, f)
            except Exception:
                pass

        # Helper to build result dict from raw CF output
        def _build_result(cf_result: dict, method_name: str) -> dict:
            """Convert raw CF function output (tensors) to serializable result."""
            y_cf = cf_result.get("y_cf")
            theta_star = cf_result.get("theta_star")

            # Denormalize predictions to physical units
            original_raw = []
            counterfactual_raw = []
            if y_cf is not None:
                y_cf_np = y_cf.detach().cpu().numpy().flatten()
                counterfactual_raw = (y_cf_np * sigma_target + mu_target).tolist()

            # Generate original prediction (no perturbation)
            with torch.no_grad():
                if method_name in ("physcf", "optuna"):
                    # Normalize s_obs_phys for model input
                    s_norm = torch.tensor(s_obs_phys, dtype=torch.float32).clone()
                    stress_cols = PerturbationLayer.STRESS_COLUMNS
                    for j, col in enumerate(stress_cols):
                        if col in physcf_scaler:
                            mu_c = physcf_scaler[col]["mean"]
                            sigma_c = physcf_scaler[col]["std"]
                            if sigma_c > 0:
                                s_norm[..., j] = (s_norm[..., j] - mu_c) / sigma_c
                else:
                    s_norm = torch.tensor(s_obs_norm, dtype=torch.float32)

                adapter_dev = adapter.to(device)
                y_orig = adapter_dev(
                    h_t.to(device),
                    s_norm.to(device),
                ).squeeze(0).detach().cpu().numpy().flatten()
                original_raw = (y_orig * sigma_target + mu_target).tolist()

            dates_list = [d.strftime("%Y-%m-%d") for d in horizon_dates[:H_model]]

            res = {
                "method": method_name,
                "status": "completed",
                "original": clean_nans(original_raw),
                "counterfactual": clean_nans(counterfactual_raw),
                "dates": dates_list,
                "theta": clean_nans(theta_star) if theta_star else {},
                "metrics": clean_nans({
                    "converged": cf_result.get("converged", False),
                    "wall_clock_s": cf_result.get("wall_clock_s", 0),
                    "n_params": cf_result.get("n_params", 0),
                    "n_iter": cf_result.get("n_iter", 0),
                    "best_loss": cf_result.get("best_loss"),
                }),
                "convergence": clean_nans(cf_result.get("loss_history", [])),
            }
            if cf_result.get("n_trials") is not None:
                res["best_trial"] = {"n_trials": cf_result["n_trials"], "best_loss": cf_result.get("best_loss")}
            return res

        if method == "physcf":
            from dashboard.utils.counterfactual.physcf_optim import generate_counterfactual

            write_progress(0, req.n_iter, "Starting PhysCF optimization")

            cf_result = generate_counterfactual(
                h_obs=h_t,
                s_obs_phys=torch.tensor(s_obs_phys, dtype=torch.float32),
                model=adapter,
                target_bounds=target_bounds,
                scaler=physcf_scaler,
                months=months_t,
                lambda_prox=req.lambda_prox,
                n_iter=req.n_iter,
                lr=req.lr,
                cc_rate=req.cc_rate,
                device=device,
            )

            result = _build_result(cf_result, "physcf")

        elif method == "optuna":
            from dashboard.utils.counterfactual.optuna_optim import generate_counterfactual_optuna

            write_progress(0, req.n_trials, "Starting Optuna search")

            cf_result = generate_counterfactual_optuna(
                h_obs=h_t,
                s_obs_phys=torch.tensor(s_obs_phys, dtype=torch.float32),
                model=adapter,
                target_bounds=target_bounds,
                scaler=physcf_scaler,
                months=months_t,
                lambda_prox=req.lambda_prox,
                n_trials=req.n_trials,
                cc_rate=req.cc_rate,
                device=device,
                seed=req.seed,
            )

            result = _build_result(cf_result, "optuna")

        elif method == "comet":
            from dashboard.utils.counterfactual.comet_hydro import generate_counterfactual_comet

            write_progress(0, 1, "Running COMET-Hydro")

            cf_result = generate_counterfactual_comet(
                h_obs=h_t,
                s_obs_norm=torch.tensor(s_obs_norm, dtype=torch.float32),
                model=adapter,
                target_bounds=target_bounds,
                scaler=physcf_scaler,
                k_sigma=req.k_sigma,
                lambda_smooth=req.lambda_smooth,
                n_iter=req.n_iter,
                lr=req.lr,
                device=device,
            )

            result = _build_result(cf_result, "comet")

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


@router.post("/run", response_model=CFResult, status_code=202)
async def run_counterfactual(req: CFGenerateRequest):
    """Unified counterfactual generation endpoint. Dispatches to the correct method."""
    method = req.method or "physcf"
    task = task_manager.create(task_type="counterfactual", config={"method": method, **req.model_dump()})
    thread = threading.Thread(
        target=_run_cf_thread, args=(task.task_id, method, req), daemon=True, name=f"cf-{method}-{task.task_id}"
    )
    task.thread = thread
    thread.start()
    return CFResult(task_id=task.task_id, status=task.status.value)


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


@router.get("/ips-bounds")
async def ips_bounds(
    model_id: str = Query(...),
    window: int = Query(1, ge=1, le=12),
):
    """Return monthly IPS class bounds (m NGF) for the test set date range."""
    from dashboard.utils.model_registry import ModelRegistry
    from dashboard.utils.counterfactual.ips import (
        compute_ips_reference_n,
        compute_monthly_ips_bounds,
        IPS_CLASSES,
    )
    import pandas as pd

    # Human-readable labels for IPS classes
    ips_labels = {
        "very_low": "Très bas",
        "low": "Bas",
        "moderately_low": "Modérément bas",
        "normal": "Normal",
        "moderately_high": "Modérément haut",
        "high": "Haut",
        "very_high": "Très haut",
    }
    # Colors for IPS classes (red → green spectrum)
    ips_colors = {
        "very_low": "#d73027",
        "low": "#fc8d59",
        "moderately_low": "#fee08b",
        "normal": "#ffffbf",
        "moderately_high": "#d9ef8b",
        "high": "#91cf60",
        "very_high": "#1a9850",
    }

    registry = ModelRegistry(checkpoints_dir=Path(settings.checkpoints_dir))
    entry = registry.get_model(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Model not found")

    # Load test data for date range
    test_df = registry.load_data(entry, "test")
    if test_df is None:
        raise HTTPException(status_code=404, detail="Test data not found")

    # Get IPS reference (reuse existing endpoint logic)
    ref_response = await ips_reference(model_id=model_id, window=window, aquifer_type=None)
    ref_stats_raw = ref_response.get("ref_stats", {})
    ref_stats = {int(k): tuple(v) for k, v in ref_stats_raw.items()}

    # Compute bounds
    bounds_df = compute_monthly_ips_bounds(test_df.index, ref_stats)
    if bounds_df.empty:
        return {"bounds": [], "classes": ips_labels, "colors": ips_colors}

    # Serialize
    rows = []
    for _, row in bounds_df.iterrows():
        r = {
            "month_start": row["month_start"].isoformat(),
            "month_end": row["month_end"].isoformat(),
            "month": int(row["month"]),
            "mu": float(row["mu"]),
            "sigma": float(row["sigma"]),
        }
        for cls_name in IPS_CLASSES:
            r[f"{cls_name}_lower"] = float(row[f"{cls_name}_lower"])
            r[f"{cls_name}_upper"] = float(row[f"{cls_name}_upper"])
        rows.append(r)

    return {
        "bounds": rows,
        "classes": ips_labels,
        "colors": ips_colors,
    }


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
