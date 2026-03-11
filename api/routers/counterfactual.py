"""Counterfactual API router.

PhysCF gradient-based, Optuna black-box, CoMTE feature-swapping generation.
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

        # Build full DataFrame — covariate_cols is UPDATED to actual available column order
        df_full, covariate_cols = build_complete_dataframe(data_dict, target_col, covariate_cols)
        if df_full is None or len(df_full) == 0:
            raise ValueError("Could not reconstruct data from model artifacts")

        # G2 fix: Validate covariate columns actually exist in df_full
        missing_covs = [c for c in covariate_cols if c not in df_full.columns]
        if missing_covs:
            logger.warning("Covariates missing from df_full: %s — removing", missing_covs)
            covariate_cols = [c for c in covariate_cols if c in df_full.columns]

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
        # G3 fix: cov_params keys are indexed (cov_0, cov_1, ...) — map by position to actual column names
        for i, col in enumerate(covariate_cols):
            indexed_key = f"cov_{i}"
            if indexed_key in cov_params:
                physcf_scaler[col] = cov_params[indexed_key]
            elif col in cov_params:
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
            # G5 fix: Compute from train+val data only (no test leakage)
            train_target = data_dict["train"][target_col] if target_col in data_dict["train"].columns else None
            val_target = data_dict.get("val", pd.DataFrame()).get(target_col) if "val" in data_dict else None
            ref_parts = [s for s in [train_target, val_target] if s is not None]
            if ref_parts:
                gwl_ref_norm = pd.concat(ref_parts).sort_index()
            else:
                gwl_ref_norm = df_full[target_col]
            gwl_ref_raw = gwl_ref_norm.values * sigma_target + mu_target
            gwl_series = pd.Series(gwl_ref_raw, index=gwl_ref_norm.index)
            ref_stats = compute_ips_reference_n(gwl_series, window=1, aggregate_to_monthly=True)

        # Resolve IPS class name (handles aliases like "bas" -> "low")
        ips_name_map = {
            "very_low": "very_low", "tres_bas": "very_low",
            "low": "low", "bas": "low",
            "moderately_low": "moderately_low", "moderement_bas": "moderately_low",
            "normal": "normal",
            "moderately_high": "moderately_high", "moderement_haut": "moderately_high",
            "high": "high", "haut": "high",
            "very_high": "very_high", "tres_haut": "very_high",
        }

        def _resolve_ips_class(name: str) -> str:
            if name in IPS_CLASSES:
                return name
            return ips_name_map.get(name, "normal")

        # Default IPS class (used when target_ips_classes has no entry for a month)
        default_ips_key = _resolve_ips_class(req.target_ips_class)

        # Per-month overrides: target_ips_classes takes priority over target_ips_class
        per_month_ips: dict[int, str] = {}
        for month_str, cls_name in req.target_ips_classes.items():
            per_month_ips[int(month_str)] = _resolve_ips_class(cls_name)

        # Compute target bounds per timestep using per-month IPS classes
        horizon_months = horizon_df.index.month.values if len(horizon_df) >= H_model else np.full(H_model, 6)
        lower_norm_arr = np.zeros(H_model, dtype=np.float32)
        upper_norm_arr = np.zeros(H_model, dtype=np.float32)

        for t in range(min(H_model, len(horizon_months))):
            m = int(horizon_months[t])
            ips_key = per_month_ips.get(m, default_ips_key)
            z_min, z_max = IPS_CLASSES[ips_key]
            z_min_c = max(z_min, -5.0)
            z_max_c = min(z_max, 5.0)
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

        # Compute factual prediction ONCE (shared across all methods)
        # Use s_obs_norm (the authoritative normalized stresses from the data pipeline)
        # to ensure identical baselines regardless of method.
        dates_list = [d.strftime("%Y-%m-%d") for d in horizon_dates[:H_model]]
        with torch.no_grad():
            s_norm_ref = torch.tensor(s_obs_norm, dtype=torch.float32)
            adapter_dev = adapter.to(device)
            y_orig_shared = adapter_dev(
                h_t.to(device),
                s_norm_ref.to(device),
            ).squeeze(0).detach().cpu().numpy().flatten()
            original_raw_shared = clean_nans((y_orig_shared * sigma_target + mu_target).tolist())

        # Helper to build result dict from raw CF output
        def _build_result(cf_result: dict, method_name: str) -> dict:
            """Convert raw CF function output (tensors) to serializable result."""
            y_cf = cf_result.get("y_cf")
            theta_star = cf_result.get("theta_star")

            # Denormalize predictions to physical units
            counterfactual_raw = []
            if y_cf is not None:
                y_cf_np = y_cf.detach().cpu().numpy().flatten()
                counterfactual_raw = (y_cf_np * sigma_target + mu_target).tolist()

            res = {
                "method": method_name,
                "status": "completed",
                "original": original_raw_shared,
                "counterfactual": clean_nans(counterfactual_raw),
                "dates": dates_list,
                "theta": clean_nans(theta_star) if theta_star else {},
                "metrics": clean_nans({
                    "converged": cf_result.get("converged", False),
                    "wall_clock_s": cf_result.get("wall_clock_s", 0),
                    "n_params": cf_result.get("n_params", 0),
                    "n_iter": cf_result.get("n_iter", 0),
                    "best_loss": cf_result.get("best_loss"),
                    # Final values of each loss component
                    "target_loss_final": cf_result["target_history"][-1] if cf_result.get("target_history") else None,
                    "prox_loss_final": cf_result["prox_history"][-1] if cf_result.get("prox_history") else None,
                    "smooth_loss_final": cf_result["smooth_history"][-1] if cf_result.get("smooth_history") else None,
                }),
                "convergence": clean_nans(cf_result.get("loss_history", [])),
                "target_history": clean_nans(cf_result.get("target_history", [])),
                "prox_history": clean_nans(cf_result.get("prox_history", [])),
                "smooth_history": clean_nans(cf_result.get("smooth_history", [])),
            }
            if cf_result.get("n_trials") is not None:
                res["best_trial"] = {"n_trials": cf_result["n_trials"], "best_loss": cf_result.get("best_loss")}

            # Save CF stresses and lookback dates for Pastas dual validation
            res["s_cf_phys"] = (
                cf_result["s_cf_phys"].detach().cpu().numpy().tolist()
                if "s_cf_phys" in cf_result and cf_result["s_cf_phys"] is not None
                else None
            )
            res["lookback_dates"] = (
                lookback_df.index.strftime("%Y-%m-%d").tolist()
                if lookback_df is not None
                else None
            )

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

        elif method == "comte":
            from dashboard.utils.counterfactual.comte import generate_counterfactual_comte

            write_progress(0, 1, "Running CoMTE (feature swapping)")

            # G1 fix: Build train+val only DataFrame for distractor pools (no test leakage)
            train_val_dict = {k: v for k, v in data_dict.items() if "test" not in k}
            df_train_val, _ = build_complete_dataframe(train_val_dict, target_col, covariate_cols)
            if df_train_val is None or len(df_train_val) == 0:
                df_train_val = df_full.loc[df_full.index < test_df.index[0]]

            # C2 fix: Derive dominant target class from per-month overrides
            from collections import Counter
            all_target_classes = list(per_month_ips.values()) if per_month_ips else [default_ips_key]
            dominant_target_class = Counter(all_target_classes).most_common(1)[0][0] if all_target_classes else default_ips_key

            cf_result = generate_counterfactual_comte(
                h_obs=h_t,
                s_obs_norm=torch.tensor(s_obs_norm, dtype=torch.float32),
                model=adapter,
                target_bounds=target_bounds,
                df_full=df_train_val,
                target_col=target_col,
                covariate_cols=covariate_cols,
                mu_target=mu_target,
                sigma_target=sigma_target,
                ref_stats=ref_stats,
                L=L_model,
                H=H_model,
                scaler=physcf_scaler,
                target_ips_class=dominant_target_class,
                num_distractors=req.num_distractors,
                tau=req.tau,
                device=device,
            )

            result = _build_result(cf_result, "comte")
            # Attach CoMTE-specific info
            if "comte_info" in cf_result:
                result["comte_info"] = cf_result["comte_info"]

        else:
            raise ValueError(f"Unknown CF method: {method}")

        # Attach context needed for Pastas dual validation
        result["_pastas_context"] = {
            "model_id": req.model_id,
            "target_col": target_col,
            "covariate_cols": covariate_cols,
            "mu_target": mu_target,
            "sigma_target": sigma_target,
            "physcf_scaler": physcf_scaler,
            "horizon_start": str(horizon_dates[0].date()) if len(horizon_dates) > 0 else None,
            "horizon_end": str(horizon_dates[-1].date()) if len(horizon_dates) > 0 else None,
        }

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
    # Alias: "comet" → "comte" (backwards compat)
    if method == "comet":
        method = "comte"
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


@router.post("/generate-comte", response_model=CFResult, status_code=202)
async def generate_comte(req: CFGenerateRequest):
    """Start CoMTE feature-swapping counterfactual generation (Ates et al. 2021)."""
    task = task_manager.create(task_type="counterfactual", config={"method": "comte", **req.model_dump()})
    thread = threading.Thread(
        target=_run_cf_thread, args=(task.task_id, "comte", req), daemon=True, name=f"cf-comte-{task.task_id}"
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
    """Run Pastas dual validation on a counterfactual result.

    Fits a Pastas TFN model on the same data used for CF generation,
    then checks TFT-Pastas agreement on the counterfactual stresses.
    CPU-bound Pastas fitting is offloaded to a thread via asyncio.to_thread().
    """
    from dashboard.utils.model_registry import ModelRegistry

    try:
        from dashboard.utils.counterfactual.pastas_validation import (
            PASTAS_AVAILABLE,
            build_pastas_series_from_data,
            cf_stresses_to_pastas_series,
            fit_pastas_for_station,
            validate_with_pastas,
        )
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=f"Pastas not available: {exc}")

    if not PASTAS_AVAILABLE:
        raise HTTPException(status_code=501, detail="Pastas is not installed")

    # --- Load CF task result ---
    cf_task = task_manager.get(req.cf_task_id)
    if cf_task is None:
        raise HTTPException(status_code=404, detail="Counterfactual task not found")
    if cf_task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"CF task status: {cf_task.status.value}, must be COMPLETED")
    if cf_task.result is None:
        raise HTTPException(status_code=400, detail="CF task has no result")

    cf_result = cf_task.result

    # Validate required fields from CF result
    s_cf_phys_list = cf_result.get("s_cf_phys")
    if s_cf_phys_list is None:
        raise HTTPException(
            status_code=400,
            detail="CF task result does not contain s_cf_phys. "
            "Re-run the CF generation to include stress data.",
        )

    lookback_dates_str = cf_result.get("lookback_dates")
    if lookback_dates_str is None:
        raise HTTPException(
            status_code=400,
            detail="CF task result does not contain lookback_dates.",
        )

    pastas_ctx = cf_result.get("_pastas_context", {})

    # --- Load model and data (same pattern as _run_cf_thread) ---
    registry = ModelRegistry(checkpoints_dir=Path(settings.checkpoints_dir))
    model_id = pastas_ctx.get("model_id", req.model_id)
    entry = registry.get_model(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Model not found")

    def _run_pastas_validation() -> dict:
        """CPU-bound Pastas fitting and validation (runs in thread)."""
        import numpy as np
        import pandas as pd

        from dashboard.utils.counterfactual.ips import extract_scaler_params
        from dashboard.utils.preprocessing import detect_columns_from_config

        scalers = registry.load_scalers(entry)
        config = registry.load_model_config(entry)

        # Load data splits
        data_dict = {}
        for split in ["train", "val", "test", "train_cov", "val_cov", "test_cov"]:
            df_split = registry.load_data(entry, split)
            if df_split is not None:
                data_dict[split] = df_split

        if "train" not in data_dict:
            raise ValueError("Model training data not found")

        # Use saved context if available, otherwise re-derive
        target_col = pastas_ctx.get("target_col")
        covariate_cols = pastas_ctx.get("covariate_cols")
        mu_target = pastas_ctx.get("mu_target")
        sigma_target = pastas_ctx.get("sigma_target")
        physcf_scaler = pastas_ctx.get("physcf_scaler")

        if target_col is None or covariate_cols is None:
            target_col, covariate_cols = detect_columns_from_config(config, data_dict)

        if mu_target is None or sigma_target is None:
            mu_t, sigma_t, _ = extract_scaler_params(scalers)
            mu_target_f = mu_t if mu_t is not None else float(data_dict["train"][target_col].mean())
            sigma_target_f = sigma_t if sigma_t is not None else float(data_dict["train"][target_col].std())
        else:
            mu_target_f = mu_target
            sigma_target_f = sigma_target

        if physcf_scaler is None:
            # Rebuild minimal scaler (needed for denormalization)
            _, _, cov_params = extract_scaler_params(scalers)
            physcf_scaler_f: dict = {target_col: {"mean": mu_target_f, "std": sigma_target_f}}
            for col in covariate_cols:
                if col in cov_params:
                    physcf_scaler_f[col] = cov_params[col]
            # Map to canonical names
            for c in covariate_cols:
                cl = c.lower()
                if ("precip" in cl or "rain" in cl) and c in physcf_scaler_f and "precip" not in physcf_scaler_f:
                    physcf_scaler_f["precip"] = physcf_scaler_f[c]
                elif "temp" in cl and c in physcf_scaler_f and "temp" not in physcf_scaler_f:
                    physcf_scaler_f["temp"] = physcf_scaler_f[c]
                elif ("evap" in cl or "etp" in cl) and c in physcf_scaler_f and "evap" not in physcf_scaler_f:
                    physcf_scaler_f["evap"] = physcf_scaler_f[c]
        else:
            physcf_scaler_f = physcf_scaler

        # Step 1: Build Pastas series from data
        gwl_series, precip_series, evap_series, train_end = build_pastas_series_from_data(
            data_dict, target_col, covariate_cols,
            mu_target_f, sigma_target_f, physcf_scaler_f,
        )

        # Step 2: Fit Pastas model
        pastas_model = fit_pastas_for_station(
            gwl_series, precip_series, evap_series, train_end,
        )
        if pastas_model is None:
            raise ValueError("Pastas model fitting failed")

        # Step 3: Convert CF stresses to Pastas series
        lookback_index = pd.DatetimeIndex(lookback_dates_str)
        s_cf_phys_arr = np.array(s_cf_phys_list, dtype=np.float32)
        cf_series = cf_stresses_to_pastas_series(s_cf_phys_arr, lookback_index)

        # Step 4: Get factual predictions for baseline RMSE
        horizon_start = pastas_ctx.get("horizon_start")
        horizon_end = pastas_ctx.get("horizon_end")

        # Factual TFT prediction (from CF result)
        y_factual_tft_raw = np.array(cf_result.get("original", []), dtype=np.float64)

        # Factual Pastas prediction
        y_factual_pastas = pastas_model.predict(tmin=horizon_start, tmax=horizon_end)
        min_factual = min(len(y_factual_tft_raw), len(y_factual_pastas))
        y_factual_tft_aligned = y_factual_tft_raw[:min_factual]
        y_factual_pastas_aligned = y_factual_pastas[:min_factual]

        # Step 5: CF TFT prediction (already denormalized in result)
        y_cf_tft_raw = np.array(cf_result.get("counterfactual", []), dtype=np.float64)

        # Step 6: Run dual validation
        validation = validate_with_pastas(
            s_cf_phys=cf_series,
            pastas_model=pastas_model,
            y_cf_tft=y_cf_tft_raw,
            y_factual_tft=y_factual_tft_aligned,
            y_factual_pastas=y_factual_pastas_aligned,
            gamma=req.gamma,
        )

        # Serialize result
        result = {
            "model_id": model_id,
            "cf_task_id": req.cf_task_id,
            "gamma": req.gamma,
            "status": "completed",
            "accepted": validation["accepted"],
            "rmse_cf": validation["rmse_cf"],
            "rmse_0": validation["rmse_0"],
            "epsilon": validation["epsilon"],
            "y_cf_pastas": (
                validation["y_cf_pastas"].tolist()
                if validation.get("y_cf_pastas") is not None
                else None
            ),
            "y_factual_pastas": y_factual_pastas_aligned.tolist(),
            "pastas_params": pastas_model.get_response_params(),
        }
        return result

    try:
        result = await asyncio.to_thread(_run_pastas_validation)
        return clean_nans(result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Pastas validation failed for CF task %s", req.cf_task_id)
        raise HTTPException(status_code=500, detail=f"Pastas validation failed: {exc}")


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
