"""Counterfactual Analysis Page - PhysCF Integration.

Fully autonomous: loads model, data, and scalers from MLflow artifacts.
Workflow:
1. Load trained model from MLflow registry
2. Display test set with ground truth + sliding window predictions
3. Show IPS classification bands with colored levels
4. Indicate per-window whether prediction matches ground truth IPS class
5. Allow user to select target IPS class change (e.g., normal -> dry)
6. Run 1, 2 or 3 CF methods (checkboxes) and overlay results
7. Display comparative metrics tables and scenario interpretation
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Ensure MLflow tracking URI is set before any registry call
import mlflow
from dashboard.config import MLFLOW_TRACKING_URI, CHECKPOINTS_DIR
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

from dashboard.utils.model_registry import get_registry
from dashboard.utils.forecasting import generate_single_window_forecast
from dashboard.utils.preprocessing import prepare_dataframe_for_darts

# PhysCF counterfactual module
from dashboard.utils.counterfactual import (
    PerturbationLayer,
    generate_counterfactual,
    generate_counterfactual_optuna,
    generate_counterfactual_comet,
    IPS_CLASSES,
    compute_ips_reference,
    ips_class_to_gwl_bounds,
    validity_ratio,
    proximity_theta,
    cc_compliance,
    param_count,
)
from dashboard.utils.counterfactual.ips import (
    IPS_ORDER,
    gwl_to_ips_class,
    gwl_to_ips_zscore,
)
from dashboard.utils.counterfactual.darts_adapter import (
    DartsModelAdapter,
    StandaloneGRUAdapter,
)

# ---- Page Config ----
st.set_page_config(page_title="PhysCF - Counterfactual Analysis", layout="wide")
st.title("PhysCF - Analyse Contrefactuelle")

# ---- Constants ----
IPS_COLORS = {
    "very_low":       "#8B0000",
    "low":            "#DC143C",
    "moderately_low": "#FF8C00",
    "normal":         "#228B22",
    "moderately_high":"#4169E1",
    "high":           "#0000CD",
    "very_high":      "#00008B",
}

IPS_LABELS = {
    "very_low":        "Tres bas",
    "low":             "Bas",
    "moderately_low":  "Moderement bas",
    "normal":          "Normal",
    "moderately_high": "Moderement haut",
    "high":            "Haut",
    "very_high":       "Tres haut",
}

# ---- Registry ----
registry = get_registry(CHECKPOINTS_DIR.parent)
models_list = registry.list_all_models()

if not models_list:
    st.error("Aucun modele entraine dans le registry MLflow. "
             "Entrainez un modele dans l'onglet Train Models d'abord.")
    st.stop()


# ====================
# Data loading (cached)
# ====================

@st.cache_resource(show_spinner="Chargement du modele et des donnees depuis MLflow...")
def _load_model_and_data(run_id: str):
    """Load model, scalers, data, config from MLflow artifacts. Cached."""
    reg = get_registry(CHECKPOINTS_DIR.parent)
    entry = None
    for m in reg.list_all_models():
        if m.run_id == run_id:
            entry = m
            break
    if entry is None:
        raise ValueError(f"Run {run_id} introuvable")

    model = reg.load_model(entry)
    scalers = reg.load_scalers(entry)
    model_config = reg.load_model_config(entry)

    data_dict = {}
    for split in ["train", "val", "test", "train_cov", "val_cov", "test_cov"]:
        df_split = reg.load_data(entry, split)
        if df_split is not None:
            data_dict[split] = df_split

    return model, scalers, data_dict, model_config, entry


def _detect_columns(model_config, data_dict):
    """Detect target and covariate columns from model config and data."""
    target_col = None
    covariate_cols = []

    if model_config:
        if isinstance(model_config, dict):
            cols = model_config.get("columns", {})
            if not cols:
                cols = model_config.get("preprocessing", {}).get("columns", {})
        elif hasattr(model_config, "columns"):
            cols = model_config.columns if isinstance(model_config.columns, dict) else {}
        else:
            cols = {}
        target_col = cols.get("target")
        covariate_cols = cols.get("covariates", [])
        if isinstance(covariate_cols, str):
            import json as _json
            try:
                covariate_cols = _json.loads(covariate_cols)
            except Exception:
                covariate_cols = [covariate_cols]

    if not target_col and "train" in data_dict:
        df_train = data_dict["train"]
        for c in ["gwl", "Water_Level", "water_level", "niveau_nappe_eau", "piezo", "level"]:
            if c in df_train.columns:
                target_col = c
                break
        if not target_col:
            target_col = df_train.columns[0]

    if not covariate_cols and "train_cov" in data_dict:
        covariate_cols = list(data_dict["train_cov"].columns)

    return target_col, covariate_cols


def _build_full_df(data_dict, target_col, covariate_cols):
    """Merge train/val/test + covariates into a single DataFrame."""
    parts = []
    for split in ["train", "val", "test"]:
        if split in data_dict and target_col in data_dict[split].columns:
            parts.append(data_dict[split][[target_col]])
    if not parts:
        return None, []

    df_target = pd.concat(parts).sort_index()
    df_target = df_target[~df_target.index.duplicated(keep="first")]

    cov_parts = []
    for split in ["train_cov", "val_cov", "test_cov"]:
        if split in data_dict:
            cov_parts.append(data_dict[split])

    if cov_parts:
        df_cov = pd.concat(cov_parts).sort_index()
        df_cov = df_cov[~df_cov.index.duplicated(keep="first")]
        available_covs = [c for c in covariate_cols if c in df_cov.columns]
        if not available_covs:
            available_covs = list(df_cov.columns)[:3]
        df_full = df_target.join(df_cov[available_covs], how="inner")
        return df_full, available_covs
    else:
        return df_target, []


def _build_scaler_dict(data_dict, target_col, covariate_cols):
    """Build a simple z-score scaler dict from training data for PhysCF."""
    scaler = {}
    if "train" in data_dict:
        df_t = data_dict["train"]
        if target_col in df_t.columns:
            scaler[target_col] = {"mean": float(df_t[target_col].mean()),
                                  "std": float(df_t[target_col].std())}
    if "train_cov" in data_dict:
        df_c = data_dict["train_cov"]
        for col in covariate_cols:
            if col in df_c.columns:
                scaler[col] = {"mean": float(df_c[col].mean()),
                               "std": float(df_c[col].std())}
    # Map generic names for PhysCF perturbation layer
    for c in covariate_cols:
        cl = c.lower()
        if "precip" in cl or "rain" in cl:
            if c in scaler and "precip" not in scaler:
                scaler["precip"] = scaler[c]
        elif "temp" in cl:
            if c in scaler and "temp" not in scaler:
                scaler["temp"] = scaler[c]
        elif "evap" in cl or "etp" in cl:
            if c in scaler and "evap" not in scaler:
                scaler["evap"] = scaler[c]

    return scaler


# ====================
# Sidebar
# ====================

st.sidebar.header("1. Modele")
model_display = {m.display_name: m for m in models_list}
selected_model_name = st.sidebar.selectbox("Modele entraine", list(model_display.keys()))
selected_model_entry = model_display[selected_model_name]

# Load model + data
try:
    darts_model, scalers, data_dict, model_config, model_entry = \
        _load_model_and_data(selected_model_entry.run_id)
except Exception as e:
    st.error(f"Erreur chargement modele: {e}")
    st.code(traceback.format_exc())
    st.stop()

target_col, covariate_cols = _detect_columns(model_config, data_dict)
df_full, covariate_cols = _build_full_df(data_dict, target_col, covariate_cols)

if df_full is None or len(df_full) == 0:
    st.error("Impossible de reconstruire les donnees depuis MLflow.")
    st.stop()

L_model = getattr(darts_model, "input_chunk_length", 365)
H_model = getattr(darts_model, "output_chunk_length", 90)

# Build PhysCF scaler
physcf_scaler = _build_scaler_dict(data_dict, target_col, covariate_cols)

# ---- IPS reference ----
target_scaler = physcf_scaler.get(target_col, {"mean": 0, "std": 1})
mu_target = target_scaler["mean"]
sigma_target = target_scaler["std"]

gwl_train_norm = data_dict["train"][target_col] if "train" in data_dict else df_full[target_col]
gwl_train_raw = gwl_train_norm * sigma_target + mu_target
ref_stats = compute_ips_reference(gwl_train_raw)

# ---- Sidebar: scenario ----
st.sidebar.markdown("---")
st.sidebar.header("2. Scenario IPS")
st.sidebar.markdown("Changer la classe IPS de la prediction")

ips_labels_list = list(IPS_LABELS.values())
col_from, col_to = st.sidebar.columns(2)
with col_from:
    ips_from = st.selectbox("De", ips_labels_list, index=3, key="ips_from")
with col_to:
    ips_to = st.selectbox("Vers", ips_labels_list, index=1, key="ips_to")

ips_from_key = [k for k, v in IPS_LABELS.items() if v == ips_from][0]
ips_to_key = [k for k, v in IPS_LABELS.items() if v == ips_to][0]
st.sidebar.caption(f"Scenario: {ips_from} -> **{ips_to}**")

# ---- Sidebar: methods ----
st.sidebar.markdown("---")
st.sidebar.header("3. Methodes CF")
use_physcf = st.sidebar.checkbox("PhysCF (gradient)", value=True)
use_optuna = st.sidebar.checkbox("PhysCF (Optuna)", value=False)
use_comet = st.sidebar.checkbox("COMET-Hydro", value=False)

# ---- Sidebar: hyperparams ----
st.sidebar.markdown("---")
st.sidebar.header("4. Hyperparametres")
lambda_prox = st.sidebar.slider("lambda_prox", 0.001, 2.0, 0.1, 0.01)
n_iter = st.sidebar.number_input("Iterations", 50, 2000, 500, 50)
lr_cf = st.sidebar.number_input("Learning rate", 0.001, 0.1, 0.02, 0.005, format="%.3f")

device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.caption(f"Device: **{device}** | L={L_model} H={H_model}")


# ====================
# SECTION 1: Test set with ground truth + sliding window
# ====================

st.markdown("---")
st.subheader("1. Serie de test et prediction du modele")

if "test" not in data_dict:
    st.error("Pas de donnees de test dans les artifacts MLflow.")
    st.stop()

test_df = data_dict["test"]
test_len = len(test_df)

# Raw values for display
test_raw_values = test_df[target_col].values * sigma_target + mu_target
test_dates = test_df.index

# Sliding window
valid_end = test_len - H_model
if valid_end <= 0:
    st.error(f"Test set trop court ({test_len}j) pour H={H_model}.")
    st.stop()

start_idx = st.slider(
    f"Fenetre glissante sur le test ({H_model}j de prediction)",
    min_value=0, max_value=valid_end,
    value=min(valid_end // 2, valid_end),
    help=f"Contexte: {L_model}j | Prediction: {H_model}j"
)

window_pred_start = test_dates[start_idx]
window_pred_end = test_dates[min(start_idx + H_model - 1, test_len - 1)]

# Find context start in full df
full_pred_loc = df_full.index.get_loc(window_pred_start)
if isinstance(full_pred_loc, slice):
    full_pred_loc = full_pred_loc.start
context_start_loc = max(0, full_pred_loc - L_model)
window_context_start = df_full.index[context_start_loc]

st.caption(f"**Contexte:** {window_context_start.strftime('%Y-%m-%d')} | "
           f"**Prediction:** {window_pred_start.strftime('%Y-%m-%d')} -> "
           f"{window_pred_end.strftime('%Y-%m-%d')} ({H_model}j)")

# Generate prediction
pred_cache_key = f"cf_pred_{model_entry.run_id}_{start_idx}"
if pred_cache_key not in st.session_state:
    with st.spinner("Generation de la prediction..."):
        try:
            full_df_processed = pd.concat(
                [data_dict.get(s, pd.DataFrame()) for s in ["train", "val", "test"]]
            ).sort_index()
            full_df_processed = full_df_processed[~full_df_processed.index.duplicated(keep="first")]

            if covariate_cols:
                full_cov = pd.concat(
                    [data_dict.get(s, pd.DataFrame()) for s in ["train_cov", "val_cov", "test_cov"]]
                ).sort_index()
                full_cov = full_cov[~full_cov.index.duplicated(keep="first")]
                full_df_processed = full_df_processed.join(full_cov, how="inner")

            preproc_config = model_config.get("preprocessing", {}) if isinstance(model_config, dict) else {}
            is_global = (model_config.get("type", "single") == "global") if isinstance(model_config, dict) else False

            results = generate_single_window_forecast(
                model=darts_model,
                full_df=full_df_processed,
                target_col=target_col,
                covariate_cols=covariate_cols or None,
                preprocessing_config=preproc_config,
                scalers=scalers,
                start_date=window_pred_start,
                use_covariates=bool(covariate_cols),
                already_processed=True,
                is_global_model=is_global,
            )
            st.session_state[pred_cache_key] = {
                "prediction": results[0],
                "target": results[2],
                "metrics": results[3],
            }
        except Exception as e:
            st.error(f"Erreur prediction: {e}")
            st.code(traceback.format_exc())
            st.session_state[pred_cache_key] = None

cached = st.session_state.get(pred_cache_key)


# ====================
# SECTION 2: IPS bands + main chart
# ====================

st.subheader("2. Classification IPS et qualite de la prediction")

fig = go.Figure()

# IPS colored bands
# For each month in the test period, we could compute per-month bands,
# but for simplicity use the test median month
test_month_median = int(np.median(test_dates.month))
for cls_name in IPS_ORDER:
    z_lo, z_hi = IPS_CLASSES[cls_name]
    z_lo_c = max(z_lo, -5.0)
    z_hi_c = min(z_hi, 5.0)
    mu_m, sigma_m = ref_stats.get(test_month_median, (mu_target, sigma_target))
    y_lo = mu_m + z_lo_c * sigma_m
    y_hi = mu_m + z_hi_c * sigma_m

    fig.add_hrect(
        y0=y_lo, y1=y_hi,
        fillcolor=IPS_COLORS[cls_name], opacity=0.08,
        layer="below", line_width=0,
        annotation_text=IPS_LABELS[cls_name],
        annotation_position="right",
        annotation=dict(font_size=9, font_color=IPS_COLORS[cls_name]),
    )

# Context window highlight
fig.add_vrect(
    x0=window_context_start, x1=window_pred_start,
    fillcolor="rgba(46,134,171,0.12)", layer="below", line_width=1,
    line=dict(color="rgba(46,134,171,0.3)"),
    annotation_text=f"Contexte ({L_model}j)", annotation_position="bottom left",
    annotation=dict(font_size=9),
)

# Prediction window highlight
fig.add_vrect(
    x0=window_pred_start, x1=window_pred_end,
    fillcolor="rgba(255,200,0,0.2)", layer="below", line_width=1,
    line=dict(color="rgba(255,200,0,0.5)"),
    annotation_text=f"Prediction ({H_model}j)", annotation_position="top right",
    annotation=dict(font_size=9),
)

# Ground truth (blue)
fig.add_trace(go.Scatter(
    x=test_dates, y=test_raw_values,
    mode="lines", name="Ground Truth",
    line=dict(color="#2E86AB", width=2),
))

# Model prediction (pink)
if cached and cached.get("prediction") is not None:
    pred_ts = cached["prediction"]
    fig.add_trace(go.Scatter(
        x=pred_ts.time_index, y=pred_ts.values().flatten(),
        mode="lines+markers", name="Prediction modele",
        line=dict(color="#E91E63", width=3),
        marker=dict(size=4),
    ))

fig.update_layout(
    title=f"{target_col} - Test set avec bandes IPS",
    xaxis_title="Date", yaxis_title=f"{target_col} (raw)",
    height=500, hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

# --- Per-window IPS assessment ---
if cached and cached.get("prediction") is not None and cached.get("target") is not None:
    pred_ts = cached["prediction"]
    target_ts = cached["target"]

    pred_mean_raw = float(np.mean(pred_ts.values().flatten()))
    gt_mean_raw = float(np.mean(target_ts.values().flatten()))

    gt_month = int(pd.Timestamp(target_ts.time_index[len(target_ts) // 2]).month)
    gt_ips = gwl_to_ips_class(gt_mean_raw, gt_month, ref_stats)
    pred_ips = gwl_to_ips_class(pred_mean_raw, gt_month, ref_stats)

    col_gt, col_pred, col_match = st.columns(3)
    with col_gt:
        st.metric("IPS Ground Truth", IPS_LABELS.get(gt_ips, gt_ips))
    with col_pred:
        st.metric("IPS Prediction", IPS_LABELS.get(pred_ips, pred_ips))
    with col_match:
        match = gt_ips == pred_ips
        st.metric("Correspondance", "OUI" if match else "NON",
                  delta="Bonne prediction" if match else "Classe differente",
                  delta_color="normal" if match else "inverse")

    if cached.get("metrics"):
        metrics = cached["metrics"]
        m_cols = st.columns(min(len(metrics), 4))
        for i, (name, val) in enumerate(metrics.items()):
            if val is not None and not pd.isna(val):
                with m_cols[i % len(m_cols)]:
                    st.metric(name, f"{val:.4f}")


# ====================
# SECTION 3: Counterfactual generation
# ====================

st.markdown("---")
st.subheader(f"3. Generation contrefactuelle: {ips_from} -> {ips_to}")

if not any([use_physcf, use_optuna, use_comet]):
    st.info("Cochez au moins une methode CF dans la sidebar (section 3).")

if st.button("Lancer la generation contrefactuelle", type="primary",
             use_container_width=True, disabled=not any([use_physcf, use_optuna, use_comet])):

    # Extract window data
    lookback_df = df_full.iloc[context_start_loc: context_start_loc + L_model]
    horizon_df = df_full.iloc[context_start_loc + L_model: context_start_loc + L_model + H_model]

    h_obs_norm = lookback_df[target_col].values.astype(np.float32)
    s_obs_norm = lookback_df[covariate_cols].values.astype(np.float32)
    months_arr = lookback_df.index.month.values.astype(np.int64)

    # Denormalize stresses for PhysCF perturbation layer
    s_obs_phys = s_obs_norm.copy()
    for j, col in enumerate(covariate_cols):
        if col in physcf_scaler:
            mu_c = physcf_scaler[col]["mean"]
            sigma_c = physcf_scaler[col]["std"]
            if sigma_c > 0:
                s_obs_phys[:, j] = s_obs_norm[:, j] * sigma_c + mu_c

    # Target bounds
    z_min, z_max = IPS_CLASSES[ips_to_key]
    z_min_c = max(z_min, -5.0)
    z_max_c = min(z_max, 5.0)
    lower_norm = torch.full((H_model,), z_min_c, dtype=torch.float32)
    upper_norm = torch.full((H_model,), z_max_c, dtype=torch.float32)

    # Raw bounds for display
    horizon_months = horizon_df.index.month.values if len(horizon_df) >= H_model else np.full(H_model, 6)
    lower_raw_arr = np.zeros(H_model)
    upper_raw_arr = np.zeros(H_model)
    for t in range(min(H_model, len(horizon_months))):
        m = int(horizon_months[t])
        mu_m, sigma_m = ref_stats.get(m, (mu_target, sigma_target))
        lower_raw_arr[t] = mu_m + z_min_c * sigma_m if sigma_m > 0 else mu_m
        upper_raw_arr[t] = mu_m + z_max_c * sigma_m if sigma_m > 0 else mu_m

    # Model adapter
    try:
        model_adapter = DartsModelAdapter(darts_model, L_model, H_model)
    except Exception as e:
        st.error(f"Erreur adaptateur: {e}")
        st.code(traceback.format_exc())
        st.stop()

    # Run methods
    methods_to_run = {}
    if use_physcf:
        methods_to_run["PhysCF (gradient)"] = "physcf_gradient"
    if use_optuna:
        methods_to_run["PhysCF (Optuna)"] = "physcf_optuna"
    if use_comet:
        methods_to_run["COMET-Hydro"] = "comet_hydro"

    results_dict = {}
    progress = st.progress(0)
    status_text = st.empty()

    for i, (display_name, method_key) in enumerate(methods_to_run.items()):
        status_text.text(f"Optimisation {display_name}...")
        progress.progress(i / len(methods_to_run))

        try:
            h_t = torch.tensor(h_obs_norm, dtype=torch.float32)
            months_t = torch.tensor(months_arr, dtype=torch.long)

            if method_key == "physcf_gradient":
                result = generate_counterfactual(
                    h_t, torch.tensor(s_obs_phys, dtype=torch.float32),
                    model_adapter, (lower_norm, upper_norm),
                    physcf_scaler, months_t,
                    lambda_prox=lambda_prox, n_iter=n_iter, lr=lr_cf, device=device,
                )
            elif method_key == "physcf_optuna":
                result = generate_counterfactual_optuna(
                    h_t, torch.tensor(s_obs_phys, dtype=torch.float32),
                    model_adapter, (lower_norm, upper_norm),
                    physcf_scaler, months_t,
                    lambda_prox=lambda_prox, n_trials=n_iter, device=device,
                )
            elif method_key == "comet_hydro":
                result = generate_counterfactual_comet(
                    h_t, torch.tensor(s_obs_norm, dtype=torch.float32),
                    model_adapter, (lower_norm, upper_norm),
                    physcf_scaler, n_iter=n_iter, lr=lr_cf, device=device,
                )

            results_dict[display_name] = result
        except Exception as e:
            st.error(f"Erreur {display_name}: {e}")
            st.code(traceback.format_exc())

    progress.progress(1.0)
    status_text.text("Terminee!")
    st.session_state["cf_results_latest"] = results_dict

    if not results_dict:
        st.error("Aucune methode n'a reussi.")
        st.stop()

    # ====================================
    # SECTION 4: Display results
    # ====================================

    st.markdown("---")
    st.subheader("4. Resultats contrefactuels")

    # Metrics table
    rows = []
    for mn, res in results_dict.items():
        row = {"Methode": mn}
        y_cf = res.get("y_cf")
        if y_cf is not None:
            row["Validite"] = f"{validity_ratio(y_cf, lower_norm, upper_norm):.2%}"
        if "theta_star" in res:
            row["Proximite"] = f"{proximity_theta(res['theta_star']):.4f}"
            row["CC residuel"] = f"{cc_compliance(res['theta_star']):.6f}"
        row["Converge"] = "Oui" if res.get("converged") else "Non"
        row["Temps"] = f"{res.get('wall_clock_s', 0):.1f}s"
        row["Params"] = param_count(res.get("method", mn))
        rows.append(row)

    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Methode"), use_container_width=True)

    # CF overlay chart
    fig_cf = go.Figure()

    gt_window = test_raw_values[start_idx: start_idx + H_model]
    gt_dates = test_dates[start_idx: start_idx + H_model]

    fig_cf.add_trace(go.Scatter(
        x=gt_dates, y=gt_window,
        mode="lines", name="Ground Truth",
        line=dict(color="#2E86AB", width=2),
    ))

    if cached and cached.get("prediction") is not None:
        pred_ts = cached["prediction"]
        fig_cf.add_trace(go.Scatter(
            x=pred_ts.time_index, y=pred_ts.values().flatten(),
            mode="lines", name="Prediction factuelle",
            line=dict(color="#E91E63", width=2, dash="dot"),
        ))

    # Target IPS band
    ips_color = IPS_COLORS[ips_to_key]
    r, g, b = int(ips_color[1:3], 16), int(ips_color[3:5], 16), int(ips_color[5:7], 16)
    fig_cf.add_trace(go.Scatter(
        x=gt_dates, y=lower_raw_arr[:len(gt_dates)],
        mode="lines", name=f"Borne inf ({ips_to})",
        line=dict(color=ips_color, width=1, dash="dot"),
    ))
    fig_cf.add_trace(go.Scatter(
        x=gt_dates, y=upper_raw_arr[:len(gt_dates)],
        mode="lines", name=f"Borne sup ({ips_to})",
        line=dict(color=ips_color, width=1, dash="dot"),
        fill="tonexty", fillcolor=f"rgba({r},{g},{b},0.1)",
    ))

    # CF curves
    cf_colors = {"PhysCF (gradient)": "#FF6B35", "PhysCF (Optuna)": "#9B59B6", "COMET-Hydro": "#2ECC71"}
    for mn, res in results_dict.items():
        y_cf_norm = res["y_cf"].numpy() if hasattr(res["y_cf"], "numpy") else np.array(res["y_cf"])
        y_cf_raw = y_cf_norm * sigma_target + mu_target
        fig_cf.add_trace(go.Scatter(
            x=gt_dates[:len(y_cf_raw)], y=y_cf_raw,
            mode="lines+markers", name=f"CF: {mn}",
            line=dict(color=cf_colors.get(mn, "#888"), width=2, dash="dash"),
            marker=dict(size=3),
        ))

    fig_cf.update_layout(
        title=f"Contrefactuel: {ips_from} -> {ips_to}",
        xaxis_title="Date", yaxis_title=f"{target_col} (raw)",
        height=450, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_cf, use_container_width=True)

    # Theta* table
    theta_rows = []
    for mn, res in results_dict.items():
        if "theta_star" in res:
            theta_rows.append({"Methode": mn, **res["theta_star"]})
    if theta_rows:
        st.subheader("5. Parametres theta* (scenarios climatiques)")
        st.dataframe(pd.DataFrame(theta_rows).set_index("Methode").round(3), use_container_width=True)

        for mn, res in results_dict.items():
            if "theta_star" not in res:
                continue
            theta = res["theta_star"]
            st.markdown(f"**{mn}** - Scenario pour passer en *{ips_to}*:")
            parts = []
            for s in ["DJF", "MAM", "JJA", "SON"]:
                k = f"s_P_{s}"
                if k in theta:
                    v = theta[k]
                    if v > 1.05:
                        parts.append(f"  - Precipitation {s}: **+{(v-1)*100:.0f}%**")
                    elif v < 0.95:
                        parts.append(f"  - Precipitation {s}: **{(v-1)*100:.0f}%**")
            if "delta_T" in theta and abs(theta["delta_T"]) > 0.1:
                parts.append(f"  - Temperature: **{theta['delta_T']:+.1f} C**")
            if "delta_s" in theta and abs(theta["delta_s"]) > 1:
                parts.append(f"  - Decalage temporel: **{theta['delta_s']:+.0f} jours**")
            st.markdown("\n".join(parts) if parts else "  - Perturbations negligeables")

    # Stress comparison
    for mn, res in results_dict.items():
        if "s_cf_phys" not in res:
            continue
        s_cf_phys = res["s_cf_phys"].numpy() if hasattr(res["s_cf_phys"], "numpy") else res["s_cf_phys"]

        st.subheader("6. Comparaison des stresses climatiques")
        n_cov = min(len(covariate_cols), s_obs_phys.shape[-1])
        fig_s = make_subplots(rows=n_cov, cols=1, shared_xaxes=True,
                              subplot_titles=covariate_cols[:n_cov])
        lb_dates = lookback_df.index
        colors_s = ["#2E86AB", "#FF8C00", "#9B59B6"]
        for j in range(n_cov):
            fig_s.add_trace(go.Scatter(
                x=lb_dates, y=s_obs_phys[:, j], name=f"{covariate_cols[j]} (obs)",
                line=dict(color=colors_s[j%3], width=1),
                showlegend=(j==0), legendgroup="obs",
            ), row=j+1, col=1)
            fig_s.add_trace(go.Scatter(
                x=lb_dates, y=s_cf_phys[:, j], name=f"{covariate_cols[j]} (CF)",
                line=dict(color=colors_s[j%3], width=2, dash="dash"),
                showlegend=(j==0), legendgroup="cf",
            ), row=j+1, col=1)
        fig_s.update_layout(title=f"Stresses ({mn})", height=150*n_cov+100)
        st.plotly_chart(fig_s, use_container_width=True)
        break

    # Loss curves
    has_loss = any("loss_history" in r for r in results_dict.values())
    if has_loss:
        st.subheader("7. Convergence")
        fig_l = go.Figure()
        for mn, res in results_dict.items():
            c = cf_colors.get(mn, "#888")
            if "loss_history" in res:
                fig_l.add_trace(go.Scatter(y=res["loss_history"], name=f"{mn} (total)",
                                           mode="lines", line=dict(color=c)))
            if "target_history" in res:
                fig_l.add_trace(go.Scatter(y=res["target_history"], name=f"{mn} (cible)",
                                           mode="lines", line=dict(color=c, dash="dash")))
        fig_l.update_layout(xaxis_title="Iteration", yaxis_title="Loss",
                            yaxis_type="log", height=350)
        st.plotly_chart(fig_l, use_container_width=True)

    st.success(f"Analyse terminee! {len(results_dict)} methode(s), scenario: {ips_from} -> {ips_to}")
