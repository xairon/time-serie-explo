"""Counterfactual Analysis Page - PhysCF Integration.

Fully autonomous: loads model, data, and scalers from MLflow artifacts.
IPS is computed on RAW physical values (m NGF) using real scaler params.

Workflow:
1. Load trained model from MLflow registry
2. Extract real mu/sigma from Darts scalers (inverse_transform)
3. Denormalize data to physical units (m NGF) for IPS computation
4. Display test set with ground truth + sliding window predictions
5. Show IPS classification bands with colored levels
6. Indicate per-window whether prediction matches ground truth IPS class
7. Allow user to select target IPS class change (e.g., normal -> dry)
8. Run 1, 2 or 3 CF methods (checkboxes) and overlay results
9. Display comparative metrics tables and scenario interpretation
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

# PhysCF counterfactual module
from dashboard.utils.counterfactual import (
    PerturbationLayer,
    generate_counterfactual,
    generate_counterfactual_optuna,
    generate_counterfactual_comet,
    IPS_CLASSES,
    IPS_ORDER,
    BRGM_MIN_YEARS,
    AQUIFER_IPS_WINDOW,
    compute_ips_reference,
    validate_ips_data,
    get_aquifer_ips_info,
    daily_to_monthly_mean,
    ips_class_to_gwl_bounds,
    extract_scaler_params,
    validity_ratio,
    proximity_theta,
    cc_compliance,
    param_count,
)
from dashboard.utils.counterfactual.ips import (
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
    """Load model, scalers, data, config, and IPS reference from MLflow artifacts."""
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

    # Try to load pre-computed IPS reference stats
    import json as _json
    ips_reference = None
    try:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="model/ips_reference.json"
        )
        with open(local_path, 'r') as f:
            ips_reference = _json.load(f)
    except Exception:
        pass  # IPS reference not available (older model), will compute on-the-fly

    return model, scalers, data_dict, model_config, entry, ips_reference


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
    """Merge train/val/test + covariates into a single DataFrame (NORMALIZED)."""
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


def _extract_real_scaler_params(scalers_dict):
    """Extract real mu/sigma from Darts scalers.

    Returns (mu_target, sigma_target, cov_params_dict) or (None, None, {}).
    cov_params_dict: {col_name: {'mean': mu, 'std': sigma}}
    """
    if not scalers_dict:
        return None, None, {}
    return extract_scaler_params(scalers_dict)


def _build_physcf_scaler(mu_target, sigma_target, target_col, cov_params, covariate_cols):
    """Build the PhysCF scaler dict from real physical parameters.

    This scaler maps column names to {mean, std} in physical units.
    PhysCF uses it for Clausius-Clapeyron constraint and normalization.
    """
    scaler = {}

    if mu_target is not None and sigma_target is not None:
        scaler[target_col] = {"mean": mu_target, "std": sigma_target}

    # Map covariate params
    for col in covariate_cols:
        if col in cov_params:
            scaler[col] = cov_params[col]

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
    darts_model, scalers, data_dict, model_config, model_entry, ips_reference_cached = \
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


# ====================
# CRITICAL: Extract real scaler parameters from Darts scalers
# ====================

mu_target, sigma_target, cov_params = _extract_real_scaler_params(scalers)

if mu_target is None or sigma_target is None:
    st.warning(
        "Impossible d'extraire les parametres de normalisation depuis les scalers Darts. "
        "L'IPS sera calcule sur les donnees normalisees (z-scores), ce qui est INCORRECT. "
        "Verifiez que le modele a ete entraine avec des scalers."
    )
    # Fallback: compute from normalized data (INCORRECT but avoids crash)
    if "train" in data_dict and target_col in data_dict["train"].columns:
        mu_target = float(data_dict["train"][target_col].mean())
        sigma_target = float(data_dict["train"][target_col].std())
    else:
        mu_target = 0.0
        sigma_target = 1.0

# Build PhysCF scaler with REAL physical units
physcf_scaler = _build_physcf_scaler(mu_target, sigma_target, target_col, cov_params, covariate_cols)


# ====================
# IPS Reference Stats (on RAW physical values, monthly aggregation)
# ====================

# Denormalize ALL available data (train+val+test) to physical units for IPS reference
# BRGM recommends using the longest possible series for reference computation
gwl_all_norm = df_full[target_col]
gwl_all_raw = gwl_all_norm * sigma_target + mu_target

# Also get train-only for reference period
gwl_train_norm = data_dict["train"][target_col] if "train" in data_dict else gwl_all_norm
gwl_train_raw = gwl_train_norm * sigma_target + mu_target

# Try to detect aquifer type from model metadata
aquifer_type = None
if isinstance(model_config, dict):
    aquifer_type = model_config.get("aquifer_type")
if not aquifer_type and hasattr(model_entry, "hyperparams"):
    aquifer_type = model_entry.hyperparams.get("aquifer_type")

# Sidebar: aquifer type selection (user can override)
st.sidebar.markdown("---")
st.sidebar.header("5. Type de nappe")
aquifer_options = ["auto", "chalk", "limestone", "karst", "alluvial", "sand", "volcanic"]
aquifer_labels = {
    "auto": "Auto-detect",
    "chalk": "Craie (inertielle)",
    "limestone": "Calcaire (inertielle)",
    "karst": "Karst (reactive)",
    "alluvial": "Alluviale (reactive)",
    "sand": "Sable (reactive)",
    "volcanic": "Volcanique (inertielle)",
}
selected_aquifer = st.sidebar.selectbox(
    "Type d'aquifere",
    options=aquifer_options,
    format_func=lambda x: aquifer_labels.get(x, x),
    index=aquifer_options.index(aquifer_type) if (aquifer_type and aquifer_type in aquifer_options) else 0,
    help="Influence les recommandations IPS (fenetre d'aggregation)"
)
if selected_aquifer == "auto":
    selected_aquifer = aquifer_type  # May be None

# Validate data for IPS (with aquifer info)
ips_validation = validate_ips_data(gwl_all_raw, aquifer_type=selected_aquifer)

# Show IPS data quality and methodology
with st.expander("Qualite des donnees et methodologie IPS (BRGM)", expanded=not ips_validation["valid"]):
    st.markdown("""
    **Methodologie IPS** (ref: Seguin 2014, BRGM/RP-64147-FR):
    - L'IPS est un indicateur **mensuel**: les donnees journalieres sont d'abord
      agregees en **moyennes mensuelles** (min 15 valeurs/mois)
    - Pour chaque mois calendaire (1-12), on calcule la moyenne et l'ecart-type
      sur la periode de reference (idealement 1981-2010, 30 ans)
    - Le z-score est: `z = (gwl_mois - mu_mois) / sigma_mois`
    - Classification en 7 classes (Tres bas a Tres haut) selon les seuils z
    """)

    col_v1, col_v2, col_v3, col_v4 = st.columns(4)
    with col_v1:
        color = "normal" if ips_validation["n_years"] >= BRGM_MIN_YEARS else "inverse"
        st.metric("Annees de donnees", ips_validation["n_years"],
                  delta=f"min BRGM: {BRGM_MIN_YEARS}",
                  delta_color=color)
    with col_v2:
        st.metric("Mois couverts", f"{ips_validation['n_months_covered']}/12")
    with col_v3:
        st.metric("Moyennes mensuelles", ips_validation.get("n_monthly_values", "?"))
    with col_v4:
        st.metric("mu (m NGF)", f"{mu_target:.2f}")
        st.caption(f"sigma = {sigma_target:.4f} m")

    # Aquifer-specific info
    if selected_aquifer and ips_validation.get("aquifer_info"):
        aq_info = ips_validation["aquifer_info"]
        cat = aq_info.get("category", "?")
        win = aq_info.get("recommended_window", 1)
        st.info(f"**Nappe {cat}** ({selected_aquifer}) - IPS-{win} recommande")

    for w in ips_validation["warnings"]:
        st.warning(w)
    for e in ips_validation["errors"]:
        st.error(e)

    if not ips_validation["valid"]:
        st.error("L'IPS ne peut pas etre calcule de maniere fiable avec ces donnees.")
        st.stop()

# Use pre-computed IPS reference from MLflow if available, otherwise compute on-the-fly
if ips_reference_cached and "ref_stats" in ips_reference_cached:
    ref_stats = {int(k): tuple(v) for k, v in ips_reference_cached["ref_stats"].items()}
    st.caption("IPS reference chargee depuis les artefacts MLflow (pre-calculee a l'entrainement)")

    # Also use cached scaler params if they are more reliable
    if ips_reference_cached.get("mu_target") is not None:
        cached_mu = ips_reference_cached["mu_target"]
        cached_sigma = ips_reference_cached["sigma_target"]
        if abs(cached_mu - mu_target) > 0.01 or abs(cached_sigma - sigma_target) > 0.01:
            st.info(
                f"Scalers caches: mu={cached_mu:.4f}, sigma={cached_sigma:.4f} "
                f"(vs extraits: mu={mu_target:.4f}, sigma={sigma_target:.4f})"
            )
else:
    # Compute on-the-fly (for older models without ips_reference.json)
    ref_stats = compute_ips_reference(gwl_all_raw, aggregate_to_monthly=True)
    st.caption("IPS reference calculee a la volee (modele entraine avant cette fonctionnalite)")


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

# Raw values for display (denormalized to m NGF)
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

# IPS colored bands - use per-month stats for the median month of the test period
# We show bands that vary month by month across the test timeline
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

# Ground truth (blue) - in m NGF
fig.add_trace(go.Scatter(
    x=test_dates, y=test_raw_values,
    mode="lines", name="Ground Truth (m NGF)",
    line=dict(color="#2E86AB", width=2),
))

# Model prediction (pink) - denormalized to m NGF
if cached and cached.get("prediction") is not None:
    pred_ts = cached["prediction"]
    pred_values_norm = pred_ts.values().flatten()
    pred_values_raw = pred_values_norm * sigma_target + mu_target
    fig.add_trace(go.Scatter(
        x=pred_ts.time_index, y=pred_values_raw,
        mode="lines+markers", name="Prediction modele (m NGF)",
        line=dict(color="#E91E63", width=3),
        marker=dict(size=4),
    ))

fig.update_layout(
    title=f"{target_col} - Test set avec bandes IPS (m NGF)",
    xaxis_title="Date", yaxis_title=f"Niveau piezometrique (m NGF)",
    height=500, hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

# --- Per-window IPS assessment ---
if cached and cached.get("prediction") is not None and cached.get("target") is not None:
    pred_ts = cached["prediction"]
    target_ts = cached["target"]

    # Denormalize to physical units for IPS
    pred_values_raw = pred_ts.values().flatten() * sigma_target + mu_target
    gt_values_raw = target_ts.values().flatten() * sigma_target + mu_target

    pred_mean_raw = float(np.mean(pred_values_raw))
    gt_mean_raw = float(np.mean(gt_values_raw))

    gt_month = int(pd.Timestamp(target_ts.time_index[len(target_ts) // 2]).month)
    gt_ips = gwl_to_ips_class(gt_mean_raw, gt_month, ref_stats)
    pred_ips = gwl_to_ips_class(pred_mean_raw, gt_month, ref_stats)

    gt_zscore = gwl_to_ips_zscore(gt_mean_raw, gt_month, ref_stats)
    pred_zscore = gwl_to_ips_zscore(pred_mean_raw, gt_month, ref_stats)

    col_gt, col_pred, col_match = st.columns(3)
    with col_gt:
        st.metric(
            "IPS Ground Truth",
            IPS_LABELS.get(gt_ips, gt_ips),
            delta=f"z = {gt_zscore:+.2f} | {gt_mean_raw:.2f} m NGF",
        )
    with col_pred:
        st.metric(
            "IPS Prediction",
            IPS_LABELS.get(pred_ips, pred_ips),
            delta=f"z = {pred_zscore:+.2f} | {pred_mean_raw:.2f} m NGF",
        )
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

    # Extract window data (NORMALIZED - as the model expects)
    lookback_df = df_full.iloc[context_start_loc: context_start_loc + L_model]
    horizon_df = df_full.iloc[context_start_loc + L_model: context_start_loc + L_model + H_model]

    h_obs_norm = lookback_df[target_col].values.astype(np.float32)
    s_obs_norm = lookback_df[covariate_cols].values.astype(np.float32) if covariate_cols else np.zeros((L_model, 1), dtype=np.float32)
    months_arr = lookback_df.index.month.values.astype(np.int64)

    # Denormalize stresses to physical units for PhysCF perturbation layer
    s_obs_phys = s_obs_norm.copy()
    for j, col in enumerate(covariate_cols):
        if col in physcf_scaler:
            mu_c = physcf_scaler[col]["mean"]
            sigma_c = physcf_scaler[col]["std"]
            if sigma_c > 0:
                s_obs_phys[:, j] = s_obs_norm[:, j] * sigma_c + mu_c

    # Target bounds: convert IPS class to z-score bounds, then to NORMALIZED bounds
    # The model predicts in normalized space, so bounds must be in normalized space too.
    z_min, z_max = IPS_CLASSES[ips_to_key]
    z_min_c = max(z_min, -5.0)
    z_max_c = min(z_max, 5.0)

    # Convert IPS z-scores to normalized model space:
    # IPS z-score: z_ips = (gwl_raw - mu_month) / sigma_month
    # Model normalized: gwl_norm = (gwl_raw - mu_target) / sigma_target
    # So: gwl_raw = mu_month + z_ips * sigma_month
    # And: gwl_norm = (mu_month + z_ips * sigma_month - mu_target) / sigma_target
    horizon_months = horizon_df.index.month.values if len(horizon_df) >= H_model else np.full(H_model, 6)
    lower_norm_arr = np.zeros(H_model, dtype=np.float32)
    upper_norm_arr = np.zeros(H_model, dtype=np.float32)
    lower_raw_arr = np.zeros(H_model)
    upper_raw_arr = np.zeros(H_model)

    for t in range(min(H_model, len(horizon_months))):
        m = int(horizon_months[t])
        mu_m, sigma_m = ref_stats.get(m, (mu_target, sigma_target))

        # Raw bounds (m NGF) for display
        lower_raw_arr[t] = mu_m + z_min_c * sigma_m if sigma_m > 0 else mu_m
        upper_raw_arr[t] = mu_m + z_max_c * sigma_m if sigma_m > 0 else mu_m

        # Normalized bounds for the model
        if sigma_target > 0:
            lower_norm_arr[t] = (lower_raw_arr[t] - mu_target) / sigma_target
            upper_norm_arr[t] = (upper_raw_arr[t] - mu_target) / sigma_target
        else:
            lower_norm_arr[t] = z_min_c
            upper_norm_arr[t] = z_max_c

    lower_norm = torch.tensor(lower_norm_arr, dtype=torch.float32)
    upper_norm = torch.tensor(upper_norm_arr, dtype=torch.float32)

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

    # CF overlay chart - ALL IN m NGF
    fig_cf = go.Figure()

    gt_window_raw = test_raw_values[start_idx: start_idx + H_model]
    gt_dates = test_dates[start_idx: start_idx + H_model]

    fig_cf.add_trace(go.Scatter(
        x=gt_dates, y=gt_window_raw,
        mode="lines", name="Ground Truth (m NGF)",
        line=dict(color="#2E86AB", width=2),
    ))

    if cached and cached.get("prediction") is not None:
        pred_ts = cached["prediction"]
        pred_raw = pred_ts.values().flatten() * sigma_target + mu_target
        fig_cf.add_trace(go.Scatter(
            x=pred_ts.time_index, y=pred_raw,
            mode="lines", name="Prediction factuelle (m NGF)",
            line=dict(color="#E91E63", width=2, dash="dot"),
        ))

    # Target IPS band (raw m NGF)
    ips_color = IPS_COLORS[ips_to_key]
    r, g, b = int(ips_color[1:3], 16), int(ips_color[3:5], 16), int(ips_color[5:7], 16)
    fig_cf.add_trace(go.Scatter(
        x=gt_dates, y=lower_raw_arr[:len(gt_dates)],
        mode="lines", name=f"Borne inf IPS ({ips_to})",
        line=dict(color=ips_color, width=1, dash="dot"),
    ))
    fig_cf.add_trace(go.Scatter(
        x=gt_dates, y=upper_raw_arr[:len(gt_dates)],
        mode="lines", name=f"Borne sup IPS ({ips_to})",
        line=dict(color=ips_color, width=1, dash="dot"),
        fill="tonexty", fillcolor=f"rgba({r},{g},{b},0.1)",
    ))

    # CF curves - denormalized to m NGF
    cf_colors = {"PhysCF (gradient)": "#FF6B35", "PhysCF (Optuna)": "#9B59B6", "COMET-Hydro": "#2ECC71"}
    for mn, res in results_dict.items():
        y_cf_norm = res["y_cf"].numpy() if hasattr(res["y_cf"], "numpy") else np.array(res["y_cf"])
        y_cf_raw = y_cf_norm * sigma_target + mu_target
        fig_cf.add_trace(go.Scatter(
            x=gt_dates[:len(y_cf_raw)], y=y_cf_raw,
            mode="lines+markers", name=f"CF: {mn} (m NGF)",
            line=dict(color=cf_colors.get(mn, "#888"), width=2, dash="dash"),
            marker=dict(size=3),
        ))

    fig_cf.update_layout(
        title=f"Contrefactuel: {ips_from} -> {ips_to} (m NGF)",
        xaxis_title="Date", yaxis_title="Niveau piezometrique (m NGF)",
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
