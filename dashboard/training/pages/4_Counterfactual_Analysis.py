"""Counterfactual Analysis Page - PhysCF Integration.

Fully autonomous: loads model, data, and scalers from MLflow artifacts.
IPS is computed on RAW physical values (m NGF) using real scaler params.

Workflow:
1. Load trained model from MLflow registry
2. Extract real mu/sigma from Darts scalers (inverse_transform)
3. Denormalize data to physical units (m NGF) for IPS computation
4. Display test set with ground truth + sliding window predictions
5. Show IPS-N classification bands with colored levels
6. Show context: climate data + parameter explanations
7. Allow user to select target IPS class change (e.g., normal -> dry)
8. Run 1, 2 or 3 CF methods (checkboxes) and overlay results
9. Tabbed results: resume, scenario interpretation, climate comparison, export
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
from dashboard.utils.preprocessing import detect_columns_from_config, build_complete_dataframe

# PhysCF counterfactual module
from dashboard.utils.counterfactual import (
    PerturbationLayer,
    generate_counterfactual,
    generate_counterfactual_optuna,
    generate_counterfactual_comet,
    IPS_CLASSES,
    IPS_ORDER,
    IPS_WINDOWS,
    BRGM_MIN_YEARS,
    AQUIFER_IPS_WINDOW,
    compute_ips_reference,
    compute_ips_reference_n,
    validate_ips_data,
    ips_class_to_gwl_bounds,
    extract_scaler_params,
    validity_ratio,
    proximity_theta,
    cc_compliance,
    cc_compliance_from_theta,
    param_count,
)
from dashboard.utils.counterfactual.ips import (
    gwl_to_ips_class,
    gwl_to_ips_zscore,
    compute_monthly_ips_bounds,
    classify_prediction_monthly,
)
from dashboard.utils.counterfactual.darts_adapter import (
    DartsModelAdapter,
    StandaloneGRUAdapter,
)
from dashboard.utils.counterfactual.constants import (
    IPS_COLORS,
    IPS_LABELS,
    CF_METHOD_COLORS,
    PHYSCF_PARAM_INFO,
    PRESET_SCENARIOS,
    MONTH_TO_SEASON_NAME,
)
from dashboard.utils.counterfactual.viz import (
    plot_theta_radar,
    plot_cf_overlay,
    add_monthly_ips_bars,
    plot_stress_comparison,
    plot_convergence,
    compute_seasonal_summary,
    generate_cf_narrative,
    build_cf_export_df,
)
from dashboard.utils.counterfactual.pastas_validation import (
    PASTAS_AVAILABLE,
    fit_pastas_for_station,
    build_pastas_series_from_data,
    run_dual_validation_for_results,
)

# ---- Page Config ----
# st.set_page_config is called by Home.py (multipage entrypoint)
# Only call it if running standalone (not as a sub-page)
try:
    st.set_page_config(page_title="PhysCF - Counterfactual Analysis", layout="wide")
except st.errors.StreamlitAPIException:
    pass  # Already set by Home.py
st.title("PhysCF - Analyse Contrefactuelle")

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
        pass

    return model, scalers, data_dict, model_config, entry, ips_reference


# Aliases for backward compatibility (functions extracted to preprocessing.py)
_detect_columns = detect_columns_from_config
_build_full_df = build_complete_dataframe


def _extract_real_scaler_params(scalers_dict):
    """Extract real mu/sigma from Darts scalers."""
    if not scalers_dict:
        return None, None, {}
    return extract_scaler_params(scalers_dict)


def _build_physcf_scaler(mu_target, sigma_target, target_col, cov_params, covariate_cols):
    """Build the PhysCF scaler dict from real physical parameters."""
    scaler = {}
    if mu_target is not None and sigma_target is not None:
        scaler[target_col] = {"mean": mu_target, "std": sigma_target}

    for col in covariate_cols:
        if col in cov_params:
            scaler[col] = cov_params[col]

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
selected_model_name = st.sidebar.selectbox(
    "Modele entraine", list(model_display.keys()),
    help="Modele entraine depuis MLflow. Determine L (contexte) et H (horizon de prediction).",
)
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

# ---- Sidebar: prediction window slider (under model section) ----
if "test" in data_dict:
    _test_len_sidebar = len(data_dict["test"])
    _valid_end_sidebar = _test_len_sidebar - H_model
    if _valid_end_sidebar > 0:
        st.sidebar.markdown("**Fenetre de prediction**")
        start_idx = st.sidebar.slider(
            f"Position sur le test ({H_model}j)",
            min_value=0, max_value=_valid_end_sidebar,
            value=min(_valid_end_sidebar // 2, _valid_end_sidebar),
            help=f"Contexte: {L_model}j | Prediction: {H_model}j",
        )
    else:
        start_idx = 0
else:
    start_idx = 0


# ====================
# CRITICAL: Extract real scaler parameters
# ====================

mu_target, sigma_target, cov_params = _extract_real_scaler_params(scalers)

if mu_target is None or sigma_target is None:
    st.warning(
        "Impossible d'extraire les parametres de normalisation depuis les scalers Darts. "
        "L'IPS sera calcule sur les donnees normalisees (z-scores), ce qui est INCORRECT."
    )
    if "train" in data_dict and target_col in data_dict["train"].columns:
        mu_target = float(data_dict["train"][target_col].mean())
        sigma_target = float(data_dict["train"][target_col].std())
    else:
        mu_target, sigma_target = 0.0, 1.0

physcf_scaler = _build_physcf_scaler(mu_target, sigma_target, target_col, cov_params, covariate_cols)


# ====================
# IPS Reference Stats
# ====================

gwl_all_norm = df_full[target_col]
gwl_all_raw = gwl_all_norm * sigma_target + mu_target

aquifer_type = None
if isinstance(model_config, dict):
    aquifer_type = model_config.get("aquifer_type")
if not aquifer_type and hasattr(model_entry, "hyperparams"):
    aquifer_type = model_entry.hyperparams.get("aquifer_type") if model_entry.hyperparams else None

# ---- Sidebar: aquifer & IPS-N ----
st.sidebar.markdown("---")
st.sidebar.header("2. Type de nappe & IPS-N")
aquifer_options = ["auto", "chalk", "limestone", "karst", "alluvial", "sand", "volcanic"]
aquifer_labels_map = {
    "auto": "Auto-detect", "chalk": "Craie (inertielle)", "limestone": "Calcaire (inertielle)",
    "karst": "Karst (reactive)", "alluvial": "Alluviale (reactive)",
    "sand": "Sable (reactive)", "volcanic": "Volcanique (inertielle)",
}
selected_aquifer = st.sidebar.selectbox(
    "Type d'aquifere", options=aquifer_options,
    format_func=lambda x: aquifer_labels_map.get(x, x),
    index=aquifer_options.index(aquifer_type) if (aquifer_type and aquifer_type in aquifer_options) else 0,
    help="Influence les recommandations IPS (fenetre d'aggregation)",
)
if selected_aquifer == "auto":
    selected_aquifer = aquifer_type

_default_window = AQUIFER_IPS_WINDOW.get(selected_aquifer, 1) if selected_aquifer else 1
_window_labels = {1: "IPS-1 (mensuel)", 3: "IPS-3 (trimestriel)", 6: "IPS-6 (semestriel)", 12: "IPS-12 (annuel)"}
_default_idx = IPS_WINDOWS.index(_default_window) if _default_window in IPS_WINDOWS else 0
selected_ips_window = st.sidebar.selectbox(
    "Fenetre IPS-N", options=IPS_WINDOWS,
    format_func=lambda x: _window_labels.get(x, f"IPS-{x}"),
    index=_default_idx,
    help="IPS-1: mensuel. IPS-3: karst/alluvial. IPS-6: craie. IPS-12: calcaire.",
)
st.sidebar.caption(f"Recommandation: **IPS-{_default_window}** ({'auto' if not selected_aquifer else selected_aquifer})")

# Validate data
ips_validation = validate_ips_data(gwl_all_raw, aquifer_type=selected_aquifer)

with st.expander("Qualite des donnees et methodologie IPS (BRGM)", expanded=not ips_validation["valid"]):
    st.markdown(f"""
    **Methodologie IPS-{selected_ips_window}** (ref: Seguin 2014, BRGM/RP-64147-FR):
    - Donnees journalieres agregees en **moyennes mensuelles** (min 15 valeurs/mois)
    - **IPS-{selected_ips_window}**: moyenne glissante sur **{selected_ips_window} mois**
      {'(pas de lissage)' if selected_ips_window == 1 else 'avant calcul des statistiques'}
    - z-score: `z = (gwl_lisse - mu_m) / sigma_m` par mois calendaire
    - 7 classes: Tres bas (z < -1.28) a Tres haut (z > 1.28)
    """)

    col_v1, col_v2, col_v3, col_v4 = st.columns(4)
    with col_v1:
        st.metric("Annees", ips_validation["n_years"],
                  delta=f"min BRGM: {BRGM_MIN_YEARS}",
                  delta_color="normal" if ips_validation["n_years"] >= BRGM_MIN_YEARS else "inverse")
    with col_v2:
        st.metric("Mois couverts", f"{ips_validation['n_months_covered']}/12")
    with col_v3:
        st.metric("Moy. mensuelles", ips_validation.get("n_monthly_values", "?"))
    with col_v4:
        st.metric("mu (m NGF)", f"{mu_target:.2f}")
        st.caption(f"sigma = {sigma_target:.4f} m")

    if selected_aquifer and ips_validation.get("aquifer_info"):
        aq_info = ips_validation["aquifer_info"]
        st.info(f"**Nappe {aq_info.get('category', '?')}** ({selected_aquifer}) - IPS-{aq_info.get('recommended_window', 1)} recommande")

    for w in ips_validation["warnings"]:
        st.warning(w)
    for e in ips_validation["errors"]:
        st.error(e)
    if not ips_validation["valid"]:
        st.error("L'IPS ne peut pas etre calcule de maniere fiable.")
        st.stop()

# Load IPS reference stats
all_ref_stats = {}
if ips_reference_cached and "ref_stats_all" in ips_reference_cached:
    for w_str, month_dict in ips_reference_cached["ref_stats_all"].items():
        all_ref_stats[int(w_str)] = {int(m): tuple(v) for m, v in month_dict.items()}
elif ips_reference_cached and "ref_stats" in ips_reference_cached:
    all_ref_stats[1] = {int(k): tuple(v) for k, v in ips_reference_cached["ref_stats"].items()}

if selected_ips_window not in all_ref_stats:
    with st.spinner(f"Calcul IPS-{selected_ips_window}..."):
        all_ref_stats[selected_ips_window] = compute_ips_reference_n(
            gwl_all_raw, window=selected_ips_window, aggregate_to_monthly=True
        )
if 1 not in all_ref_stats:
    all_ref_stats[1] = compute_ips_reference(gwl_all_raw, aggregate_to_monthly=True)

ref_stats = all_ref_stats[selected_ips_window]


# ---- Sidebar: scenario IPS ----
st.sidebar.markdown("---")
st.sidebar.header("3. Scenario IPS")

# Preset scenarios
selected_preset = st.sidebar.selectbox(
    "Scenario type", list(PRESET_SCENARIOS.keys()),
    help="Scenarios pre-definis bases sur des analogs climatiques reels",
)
if PRESET_SCENARIOS[selected_preset] is not None:
    preset = PRESET_SCENARIOS[selected_preset]
    st.sidebar.caption(f"_{preset['description']}_")

ips_labels_list = list(IPS_LABELS.values())
col_from, col_to = st.sidebar.columns(2)

# Default indices (may be overridden by preset)
_default_from_idx = 3  # Normal
_default_to_idx = 1    # Bas
if PRESET_SCENARIOS[selected_preset] is not None:
    p = PRESET_SCENARIOS[selected_preset]
    _from_label = IPS_LABELS.get(p["suggested_from"], "Normal")
    _to_label = IPS_LABELS.get(p["suggested_to"], "Bas")
    _default_from_idx = ips_labels_list.index(_from_label) if _from_label in ips_labels_list else 3
    _default_to_idx = ips_labels_list.index(_to_label) if _to_label in ips_labels_list else 1

with col_from:
    ips_from = st.selectbox("De", ips_labels_list, index=_default_from_idx, key="ips_from",
                            help="Classe IPS actuelle de la prediction")
with col_to:
    ips_to = st.selectbox("Vers", ips_labels_list, index=_default_to_idx, key="ips_to",
                          help="Classe IPS cible pour le contrefactuel")

ips_from_key = [k for k, v in IPS_LABELS.items() if v == ips_from][0]
ips_to_key = [k for k, v in IPS_LABELS.items() if v == ips_to][0]

# Visual IPS color badges
color_from = IPS_COLORS[ips_from_key]
color_to = IPS_COLORS[ips_to_key]
st.sidebar.markdown(
    f'<div style="text-align:center;padding:8px 0">'
    f'<span style="background:{color_from};color:white;padding:4px 12px;'
    f'border-radius:4px;font-weight:bold">{ips_from}</span>'
    f' &nbsp;&rarr;&nbsp; '
    f'<span style="background:{color_to};color:white;padding:4px 12px;'
    f'border-radius:4px;font-weight:bold">{ips_to}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

# ---- Sidebar: methods ----
st.sidebar.markdown("---")
st.sidebar.header("4. Methodes CF")
use_physcf = st.sidebar.checkbox("PhysCF (gradient)", value=True,
    help="Methode PhysCF par descente de gradient. 7 parametres physiques, rapide.")
use_optuna = st.sidebar.checkbox("PhysCF (Optuna)", value=False,
    help="Methode PhysCF par optimisation boite noire (TPE). Plus lent mais sans gradient.")
use_comet = st.sidebar.checkbox("COMET-Hydro", value=False,
    help="Baseline COMET: perturbe chaque pas de temps x covariable. L*3 parametres, pas de CC.")

# ---- Sidebar: hyperparams ----
st.sidebar.markdown("---")
st.sidebar.header("5. Hyperparametres")
lambda_prox = st.sidebar.slider("lambda_prox", 0.001, 2.0, 0.1, 0.01,
    help="Poids de la proximite. Plus eleve = perturbations plus petites mais CF potentiellement hors cible.")
n_iter = st.sidebar.number_input("Iterations", 50, 2000, 500, 50,
    help="Nombre d'iterations d'optimisation. 500 suffit pour PhysCF gradient.")
lr_cf = st.sidebar.number_input("Learning rate", 0.001, 0.1, 0.02, 0.005, format="%.3f",
    help="Taux d'apprentissage (Adam). Reduire si oscillations.")

device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.caption(f"Device: **{device}** | L={L_model} H={H_model}")

# ---- Sidebar: Pastas dual validation ----
st.sidebar.markdown("---")
st.sidebar.header("6. Validation Pastas")
use_pastas_validation = st.sidebar.checkbox(
    "Validation duale (TFT + Pastas TFN)",
    value=False,
    disabled=not PASTAS_AVAILABLE,
    help=(
        "Ajoute un modele Pastas TFN (Transfer Function Noise) "
        "comme second modele independant. Valide les CF par accord "
        "TFT-Pastas (RMSE_cf < gamma x RMSE_baseline). "
        "Necessite pastas>=1.7."
    ),
)
if use_pastas_validation and not PASTAS_AVAILABLE:
    st.sidebar.warning("pastas n'est pas installe. `pip install pastas>=1.7`")
    use_pastas_validation = False

if use_pastas_validation:
    gamma_pastas = st.sidebar.slider(
        "gamma (tolerance)", 1.0, 3.0, 1.5, 0.1,
        help="Multiplicateur sur le RMSE factuel baseline. Plus eleve = plus tolerant.",
    )
else:
    gamma_pastas = 1.5


# ---- Pastas model fitting (cached per station) ----
@st.cache_resource(show_spinner="Calibration du modele Pastas (TFN)...")
def _fit_pastas_cached(_run_id: str, _gwl_s, _precip_s, _evap_s, _train_end: str):
    """Fit and cache Pastas model per run_id (= station/model combo)."""
    return fit_pastas_for_station(_gwl_s, _precip_s, _evap_s, _train_end)


pastas_model = None
if use_pastas_validation:
    try:
        gwl_pastas, precip_pastas, evap_pastas, train_end_pastas = build_pastas_series_from_data(
            data_dict, target_col, covariate_cols,
            mu_target, sigma_target, physcf_scaler,
        )
        pastas_model = _fit_pastas_cached(
            selected_model_entry.run_id,
            gwl_pastas, precip_pastas, evap_pastas, train_end_pastas,
        )
        if pastas_model is None:
            st.sidebar.warning("Echec calibration Pastas. Validation desactivee.")
            use_pastas_validation = False
        else:
            params = pastas_model.get_response_params()
            st.sidebar.success(f"Pastas calibre ({len(params)} params)")
    except Exception as e:
        st.sidebar.warning(f"Erreur Pastas: {e}")
        use_pastas_validation = False


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
test_raw_values = test_df[target_col].values * sigma_target + mu_target
test_dates = test_df.index

valid_end = test_len - H_model
if valid_end <= 0:
    st.error(f"Test set trop court ({test_len}j) pour H={H_model}.")
    st.stop()

window_pred_start = test_dates[start_idx]
window_pred_end = test_dates[min(start_idx + H_model - 1, test_len - 1)]

try:
    full_pred_loc = df_full.index.get_loc(window_pred_start)
except KeyError:
    full_pred_loc = df_full.index.searchsorted(window_pred_start)
    if full_pred_loc >= len(df_full):
        st.error(f"Date {window_pred_start} introuvable dans les donnees.")
        st.stop()
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
                model=darts_model, full_df=full_df_processed,
                target_col=target_col, covariate_cols=covariate_cols or None,
                preprocessing_config=preproc_config, scalers=scalers,
                start_date=window_pred_start, use_covariates=bool(covariate_cols),
                already_processed=True, is_global_model=is_global,
            )
            st.session_state[pred_cache_key] = {
                "prediction": results[0], "target": results[2], "metrics": results[3],
            }
        except Exception as e:
            st.error(f"Erreur prediction: {e}")
            st.code(traceback.format_exc())
            st.session_state[pred_cache_key] = None

cached = st.session_state.get(pred_cache_key)


# ====================
# SECTION 2: IPS bands + main chart
# ====================

st.subheader(f"2. Classification IPS-{selected_ips_window} et qualite de la prediction")

fig = go.Figure()

# Monthly IPS stacked bars as background (varies per calendar month)
add_monthly_ips_bars(fig, test_dates, ref_stats)

fig.add_vrect(x0=window_context_start, x1=window_pred_start,
    fillcolor="rgba(46,134,171,0.12)", layer="below", line_width=1,
    line=dict(color="rgba(46,134,171,0.3)"),
    annotation_text=f"Contexte ({L_model}j)", annotation_position="bottom left",
    annotation=dict(font_size=9))

fig.add_vrect(x0=window_pred_start, x1=window_pred_end,
    fillcolor="rgba(255,200,0,0.2)", layer="below", line_width=1,
    line=dict(color="rgba(255,200,0,0.5)"),
    annotation_text=f"Prediction ({H_model}j)", annotation_position="top right",
    annotation=dict(font_size=9))

fig.add_trace(go.Scatter(x=test_dates, y=test_raw_values,
    mode="lines", name="Verite terrain (m NGF)", line=dict(color="#2E86AB", width=2)))

if cached and cached.get("prediction") is not None:
    pred_ts = cached["prediction"]
    pred_values_raw = pred_ts.values().flatten()
    fig.add_trace(go.Scatter(x=pred_ts.time_index, y=pred_values_raw,
        mode="lines+markers", name="Prediction modele (m NGF)",
        line=dict(color="#E91E63", width=3), marker=dict(size=4)))

fig.update_layout(
    title=f"{target_col} - Test set avec bandes IPS-{selected_ips_window} (m NGF)",
    xaxis_title="Date", yaxis_title="Niveau piezometrique (m NGF)",
    height=500, hovermode="x unified",
    barmode="overlay",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig, use_container_width=True)

# --- Per-window IPS assessment (month-by-month) ---
if cached and cached.get("prediction") is not None and cached.get("target") is not None:
    pred_ts = cached["prediction"]
    target_ts = cached["target"]
    pred_values_raw = pred_ts.values().flatten()
    gt_values_raw = target_ts.values().flatten()

    # Month-by-month IPS classification
    gt_monthly_ips = classify_prediction_monthly(gt_values_raw, target_ts.time_index, ref_stats)
    pred_monthly_ips = classify_prediction_monthly(pred_values_raw, pred_ts.time_index, ref_stats)

    if not gt_monthly_ips.empty and not pred_monthly_ips.empty:
        st.markdown("**Classification IPS mois par mois**")

        # Build a display table merging GT and pred monthly IPS
        _ips_months = sorted(set(gt_monthly_ips["month_start"].tolist()) | set(pred_monthly_ips["month_start"].tolist()))
        _gt_map = {row["month_start"]: row for _, row in gt_monthly_ips.iterrows()}
        _pred_map = {row["month_start"]: row for _, row in pred_monthly_ips.iterrows()}

        ips_display_cols = st.columns(min(len(_ips_months), 6))
        n_match = 0
        n_total = 0
        for i, ms in enumerate(_ips_months):
            gt_row = _gt_map.get(ms)
            pred_row = _pred_map.get(ms)
            month_label = ms.strftime("%b %Y")

            with ips_display_cols[i % len(ips_display_cols)]:
                st.markdown(f"**{month_label}**")
                if gt_row is not None:
                    gt_cls = gt_row["ips_class"]
                    gt_z = gt_row["z_score"]
                    gt_lbl = IPS_LABELS.get(gt_cls, gt_cls)
                    gt_color = IPS_COLORS.get(gt_cls, "#888")
                    st.markdown(
                        f'<span style="color:{gt_color};font-weight:bold">GT: {gt_lbl}</span> '
                        f'<small>(z={gt_z:+.2f})</small>',
                        unsafe_allow_html=True,
                    )
                if pred_row is not None:
                    pred_cls = pred_row["ips_class"]
                    pred_z = pred_row["z_score"]
                    pred_lbl = IPS_LABELS.get(pred_cls, pred_cls)
                    pred_color = IPS_COLORS.get(pred_cls, "#888")
                    st.markdown(
                        f'<span style="color:{pred_color};font-weight:bold">Pred: {pred_lbl}</span> '
                        f'<small>(z={pred_z:+.2f})</small>',
                        unsafe_allow_html=True,
                    )

                if gt_row is not None and pred_row is not None:
                    n_total += 1
                    if gt_row["ips_class"] == pred_row["ips_class"]:
                        n_match += 1
                        st.caption("Correspondance")
                    else:
                        st.caption("Classe differente")

        # Overall match rate
        if n_total > 0:
            match_pct = n_match / n_total * 100
            col_summary_a, col_summary_b = st.columns(2)
            with col_summary_a:
                st.metric("Correspondance IPS mensuelle", f"{n_match}/{n_total} ({match_pct:.0f}%)",
                          delta="Bonne prediction" if match_pct >= 50 else "Predictions divergentes",
                          delta_color="normal" if match_pct >= 50 else "inverse")
            with col_summary_b:
                # Check if prediction spans multiple IPS classes
                pred_classes = pred_monthly_ips["ips_class"].unique().tolist()
                if len(pred_classes) > 1:
                    labels = [IPS_LABELS.get(c, c) for c in pred_classes]
                    st.info(f"La prediction couvre **{len(pred_classes)} classes IPS**: {', '.join(labels)}")
                elif len(pred_classes) == 1:
                    st.info(f"La prediction est entierement en classe **{IPS_LABELS.get(pred_classes[0], pred_classes[0])}**")

    if cached.get("metrics"):
        metrics = cached["metrics"]
        m_cols = st.columns(min(len(metrics), 4))
        for i, (name, val) in enumerate(metrics.items()):
            if val is not None and not pd.isna(val):
                with m_cols[i % len(m_cols)]:
                    st.metric(name, f"{val:.4f}")


# ====================
# SECTION 3: Context + CF generation
# ====================

st.markdown("---")
st.subheader(f"3. Generation contrefactuelle: {ips_from} -> {ips_to}")

# Pre-compute lookback data (needed BEFORE button for context display)
lookback_df = df_full.iloc[context_start_loc: context_start_loc + L_model]
horizon_df = df_full.iloc[context_start_loc + L_model: context_start_loc + L_model + H_model]

h_obs_norm = lookback_df[target_col].values.astype(np.float32)
s_obs_norm = lookback_df[covariate_cols].values.astype(np.float32) if covariate_cols else np.zeros((L_model, 1), dtype=np.float32)
months_arr = lookback_df.index.month.values.astype(np.int64)

s_obs_phys = s_obs_norm.copy()
for j, col in enumerate(covariate_cols):
    if col in physcf_scaler:
        mu_c = physcf_scaler[col]["mean"]
        sigma_c = physcf_scaler[col]["std"]
        if sigma_c > 0:
            s_obs_phys[:, j] = s_obs_norm[:, j] * sigma_c + mu_c

# IPS transition explanation
horizon_months = horizon_df.index.month.values if len(horizon_df) >= H_model else np.full(H_model, 6)
horizon_month_med = int(np.median(horizon_months)) if len(horizon_months) > 0 else 6
mu_m_h, sigma_m_h = ref_stats.get(horizon_month_med, (mu_target, sigma_target))
z_from_lo, z_from_hi = IPS_CLASSES[ips_from_key]
z_to_lo, z_to_hi = IPS_CLASSES[ips_to_key]
from_mid_raw = mu_m_h + ((max(z_from_lo, -5) + min(z_from_hi, 5)) / 2) * sigma_m_h
to_mid_raw = mu_m_h + ((max(z_to_lo, -5) + min(z_to_hi, 5)) / 2) * sigma_m_h
delta_m = to_mid_raw - from_mid_raw
direction = "baisser" if delta_m < 0 else "monter"

st.info(
    f"**Transition {ips_from} -> {ips_to}**: le niveau moyen doit {direction} "
    f"d'environ **{abs(delta_m):.2f} m** sur la fenetre de prediction "
    f"(de ~{from_mid_raw:.2f} m a ~{to_mid_raw:.2f} m NGF, mois ref: {horizon_month_med})."
)

# Context: current climate and PhysCF parameters
with st.expander("Contexte climatique et parametres PhysCF", expanded=False):
    # Climate context
    st.markdown("##### Conditions climatiques de la fenetre de contexte")
    if len(covariate_cols) >= 1:
        seasons_arr = np.array([MONTH_TO_SEASON_NAME.get(m, "?") for m in lookback_df.index.month])
        context_cols = st.columns(min(len(covariate_cols), 3))
        for j, col in enumerate(covariate_cols[:3]):
            with context_cols[j]:
                st.markdown(f"**{col}**")
                obs_vals = s_obs_phys[:, j]
                st.caption(f"Moy: {obs_vals.mean():.2f} | Min: {obs_vals.min():.2f} | Max: {obs_vals.max():.2f}")
                # Seasonal bar chart
                season_sums = {}
                for s_name in ["DJF", "MAM", "JJA", "SON"]:
                    mask = seasons_arr == s_name
                    if np.any(mask):
                        season_sums[s_name] = float(np.sum(obs_vals[mask]))
                    else:
                        season_sums[s_name] = 0.0
                fig_ctx = go.Figure(go.Bar(
                    x=list(season_sums.keys()), y=list(season_sums.values()),
                    marker_color=["#4169E1", "#228B22", "#FF8C00", "#DC143C"],
                ))
                fig_ctx.update_layout(height=180, margin=dict(l=20, r=20, t=10, b=20), yaxis_title="Cumul")
                st.plotly_chart(fig_ctx, use_container_width=True)

    # PhysCF parameter glossary
    st.markdown("##### Parametres PhysCF (couche de perturbation)")
    st.markdown("La couche de perturbation a **7 parametres** physiquement contraints:")

    col_precip = st.columns(4)
    for i, season in enumerate(["DJF", "MAM", "JJA", "SON"]):
        key = f"s_P_{season}"
        info = PHYSCF_PARAM_INFO[key]
        with col_precip[i]:
            st.markdown(f"**{info['label_short']}**")
            st.caption(f"[{info['range_min']}, {info['range_max']}] | Identite: {info['identity']}")
            st.markdown(f"_{info['explanation']}_", help=info["explanation"])

    col_other = st.columns(3)
    for i, key in enumerate(["delta_T", "delta_s", "delta_etp"]):
        info = PHYSCF_PARAM_INFO[key]
        with col_other[i]:
            st.markdown(f"**{info['label_short']}**")
            st.caption(f"[{info['range_min']}, {info['range_max']}] {info['unit']} | Identite: {info['identity']}")
            st.markdown(f"_{info['explanation']}_", help=info["explanation"])


# ---- CF Generation button ----
if not any([use_physcf, use_optuna, use_comet]):
    st.info("Cochez au moins une methode CF dans la sidebar (section 4).")

if st.button("Lancer la generation contrefactuelle", type="primary",
             use_container_width=True, disabled=not any([use_physcf, use_optuna, use_comet])):

    # Target bounds
    z_min, z_max = IPS_CLASSES[ips_to_key]
    z_min_c, z_max_c = max(z_min, -5.0), min(z_max, 5.0)

    lower_norm_arr = np.zeros(H_model, dtype=np.float32)
    upper_norm_arr = np.zeros(H_model, dtype=np.float32)
    lower_raw_arr = np.zeros(H_model)
    upper_raw_arr = np.zeros(H_model)

    for t in range(min(H_model, len(horizon_months))):
        m = int(horizon_months[t])
        mu_m, sigma_m = ref_stats.get(m, (mu_target, sigma_target))
        lower_raw_arr[t] = mu_m + z_min_c * sigma_m if sigma_m > 0 else mu_m
        upper_raw_arr[t] = mu_m + z_max_c * sigma_m if sigma_m > 0 else mu_m
        if sigma_target > 0:
            lower_norm_arr[t] = (lower_raw_arr[t] - mu_target) / sigma_target
            upper_norm_arr[t] = (upper_raw_arr[t] - mu_target) / sigma_target
        else:
            lower_norm_arr[t], upper_norm_arr[t] = z_min_c, z_max_c

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
    status_container = st.empty()

    for i, (display_name, method_key) in enumerate(methods_to_run.items()):
        with status_container.container():
            st.markdown(f"**Optimisation {display_name}** ({i+1}/{len(methods_to_run)})")
            st.caption(f"Parametres: {param_count(method_key, L_model)} | Iterations: {n_iter}")
        progress.progress(i / len(methods_to_run))

        try:
            h_t = torch.tensor(h_obs_norm, dtype=torch.float32)
            months_t = torch.tensor(months_arr, dtype=torch.long)

            if method_key == "physcf_gradient":
                result = generate_counterfactual(
                    h_t, torch.tensor(s_obs_phys, dtype=torch.float32),
                    model_adapter, (lower_norm, upper_norm),
                    physcf_scaler, months_t,
                    lambda_prox=lambda_prox, n_iter=n_iter, lr=lr_cf, device=device)
            elif method_key == "physcf_optuna":
                result = generate_counterfactual_optuna(
                    h_t, torch.tensor(s_obs_phys, dtype=torch.float32),
                    model_adapter, (lower_norm, upper_norm),
                    physcf_scaler, months_t,
                    lambda_prox=lambda_prox, n_trials=n_iter, device=device)
            elif method_key == "comet_hydro":
                result = generate_counterfactual_comet(
                    h_t, torch.tensor(s_obs_norm, dtype=torch.float32),
                    model_adapter, (lower_norm, upper_norm),
                    physcf_scaler, n_iter=n_iter, lr=lr_cf, device=device)

            result["method_key"] = method_key
            results_dict[display_name] = result
        except Exception as e:
            st.error(f"Erreur {display_name}: {e}")
            st.code(traceback.format_exc())
            if "dtype" in str(e).lower() or "float" in str(e).lower():
                st.info("Essayez de re-entrainer le modele ou verifiez la compatibilite des types de donnees.")
            elif "shape" in str(e).lower() or "dimension" in str(e).lower():
                st.info("Verifiez que le modele a ete entraine avec les memes covariables.")
            elif "memory" in str(e).lower() or "cuda" in str(e).lower():
                st.info("Essayez de reduire le nombre d'iterations ou de passer en mode CPU.")

    progress.progress(1.0)
    status_container.empty()
    progress.empty()

    total_time = sum(r.get("wall_clock_s", 0) for r in results_dict.values())

    # ---- Pastas dual validation ----
    pastas_validation_results = {}
    n_accepted = 0
    n_total = 0
    if use_pastas_validation and pastas_model is not None and results_dict:
        with st.spinner("Validation Pastas (dual model)..."):
            try:
                # Get factual TFT prediction (denormalized)
                _cached_pred = st.session_state.get(pred_cache_key)
                if _cached_pred and _cached_pred.get("prediction") is not None:
                    _y_factual_tft_raw = _cached_pred["prediction"].values().flatten()
                else:
                    _y_factual_tft_raw = test_raw_values[start_idx: start_idx + H_model]

                pastas_validation_results = run_dual_validation_for_results(
                    results_dict=results_dict,
                    pastas_model=pastas_model,
                    lookback_dates=lookback_df.index,
                    y_factual_tft_raw=_y_factual_tft_raw,
                    horizon_start=str(window_pred_start.date()),
                    horizon_end=str(window_pred_end.date()),
                    mu_target=mu_target,
                    sigma_target=sigma_target,
                    gamma=gamma_pastas,
                )
                n_accepted = sum(1 for v in pastas_validation_results.values() if v.get("accepted"))
                n_total = len(pastas_validation_results)
                total_time += 0.1  # negligible overhead
            except Exception as e:
                st.warning(f"Erreur validation Pastas: {e}")

    pastas_msg = ""
    if pastas_validation_results:
        pastas_msg = f" | Pastas: {n_accepted}/{n_total} accepte(s)"
    st.success(f"Analyse terminee en {total_time:.1f}s | {len(results_dict)} methode(s) | Scenario: {ips_from} -> {ips_to}{pastas_msg}")

    # Store in session state
    st.session_state["cf_results_latest"] = results_dict
    st.session_state["cf_pastas_validation"] = pastas_validation_results
    st.session_state["cf_context_latest"] = {
        "lookback_df": lookback_df,
        "s_obs_phys": s_obs_phys,
        "s_obs_norm": s_obs_norm,
        "horizon_months": horizon_months,
        "lower_norm": lower_norm,
        "upper_norm": upper_norm,
        "lower_raw_arr": lower_raw_arr,
        "upper_raw_arr": upper_raw_arr,
        "ips_from": ips_from,
        "ips_to": ips_to,
        "ips_from_key": ips_from_key,
        "ips_to_key": ips_to_key,
    }

    if not results_dict:
        st.error("Aucune methode n'a reussi.")
        st.stop()


# ====================
# SECTION 4: Display results (if available)
# ====================

if "cf_results_latest" in st.session_state and st.session_state["cf_results_latest"]:
    results_dict = st.session_state["cf_results_latest"]
    ctx = st.session_state.get("cf_context_latest", {})

    _lookback_df = ctx.get("lookback_df", lookback_df)
    _s_obs_phys = ctx.get("s_obs_phys", s_obs_phys)
    _lower_norm = ctx.get("lower_norm")
    _upper_norm = ctx.get("upper_norm")
    _lower_raw_arr = ctx.get("lower_raw_arr", np.zeros(H_model))
    _upper_raw_arr = ctx.get("upper_raw_arr", np.zeros(H_model))
    _horizon_months = ctx.get("horizon_months", np.full(H_model, 6))
    _ips_from = ctx.get("ips_from", ips_from)
    _ips_to = ctx.get("ips_to", ips_to)
    _ips_from_key = ctx.get("ips_from_key", ips_from_key)
    _ips_to_key = ctx.get("ips_to_key", ips_to_key)

    st.markdown("---")
    st.subheader("4. Resultats contrefactuels")

    # ---- Tabbed results ----
    tab_summary, tab_scenario, tab_climate, tab_convergence, tab_export = st.tabs([
        "Resume", "Interpretation du scenario", "Comparaison climatique", "Convergence", "Export",
    ])

    # ========== TAB: Resume ==========
    with tab_summary:
        # Metrics table
        rows = []
        for mn, res in results_dict.items():
            row = {"Methode": mn}
            y_cf = res.get("y_cf")
            if y_cf is not None and _lower_norm is not None:
                val = validity_ratio(y_cf, _lower_norm, _upper_norm)
                row["Validite"] = f"{val:.0%}"
            if "theta_star" in res:
                prox = proximity_theta(res["theta_star"])
                cc = cc_compliance_from_theta(res["theta_star"])
                row["Proximite"] = f"{prox:.4f}"
                row["CC residuel"] = f"{cc:.6f}"
            row["Converge"] = "Oui" if res.get("converged") else "Non"
            row["Temps"] = f"{res.get('wall_clock_s', 0):.1f}s"
            row["Params"] = param_count(res.get("method_key", mn), L_model)
            # Pastas validation columns
            pv = st.session_state.get("cf_pastas_validation", {}).get(mn)
            if pv:
                row["Pastas RMSE"] = f"{pv['rmse_cf']:.3f}"
                row["Pastas Seuil"] = f"{pv['epsilon']:.3f}"
                row["Pastas"] = "Accepte" if pv["accepted"] else "Rejete"
            rows.append(row)

        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("Methode"), use_container_width=True)

        # Plausibility indicators
        st.markdown("#### Indicateurs de plausibilite physique")
        for mn, res in results_dict.items():
            st.markdown(f"**{mn}**")
            ind_cols = st.columns(6)
            y_cf = res.get("y_cf")
            if y_cf is not None and _lower_norm is not None:
                val = validity_ratio(y_cf, _lower_norm, _upper_norm)
                with ind_cols[0]:
                    st.metric("Validite", f"{val:.0%}",
                              delta="OK" if val >= 0.95 else "Partiel" if val >= 0.8 else "Insuffisant",
                              delta_color="normal" if val >= 0.95 else "inverse")
            if "theta_star" in res:
                prox = proximity_theta(res["theta_star"])
                cc = cc_compliance_from_theta(res["theta_star"])
                with ind_cols[1]:
                    st.metric("Proximite", f"{prox:.3f}",
                              delta="Minimal" if prox <= 0.1 else "Modere" if prox <= 0.5 else "Fort",
                              delta_color="normal" if prox <= 0.1 else "inverse")
                with ind_cols[2]:
                    st.metric("CC Conformite", f"{cc:.4f}",
                              delta="Conforme" if cc <= 0.01 else "Acceptable" if cc <= 0.03 else "Violation",
                              delta_color="normal" if cc <= 0.01 else "inverse")
            with ind_cols[3]:
                conv = res.get("converged", False)
                st.metric("Convergence", "Oui" if conv else "Non",
                          delta_color="normal" if conv else "inverse",
                          delta="OK" if conv else "Non converge")
            if "theta_star" in res:
                theta = res["theta_star"]
                warnings = []
                if all(theta.get(f"s_P_{s}", 1.0) < 0.6 for s in ["DJF", "MAM", "JJA", "SON"]):
                    warnings.append("Secheresse toutes saisons")
                if abs(theta.get("delta_s", 0)) > 20:
                    warnings.append("Decalage > 20j")
                if abs(theta.get("delta_T", 0)) > 4:
                    warnings.append("Temperature extreme")
                with ind_cols[4]:
                    if not warnings:
                        st.metric("Plausibilite", "OK", delta="Scenario realiste", delta_color="normal")
                    else:
                        st.metric("Plausibilite", "Attention",
                                  delta="; ".join(warnings), delta_color="inverse")
            # Pastas agreement indicator (always show column 5)
            pv = st.session_state.get("cf_pastas_validation", {}).get(mn)
            with ind_cols[5]:
                if pv:
                    rmse_str = f"{pv['rmse_cf']:.3f}"
                    if pv["accepted"]:
                        st.metric("Accord Pastas", rmse_str,
                                  delta=f"< {pv['epsilon']:.3f} (accepte)",
                                  delta_color="normal")
                    else:
                        st.metric("Accord Pastas", rmse_str,
                                  delta=f"> {pv['epsilon']:.3f} (rejete)",
                                  delta_color="inverse")
                else:
                    st.metric("Accord Pastas", "N/A",
                              delta="Non active", delta_color="off")

        # CF overlay chart
        gt_window_raw = test_raw_values[start_idx: start_idx + H_model]
        gt_dates = test_dates[start_idx: start_idx + H_model]
        pred_raw_for_chart = None
        pred_dates_for_chart = None
        if cached and cached.get("prediction") is not None:
            pred_ts_c = cached["prediction"]
            pred_raw_for_chart = pred_ts_c.values().flatten() * sigma_target + mu_target
            pred_dates_for_chart = pred_ts_c.time_index

        fig_cf = plot_cf_overlay(
            gt_dates, gt_window_raw, pred_raw_for_chart, pred_dates_for_chart,
            results_dict, _lower_raw_arr, _upper_raw_arr,
            _ips_to_key, _ips_to, ref_stats,
            mu_target, sigma_target, _horizon_months,
        )

        # Overlay Pastas CF predictions (dashdot lines)
        _pastas_val = st.session_state.get("cf_pastas_validation", {})
        for mn in results_dict:
            pv = _pastas_val.get(mn)
            if pv and pv.get("y_cf_pastas") is not None:
                y_pastas = pv["y_cf_pastas"]
                _color = CF_METHOD_COLORS.get(results_dict[mn].get("method_key", ""), "#888888")
                fig_cf.add_trace(go.Scatter(
                    x=gt_dates[:len(y_pastas)],
                    y=y_pastas[:len(gt_dates)],
                    mode="lines",
                    name=f"Pastas CF ({mn})",
                    line=dict(color=_color, width=1.5, dash="dashdot"),
                    opacity=0.6,
                ))

        st.plotly_chart(fig_cf, use_container_width=True)

    # ========== TAB: Scenario interpretation ==========
    with tab_scenario:
        theta_dicts = {mn: res["theta_star"] for mn, res in results_dict.items() if "theta_star" in res}

        if theta_dicts:
            col_radar, col_table = st.columns([1, 1])
            with col_radar:
                fig_radar = plot_theta_radar(theta_dicts)
                st.plotly_chart(fig_radar, use_container_width=True)
            with col_table:
                theta_rows = []
                for mn, theta in theta_dicts.items():
                    theta_rows.append({"Methode": mn, **theta})
                st.dataframe(pd.DataFrame(theta_rows).set_index("Methode").round(3), use_container_width=True)

            # Natural language narrative
            st.markdown("#### Interpretation")
            for mn, theta in theta_dicts.items():
                narrative = generate_cf_narrative(theta, _ips_from, _ips_to)
                st.markdown(f"**{mn}**: {narrative}")
        else:
            st.info("Aucune methode PhysCF selectionnee (COMET-Hydro n'a pas de theta structuree).")

    # ========== TAB: Climate comparison ==========
    with tab_climate:
        for mn, res in results_dict.items():
            # Handle both PhysCF (s_cf_phys) and COMET (s_cf_norm)
            s_cf_data = None
            if "s_cf_phys" in res:
                s_cf_data = res["s_cf_phys"].numpy() if hasattr(res["s_cf_phys"], "numpy") else np.array(res["s_cf_phys"])
            elif "s_cf_norm" in res:
                s_cf_norm_arr = res["s_cf_norm"].numpy() if hasattr(res["s_cf_norm"], "numpy") else np.array(res["s_cf_norm"])
                s_cf_data = s_cf_norm_arr.copy()
                for j, col in enumerate(covariate_cols):
                    if col in physcf_scaler:
                        s_cf_data[:, j] = s_cf_norm_arr[:, j] * physcf_scaler[col]["std"] + physcf_scaler[col]["mean"]

            if s_cf_data is None:
                continue

            st.markdown(f"#### {mn}")
            col_ts, col_season = st.columns([2, 1])

            with col_ts:
                fig_s = plot_stress_comparison(
                    _lookback_df.index, _s_obs_phys, s_cf_data, covariate_cols, mn,
                )
                st.plotly_chart(fig_s, use_container_width=True)

            with col_season:
                st.markdown("**Bilan saisonnier**")
                season_df = compute_seasonal_summary(
                    _s_obs_phys, s_cf_data,
                    _lookback_df.index.month.values, covariate_cols,
                )
                for _, row in season_df.iterrows():
                    st.metric(
                        f"{row['covariate'][:15]} {row['season']}",
                        f"{row['cf_total']:.0f}",
                        delta=f"{row['delta_pct']:+.0f}%",
                        delta_color="normal" if row['delta_pct'] >= 0 else "inverse",
                    )

    # ========== TAB: Convergence ==========
    with tab_convergence:
        has_loss = any("loss_history" in r for r in results_dict.values())
        if has_loss:
            fig_l = plot_convergence(results_dict)
            st.plotly_chart(fig_l, use_container_width=True)
        else:
            st.info("Pas de courbe de convergence disponible pour les methodes selectionnees.")

    # ========== TAB: Export ==========
    with tab_export:
        st.markdown("#### Export CSV")
        metadata = {
            "model": selected_model_name,
            "station": getattr(model_entry, "primary_station", "unknown"),
            "ips_from": _ips_from_key,
            "ips_to": _ips_to_key,
            "ips_window": selected_ips_window,
            "window_start": window_pred_start.strftime("%Y-%m-%d"),
            "window_end": window_pred_end.strftime("%Y-%m-%d"),
            "lambda_prox": lambda_prox,
            "n_iter": n_iter,
            "lr": lr_cf,
        }
        if _lower_norm is not None:
            export_df = build_cf_export_df(results_dict, metadata, _lower_norm, _upper_norm, L_model)
            st.dataframe(export_df, use_container_width=True)
            st.download_button(
                label="Telecharger CSV",
                data=export_df.to_csv(index=False),
                file_name=f"physcf_cf_{metadata.get('station', 'unknown')}_{_ips_from_key}_to_{_ips_to_key}.csv",
                mime="text/csv",
            )

        # LaTeX export
        st.markdown("#### LaTeX (pour le papier)")
        try:
            _pastas_val_export = st.session_state.get("cf_pastas_validation", {})
            _has_pastas = bool(_pastas_val_export)
            latex_rows = []
            for mn, res in results_dict.items():
                y_cf = res.get("y_cf")
                v = validity_ratio(y_cf, _lower_norm, _upper_norm) if y_cf is not None and _lower_norm is not None else 0
                prox = proximity_theta(res["theta_star"]) if "theta_star" in res else "-"
                cc = cc_compliance_from_theta(res["theta_star"]) if "theta_star" in res else "-"
                n_p = param_count(res.get("method_key", mn), L_model)
                conv = "Y" if res.get("converged") else "N"
                t_s = f"{res.get('wall_clock_s', 0):.1f}"
                pv_export = _pastas_val_export.get(mn)
                pastas_str = f"{pv_export['rmse_cf']:.3f}" if pv_export else "-"
                if _has_pastas:
                    latex_rows.append(f"  {mn} & {v:.2f} & {prox:.3f} & {cc:.3f} & {pastas_str} & {n_p} & {conv} & {t_s}s \\\\")
                else:
                    latex_rows.append(f"  {mn} & {v:.2f} & {prox:.3f} & {cc:.3f} & {n_p} & {conv} & {t_s}s \\\\")

            if _has_pastas:
                _tabular = "\\begin{tabular}{l c c c c c c c}\n"
                _header = "Method & Valid. & Prox. & CC & Pastas & $|\\theta|$ & Conv. & Time \\\\\n"
            else:
                _tabular = "\\begin{tabular}{l c c c c c c}\n"
                _header = "Method & Valid. & Prox. & CC & $|\\theta|$ & Conv. & Time \\\\\n"

            latex_str = (
                "\\begin{table}[t]\n"
                "\\centering\n"
                f"\\caption{{CF results: {_ips_from} $\\to$ {_ips_to} (IPS-{selected_ips_window})}}\n"
                "\\label{tab:cf_results}\n"
                + _tabular
                + "\\hline\n"
                + _header
                + "\\hline\n"
                + "\n".join(latex_rows) + "\n"
                "\\hline\n"
                "\\end{tabular}\n"
                "\\end{table}"
            )
            st.code(latex_str, language="latex")
        except Exception as e:
            st.warning(f"Erreur generation LaTeX: {e}")

        # Session history
        if "cf_results_history" not in st.session_state:
            st.session_state["cf_results_history"] = []
        run_entry = {
            "timestamp": pd.Timestamp.now().isoformat()[:19],
            "model": selected_model_name,
            "scenario": f"{_ips_from} -> {_ips_to}",
            "methods": list(results_dict.keys()),
            "n_methods": len(results_dict),
        }
        # Avoid duplicate entries
        existing_timestamps = [e["timestamp"] for e in st.session_state["cf_results_history"]]
        if run_entry["timestamp"] not in existing_timestamps:
            st.session_state["cf_results_history"].append(run_entry)

        if len(st.session_state["cf_results_history"]) > 1:
            st.markdown("#### Historique des runs (session)")
            st.dataframe(pd.DataFrame(st.session_state["cf_results_history"]), use_container_width=True)
