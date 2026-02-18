"""Counterfactual Analysis Page - PhysCF Integration.

This page allows users to:
1. Load a trained model from the registry
2. Select a test window and target IPS class
3. Generate counterfactual scenarios using PhysCF (gradient), Optuna, or COMET-Hydro
4. Visualize factual vs counterfactual predictions
5. Interpret the perturbation parameters (theta*)
"""

import sys
import time
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

from dashboard.utils.model_registry import get_registry

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
from dashboard.utils.counterfactual.darts_adapter import (
    DartsModelAdapter,
    StandaloneGRUAdapter,
)

# ---- Page Config ----
st.set_page_config(page_title="PhysCF - Counterfactual Analysis", layout="wide")
st.title("PhysCF - Analyse Contrefactuelle")
st.markdown("""
Generez des scenarios meteorologiques contrefactuels pour expliquer les previsions
de niveau piezometrique. PhysCF optimise **7 parametres physiques** (precipitation saisonniere,
temperature, ETP, decalage temporel) avec le couplage Clausius-Clapeyron en contrainte dure.
""")

# ---- Sidebar: Model & Data Selection ----
st.sidebar.header("Configuration")

# Check if data is in session
if "df" not in st.session_state or st.session_state.df is None:
    st.warning("Chargez d'abord des donnees dans l'onglet Dataset Preparation.")
    st.stop()

df = st.session_state.df

# Model selection from registry
registry = get_registry()
models_list = registry.list_all_models() if hasattr(registry, "list_all_models") else []

if not models_list:
    st.sidebar.warning("Aucun modele dans le registry MLflow.")

model_display = {m.display_name: m for m in models_list}
selected_model_name = st.sidebar.selectbox(
    "Modele entraine",
    list(model_display.keys()) if model_display else ["(aucun)"],
    help="Selectionnez un modele entraine depuis MLflow."
)

# ---- Method Selection ----
st.sidebar.subheader("Methode CF")
cf_method = st.sidebar.selectbox(
    "Methode de generation",
    ["PhysCF (gradient)", "PhysCF (Optuna)", "COMET-Hydro", "Comparer les 3"],
    help="PhysCF: 7 params physiques. COMET-Hydro: L x 3 params bruts."
)

# ---- Hyperparameters ----
st.sidebar.subheader("Hyperparametres")
lambda_prox = st.sidebar.slider(
    "lambda_prox (regularisation)",
    min_value=0.001, max_value=2.0, value=0.1, step=0.01,
    help="Poids de la proximite. Bas = plus de liberte pour les perturbations."
)
n_iter = st.sidebar.number_input("Iterations", min_value=50, max_value=2000, value=500, step=50)
lr = st.sidebar.number_input(
    "Learning rate", min_value=0.001, max_value=0.1, value=0.02,
    step=0.005, format="%.3f"
)

# ---- Target IPS Class ----
st.sidebar.subheader("Classe IPS cible")
target_ips = st.sidebar.selectbox(
    "Classe cible",
    list(IPS_CLASSES.keys()),
    index=3,  # 'normal' by default
    help="Classe IPS que le contrefactuel doit atteindre."
)

# ---- Device selection ----
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.caption(f"Device: **{device}** {'(GPU)' if device == 'cuda' else '(CPU)'}")


# ====================
# Helper functions
# ====================

def _get_columns():
    """Get target and covariate column names from session state."""
    target_col = st.session_state.get("target_col", None)
    covariate_cols = st.session_state.get("covariate_cols", None)

    # Auto-detect if not set
    if target_col is None:
        # Common target column names
        for candidate in ["gwl", "Water_Level", "water_level", "piezo", "level"]:
            if candidate in df.columns:
                target_col = candidate
                break
        if target_col is None:
            target_col = df.columns[0]

    if covariate_cols is None:
        # Common covariate patterns
        covariate_cols = []
        for candidate in ["precip", "Precipitation", "temp", "Temperature", "evap", "ETP", "etp"]:
            if candidate in df.columns:
                covariate_cols.append(candidate)
        if not covariate_cols:
            covariate_cols = [c for c in df.columns if c != target_col][:3]

    return target_col, covariate_cols


def _build_scaler_from_df(df_train: pd.DataFrame, cols: list) -> dict:
    """Build a simple z-score scaler dict from training data."""
    scaler = {}
    for col in cols:
        if col in df_train.columns:
            scaler[col] = {
                "mean": float(df_train[col].mean()),
                "std": float(df_train[col].std()),
            }
    return scaler


def _normalize(arr: np.ndarray, scaler: dict, cols: list) -> np.ndarray:
    """Z-score normalize array using scaler."""
    out = arr.copy()
    for j, col in enumerate(cols):
        if col in scaler:
            mu = scaler[col]["mean"]
            sigma = scaler[col]["std"]
            if sigma > 0:
                out[..., j] = (out[..., j] - mu) / sigma
    return out


def _denormalize_gwl(arr: np.ndarray, scaler: dict, target_col: str) -> np.ndarray:
    """Inverse z-score for gwl."""
    if target_col in scaler:
        mu = scaler[target_col]["mean"]
        sigma = scaler[target_col]["std"]
        return arr * sigma + mu
    return arr


def _extract_window(
    df: pd.DataFrame, start_idx: int, L: int, H: int,
    target_col: str, covariate_cols: list
) -> tuple:
    """Extract a (h_obs, s_obs, y_true, months, dates) window from the dataframe."""
    lookback_df = df.iloc[start_idx : start_idx + L]
    horizon_df = df.iloc[start_idx + L : start_idx + L + H]

    h_obs = lookback_df[target_col].values.astype(np.float32)
    s_obs = lookback_df[covariate_cols].values.astype(np.float32)
    y_true = horizon_df[target_col].values.astype(np.float32) if len(horizon_df) >= H else None
    months = lookback_df.index.month.values.astype(np.int64)
    dates_lookback = lookback_df.index
    dates_horizon = horizon_df.index

    return h_obs, s_obs, y_true, months, dates_lookback, dates_horizon


def _run_single_cf(
    method: str, h_obs_norm, s_obs_phys, s_obs_norm, model_adapter,
    target_bounds, scaler, months, L, H,
    lambda_prox, n_iter, lr, device
) -> dict:
    """Run a single counterfactual method and return results."""
    h_t = torch.tensor(h_obs_norm, dtype=torch.float32)
    months_t = torch.tensor(months, dtype=torch.long)
    lower_t, upper_t = target_bounds

    if method == "physcf_gradient":
        s_phys_t = torch.tensor(s_obs_phys, dtype=torch.float32)
        result = generate_counterfactual(
            h_t, s_phys_t, model_adapter, (lower_t, upper_t),
            scaler, months_t,
            lambda_prox=lambda_prox, n_iter=n_iter, lr=lr, device=device,
        )
    elif method == "physcf_optuna":
        s_phys_t = torch.tensor(s_obs_phys, dtype=torch.float32)
        result = generate_counterfactual_optuna(
            h_t, s_phys_t, model_adapter, (lower_t, upper_t),
            scaler, months_t,
            lambda_prox=lambda_prox, n_trials=n_iter, device=device,
        )
    elif method == "comet_hydro":
        s_norm_t = torch.tensor(s_obs_norm, dtype=torch.float32)
        result = generate_counterfactual_comet(
            h_t, s_norm_t, model_adapter, (lower_t, upper_t),
            scaler, n_iter=n_iter, lr=lr, device=device,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return result


def _plot_factual_vs_cf(
    dates_lookback, dates_horizon, h_obs_raw, y_factual_raw,
    y_cf_raw, target_bounds_raw, target_col, method_name
):
    """Create a plotly figure comparing factual vs CF predictions."""
    fig = make_subplots(rows=1, cols=1)

    # Lookback period
    fig.add_trace(go.Scatter(
        x=dates_lookback, y=h_obs_raw,
        name="Historique (lookback)", mode="lines",
        line=dict(color="gray", width=1),
    ))

    # Factual prediction
    fig.add_trace(go.Scatter(
        x=dates_horizon, y=y_factual_raw,
        name="Prediction factuelle", mode="lines",
        line=dict(color="blue", width=2),
    ))

    # CF prediction
    fig.add_trace(go.Scatter(
        x=dates_horizon, y=y_cf_raw,
        name=f"Contrefactuel ({method_name})", mode="lines",
        line=dict(color="red", width=2, dash="dash"),
    ))

    # Target bounds
    lower_raw, upper_raw = target_bounds_raw
    fig.add_trace(go.Scatter(
        x=dates_horizon, y=lower_raw,
        name="Borne inf IPS", mode="lines",
        line=dict(color="green", width=1, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=dates_horizon, y=upper_raw,
        name="Borne sup IPS", mode="lines",
        line=dict(color="green", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(0,255,0,0.08)",
    ))

    fig.update_layout(
        title=f"Factuel vs Contrefactuel - {method_name}",
        xaxis_title="Date",
        yaxis_title=f"{target_col} (m NGF)",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def _plot_stress_comparison(
    dates_lookback, s_obs_phys, s_cf_phys, covariate_cols
):
    """Plot original vs CF stresses (precipitation, temperature, ETP)."""
    n_cov = min(len(covariate_cols), s_obs_phys.shape[-1])
    fig = make_subplots(rows=n_cov, cols=1, shared_xaxes=True,
                        subplot_titles=[c for c in covariate_cols[:n_cov]])

    colors = ["blue", "orange", "purple"]
    for j in range(n_cov):
        fig.add_trace(go.Scatter(
            x=dates_lookback, y=s_obs_phys[:, j],
            name=f"{covariate_cols[j]} (obs)", mode="lines",
            line=dict(color=colors[j % 3], width=1),
            showlegend=(j == 0),
            legendgroup="obs",
        ), row=j + 1, col=1)

        if s_cf_phys is not None:
            fig.add_trace(go.Scatter(
                x=dates_lookback, y=s_cf_phys[:, j],
                name=f"{covariate_cols[j]} (CF)", mode="lines",
                line=dict(color=colors[j % 3], width=1, dash="dash"),
                showlegend=(j == 0),
                legendgroup="cf",
            ), row=j + 1, col=1)

    fig.update_layout(
        title="Stresses: Factuel vs Contrefactuel",
        height=150 * n_cov + 100,
    )
    return fig


def _plot_loss_curves(results_dict: dict):
    """Plot convergence curves for all methods."""
    fig = go.Figure()
    for method, result in results_dict.items():
        if "loss_history" in result:
            fig.add_trace(go.Scatter(
                y=result["loss_history"], name=f"{method} (total)",
                mode="lines",
            ))
        if "target_history" in result:
            fig.add_trace(go.Scatter(
                y=result["target_history"], name=f"{method} (target)",
                mode="lines", line=dict(dash="dash"),
            ))
    fig.update_layout(
        title="Courbes de convergence",
        xaxis_title="Iteration", yaxis_title="Loss",
        yaxis_type="log", height=350,
    )
    return fig


def _make_theta_table(results_dict: dict) -> pd.DataFrame:
    """Create a comparison table of theta* parameters."""
    rows = []
    for method, result in results_dict.items():
        if "theta_star" in result:
            row = {"Methode": method, **result["theta_star"]}
            rows.append(row)
    if rows:
        return pd.DataFrame(rows).set_index("Methode").round(3)
    return pd.DataFrame()


def _make_metrics_table(results_dict: dict, target_bounds) -> pd.DataFrame:
    """Create a comparison table of metrics."""
    lower, upper = target_bounds
    rows = []
    for method, result in results_dict.items():
        row = {"Methode": method}
        y_cf = result.get("y_cf")
        if y_cf is not None:
            row["Validite"] = validity_ratio(y_cf, lower, upper)
        if "theta_star" in result:
            row["Proximite"] = proximity_theta(result["theta_star"])
            row["CC compliance"] = cc_compliance(result["theta_star"])
        row["Converge"] = result.get("converged", False)
        row["Temps (s)"] = round(result.get("wall_clock_s", 0), 2)
        row["N params"] = param_count(result.get("method", method))
        rows.append(row)
    if rows:
        return pd.DataFrame(rows).set_index("Methode")
    return pd.DataFrame()


# ====================
# Main Content
# ====================

target_col, covariate_cols = _get_columns()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Donnees et fenetre de test")
    st.write(f"**Cible:** {target_col} | **Covariables:** {', '.join(covariate_cols)}")
    st.write(f"**Periode:** {df.index.min().date()} - {df.index.max().date()} ({len(df)} jours)")

    # Window selection
    L = st.number_input("Lookback L (jours)", min_value=30, max_value=730, value=365, step=30)
    H = st.number_input("Horizon H (jours)", min_value=7, max_value=365, value=90, step=7)

    max_start = len(df) - L - H
    if max_start <= 0:
        st.error(f"Donnees insuffisantes pour L={L} + H={H} = {L + H} jours.")
        st.stop()

    # Date-based window selection
    min_date = df.index[L].date()
    max_date = df.index[max_start + L - 1].date()
    window_start_date = st.date_input(
        "Date de debut de la fenetre de prevision",
        value=max_date,  # Default to latest possible
        min_value=min_date,
        max_value=max_date,
    )

    # Find the index for this date
    window_start_dt = pd.Timestamp(window_start_date)
    window_start_idx = df.index.searchsorted(window_start_dt) - L

with col2:
    st.subheader("PerturbationLayer (PhysCF)")
    st.markdown("""
    | Param | Bornes |
    |-------|--------|
    | s_P (x4 saisons) | [0.3, 2.0] |
    | delta_T | [-5, +5] C |
    | delta_ETP | CC coupling |
    | delta_s | [-30, +30] j |
    """)
    st.caption(f"IPS cible: **{target_ips}** "
               f"(z in [{IPS_CLASSES[target_ips][0]:.2f}, {IPS_CLASSES[target_ips][1]:.2f}])")


# ---- Run Counterfactual ----
st.markdown("---")

if st.button("Generer les contrefactuels", type="primary", use_container_width=True):

    # Extract data window
    h_obs_raw, s_obs_raw, y_true_raw, months_arr, dates_lookback, dates_horizon = \
        _extract_window(df, window_start_idx, L, H, target_col, covariate_cols)

    # Build scaler from training data (first 70% of the series)
    train_end = int(len(df) * 0.7)
    df_train = df.iloc[:train_end]
    scaler = _build_scaler_from_df(df_train, [target_col] + covariate_cols)

    # Normalize
    h_obs_norm = h_obs_raw.copy()
    if target_col in scaler:
        mu_t, sigma_t = scaler[target_col]["mean"], scaler[target_col]["std"]
        if sigma_t > 0:
            h_obs_norm = (h_obs_norm - mu_t) / sigma_t

    s_obs_norm = _normalize(
        s_obs_raw.reshape(-1, len(covariate_cols)), scaler, covariate_cols
    )

    # IPS reference stats (for bounds)
    gwl_series = df[target_col].dropna()
    ref_stats = compute_ips_reference(gwl_series)

    # Target bounds in normalized units
    z_min, z_max = IPS_CLASSES[target_ips]
    # Clamp infinities to +-5 sigma
    z_min_clamped = max(z_min, -5.0)
    z_max_clamped = min(z_max, 5.0)
    lower_norm = torch.full((H,), z_min_clamped, dtype=torch.float32)
    upper_norm = torch.full((H,), z_max_clamped, dtype=torch.float32)

    # Target bounds in raw units (for display) - use mean month of horizon
    horizon_months = dates_horizon.month.values if len(dates_horizon) >= H else np.array([6] * H)
    lower_raw_arr = np.zeros(H)
    upper_raw_arr = np.zeros(H)
    for t in range(min(H, len(horizon_months))):
        m = int(horizon_months[t])
        mu_m, sigma_m = ref_stats.get(m, (0, 1))
        if sigma_m > 0:
            lower_raw_arr[t] = mu_m + z_min_clamped * sigma_m
            upper_raw_arr[t] = mu_m + z_max_clamped * sigma_m
        else:
            lower_raw_arr[t] = mu_m
            upper_raw_arr[t] = mu_m

    # ---- Load or create model adapter ----
    model_adapter = None

    if selected_model_name != "(aucun)" and selected_model_name in model_display:
        model_entry = model_display[selected_model_name]
        with st.spinner(f"Chargement du modele {model_entry.model_name}..."):
            try:
                darts_model = registry.load_model(model_entry)
                model_adapter = DartsModelAdapter(
                    darts_model,
                    input_chunk_length=L,
                    output_chunk_length=H,
                )
                st.success(f"Modele {model_entry.model_name} charge (type: {model_adapter._model_type})")
            except Exception as e:
                st.error(f"Erreur chargement modele: {e}")
                st.code(traceback.format_exc())

    if model_adapter is None:
        st.warning("Pas de modele charge. Utilisation d'un GRU factice pour la demo.")

        # Create a simple demo model
        class DemoGRU(torch.nn.Module):
            def __init__(self, H):
                super().__init__()
                self.gru = torch.nn.GRU(4, 64, batch_first=True)
                self.fc = torch.nn.Linear(64, H)

            def forward(self, h_obs, s_obs):
                if h_obs.dim() == 2:
                    h_obs = h_obs.unsqueeze(-1)
                x = torch.cat([h_obs, s_obs], dim=-1)
                out, _ = self.gru(x)
                return self.fc(out[:, -1, :])

            def to_train_mode(self):
                self.train()
                for p in self.parameters():
                    p.requires_grad_(False)

        model_adapter = StandaloneGRUAdapter(DemoGRU(H))
        st.info("Demo: le GRU n'est pas entraine, les resultats sont illustratifs.")

    # ---- Factual prediction ----
    with st.spinner("Prediction factuelle..."):
        model_adapter.eval()
        h_t = torch.tensor(h_obs_norm, dtype=torch.float32).unsqueeze(0)
        s_t = torch.tensor(s_obs_norm, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y_factual_norm = model_adapter(h_t.to(device), s_t.to(device)).squeeze(0).cpu().numpy()

    y_factual_raw = _denormalize_gwl(y_factual_norm, scaler, target_col)

    # ---- Run CF methods ----
    methods_to_run = {
        "PhysCF (gradient)": "physcf_gradient",
        "PhysCF (Optuna)": "physcf_optuna",
        "COMET-Hydro": "comet_hydro",
    }

    if cf_method == "Comparer les 3":
        selected_methods = methods_to_run
    else:
        selected_methods = {cf_method: methods_to_run[cf_method]}

    results_dict = {}
    progress = st.progress(0)
    status = st.empty()

    for i, (display_name, method_key) in enumerate(selected_methods.items()):
        status.text(f"Optimisation {display_name}...")
        progress.progress((i) / len(selected_methods))

        try:
            result = _run_single_cf(
                method_key, h_obs_norm, s_obs_raw, s_obs_norm,
                model_adapter, (lower_norm, upper_norm), scaler,
                months_arr, L, H, lambda_prox, n_iter, lr, device,
            )
            results_dict[display_name] = result
        except Exception as e:
            st.error(f"Erreur {display_name}: {e}")
            st.code(traceback.format_exc())

    progress.progress(1.0)
    status.text("Terminee!")

    if not results_dict:
        st.error("Aucune methode n'a converge.")
        st.stop()

    # ============================================
    # DISPLAY RESULTS
    # ============================================

    st.subheader("Resultats")

    # ---- Metrics table ----
    metrics_df = _make_metrics_table(results_dict, (lower_norm, upper_norm))
    if not metrics_df.empty:
        st.dataframe(metrics_df, use_container_width=True)

    # ---- Factual vs CF plots ----
    for display_name, result in results_dict.items():
        y_cf_norm = result["y_cf"].numpy() if hasattr(result["y_cf"], "numpy") else result["y_cf"]
        y_cf_raw = _denormalize_gwl(y_cf_norm, scaler, target_col)

        fig = _plot_factual_vs_cf(
            dates_lookback, dates_horizon,
            h_obs_raw, y_factual_raw, y_cf_raw,
            (lower_raw_arr, upper_raw_arr),
            target_col, display_name,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- Stress comparison (for PhysCF methods) ----
    for display_name, result in results_dict.items():
        if "s_cf_phys" in result:
            s_cf_phys = result["s_cf_phys"].numpy() if hasattr(result["s_cf_phys"], "numpy") else result["s_cf_phys"]
            fig_stress = _plot_stress_comparison(
                dates_lookback, s_obs_raw, s_cf_phys, covariate_cols
            )
            st.plotly_chart(fig_stress, use_container_width=True)
            break  # Only show once for PhysCF

    # ---- Theta* table ----
    theta_df = _make_theta_table(results_dict)
    if not theta_df.empty:
        st.subheader("Parametres theta* (interpretables)")
        st.dataframe(theta_df, use_container_width=True)

        # Interpretation
        for method_name, result in results_dict.items():
            if "theta_star" not in result:
                continue
            theta = result["theta_star"]
            st.markdown(f"**{method_name}:**")
            parts = []
            for s in ["DJF", "MAM", "JJA", "SON"]:
                k = f"s_P_{s}"
                if k in theta:
                    v = theta[k]
                    if v > 1.05:
                        parts.append(f"  - Precipitation {s}: **+{(v - 1) * 100:.0f}%**")
                    elif v < 0.95:
                        parts.append(f"  - Precipitation {s}: **{(v - 1) * 100:.0f}%**")
            if "delta_T" in theta and abs(theta["delta_T"]) > 0.1:
                parts.append(f"  - Temperature: **{theta['delta_T']:+.1f} C**")
            if "delta_s" in theta and abs(theta["delta_s"]) > 1:
                parts.append(f"  - Decalage temporel: **{theta['delta_s']:+.0f} jours**")
            if parts:
                st.markdown("\n".join(parts))
            else:
                st.markdown("  - Perturbations negligeables (proche de l'identite)")

    # ---- Loss curves ----
    fig_loss = _plot_loss_curves(results_dict)
    st.plotly_chart(fig_loss, use_container_width=True)

    # ---- Radar chart for PhysCF methods ----
    physcf_results = {k: v for k, v in results_dict.items() if "theta_star" in v}
    if len(physcf_results) > 0:
        fig_radar = go.Figure()
        labels = ["P DJF", "P MAM", "P JJA", "P SON", "dT (norm)", "ds (norm)"]

        for method_name, result in physcf_results.items():
            theta = result["theta_star"]
            values = [
                theta.get("s_P_DJF", 1.0),
                theta.get("s_P_MAM", 1.0),
                theta.get("s_P_JJA", 1.0),
                theta.get("s_P_SON", 1.0),
                theta.get("delta_T", 0.0) / 5.0 + 1.0,
                theta.get("delta_s", 0.0) / 30.0 + 1.0,
            ]
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=labels + [labels[0]],
                name=method_name,
                fill="toself",
                opacity=0.4,
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0.0, 2.0])),
            title="Profil des perturbations theta*",
            showlegend=True,
            height=450,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ---- Save results to session ----
    st.session_state["cf_results"] = results_dict
    st.success(f"Analyse terminee! {len(results_dict)} methode(s) executee(s).")
