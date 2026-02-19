"""Visualization helpers for the Counterfactual Analysis page.

All chart-building functions return plotly go.Figure objects.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .constants import (
    IPS_COLORS,
    IPS_LABELS,
    CF_METHOD_COLORS,
    PHYSCF_PARAM_INFO,
    MONTH_TO_SEASON_NAME,
    SEASON_NAMES_FR,
)
from .ips import IPS_CLASSES, IPS_ORDER


# ---- Theta radar chart ----

def plot_theta_radar(
    theta_dicts: dict[str, dict[str, float]],
    title: str = "Parametres theta* (normalises)",
) -> go.Figure:
    """Radar chart of theta* parameters for one or more CF methods.

    Each parameter is normalized to [0, 1] using its physical range
    (from PerturbationLayer). The identity perturbation is shown
    as a dashed reference circle.

    Args:
        theta_dicts: {method_name: theta_star dict}
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    param_keys = list(PHYSCF_PARAM_INFO.keys())
    categories = [PHYSCF_PARAM_INFO[k]["label_short"] for k in param_keys]
    categories_closed = categories + [categories[0]]  # close polygon

    fig = go.Figure()

    # Identity reference circle (dashed gray)
    identity_values = []
    for key in param_keys:
        pc = PHYSCF_PARAM_INFO[key]
        identity_values.append(
            (pc["identity"] - pc["range_min"]) / (pc["range_max"] - pc["range_min"])
        )
    identity_values.append(identity_values[0])

    fig.add_trace(go.Scatterpolar(
        r=identity_values, theta=categories_closed,
        mode="lines", name="Identite (pas de perturbation)",
        line=dict(color="gray", dash="dash", width=2),
        fill=None,
    ))

    # Method traces
    for method_name, theta in theta_dicts.items():
        values = []
        hover_texts = []
        for key in param_keys:
            pc = PHYSCF_PARAM_INFO[key]
            val = theta.get(key, pc["identity"])
            normalized = (val - pc["range_min"]) / (pc["range_max"] - pc["range_min"])
            values.append(max(0, min(1, normalized)))

            # Hover text with actual value
            if pc["unit"] == "multiplicateur":
                change_pct = (val - 1.0) * 100
                hover_texts.append(f"{pc['label_short']}: x{val:.2f} ({change_pct:+.0f}%)")
            elif pc["unit"] == "degC":
                hover_texts.append(f"{pc['label_short']}: {val:+.1f} C")
            elif pc["unit"] == "jours":
                hover_texts.append(f"{pc['label_short']}: {val:+.0f} jours")
            else:
                hover_texts.append(f"{pc['label_short']}: {val:+.4f}")

        values.append(values[0])
        hover_texts.append(hover_texts[0])

        color = CF_METHOD_COLORS.get(method_name, "#888888")
        fig.add_trace(go.Scatterpolar(
            r=values, theta=categories_closed,
            fill="toself", name=method_name,
            line=dict(color=color, width=2),
            opacity=0.7,
            text=hover_texts,
            hoverinfo="text",
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickvals=[0, 0.25, 0.5, 0.75, 1]),
        ),
        showlegend=True,
        title=title,
        height=420,
        margin=dict(t=60, b=30),
    )
    return fig


# ---- CF overlay chart ----

def plot_cf_overlay(
    gt_dates: pd.DatetimeIndex,
    gt_raw: np.ndarray,
    pred_raw: Optional[np.ndarray],
    pred_dates: Optional[pd.DatetimeIndex],
    cf_results: dict[str, dict[str, Any]],
    lower_raw: np.ndarray,
    upper_raw: np.ndarray,
    ips_to_key: str,
    ips_to_label: str,
    ref_stats: dict[int, tuple[float, float]],
    mu_target: float,
    sigma_target: float,
    horizon_months: np.ndarray,
) -> go.Figure:
    """Build the CF overlay chart with IPS bands, prediction, and CF curves.

    Args:
        gt_dates: Horizon dates for ground truth.
        gt_raw: Ground truth values in m NGF.
        pred_raw: Factual prediction in m NGF (optional).
        pred_dates: Prediction dates (optional).
        cf_results: {method_name: result_dict} with 'y_cf' key.
        lower_raw: Target IPS band lower bounds (m NGF).
        upper_raw: Target IPS band upper bounds (m NGF).
        ips_to_key: Target IPS class key.
        ips_to_label: Target IPS class label.
        ref_stats: IPS reference stats for the selected window.
        mu_target: Target scaler mean.
        sigma_target: Target scaler std.
        horizon_months: Month indices for the horizon.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    # All IPS bands as background
    horizon_month_med = int(np.median(horizon_months)) if len(horizon_months) > 0 else 6
    mu_m, sigma_m = ref_stats.get(horizon_month_med, (mu_target, sigma_target))

    for cls_name in IPS_ORDER:
        z_lo, z_hi = IPS_CLASSES[cls_name]
        z_lo_c, z_hi_c = max(z_lo, -5.0), min(z_hi, 5.0)
        y_lo = mu_m + z_lo_c * sigma_m
        y_hi = mu_m + z_hi_c * sigma_m
        fig.add_hrect(
            y0=y_lo, y1=y_hi,
            fillcolor=IPS_COLORS[cls_name], opacity=0.06,
            layer="below", line_width=0,
            annotation_text=IPS_LABELS[cls_name] if cls_name == ips_to_key else "",
            annotation_position="right",
            annotation=dict(font_size=8, font_color=IPS_COLORS[cls_name]),
        )

    # Ground truth
    fig.add_trace(go.Scatter(
        x=gt_dates, y=gt_raw,
        mode="lines", name="Ground Truth (m NGF)",
        line=dict(color="#2E86AB", width=2),
    ))

    # Factual prediction
    if pred_raw is not None and pred_dates is not None:
        fig.add_trace(go.Scatter(
            x=pred_dates, y=pred_raw,
            mode="lines", name="Prediction factuelle (m NGF)",
            line=dict(color="#E91E63", width=2, dash="dot"),
        ))

    # Target IPS band (highlighted)
    ips_color = IPS_COLORS[ips_to_key]
    r, g, b = int(ips_color[1:3], 16), int(ips_color[3:5], 16), int(ips_color[5:7], 16)
    fig.add_trace(go.Scatter(
        x=gt_dates, y=lower_raw[:len(gt_dates)],
        mode="lines", name=f"Borne inf ({ips_to_label})",
        line=dict(color=ips_color, width=1, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=gt_dates, y=upper_raw[:len(gt_dates)],
        mode="lines", name=f"Borne sup ({ips_to_label})",
        line=dict(color=ips_color, width=1, dash="dot"),
        fill="tonexty", fillcolor=f"rgba({r},{g},{b},0.12)",
    ))

    # CF curves
    for mn, res in cf_results.items():
        if "y_cf" not in res:
            continue
        y_cf_norm = res["y_cf"].detach().numpy() if hasattr(res["y_cf"], "numpy") else np.array(res["y_cf"])
        y_cf_raw = y_cf_norm * sigma_target + mu_target
        color = CF_METHOD_COLORS.get(mn, "#888")
        fig.add_trace(go.Scatter(
            x=gt_dates[:len(y_cf_raw)], y=y_cf_raw,
            mode="lines+markers", name=f"CF: {mn} (m NGF)",
            line=dict(color=color, width=2, dash="dash"),
            marker=dict(size=3),
        ))

    fig.update_layout(
        title=f"Contrefactuel: cible {ips_to_label} (m NGF)",
        xaxis_title="Date", yaxis_title="Niveau piezometrique (m NGF)",
        height=450, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ---- Stress comparison chart ----

def plot_stress_comparison(
    lookback_dates: pd.DatetimeIndex,
    s_obs_phys: np.ndarray,
    s_cf_phys: np.ndarray,
    covariate_cols: list[str],
    method_name: str,
) -> go.Figure:
    """Side-by-side comparison of original vs perturbed climate stresses.

    Shows obs (solid) and CF (dashed) for each covariate, with
    difference shading to highlight perturbation zones.

    Args:
        lookback_dates: Dates for the lookback window.
        s_obs_phys: Original stresses in physical units (L, C).
        s_cf_phys: CF stresses in physical units (L, C).
        covariate_cols: Covariate column names.
        method_name: Name of the CF method.

    Returns:
        Plotly Figure with subplots.
    """
    n_cov = min(len(covariate_cols), s_obs_phys.shape[-1])
    colors_s = ["#2E86AB", "#FF8C00", "#9B59B6"]

    fig = make_subplots(
        rows=n_cov, cols=1, shared_xaxes=True,
        subplot_titles=[f"{c} (obs vs CF)" for c in covariate_cols[:n_cov]],
        vertical_spacing=0.08,
    )

    for j in range(n_cov):
        # Original (solid)
        fig.add_trace(go.Scatter(
            x=lookback_dates, y=s_obs_phys[:, j],
            name=f"Obs" if j == 0 else None,
            line=dict(color=colors_s[j % 3], width=1),
            showlegend=(j == 0), legendgroup="obs",
        ), row=j + 1, col=1)

        # CF (dashed, thicker)
        fig.add_trace(go.Scatter(
            x=lookback_dates, y=s_cf_phys[:, j],
            name=f"CF ({method_name})" if j == 0 else None,
            line=dict(color=colors_s[j % 3], width=2, dash="dash"),
            showlegend=(j == 0), legendgroup="cf",
        ), row=j + 1, col=1)

        # Difference shading: fill between obs and CF
        # Use two traces with fill='tonexty' - upper/lower sorted
        y_upper = np.maximum(s_obs_phys[:, j], s_cf_phys[:, j])
        y_lower = np.minimum(s_obs_phys[:, j], s_cf_phys[:, j])
        diff = s_cf_phys[:, j] - s_obs_phys[:, j]
        avg_diff = float(np.mean(diff))
        fill_color = "rgba(65,105,225,0.1)" if avg_diff >= 0 else "rgba(220,20,60,0.1)"

        fig.add_trace(go.Scatter(
            x=lookback_dates, y=y_lower,
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ), row=j + 1, col=1)
        fig.add_trace(go.Scatter(
            x=lookback_dates, y=y_upper,
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=fill_color,
            showlegend=False, hoverinfo="skip",
        ), row=j + 1, col=1)

    fig.update_layout(
        title=f"Stresses climatiques: obs vs CF ({method_name})",
        height=180 * n_cov + 80,
        margin=dict(t=60, b=30),
    )
    return fig


# ---- Convergence chart ----

def plot_convergence(
    results_dict: dict[str, dict[str, Any]],
) -> go.Figure:
    """Loss convergence chart for CF methods.

    Args:
        results_dict: {method_name: result_dict} with 'loss_history' key.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    for mn, res in results_dict.items():
        color = CF_METHOD_COLORS.get(mn, "#888")
        if "loss_history" in res:
            fig.add_trace(go.Scatter(
                y=res["loss_history"], name=f"{mn} (total)",
                mode="lines", line=dict(color=color),
            ))
        if "target_history" in res:
            fig.add_trace(go.Scatter(
                y=res["target_history"], name=f"{mn} (cible)",
                mode="lines", line=dict(color=color, dash="dash"),
            ))

    fig.update_layout(
        xaxis_title="Iteration", yaxis_title="Loss",
        yaxis_type="log", height=350,
        title="Convergence de l'optimisation",
        margin=dict(t=60, b=30),
    )
    return fig


# ---- Seasonal summary metrics ----

def compute_seasonal_summary(
    s_obs_phys: np.ndarray,
    s_cf_phys: np.ndarray,
    months: np.ndarray,
    covariate_cols: list[str],
) -> pd.DataFrame:
    """Compute seasonal aggregated comparison between obs and CF stresses.

    Returns DataFrame with columns: covariate, season, obs_total, cf_total, delta_pct.
    """
    rows = []
    n_cov = min(len(covariate_cols), s_obs_phys.shape[-1])
    seasons_arr = np.array([MONTH_TO_SEASON_NAME.get(m, "?") for m in months])

    for j in range(n_cov):
        for s_name in ["DJF", "MAM", "JJA", "SON"]:
            mask = seasons_arr == s_name
            if not np.any(mask):
                continue
            obs_sum = float(np.sum(s_obs_phys[mask, j]))
            cf_sum = float(np.sum(s_cf_phys[mask, j]))
            delta_pct = ((cf_sum - obs_sum) / obs_sum * 100) if abs(obs_sum) > 1e-8 else 0.0
            rows.append({
                "covariate": covariate_cols[j],
                "season": s_name,
                "season_fr": SEASON_NAMES_FR.get(s_name, s_name),
                "obs_total": obs_sum,
                "cf_total": cf_sum,
                "delta_pct": delta_pct,
            })
    return pd.DataFrame(rows)


# ---- Natural language narrative ----

def generate_cf_narrative(
    theta: dict[str, float],
    ips_from_label: str,
    ips_to_label: str,
) -> str:
    """Generate a human-readable sentence describing the CF scenario.

    Args:
        theta: Interpretable theta_star dict.
        ips_from_label: French label for the source IPS class.
        ips_to_label: French label for the target IPS class.

    Returns:
        Markdown-formatted narrative string.
    """
    parts = []
    season_adj = {
        "DJF": "hivernales", "MAM": "printanieres",
        "JJA": "estivales", "SON": "automnales",
    }

    for s in ["DJF", "MAM", "JJA", "SON"]:
        k = f"s_P_{s}"
        if k in theta and abs(theta[k] - 1.0) > 0.05:
            change = (theta[k] - 1) * 100
            direction = "plus" if change > 0 else "moins"
            parts.append(f"**{abs(change):.0f}%** {direction} de precipitations {season_adj[s]}")

    if "delta_T" in theta and abs(theta["delta_T"]) > 0.1:
        direction = "plus chaud" if theta["delta_T"] > 0 else "plus froid"
        parts.append(f"**{abs(theta['delta_T']):.1f} C** {direction}")

    if "delta_s" in theta and abs(theta["delta_s"]) > 1:
        direction = "retardees" if theta["delta_s"] > 0 else "avancees"
        parts.append(f"saisons **{direction} de {abs(theta['delta_s']):.0f} jours**")

    if "delta_etp" in theta and abs(theta["delta_etp"]) > 0.005:
        direction = "augmentee" if theta["delta_etp"] > 0 else "reduite"
        parts.append(f"ETP residuelle {direction}")

    if parts:
        if len(parts) == 1:
            scenario_str = parts[0]
        else:
            scenario_str = ", ".join(parts[:-1]) + " et " + parts[-1]
        return (
            f"Pour faire passer le niveau piezometrique de **{ips_from_label}** "
            f"a **{ips_to_label}**, il faudrait : {scenario_str}."
        )
    return (
        f"Le modele n'a trouve que des perturbations negligeables "
        f"(scenario quasi-identique a l'observation)."
    )


# ---- Export helpers ----

def build_cf_export_df(
    results_dict: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
    lower_norm: Union[np.ndarray, "torch.Tensor"],
    upper_norm: Union[np.ndarray, "torch.Tensor"],
    L_model: int,
) -> pd.DataFrame:
    """Build a DataFrame for CSV export of CF results.

    Args:
        results_dict: CF method results.
        metadata: {'model': ..., 'station': ..., 'ips_from': ..., etc.}
        lower_norm: Target lower bounds (normalized).
        upper_norm: Target upper bounds (normalized).
        L_model: Lookback length.

    Returns:
        DataFrame with one row per method.
    """
    from .metrics import validity_ratio, proximity_theta, cc_compliance, param_count

    rows = []
    for mn, res in results_dict.items():
        row = {
            "method": mn,
            **metadata,
        }
        y_cf = res.get("y_cf")
        if y_cf is not None:
            row["validity"] = validity_ratio(y_cf, lower_norm, upper_norm)
        row["converged"] = res.get("converged", False)
        row["wall_clock_s"] = res.get("wall_clock_s", 0)
        row["n_params"] = param_count(res.get("method_key", mn), L_model)

        if "theta_star" in res:
            row["proximity_theta"] = proximity_theta(res["theta_star"])
            row["cc_compliance"] = cc_compliance(res["theta_star"])
            for k, v in res["theta_star"].items():
                row[k] = v
        rows.append(row)

    return pd.DataFrame(rows)
