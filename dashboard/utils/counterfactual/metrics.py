"""Evaluation metrics for counterfactual explanations.

Covers: validity, proximity, sparsity, smoothness, physical metrics, efficiency.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _to_numpy(x):
    """Convert input to numpy array (handles torch.Tensor, np.ndarray, list)."""
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# --- Validity ---

def validity_ratio(y_cf, lower, upper) -> float:
    """Fraction of CF predictions within target bounds.

    Returns value in [0, 1]. 1.0 = fully valid.
    """
    y_cf, lower, upper = _to_numpy(y_cf), _to_numpy(lower), _to_numpy(upper)
    within = (y_cf >= lower) & (y_cf <= upper)
    return float(np.mean(within))


def stepwise_validity(y_cf, lower, upper) -> float:
    """Stepwise validity: 1 if ALL timesteps are within bounds, else 0."""
    y_cf, lower, upper = _to_numpy(y_cf), _to_numpy(lower), _to_numpy(upper)
    within = (y_cf >= lower) & (y_cf <= upper)
    return 1.0 if np.all(within) else 0.0


def seasonal_validity(y_cf, lower, upper, months) -> dict:
    """Per-season validity ratio.

    Returns dict with keys 'DJF', 'MAM', 'JJA', 'SON' and values in [0,1].
    """
    y_cf, lower, upper = _to_numpy(y_cf), _to_numpy(lower), _to_numpy(upper)
    months = _to_numpy(months)
    season_map = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                  6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
    result = {}
    for season in ["DJF", "MAM", "JJA", "SON"]:
        mask = np.array([season_map.get(int(m), "") == season for m in months])
        if np.any(mask):
            within = (y_cf[mask] >= lower[mask]) & (y_cf[mask] <= upper[mask])
            result[season] = float(np.mean(within))
        else:
            result[season] = float("nan")
    return result


# --- Proximity ---

def proximity_l1(s_obs, s_cf) -> float:
    """L1 distance between original and CF stresses."""
    s_obs, s_cf = _to_numpy(s_obs), _to_numpy(s_cf)
    return float(np.mean(np.abs(s_obs - s_cf)))


def proximity_l2(s_obs, s_cf) -> float:
    """L2 distance between original and CF stresses."""
    s_obs, s_cf = _to_numpy(s_obs), _to_numpy(s_cf)
    return float(np.sqrt(np.mean((s_obs - s_cf) ** 2)))


def proximity_theta(theta_star: dict) -> float:
    """Parameter-space proximity (Eq. 7 from paper).

    Uses PerturbationLayer.PARAM_RANGES as single source of truth.
    L_prox = Σ ((param - identity) / scale)² for all parameters.
    """
    from .perturbation import PerturbationLayer
    _ranges = PerturbationLayer.PARAM_RANGES
    l = 0.0
    for key, val in theta_star.items():
        if key in _ranges:
            r = _ranges[key]
            identity = r["identity"]
            scale = max(abs(r["max"] - identity), abs(r["min"] - identity))
            if scale > 0:
                l += ((val - identity) / scale) ** 2
    return float(l)


def mean_absolute_change(s_obs, s_cf) -> float:
    """Mean absolute change per timestep (for paper Table 1)."""
    s_obs, s_cf = _to_numpy(s_obs), _to_numpy(s_cf)
    return float(np.mean(np.abs(s_cf - s_obs)))


def max_absolute_change(s_obs, s_cf) -> float:
    """Maximum absolute change over all timesteps (for paper Table 1)."""
    s_obs, s_cf = _to_numpy(s_obs), _to_numpy(s_cf)
    return float(np.max(np.abs(s_cf - s_obs)))


def relative_change_pct(s_obs, s_cf) -> float:
    """Mean relative change as percentage.

    Handles zero division gracefully.
    """
    s_obs, s_cf = _to_numpy(s_obs), _to_numpy(s_cf)
    mask = np.abs(s_obs) > 1e-8
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(s_cf[mask] - s_obs[mask]) / np.abs(s_obs[mask])) * 100)


# --- Sparsity ---

def temporal_sparsity(s_obs, s_cf, tol: float = 1e-6) -> float:
    """Fraction of timesteps where stresses are modified.

    Lower = sparser (fewer changes). Returns value in [0, 1].
    """
    s_obs, s_cf = _to_numpy(s_obs), _to_numpy(s_cf)
    diff = np.abs(s_obs - s_cf)
    modified = np.any(diff > tol, axis=-1) if diff.ndim > 1 else diff > tol
    return float(np.mean(modified))


def channel_sparsity(s_obs, s_cf, tol: float = 1e-6) -> int:
    """Number of stress channels modified."""
    s_obs, s_cf = _to_numpy(s_obs), _to_numpy(s_cf)
    if s_obs.ndim == 1:
        return 1 if np.any(np.abs(s_obs - s_cf) > tol) else 0
    diff = np.abs(s_obs - s_cf)
    return int(np.sum(np.any(diff > tol, axis=0)))


# --- Smoothness ---

def total_variation(s_obs, s_cf) -> float:
    """Total variation of the perturbation delta.

    TV = |delta[t] - delta[t-1]| averaged over channels and time.
    """
    s_obs, s_cf = _to_numpy(s_obs), _to_numpy(s_cf)
    delta = s_cf - s_obs
    if delta.ndim == 1:
        return float(np.mean(np.abs(np.diff(delta))))
    return float(np.mean(np.abs(np.diff(delta, axis=0))))


# --- Physical metrics ---

def cc_compliance_from_theta(theta_star: dict, cc_rate: float = 0.07) -> float:
    """CC compliance from PhysCF theta*: returns |delta_etp| (the CC residual).

    Lower is better. 0.0 = perfect CC compliance.
    """
    return float(abs(theta_star.get("delta_etp", 0.0)))


def cc_compliance_from_stresses(
    s_obs_phys, s_cf_phys, cc_rate: float = 0.07
) -> float:
    """CC compliance from raw stress arrays.

    Measures |ETP_cf/ETP_obs - (1 + cc_rate * (T_cf - T_obs))| averaged over t.
    Stress order: [..., 0]=precip, [..., 1]=temp, [..., 2]=evap.
    Lower is better. 0.0 = perfect CC compliance.
    """
    s_obs_phys = _to_numpy(s_obs_phys)
    s_cf_phys = _to_numpy(s_cf_phys)

    evap_obs = s_obs_phys[..., 2]
    evap_cf = s_cf_phys[..., 2]
    temp_obs = s_obs_phys[..., 1]
    temp_cf = s_cf_phys[..., 1]

    mask = np.abs(evap_obs) > 1e-8
    if not np.any(mask):
        return 0.0

    ratio = evap_cf[mask] / evap_obs[mask]
    expected = 1.0 + cc_rate * (temp_cf[mask] - temp_obs[mask])
    return float(np.mean(np.abs(ratio - expected)))


def cc_compliance(theta_star_or_obs=None, s_cf_phys=None, cc_rate: float = 0.07) -> float:
    """CC compliance (DEPRECATED -- use cc_compliance_from_theta or _from_stresses).

    .. deprecated::
        This function is deprecated and will be removed in a future release.
        Use :func:`cc_compliance_from_theta` for theta-based compliance or
        :func:`cc_compliance_from_stresses` for raw stress-based compliance.

    Kept temporarily for backward compatibility. Dispatches based on argument type:
        - If ``theta_star_or_obs`` is a dict, delegates to ``cc_compliance_from_theta``.
        - If both array arguments are provided, delegates to ``cc_compliance_from_stresses``.
        - Otherwise returns 0.0.

    Args:
        theta_star_or_obs: Either a theta_star dict or observed stresses array.
        s_cf_phys: Counterfactual stresses array (required when theta_star_or_obs
            is an array).
        cc_rate: Clausius-Clapeyron rate (default 0.07 per degC).

    Returns:
        CC compliance score (lower is better, 0.0 = perfect).
    """
    import warnings
    warnings.warn(
        "cc_compliance() is deprecated and will be removed in a future release. "
        "Use cc_compliance_from_theta() for theta-based compliance or "
        "cc_compliance_from_stresses() for raw stress-based compliance.",
        FutureWarning,
        stacklevel=2,
    )
    if isinstance(theta_star_or_obs, dict):
        return cc_compliance_from_theta(theta_star_or_obs, cc_rate)
    if theta_star_or_obs is None or s_cf_phys is None:
        return 0.0
    return cc_compliance_from_stresses(theta_star_or_obs, s_cf_phys, cc_rate)


def pastas_agreement(y_cf_tft, y_cf_pastas) -> float:
    """RMSE between TFT and Pastas predictions on CF stresses."""
    y_cf_tft, y_cf_pastas = _to_numpy(y_cf_tft), _to_numpy(y_cf_pastas)
    min_len = min(len(y_cf_tft), len(y_cf_pastas))
    return float(np.sqrt(np.mean((y_cf_tft[:min_len] - y_cf_pastas[:min_len]) ** 2)))


# --- Efficiency ---

def param_count(method: str, lookback: int = 365) -> int:
    """Number of optimizable parameters per method."""
    if method in ("physcf", "physcf_gradient", "physcf_optuna"):
        return 7  # 4 seasonal P + 1 T + 1 ETP + 1 shift
    elif method in ("comet_hydro", "comet"):
        return lookback * 3
    raise ValueError(f"Unknown CF method: {method}")


def convergence_iter(loss_history: list[float], tol: float = 1e-4) -> int:
    """Number of iterations to reach convergence (loss < tol)."""
    for i, loss in enumerate(loss_history):
        if loss < tol:
            return i + 1
    return len(loss_history)


def wall_clock_seconds(start: float, end: float) -> float:
    """Wall-clock time in seconds."""
    return end - start


# --- Paper summary ---

def build_paper_metrics(result: dict, s_obs_phys=None, lower=None, upper=None, months=None) -> dict:
    """Build a comprehensive metrics dict suitable for paper tables.

    Args:
        result: CounterfactualResult dict from any CF method
        s_obs_phys: original stresses in physical space
        lower, upper: target bounds (normalized)
        months: month indices for seasonal metrics

    Returns:
        dict with all paper-relevant metrics
    """
    metrics = {}
    y_cf = result.get("y_cf")

    # Validity
    if y_cf is not None and lower is not None and upper is not None:
        metrics["validity"] = validity_ratio(y_cf, lower, upper)
        metrics["stepwise_validity"] = stepwise_validity(y_cf, lower, upper)
        if months is not None:
            metrics["seasonal_validity"] = seasonal_validity(y_cf, lower, upper, months)

    # Proximity
    theta = result.get("theta_star")
    if theta:
        metrics["proximity_theta"] = proximity_theta(theta)

    s_cf_phys = result.get("s_cf_phys")
    if s_obs_phys is not None and s_cf_phys is not None:
        s_obs_np = _to_numpy(s_obs_phys)
        s_cf_np = _to_numpy(s_cf_phys)
        metrics["proximity_l1"] = proximity_l1(s_obs_np, s_cf_np)
        metrics["proximity_l2"] = proximity_l2(s_obs_np, s_cf_np)
        metrics["mean_abs_change"] = mean_absolute_change(s_obs_np, s_cf_np)
        metrics["max_abs_change"] = max_absolute_change(s_obs_np, s_cf_np)
        metrics["relative_change_pct"] = relative_change_pct(s_obs_np, s_cf_np)

    # Sparsity
    if s_obs_phys is not None and s_cf_phys is not None:
        s_obs_np = _to_numpy(s_obs_phys)
        s_cf_np = _to_numpy(s_cf_phys)
        metrics["temporal_sparsity"] = temporal_sparsity(s_obs_np, s_cf_np)
        metrics["channel_sparsity"] = channel_sparsity(s_obs_np, s_cf_np)

    # Smoothness
    if s_obs_phys is not None and s_cf_phys is not None:
        metrics["total_variation"] = total_variation(_to_numpy(s_obs_phys), _to_numpy(s_cf_phys))

    # CC compliance
    if theta:
        metrics["cc_residual"] = cc_compliance_from_theta(theta)
    if s_obs_phys is not None and s_cf_phys is not None:
        try:
            metrics["cc_from_stresses"] = cc_compliance_from_stresses(
                _to_numpy(s_obs_phys), _to_numpy(s_cf_phys))
        except Exception:
            pass

    # Efficiency
    metrics["converged"] = result.get("converged", False)
    metrics["wall_clock_s"] = result.get("wall_clock_s", 0)
    metrics["n_params"] = result.get("n_params", 0)
    metrics["method"] = result.get("method", "unknown")

    return metrics
