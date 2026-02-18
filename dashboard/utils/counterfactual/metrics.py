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

    L_prox = Σ(s_P[k] - 1)² + (ΔT/5)² + (δ/0.03)² + (Δs/30)²
    """
    l = 0.0
    for season in ["DJF", "MAM", "JJA", "SON"]:
        key = f"s_P_{season}"
        if key in theta_star:
            l += (theta_star[key] - 1.0) ** 2
    if "delta_T" in theta_star:
        l += (theta_star["delta_T"] / 5.0) ** 2
    if "delta_etp" in theta_star:
        l += (theta_star["delta_etp"] / 0.03) ** 2
    if "delta_s" in theta_star:
        l += (theta_star["delta_s"] / 30.0) ** 2
    return float(l)


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

def cc_compliance(theta_star_or_obs=None, s_cf_phys=None, cc_rate: float = 0.07) -> float:
    """Clausius-Clapeyron compliance.

    Two calling conventions:
    1. cc_compliance(theta_star: dict) - from PhysCF theta* (delta_etp gives CC residual)
    2. cc_compliance(s_obs_phys, s_cf_phys) - from raw stress arrays

    For PhysCF, returns |delta_etp| (the CC residual).
    For raw stresses, measures |ETP'/ETP_obs - (1 + cc_rate * (T' - T_obs))| averaged over t.
    """
    # Convention 1: theta_star dict
    if isinstance(theta_star_or_obs, dict):
        return float(abs(theta_star_or_obs.get("delta_etp", 0.0)))

    # Convention 2: raw arrays
    if theta_star_or_obs is None or s_cf_phys is None:
        return 0.0

    s_obs_phys = _to_numpy(theta_star_or_obs)
    s_cf_phys = _to_numpy(s_cf_phys)

    evap_obs = s_obs_phys[..., 2]
    evap_cf = s_cf_phys[..., 2]
    temp_obs = s_obs_phys[..., 1]
    temp_cf = s_cf_phys[..., 1]

    # Avoid division by zero
    mask = np.abs(evap_obs) > 1e-8
    if not np.any(mask):
        return 0.0

    ratio = evap_cf[mask] / evap_obs[mask]
    expected = 1.0 + cc_rate * (temp_cf[mask] - temp_obs[mask])

    return float(np.mean(np.abs(ratio - expected)))


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
    return -1


def convergence_iter(loss_history: list[float], tol: float = 1e-4) -> int:
    """Number of iterations to reach convergence (loss < tol)."""
    for i, loss in enumerate(loss_history):
        if loss < tol:
            return i + 1
    return len(loss_history)


def wall_clock_seconds(start: float, end: float) -> float:
    """Wall-clock time in seconds."""
    return end - start
