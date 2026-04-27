"""Baseflow separation using Lyne & Hollick recursive digital filter.

Applied to dh/dt (daily level change) rather than raw levels — the filter
was designed for streamflow, not absolute heads.  Working on the rate of
change gives a meaningful split between slow (sustained recharge) and rapid
(storm pulses) components of the piezometric signal.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _lyne_hollick_3pass(signal: np.ndarray, alpha: float = 0.925) -> np.ndarray:
    """Three-pass Lyne & Hollick filter (forward-backward-forward).

    Returns the quickflow component of the input signal.
    """
    n = len(signal)
    current = signal.copy()

    for pass_num in range(3):
        qf = np.zeros(n)
        if pass_num % 2 == 0:  # forward
            for t in range(1, n):
                qf[t] = alpha * qf[t - 1] + (1 + alpha) / 2 * (current[t] - current[t - 1])
        else:  # backward
            for t in range(n - 2, -1, -1):
                qf[t] = alpha * qf[t + 1] + (1 + alpha) / 2 * (current[t] - current[t + 1])
        qf = np.maximum(qf, 0)
        current = current - qf

    return signal - current


def compute_baseflow(model, tmin: str, tmax: str, alpha: float = 0.925) -> dict:
    """Separate slow from rapid piezometric variation using Lyne & Hollick."""
    obs = model.observations(tmin=tmin, tmax=tmax)
    if obs is None or len(obs) < 30:
        return {"bfi": None, "index": [], "observed": [], "baseflow": [], "quickflow": []}

    obs_clean = obs.dropna()
    values = obs_clean.values.flatten()

    dh = np.diff(values)
    if len(dh) < 30:
        return {"bfi": None, "index": [], "observed": [], "baseflow": [], "quickflow": []}

    quickflow_dh = _lyne_hollick_3pass(dh, alpha)
    baseflow_dh = dh - quickflow_dh

    baseflow = np.concatenate([[values[0]], values[0] + np.cumsum(baseflow_dh)])
    quickflow = values - baseflow

    total_var = np.sum(np.abs(dh))
    slow_var = np.sum(np.abs(baseflow_dh))
    bfi = float(slow_var / total_var) if total_var > 0 else 1.0
    bfi = min(max(bfi, 0.0), 1.0)

    return {
        "bfi": round(bfi, 3),
        "index": [d.isoformat() if hasattr(d, 'isoformat') else str(d)[:10] for d in obs_clean.index],
        "observed": values.tolist(),
        "baseflow": baseflow.tolist(),
        "quickflow": quickflow.tolist(),
    }
