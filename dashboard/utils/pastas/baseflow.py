"""Baseflow separation using Lyne & Hollick recursive digital filter."""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_baseflow(model, tmin: str, tmax: str, alpha: float = 0.925) -> dict:
    """Separate baseflow from quickflow using Lyne & Hollick filter."""
    obs = model.observations(tmin=tmin, tmax=tmax)
    if obs is None or len(obs) < 30:
        return {"bfi": None, "index": [], "observed": [], "baseflow": [], "quickflow": []}

    obs_clean = obs.dropna()
    values = obs_clean.values.flatten()
    n = len(values)

    # Three-pass Lyne & Hollick filter for stability
    quickflow = np.zeros(n)
    for _pass in range(3):
        qf_new = np.zeros(n)
        for t in range(1, n):
            qf_new[t] = alpha * qf_new[t - 1] + (1 + alpha) / 2 * (values[t] - values[t - 1])
        quickflow = qf_new

    quickflow = np.maximum(quickflow, 0)
    baseflow = values - quickflow

    # BFI = ratio of baseflow to total flow
    total = np.sum(np.abs(values))
    bfi = float(np.sum(baseflow) / total) if total > 0 else 1.0

    return {
        "bfi": round(bfi, 3),
        "index": [d.isoformat() if hasattr(d, 'isoformat') else str(d)[:10] for d in obs_clean.index],
        "observed": values.tolist(),
        "baseflow": baseflow.tolist(),
        "quickflow": quickflow.tolist(),
    }
