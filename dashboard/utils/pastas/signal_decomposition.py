"""STL decomposition of observed piezometric series."""
from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


def compute_stl_decomposition(model, tmin: str, tmax: str, period: int = 12) -> dict:
    """Decompose observed series into trend, seasonal, and residual components."""
    obs = model.observations(tmin=tmin, tmax=tmax)
    if obs is None or len(obs) < 60:
        return {"index": [], "observed": [], "trend": [], "seasonal": [],
                "residual": [], "trend_strength": None, "seasonal_strength": None}

    monthly = obs.resample("MS").mean()
    monthly = monthly.interpolate(method="linear", limit=3)
    monthly = monthly.dropna()
    if len(monthly) < 2 * period + 1:
        return {"index": [], "observed": [], "trend": [], "seasonal": [],
                "residual": [], "trend_strength": None, "seasonal_strength": None}

    stl = STL(monthly, period=period, robust=True)
    result = stl.fit()

    trend = result.trend
    seasonal = result.seasonal
    resid = result.resid

    # Strength metrics (Wang et al. 2006)
    var_resid = float(np.var(resid))
    var_trend_resid = float(np.var(trend + resid))
    var_seasonal_resid = float(np.var(seasonal + resid))

    trend_strength = max(0, 1 - var_resid / var_trend_resid) if var_trend_resid > 0 else 0
    seasonal_strength = max(0, 1 - var_resid / var_seasonal_resid) if var_seasonal_resid > 0 else 0

    index = [d.isoformat() if hasattr(d, 'isoformat') else str(d)[:10] for d in monthly.index]

    return {
        "index": index,
        "observed": monthly.values.flatten().tolist(),
        "trend": trend.values.flatten().tolist(),
        "seasonal": seasonal.values.flatten().tolist(),
        "residual": resid.values.flatten().tolist(),
        "trend_strength": round(trend_strength, 3),
        "seasonal_strength": round(seasonal_strength, 3),
    }
