"""Compute hydrological signatures using Pastas stats."""
from __future__ import annotations

import pandas as pd


def compute_signatures(observed: pd.Series, simulated: pd.Series) -> dict[str, dict[str, float]]:
    """Compute groundwater signatures on both observed and simulated.

    Returns dict with keys 'observed' and 'simulated', each containing
    signature name -> value.
    """
    import pastas as ps

    sig_functions = [
        "cv_period_mean",
        "parde_seasonality",
        "avg_seasonal_fluctuation",
        "interannual_variation",
        "rise_rate",
        "fall_rate",
        "bimodality_coefficient",
        "mean_annual_maximum",
        "autocorr_time",
    ]

    result = {}
    for label, series in [("observed", observed), ("simulated", simulated)]:
        sigs: dict[str, float] = {}
        for sig_name in sig_functions:
            try:
                func = getattr(ps.stats.signatures, sig_name)
                val = float(func(series))
                if pd.notna(val) and abs(val) < 1e10:
                    sigs[sig_name] = round(val, 6)
            except Exception:
                pass
        result[label] = sigs

    return result
