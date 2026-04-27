"""Compute hydrological signatures using Pastas stats."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SIGNATURE_CATEGORIES: dict[str, list[str]] = {
    "variability": [
        "cv_period_mean", "magnitude", "interannual_variation", "mean_annual_maximum",
    ],
    "seasonality": [
        "parde_seasonality", "avg_seasonal_fluctuation",
        "date_min", "date_max", "cv_date_min", "cv_date_max",
    ],
    "dynamics": [
        "rise_rate", "fall_rate", "cv_rise_rate", "cv_fall_rate",
    ],
    "pulses": [
        "high_pulse_count", "high_pulse_duration",
        "low_pulse_count", "low_pulse_duration",
    ],
    "reversals": [
        "reversals_avg", "reversals_cv",
    ],
    "predictability": [
        "colwell_constancy", "colwell_contingency",
    ],
    "recession": [
        "recession_constant", "recovery_constant",
    ],
    "duration_curve": [
        "duration_curve_slope", "duration_curve_ratio",
    ],
    "structure": [
        "richards_pathlength", "baselevel_index", "baselevel_stability",
        "bimodality_coefficient", "autocorr_time",
    ],
}

ALL_SIGNATURES = [s for sigs in SIGNATURE_CATEGORIES.values() for s in sigs]


def compute_signatures(observed: pd.Series, simulated: pd.Series) -> dict:
    """Compute all groundwater signatures on both observed and simulated.

    Returns dict with keys 'observed', 'simulated', and 'categories'.
    """
    import pastas as ps

    result: dict[str, dict[str, float]] = {}
    for label, series in [("observed", observed), ("simulated", simulated)]:
        sigs: dict[str, float] = {}
        for sig_name in ALL_SIGNATURES:
            try:
                func = getattr(ps.stats.signatures, sig_name)
                val = float(func(series))
                if pd.notna(val) and not np.isinf(val) and abs(val) < 1e10:
                    sigs[sig_name] = round(val, 6)
            except Exception:
                logger.debug("Signature %s failed for %s", sig_name, label)
        result[label] = sigs

    return {
        "observed": result["observed"],
        "simulated": result["simulated"],
        "categories": SIGNATURE_CATEGORIES,
    }
