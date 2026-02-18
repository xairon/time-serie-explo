"""IPS (Indicateur Piézométrique Standardisé) calculation.

Based on BRGM methodology (Seguin 2014), z-cutoffs scheme 2.5 years.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# IPS class boundaries (z-score cutoffs)
IPS_CLASSES = {
    "very_low": (-float("inf"), -1.28),
    "low": (-1.28, -0.84),
    "moderately_low": (-0.84, -0.25),
    "normal": (-0.25, 0.25),
    "moderately_high": (0.25, 0.84),
    "high": (0.84, 1.28),
    "very_high": (1.28, float("inf")),
}

# Ordered list for transitions
IPS_ORDER = [
    "very_low",
    "low",
    "moderately_low",
    "normal",
    "moderately_high",
    "high",
    "very_high",
]


def compute_ips_reference(
    gwl_series: pd.Series,
    ref_start: str = "1981",
    ref_end: str = "2010",
) -> dict[int, tuple[float, float]]:
    """Compute monthly reference statistics (mean, std) for IPS.

    For each month 1-12, computes mu_m and sigma_m over the reference period.
    If reference period has < 10 years of data, uses the full series.

    Args:
        gwl_series: Series with date index and gwl values (m NGF).
        ref_start: Start of reference period.
        ref_end: End of reference period.

    Returns:
        Dict {month: (mean, std)} for months 1-12.
    """
    ref = gwl_series.loc[ref_start:ref_end].dropna()

    # Check if we have enough data
    n_years = len(ref.index.year.unique())
    if n_years < 10:
        ref = gwl_series.dropna()

    monthly = ref.groupby(ref.index.month)
    stats = {}
    for month in range(1, 13):
        if month in monthly.groups:
            vals = monthly.get_group(month)
            stats[month] = (float(vals.mean()), float(vals.std()))
        else:
            stats[month] = (float("nan"), float("nan"))
    return stats


def gwl_to_ips_zscore(gwl: float, month: int, ref_stats: dict) -> float:
    """Convert a gwl value to a z-score using monthly reference stats."""
    mu, sigma = ref_stats[month]
    if sigma == 0 or np.isnan(sigma):
        return 0.0
    return (gwl - mu) / sigma


def gwl_to_ips_class(gwl: float, month: int, ref_stats: dict) -> str:
    """Convert gwl value to IPS class name."""
    z = gwl_to_ips_zscore(gwl, month, ref_stats)
    for cls_name, (z_min, z_max) in IPS_CLASSES.items():
        if z_min <= z < z_max:
            return cls_name
    return "very_high"  # z == inf edge case


def ips_class_to_gwl_bounds(
    target_class: str,
    month: int,
    ref_stats: dict,
) -> tuple[float, float]:
    """Convert IPS class z-cutoffs to gwl bounds (m NGF).

    lower_gwl = mu_m + z_min * sigma_m
    upper_gwl = mu_m + z_max * sigma_m
    """
    z_min, z_max = IPS_CLASSES[target_class]
    mu, sigma = ref_stats[month]

    lower = mu + z_min * sigma if z_min != -float("inf") else -float("inf")
    upper = mu + z_max * sigma if z_max != float("inf") else float("inf")
    return (lower, upper)


def compute_ips_series(
    gwl_series: pd.Series,
    ref_stats: dict,
) -> pd.DataFrame:
    """Compute IPS z-scores and classes for an entire series.

    Returns DataFrame with columns [gwl, ips_zscore, ips_class].
    """
    result = pd.DataFrame({"gwl": gwl_series})
    result["month"] = result.index.month
    result["ips_zscore"] = result.apply(
        lambda r: gwl_to_ips_zscore(r["gwl"], r["month"], ref_stats)
        if not np.isnan(r["gwl"]) else np.nan,
        axis=1,
    )
    result["ips_class"] = result.apply(
        lambda r: gwl_to_ips_class(r["gwl"], r["month"], ref_stats)
        if not np.isnan(r["gwl"]) else None,
        axis=1,
    )
    return result.drop(columns=["month"])
