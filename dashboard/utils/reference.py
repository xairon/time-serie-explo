"""Read fixed-reference IPS/SSFI grids from gold.station_reference_stats and apply them.

The warehouse stores, per (type, code, month), a 99-point empirical percentile grid over
a fixed reference window. This module turns grids into z-scores and class bounds. Pure
functions here; the DB read lives in the routers (which already own engine/session).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

PCTL_GRID = list(range(1, 100))
# 7-class BRGM cutoffs as CDF percentiles
CLASS_CUTOFF_PCTL = [4.01, 10.03, 20.05, 79.95, 89.97, 95.99]


def value_to_zscore(value, grid_month):
    if grid_month is None or value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    pct = float(np.interp(value, grid_month, PCTL_GRID)) / 100.0
    pct = float(np.clip(pct, 0.001, 0.999))
    return round(float(stats.norm.ppf(pct)), 3)


def series_zscores(dated_values, grid_by_month):
    """dated_values: list of (iso_date, value); grid_by_month: {month:int -> grid|None}."""
    import pandas as pd
    out = []
    for d, v in dated_values:
        m = pd.to_datetime(d).month
        out.append(value_to_zscore(v, grid_by_month.get(m)))
    return out


def class_bounds_ngf(grid_month):
    """6 class-boundary values (physical units) at the BRGM cutoffs, ascending."""
    if grid_month is None:
        return None
    return [float(np.interp(c, PCTL_GRID, grid_month)) for c in CLASS_CUTOFF_PCTL]
