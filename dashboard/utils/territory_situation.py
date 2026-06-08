"""Pure helpers to aggregate per-station fixed-reference indices into a
territory verdict (situation class + trend) and a coverage flag.

No DB, no Streamlit. The routers supply the per-station numbers.
The situation class of a territory is the median station z-score mapped to the
7 BRGM classes; the trend is the median of per-station normalized 3-month deltas.
"""
from __future__ import annotations

import statistics
from scipy import stats

from dashboard.utils.reference import CLASS_CUTOFF_PCTL

CLASS_ORDER = [
    "EXTREMEMENT_BAS", "TRES_BAS", "BAS", "NORMAL",
    "HAUT", "TRES_HAUT", "EXTREMEMENT_HAUT",
]
BELOW_NORMAL = {"EXTREMEMENT_BAS", "TRES_BAS", "BAS"}

_Z_CUTOFFS = [float(stats.norm.ppf(p / 100.0)) for p in CLASS_CUTOFF_PCTL]

MIN_ELIGIBLE = 3
TREND_STABLE_BAND = 0.5


def zscore_to_class(z) -> str:
    if z is None:
        return "UNKNOWN"
    for i, thr in enumerate(_Z_CUTOFFS):
        if z < thr:
            return CLASS_ORDER[i]
    return CLASS_ORDER[-1]


def aggregate_situation(index_values: list[float]) -> dict:
    vals = [v for v in index_values if v is not None]
    if len(vals) < MIN_ELIGIBLE:
        return {
            "situation_class": None, "insufficient": True,
            "n_eligible": len(vals), "pct_below_normal": None,
            "distribution": {c: 0 for c in CLASS_ORDER},
        }
    median_z = statistics.median(vals)
    classes = [zscore_to_class(v) for v in vals]
    distribution = {c: classes.count(c) for c in CLASS_ORDER}
    below = sum(1 for c in classes if c in BELOW_NORMAL)
    return {
        "situation_class": zscore_to_class(median_z),
        "insufficient": False,
        "n_eligible": len(vals),
        "pct_below_normal": round(100.0 * below / len(vals), 1),
        "distribution": distribution,
    }


def aggregate_trend(deltas: list[float]) -> str | None:
    vals = [d for d in deltas if d is not None]
    if not vals:
        return None
    m = statistics.median(vals)
    if m > TREND_STABLE_BAND:
        return "hausse"
    if m < -TREND_STABLE_BAND:
        return "baisse"
    return "stable"
