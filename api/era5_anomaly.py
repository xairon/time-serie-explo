"""Pure helpers for ERA5 temperature-anomaly window maths (no DB/Streamlit)."""
from __future__ import annotations

import math
from datetime import date, timedelta


# McKee/WMO 7-class thresholds — mirrors dashboard/utils/drought.py _THRESHOLDS_7 exactly.
_STI_THRESHOLDS = [
    (-float("inf"), -1.75, "EXTREMEMENT_BAS"),
    (-1.75, -1.28, "TRES_BAS"),
    (-1.28, -0.84, "BAS"),
    (-0.84, 0.84, "NORMAL"),
    (0.84, 1.28, "HAUT"),
    (1.28, 1.75, "TRES_HAUT"),
    (1.75, float("inf"), "EXTREMEMENT_HAUT"),
]


def classify_index(z) -> str:
    """Classify a standardized index value (z-score) into the 7 McKee/WMO class strings.

    Thresholds: ±0.84 / ±1.28 / ±1.75 (lo <= z < hi convention).
    None / NaN → 'UNKNOWN'.
    """
    if z is None:
        return "UNKNOWN"
    try:
        z = float(z)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if math.isnan(z):
        return "UNKNOWN"
    for lo, hi, label in _STI_THRESHOLDS:
        if lo <= z < hi:
            return label
    return "EXTREMEMENT_HAUT"  # fallback: +inf edge


def compute_sti(window_rows, reference, window) -> list[dict]:
    """Compute the Standardized Temperature Index (STI) per grid cell.

    Args:
        window_rows: Iterable of dicts ``{latitude, longitude, window_mean, n_months}`` —
            observed N-month window means for each cell.
        reference:  Iterable of dicts ``{latitude, longitude, mean, std, n_years}`` —
            1991-2020 reference distribution per cell.
        window:     Window length in months (1, 3, 6, or 12).

    Returns:
        List of ``{latitude, longitude, sti, index_class}`` dicts, dropping cells where:
        - ``window_mean`` is None,
        - ``n_months < window`` (incomplete observed window),
        - no reference entry exists for the cell,
        - reference ``std <= 0``.
    """
    ref_map: dict[tuple[float, float], dict] = {}
    for r in reference:
        key = (float(r["latitude"]), float(r["longitude"]))
        ref_map[key] = r

    out = []
    for r in window_rows:
        if r["window_mean"] is None:
            continue
        if int(r["n_months"]) < window:
            continue
        key = (float(r["latitude"]), float(r["longitude"]))
        ref = ref_map.get(key)
        if ref is None:
            continue
        std = float(ref["std"])
        if std <= 0:
            continue
        z = (float(r["window_mean"]) - float(ref["mean"])) / std
        out.append({
            "latitude": float(r["latitude"]),
            "longitude": float(r["longitude"]),
            "sti": z,
            "index_class": classify_index(z),
        })
    return out


def window_end_months(end_month: int, n: int) -> list[int]:
    """The n calendar months (1..12) ending at end_month, chronological, wrapping."""
    months = [((end_month - i - 1) % 12) + 1 for i in range(n)]
    return months[::-1]


def add_months(d: date, k: int) -> date:
    """First day of the month k months from d (k may be negative)."""
    total = (d.year * 12 + (d.month - 1)) + k
    year, month = divmod(total, 12)
    return date(year, month + 1, 1)


def latest_complete_month(max_date: date) -> date:
    """First day of the latest fully-complete month given the max available date.
    If max_date is the last day of its month, that month is complete; else step back one."""
    first_this = date(max_date.year, max_date.month, 1)
    first_next = add_months(first_this, 1)
    last_day_this = first_next - timedelta(days=1)
    return first_this if max_date >= last_day_this else add_months(first_this, -1)


def compute_precip_anomalies(window_rows, climatology, months, window):
    """Precipitation anomaly in % of the 1950+ normal. climatology rows carry mean_sum
    per (cell, calendar month); window_rows carry precip_sum + n_months."""
    month_set = set(months)
    norm = {}
    for c in climatology:
        if c["mo"] in month_set:
            norm.setdefault((float(c["latitude"]), float(c["longitude"])), []).append(float(c["mean_sum"]))
    out = []
    for r in window_rows:
        key = (float(r["latitude"]), float(r["longitude"]))
        vals = norm.get(key)
        if not vals or len(vals) < len(month_set) or r["precip_sum"] is None or int(r["n_months"]) < window:
            continue
        total_normal = sum(vals)
        if total_normal <= 0:
            continue
        out.append({"latitude": float(r["latitude"]), "longitude": float(r["longitude"]),
                    "anomaly": (float(r["precip_sum"]) - total_normal) / total_normal * 100.0})
    return out


def compute_anomalies(window_rows, climatology, months, window):
    """Pure: window_rows = iterable of dicts {latitude, longitude, window_mean, n_months};
    climatology = iterable of dicts {latitude, longitude, mo, mean_c}; months = the N ending
    calendar months (from window_end_months); window = N. Returns list of
    {latitude, longitude, anomaly_c}, dropping cells with incomplete normals, incomplete
    window (n_months < window), or null window_mean. Anomaly = window_mean − mean(ending-month normals)."""
    month_set = set(months)
    norm = {}
    for c in climatology:
        if c["mo"] in month_set:
            norm.setdefault((float(c["latitude"]), float(c["longitude"])), []).append(float(c["mean_c"]))
    out = []
    for r in window_rows:
        key = (float(r["latitude"]), float(r["longitude"]))
        vals = norm.get(key)
        if not vals or len(vals) < len(month_set) or r["window_mean"] is None or int(r["n_months"]) < window:
            continue
        normal = sum(vals) / len(vals)
        out.append({"latitude": float(r["latitude"]), "longitude": float(r["longitude"]), "anomaly_c": float(r["window_mean"]) - normal})
    return out
