"""Pure helpers for ERA5 temperature-anomaly window maths (no DB/Streamlit)."""
from __future__ import annotations

from datetime import date, timedelta


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
