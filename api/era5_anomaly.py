"""Pure helpers shared by the ERA5 observatory routers (no DB/Streamlit).

Historically also held the on-the-fly temperature/precipitation anomaly window maths
(``compute_anomalies``, ``compute_precip_anomalies``, ``window_end_months``,
``add_months``, ``latest_complete_month``) consumed by the ``/anomaly`` endpoint. That
endpoint and its ~71s climatology scan were removed (Task C1, Lot 2) once the frontend
dropped the phantom ``anomaly`` overlay variable in favour of the precomputed STI grid
— the window-maths helpers went with it as dead code. ``classify_index`` and
``station_spi_rows`` remain: they format the precomputed SPI/STI from
``gold.fct_era5_indices_grid`` and are still used by ``/spi``, ``/sti`` and the
per-station SPI endpoints.
"""
from __future__ import annotations

import math


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


def station_spi_rows(rows) -> list[dict]:
    """Format station SPI rows into the ``{mois, value, spi, classification}`` shape
    used by the piezo/hydro ``/stations/{code}/spi`` endpoints.

    ``rows`` are dicts ``{mois, value, spi}`` — the month, the mapped ERA5 grid
    cell's monthly precipitation total, and its precomputed SPI (from
    ``gold.fct_era5_indices_grid``). No statistics are (re)computed here; this only
    rounds and classifies, mirroring the shape the old per-station on-the-fly gamma
    fit used to return.
    """
    out = []
    for r in rows:
        spi = float(r["spi"]) if r["spi"] is not None else None
        value = float(r["value"]) if r["value"] is not None else None
        out.append({
            "mois": str(r["mois"]),
            "value": round(value, 4) if value is not None else None,
            "spi": round(spi, 3) if spi is not None else None,
            "classification": classify_index(spi),
        })
    return out
