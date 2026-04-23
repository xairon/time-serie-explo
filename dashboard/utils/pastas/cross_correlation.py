"""Cross-correlation analysis between precipitation and piezometric level."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sqlalchemy import text as sql_text
from sqlalchemy.engine import Engine


def compute_cross_correlation(
    model, code_bss: str, tmin: str, tmax: str, engine: Engine, max_lag: int = 24
) -> dict:
    """Compute normalized cross-correlation between precip and piezo."""
    # Fetch monthly data with both piezo and precip
    query = sql_text("""
        SELECT mois, niveau_moyen, precipitation_totale
        FROM gold.fct_monthly_chroniques
        WHERE code_bss = :code AND mois >= :tmin AND mois <= :tmax
        ORDER BY mois
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"code": code_bss, "tmin": tmin, "tmax": tmax})

    if df.empty or len(df) < max_lag + 12:
        return {"lags_months": [], "correlation": [], "max_lag_months": None,
                "max_correlation": None, "t95_months": None}

    piezo = df["niveau_moyen"].values
    precip = df["precipitation_totale"].values

    # Drop rows where either is NaN
    valid = ~(np.isnan(piezo) | np.isnan(precip))
    if valid.sum() < max_lag + 12:
        return {"lags_months": [], "correlation": [], "max_lag_months": None,
                "max_correlation": None, "t95_months": None}

    piezo = piezo[valid]
    precip = precip[valid]

    # Normalize (z-score)
    piezo = (piezo - np.mean(piezo)) / (np.std(piezo) + 1e-10)
    precip = (precip - np.mean(precip)) / (np.std(precip) + 1e-10)

    n = len(piezo)
    lags = list(range(-max_lag, max_lag + 1))
    correlations = []

    for lag in lags:
        if lag >= 0:
            c = np.corrcoef(precip[:n - lag], piezo[lag:])[0, 1] if n - lag > 3 else 0.0
        else:
            c = np.corrcoef(precip[-lag:], piezo[:n + lag])[0, 1] if n + lag > 3 else 0.0
        correlations.append(round(float(c) if np.isfinite(c) else 0.0, 4))

    # Find max positive correlation at positive lags (precip leads piezo)
    positive_lags = [(l, c) for l, c in zip(lags, correlations) if l >= 0]
    if positive_lags:
        max_lag_val, max_corr = max(positive_lags, key=lambda x: x[1])
    else:
        max_lag_val, max_corr = 0, 0.0

    # Try to get T95 from Pastas step response
    t95 = None
    try:
        step = model.get_step_response("recharge", add_0=True)
        if step is not None and len(step) > 0:
            vals = step.values.flatten()
            target = 0.95 * vals[-1] if vals[-1] != 0 else 0
            idx = np.argmax(vals >= target)
            t95 = round(float(idx) / 30.44, 1)  # days to months
    except Exception:
        pass

    return {
        "lags_months": lags,
        "correlation": correlations,
        "max_lag_months": max_lag_val,
        "max_correlation": round(max_corr, 3),
        "t95_months": t95,
    }
