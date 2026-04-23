"""Compare model residuals with neighboring station observations."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sqlalchemy import text as sql_text
from sqlalchemy.engine import Engine


def compute_regional_residuals(
    model, code_bss: str, tmin: str, tmax: str, engine: Engine, max_siblings: int = 8
) -> dict:
    """Compute z-scored residuals and compare with BDLISA siblings."""
    # Get model residuals, resample to monthly
    try:
        residuals = model.residuals(tmin=tmin, tmax=tmax)
        residuals_monthly = residuals.resample("MS").mean().dropna()
    except Exception:
        return {"index": [], "model_residual_zscore": [], "siblings": [],
                "correlation_with_siblings": None}

    if len(residuals_monthly) < 12:
        return {"index": [], "model_residual_zscore": [], "siblings": [],
                "correlation_with_siblings": None}

    # Z-score model residuals
    r_mean = float(residuals_monthly.mean())
    r_std = float(residuals_monthly.std())
    if r_std < 1e-10:
        r_std = 1.0
    model_zscore = ((residuals_monthly - r_mean) / r_std).values.flatten()
    model_dates = residuals_monthly.index

    # Get BDLISA siblings
    query_bdlisa = sql_text("""
        SELECT codes_bdlisa FROM gold.dim_piezo_stations WHERE code_bss = :code
    """)
    with engine.connect() as conn:
        row = conn.execute(query_bdlisa, {"code": code_bss}).mappings().first()

    if not row or not row["codes_bdlisa"]:
        return {
            "index": [d.isoformat() if hasattr(d, 'isoformat') else str(d)[:10] for d in model_dates],
            "model_residual_zscore": model_zscore.tolist(),
            "siblings": [],
            "correlation_with_siblings": None,
        }

    bdlisa = str(row["codes_bdlisa"]).split(",")[0].strip()

    query_siblings = sql_text("""
        SELECT code_bss FROM gold.dim_piezo_stations
        WHERE codes_bdlisa LIKE :pattern AND code_bss != :code
        LIMIT :lim
    """)
    with engine.connect() as conn:
        sibling_result = conn.execute(query_siblings, {"pattern": f"{bdlisa}%", "code": code_bss, "lim": max_siblings})
        sibling_codes = [r["code_bss"] for r in sibling_result.mappings()]

    if not sibling_codes:
        return {
            "index": [d.isoformat() if hasattr(d, 'isoformat') else str(d)[:10] for d in model_dates],
            "model_residual_zscore": model_zscore.tolist(),
            "siblings": [],
            "correlation_with_siblings": None,
        }

    # Fetch all sibling monthly data in one query
    placeholders = ", ".join(f":c{i}" for i in range(len(sibling_codes)))
    params = {f"c{i}": c for i, c in enumerate(sibling_codes)}
    params["tmin"] = tmin
    params["tmax"] = tmax
    query_data = sql_text(f"""
        SELECT code_bss, mois, niveau_moyen
        FROM gold.fct_monthly_chroniques
        WHERE code_bss IN ({placeholders}) AND mois >= :tmin AND mois <= :tmax
        ORDER BY code_bss, mois
    """)
    with engine.connect() as conn:
        sib_df = pd.read_sql(query_data, conn, params=params)

    siblings_output = []
    all_correlations = []

    for code in sibling_codes:
        sib_data = sib_df[sib_df["code_bss"] == code].copy()
        if sib_data.empty or len(sib_data) < 12:
            continue

        sib_data["mois"] = pd.to_datetime(sib_data["mois"])
        sib_data = sib_data.set_index("mois").sort_index()
        sib_vals = sib_data["niveau_moyen"].dropna()

        if len(sib_vals) < 12:
            continue

        # Z-score per calendar month
        sib_zscore = pd.Series(dtype=float, index=sib_vals.index)
        for month in range(1, 13):
            mask = sib_vals.index.month == month
            vals = sib_vals[mask]
            if len(vals) >= 3:
                m, s = vals.mean(), vals.std()
                if s > 1e-10:
                    sib_zscore[mask] = (vals - m) / s

        sib_zscore = sib_zscore.dropna()

        # Align with model dates
        aligned_zscore = []
        for d in model_dates:
            if d in sib_zscore.index:
                aligned_zscore.append(round(float(sib_zscore.loc[d]), 3))
            else:
                aligned_zscore.append(None)

        siblings_output.append({
            "code_bss": code,
            "zscore": aligned_zscore,
        })

        # Compute correlation with model residuals where both have data
        valid_pairs = [(mz, sz) for mz, sz in zip(model_zscore, aligned_zscore) if sz is not None]
        if len(valid_pairs) >= 12:
            m_arr = np.array([p[0] for p in valid_pairs])
            s_arr = np.array([p[1] for p in valid_pairs])
            corr = np.corrcoef(m_arr, s_arr)[0, 1]
            if np.isfinite(corr):
                all_correlations.append(corr)

    mean_corr = round(float(np.mean(all_correlations)), 3) if all_correlations else None

    return {
        "index": [d.isoformat() if hasattr(d, 'isoformat') else str(d)[:10] for d in model_dates],
        "model_residual_zscore": [round(float(v), 3) for v in model_zscore],
        "siblings": siblings_output,
        "correlation_with_siblings": mean_corr,
    }
