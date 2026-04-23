"""Compute outlier diagnostics for Pastas model residuals."""
from __future__ import annotations

import logging
from statistics import median
from typing import Any, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text as sql_text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

CATEGORY_LABELS = {
    "DATA_GAP": "Data gap",
    "CLIMATE_EXTREME": "Extreme climate event",
    "REGIONAL_SIGNAL": "Regional signal",
    "DOMINANT_CONTRIBUTION": "Dominant contribution",
    "SEASONAL_BIAS": "Seasonal bias",
    "UNKNOWN": "Undetermined",
}


def _detect_outliers(residuals: pd.Series) -> tuple[list[dict], float]:
    """Find residuals exceeding 2σ. Returns (outlier_dicts, sigma)."""
    clean = residuals.dropna()
    if len(clean) < 10:
        return [], 0.0
    sigma = float(clean.std())
    if sigma < 1e-10:
        return [], 0.0
    threshold = 2 * sigma
    outliers = []
    for date, value in clean.items():
        if abs(value) > threshold:
            outliers.append({
                "date": pd.Timestamp(date),
                "residual": float(value),
                "residual_zscore": float(abs(value) / sigma),
            })
    return outliers, sigma


def _build_climate_context(
    target_date: pd.Timestamp,
    climate_df: pd.DataFrame,
    spli_lookup: dict[str, dict],
    spi_lookup: dict[str, dict],
) -> dict[str, Any]:
    """Build climate context for one outlier month."""
    result: dict[str, Any] = {
        "precip_mm": None, "precip_zscore": None,
        "temp_c": None, "temp_zscore": None,
        "etp_mm": None, "etp_zscore": None,
        "spli": None, "spli_class": None,
        "spi": None, "spi_class": None,
    }
    if climate_df.empty:
        return result

    month_col = pd.to_datetime(climate_df["mois"])
    cal_month = target_date.month
    same_month_mask = month_col.dt.month == cal_month
    same_month = climate_df.loc[same_month_mask]

    target_row = climate_df.loc[month_col == target_date]
    if target_row.empty or same_month.empty:
        # Try lookup by SPLI/SPI even without climate row
        date_key = target_date.strftime("%Y-%m-%d")
        spli_entry = spli_lookup.get(date_key, {})
        spi_entry = spi_lookup.get(date_key, {})
        result["spli"] = spli_entry.get("spli")
        result["spli_class"] = spli_entry.get("classification")
        result["spi"] = spi_entry.get("spi")
        result["spi_class"] = spi_entry.get("classification")
        return result

    row = target_row.iloc[0]

    for col, key_val, key_z in [
        ("precipitation_totale", "precip_mm", "precip_zscore"),
        ("temperature_moyenne", "temp_c", "temp_zscore"),
        ("evaporation_moyenne", "etp_mm", "etp_zscore"),
    ]:
        if col in same_month.columns:
            vals = same_month[col].dropna()
            val = row.get(col)
            if val is not None and not pd.isna(val) and len(vals) >= 3:
                mean = float(vals.mean())
                std = float(vals.std())
                result[key_val] = float(val)
                result[key_z] = float((val - mean) / std) if std > 0 else 0.0

    date_key = target_date.strftime("%Y-%m-%d")
    spli_entry = spli_lookup.get(date_key, {})
    spi_entry = spi_lookup.get(date_key, {})
    result["spli"] = spli_entry.get("spli")
    result["spli_class"] = spli_entry.get("classification")
    result["spi"] = spi_entry.get("spi")
    result["spi_class"] = spi_entry.get("classification")

    return result


def _build_data_quality(
    target_date: pd.Timestamp,
    daily_df: pd.DataFrame,
) -> dict[str, Any]:
    """Check data quality around an outlier (±30 days)."""
    result = {"gap_days": 0, "coverage_pct": 100.0, "nearest_gap_distance_days": None}
    if daily_df.empty:
        return result

    date_col = pd.to_datetime(daily_df["date"])
    window_start = target_date - pd.Timedelta(days=30)
    window_end = target_date + pd.Timedelta(days=30)
    mask = (date_col >= window_start) & (date_col <= window_end)
    window = daily_df.loc[mask]

    total_days = 61
    if window.empty:
        result["gap_days"] = total_days
        result["coverage_pct"] = 0.0
        return result

    if "niveau_nappe_eau" in window.columns:
        non_null = window["niveau_nappe_eau"].notna().sum()
    else:
        non_null = len(window)

    result["gap_days"] = total_days - int(non_null)
    result["coverage_pct"] = round(float(non_null) / total_days * 100, 1)

    # Find nearest gap
    all_dates = date_col.sort_values()
    gaps = all_dates.diff().dt.days
    gap_mask = gaps > 1
    if gap_mask.any():
        gap_dates = all_dates[gap_mask]
        distances = (gap_dates - target_date).abs().dt.days
        result["nearest_gap_distance_days"] = int(distances.min())

    return result


def _build_neighbor_context(
    target_date: pd.Timestamp,
    sibling_codes: list[str],
    monthly_neighbors_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute z-scores for BDLISA siblings at the target month."""
    result: dict[str, Any] = {"total": len(sibling_codes), "anomalous": 0, "neighbor_zscores": []}
    if not sibling_codes or monthly_neighbors_df.empty:
        return result

    cal_month = target_date.month
    month_col = pd.to_datetime(monthly_neighbors_df["mois"])

    for code in sibling_codes:
        sib_mask = monthly_neighbors_df["code_bss"] == code
        sib_data = monthly_neighbors_df.loc[sib_mask]
        sib_months = pd.to_datetime(sib_data["mois"])

        same_cal = sib_data.loc[sib_months.dt.month == cal_month]
        target_row = sib_data.loc[sib_months == target_date]

        if same_cal.empty or target_row.empty or "niveau_moyen" not in same_cal.columns:
            continue

        vals = same_cal["niveau_moyen"].dropna()
        target_val = target_row.iloc[0].get("niveau_moyen")
        if target_val is None or pd.isna(target_val) or len(vals) < 3:
            continue

        mean = float(vals.mean())
        std = float(vals.std())
        if std == 0:
            continue
        zscore = float((target_val - mean) / std)
        result["neighbor_zscores"].append({"code_bss": code, "zscore": round(zscore, 2)})
        if abs(zscore) > 1.5:
            result["anomalous"] += 1

    return result


def _classify_outlier(
    outlier: dict,
    climate: dict,
    data_quality: dict,
    neighbors: dict,
    contributions: dict,
) -> tuple[str, list[str]]:
    """Classify an outlier. Returns (primary_category, secondary_tags)."""
    matched: list[str] = []

    # Rule 1: DATA_GAP
    if data_quality.get("gap_days", 0) >= 1:
        matched.append("DATA_GAP")

    # Rule 2: CLIMATE_EXTREME
    for key in ("precip_zscore", "temp_zscore", "etp_zscore"):
        z = climate.get(key)
        if z is not None and abs(z) > 2.0:
            if "CLIMATE_EXTREME" not in matched:
                matched.append("CLIMATE_EXTREME")
            break

    # Rule 3: REGIONAL_SIGNAL
    total = neighbors.get("total", 0)
    anomalous = neighbors.get("anomalous", 0)
    if total > 0 and anomalous / total >= 0.5:
        matched.append("REGIONAL_SIGNAL")

    # Rule 4: DOMINANT_CONTRIBUTION
    # Exclude constant/baseline terms
    stress_contribs = {k: abs(v) for k, v in contributions.items()
                       if not k.startswith("constant") and not k.startswith("Constant")}
    total_contrib = sum(stress_contribs.values())
    if total_contrib > 0:
        max_contrib = max(stress_contribs.values())
        if max_contrib / total_contrib > 0.8:
            matched.append("DOMINANT_CONTRIBUTION")

    if not matched:
        return "UNKNOWN", []

    primary = matched[0]
    secondary = matched[1:]
    return primary, secondary


def _generate_explanation(
    category: str,
    secondary_tags: list[str],
    climate: dict,
    data_quality: dict,
    neighbors: dict,
    contributions: dict,
    residual_zscore: float,
    seasonal_info: Optional[dict] = None,
) -> str:
    """Generate a natural-language explanation for the outlier."""
    parts: list[str] = []

    all_cats = [category] + secondary_tags

    for cat in all_cats:
        if cat == "DATA_GAP":
            gap = data_quality.get("gap_days", 0)
            parts.append(f"Data gap of {gap} days detected within ±30 days. Model interpolation may be unreliable.")

        elif cat == "CLIMATE_EXTREME":
            extremes = []
            for label, key in [("precipitation", "precip_zscore"), ("temperature", "temp_zscore"), ("evapotranspiration", "etp_zscore")]:
                z = climate.get(key)
                if z is not None and abs(z) > 2.0:
                    direction = "above" if z > 0 else "below"
                    extremes.append(f"{label} was {abs(z):.1f}σ {direction} normal")
            if extremes:
                parts.append(f"Monthly {', '.join(extremes)}.")

        elif cat == "REGIONAL_SIGNAL":
            n = neighbors.get("anomalous", 0)
            total = neighbors.get("total", 0)
            parts.append(f"{n}/{total} neighboring stations also show anomalous levels this month.")

        elif cat == "DOMINANT_CONTRIBUTION":
            stress_contribs = {k: abs(v) for k, v in contributions.items()
                               if not k.startswith("constant") and not k.startswith("Constant")}
            if stress_contribs:
                top = max(stress_contribs, key=stress_contribs.get)
                val = contributions[top]
                parts.append(f"The {top} contribution ({val:+.3f}m) dominates model response this month.")

        elif cat == "SEASONAL_BIAS":
            if seasonal_info:
                count = seasonal_info.get("count", 0)
                quarter = seasonal_info.get("quarter", "?")
                sign = "positive" if seasonal_info.get("sign", 1) > 0 else "negative"
                parts.append(f"{count} outliers with {sign} residuals cluster in Q{quarter}, suggesting systematic seasonal model error.")

        elif cat == "UNKNOWN":
            parts.append(f"No clear cause identified. Residual is {residual_zscore:.1f}σ from model expectation.")

    return " ".join(parts) if parts else f"Residual is {residual_zscore:.1f}σ from model expectation."


def _apply_seasonal_bias(outliers: list[dict]) -> None:
    """Pass 2: detect seasonal clustering and tag affected outliers in-place."""
    if len(outliers) < 3:
        return

    # Group by (quarter, sign)
    from collections import defaultdict
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, o in enumerate(outliers):
        q = (o["date"].month - 1) // 3 + 1
        sign = 1 if o["residual"] > 0 else -1
        groups[(q, sign)].append(i)

    for (quarter, sign), indices in groups.items():
        if len(indices) < 3:
            continue
        seasonal_info = {"quarter": quarter, "sign": sign, "count": len(indices)}
        for idx in indices:
            o = outliers[idx]
            if o["category"] == "UNKNOWN":
                o["category"] = "SEASONAL_BIAS"
                o["_seasonal_info"] = seasonal_info
            elif "SEASONAL_BIAS" not in o["secondary_tags"]:
                o["secondary_tags"].append("SEASONAL_BIAS")


# ---------------------------------------------------------------------------
# Data fetching helpers (thin wrappers for testability)
# ---------------------------------------------------------------------------

def _fetch_climate_data(code_bss: str, engine: Engine) -> pd.DataFrame:
    query = sql_text("""
        SELECT mois, precipitation_totale, temperature_moyenne, evaporation_moyenne
        FROM gold.fct_monthly_chroniques
        WHERE code_bss = :code AND niveau_moyen IS NOT NULL
        ORDER BY mois
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"code": code_bss})


def _fetch_daily_data(code_bss: str, engine: Engine) -> pd.DataFrame:
    query = sql_text("""
        SELECT date, niveau_nappe_eau
        FROM gold.hubeau_daily_chroniques
        WHERE code_bss = :code
        ORDER BY date
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"code": code_bss})


def _fetch_sibling_codes(code_bss: str, engine: Engine, limit: int = 10) -> list[str]:
    query = sql_text("""
        SELECT codes_bdlisa FROM gold.dim_piezo_stations WHERE code_bss = :code
    """)
    with engine.connect() as conn:
        row = conn.execute(query, {"code": code_bss}).mappings().first()
    if not row or not row["codes_bdlisa"]:
        return []
    bdlisa = str(row["codes_bdlisa"]).split(",")[0].strip()

    query2 = sql_text("""
        SELECT code_bss FROM gold.dim_piezo_stations
        WHERE codes_bdlisa LIKE :pattern AND code_bss != :code
        LIMIT :lim
    """)
    with engine.connect() as conn:
        result = conn.execute(query2, {"pattern": f"{bdlisa}%", "code": code_bss, "lim": limit})
        return [r["code_bss"] for r in result.mappings()]


def _fetch_neighbor_monthly(sibling_codes: list[str], engine: Engine) -> pd.DataFrame:
    if not sibling_codes:
        return pd.DataFrame()
    placeholders = ", ".join(f":c{i}" for i in range(len(sibling_codes)))
    params = {f"c{i}": c for i, c in enumerate(sibling_codes)}
    query = sql_text(f"""
        SELECT code_bss, mois, niveau_moyen
        FROM gold.fct_monthly_chroniques
        WHERE code_bss IN ({placeholders})
        ORDER BY code_bss, mois
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params=params)


def _fetch_drought_indices(
    code_bss: str, engine: Engine,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Fetch SPLI and SPI, return as date-keyed lookups."""
    from dashboard.utils.drought import compute_spli, compute_spi

    query = sql_text("""
        SELECT mois, niveau_moyen, precipitation_totale
        FROM gold.fct_monthly_chroniques
        WHERE code_bss = :code
        ORDER BY mois
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"code": code_bss})

    spli_lookup: dict[str, dict] = {}
    spi_lookup: dict[str, dict] = {}

    if not df.empty:
        months = [str(m) for m in df["mois"]]

        niveau_vals = [float(v) if pd.notna(v) else None for v in df["niveau_moyen"]]
        try:
            for entry in compute_spli(months, niveau_vals):
                spli_lookup[entry["mois"]] = entry
        except Exception:
            logger.debug("SPLI computation failed for %s", code_bss)

        precip_vals = [float(v) if pd.notna(v) else None for v in df["precipitation_totale"]]
        try:
            for entry in compute_spi(months, precip_vals):
                spi_lookup[entry["mois"]] = entry
        except Exception:
            logger.debug("SPI computation failed for %s", code_bss)

    return spli_lookup, spi_lookup


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_outlier_diagnostics(
    model,
    code_bss: str,
    cal_tmin: str,
    cal_tmax: str,
    engine: Engine,
) -> dict:
    """Compute outlier diagnostics for all residuals exceeding 2σ."""
    residuals = model.residuals(tmin=cal_tmin, tmax=cal_tmax)
    residuals_monthly = residuals.resample("MS").mean().dropna()

    outlier_list, sigma = _detect_outliers(residuals_monthly)

    if not outlier_list:
        return {
            "run_id": "",
            "code_bss": code_bss,
            "sigma": sigma,
            "threshold": 2 * sigma,
            "n_residuals": len(residuals_monthly),
            "n_outliers": 0,
            "outliers": [],
            "summary": {"by_category": {}, "seasonal_pattern": {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}, "median_severity": 0.0},
        }

    # Fetch all context data in bulk
    climate_df = _fetch_climate_data(code_bss, engine)
    daily_df = _fetch_daily_data(code_bss, engine)
    sibling_codes = _fetch_sibling_codes(code_bss, engine)
    neighbor_df = _fetch_neighbor_monthly(sibling_codes, engine)
    spli_lookup, spi_lookup = _fetch_drought_indices(code_bss, engine)

    # Get observed and simulated values
    try:
        sim = model.simulate(tmin=cal_tmin, tmax=cal_tmax)
        obs = model.observations(tmin=cal_tmin, tmax=cal_tmax)
        sim_monthly = sim.resample("MS").mean()
        obs_monthly = obs.resample("MS").mean()
    except Exception:
        sim_monthly = pd.Series(dtype=float)
        obs_monthly = pd.Series(dtype=float)

    # Get contributions
    contrib_monthly: dict[str, pd.Series] = {}
    for sm_name in model.stressmodels:
        try:
            c = model.get_contribution(sm_name, tmin=cal_tmin, tmax=cal_tmax)
            contrib_monthly[sm_name] = c.resample("MS").mean()
        except Exception:
            pass

    # Build diagnostics for each outlier
    enriched: list[dict] = []
    for o in outlier_list:
        target = o["date"]

        climate = _build_climate_context(target, climate_df, spli_lookup, spi_lookup)
        dq = _build_data_quality(target, daily_df)
        neighbors = _build_neighbor_context(target, sibling_codes, neighbor_df)

        contribs = {}
        for name, series in contrib_monthly.items():
            if target in series.index:
                contribs[name] = round(float(series.loc[target]), 4)

        observed_val = float(obs_monthly.loc[target]) if target in obs_monthly.index else 0.0
        simulated_val = float(sim_monthly.loc[target]) if target in sim_monthly.index else 0.0

        category, secondary_tags = _classify_outlier(o, climate, dq, neighbors, contribs)
        severity = min(1.0, abs(o["residual"]) / (3 * sigma))

        enriched.append({
            "date": target,
            "residual": o["residual"],
            "residual_zscore": o["residual_zscore"],
            "severity": round(severity, 2),
            "category": category,
            "secondary_tags": secondary_tags,
            "climate": climate,
            "contributions": contribs,
            "observed": round(observed_val, 4),
            "simulated": round(simulated_val, 4),
            "data_quality": dq,
            "neighbors": neighbors,
        })

    # Pass 2: seasonal bias
    _apply_seasonal_bias(enriched)

    # Generate explanations (after pass 2, so seasonal_info is available)
    for o in enriched:
        o["category_label"] = CATEGORY_LABELS.get(o["category"], o["category"])
        o["explanation"] = _generate_explanation(
            o["category"], o["secondary_tags"],
            climate=o["climate"], data_quality=o["data_quality"],
            neighbors=o["neighbors"], contributions=o["contributions"],
            residual_zscore=o["residual_zscore"],
            seasonal_info=o.pop("_seasonal_info", None),
        )
        # Serialize date
        o["date"] = o["date"].strftime("%Y-%m-%d")

    # Sort by severity descending
    enriched.sort(key=lambda x: x["severity"], reverse=True)

    # Summary
    from collections import Counter
    cat_counts = Counter(o["category"] for o in enriched)
    q_counts = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    for o in enriched:
        month = pd.Timestamp(o["date"]).month
        q = (month - 1) // 3 + 1
        q_counts[f"Q{q}"] += 1

    severities = [o["severity"] for o in enriched]

    return {
        "run_id": "",
        "code_bss": code_bss,
        "sigma": round(sigma, 4),
        "threshold": round(2 * sigma, 4),
        "n_residuals": len(residuals_monthly),
        "n_outliers": len(enriched),
        "outliers": enriched,
        "summary": {
            "by_category": dict(cat_counts),
            "seasonal_pattern": q_counts,
            "median_severity": round(median(severities), 2) if severities else 0.0,
        },
    }
