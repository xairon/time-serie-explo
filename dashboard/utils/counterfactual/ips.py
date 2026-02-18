"""IPS (Indicateur Piezometrique Standardise) calculation.

Based on BRGM methodology:
- Seguin J.J. (2014) BRGM/RP-64147-FR
- Seguin J.J. (2016) BRGM/RP-67249-FR (notes d'utilisation)
- Seguin J.J., Klinka T. (2016) BRGM/RP-67251-FR (bilans et comparaisons)

Key methodological points:
- IPS is a MONTHLY indicator: daily data -> monthly mean -> IPS per calendar month
- Minimum 15 years of data per month (min 15 daily values per month to compute mean)
- Default reference period: 1981-2010 (30 years)
- Z-score cutoffs at +/-0.25, +/-0.84, +/-1.28
- Applicable to all aquifer types but interpretability varies:
  * Inertial aquifers (chalk, limestone): well-suited, use IPS-6/IPS-12
  * Reactive aquifers (karst, alluvial): applicable but noisy, use IPS-3

IMPORTANT: IPS must be computed on RAW physical values (m NGF), NOT normalized data.
Daily data must be aggregated to monthly means BEFORE IPS computation.
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# BRGM standard: minimum 15 years for statistical validity
BRGM_MIN_YEARS = 15

# Minimum acceptable years (relaxed) - below this, IPS is unreliable
MIN_YEARS_HARD = 5

# Minimum daily values per month to compute monthly mean (BRGM standard)
MIN_DAILY_VALUES_PER_MONTH = 15

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

# Aquifer type -> recommended IPS aggregation window
AQUIFER_IPS_WINDOW = {
    "karst": 3,         # IPS-3: smooth fast dynamics
    "alluvial": 3,      # IPS-3: smooth fast dynamics
    "chalk": 6,         # IPS-6: medium-term for inertial
    "limestone": 12,    # IPS-12: long-term for multi-year cycles
    "sand": 6,          # IPS-6: intermediate
    "volcanic": 6,      # IPS-6: intermediate
    "default": 1,       # IPS-1: standard monthly
}

# Aquifer type -> response category
AQUIFER_CATEGORY = {
    "karst": "reactive",
    "alluvial": "reactive",
    "chalk": "inertial",
    "limestone": "inertial",
    "sand": "reactive",
    "volcanic": "inertial",
}


def get_aquifer_ips_info(aquifer_type: str) -> dict:
    """Get IPS configuration recommendations for an aquifer type.

    Returns dict with:
        category: 'reactive' or 'inertial'
        recommended_window: IPS aggregation window (months)
        warnings: list of warning messages
    """
    aq = aquifer_type.lower().strip()
    category = AQUIFER_CATEGORY.get(aq, "unknown")
    window = AQUIFER_IPS_WINDOW.get(aq, 1)
    warnings = []

    if category == "reactive":
        warnings.append(
            f"Nappe reactive ({aq}): l'IPS mensuel (IPS-1) peut etre volatile. "
            f"IPS-{window} recommande pour lisser les dynamiques rapides. "
            f"Les evenements infra-mensuels ne sont pas captures par l'IPS."
        )
    elif category == "inertial":
        if aq == "limestone" or aq == "chalk":
            warnings.append(
                f"Nappe inertielle ({aq}): bien adaptee a l'IPS. "
                f"IPS-{window} recommande pour capturer les cycles pluriannuels. "
                f"Necessite des series longues (idealement 30+ ans)."
            )
    else:
        warnings.append(
            f"Type de nappe non reconnu: '{aquifer_type}'. "
            f"IPS-1 sera utilise par defaut. Interpretez avec prudence."
        )

    return {
        "category": category,
        "recommended_window": window,
        "warnings": warnings,
    }


def daily_to_monthly_mean(
    gwl_series: pd.Series,
    min_daily_values: int = MIN_DAILY_VALUES_PER_MONTH,
) -> pd.Series:
    """Aggregate daily gwl series to monthly means (BRGM standard).

    A month must have at least `min_daily_values` daily observations
    for the monthly mean to be valid.

    Args:
        gwl_series: Daily Series with DatetimeIndex and gwl values (m NGF).
        min_daily_values: Minimum daily values per month (BRGM: 15).

    Returns:
        Monthly mean Series with PeriodIndex or DatetimeIndex (month-end).
    """
    clean = gwl_series.dropna()
    if len(clean) == 0:
        return pd.Series(dtype=float)

    # Resample to month-end, compute mean and count
    monthly_mean = clean.resample("ME").mean()
    monthly_count = clean.resample("ME").count()

    # Mask months with insufficient daily values
    valid = monthly_count >= min_daily_values
    monthly_mean = monthly_mean[valid]

    n_invalid = (~valid).sum()
    if n_invalid > 0:
        logger.info(
            f"IPS: {n_invalid} mois exclus (< {min_daily_values} valeurs journalieres)"
        )

    return monthly_mean


def validate_ips_data(
    gwl_series: pd.Series,
    aquifer_type: Optional[str] = None,
) -> dict:
    """Validate that the data is suitable for IPS computation.

    Args:
        gwl_series: Daily or monthly series of gwl values in m NGF.
        aquifer_type: Optional aquifer type for specific recommendations.

    Returns dict with:
        valid (bool): Whether IPS can be computed
        n_years (int): Number of years of data
        n_months_covered (int): Number of distinct calendar months with data
        n_monthly_values (int): Total number of monthly mean values
        warnings (list[str]): Warning messages
        errors (list[str]): Error messages
        aquifer_info (dict): Aquifer-specific IPS info (if aquifer_type given)
    """
    result = {
        "valid": True,
        "n_years": 0,
        "n_months_covered": 0,
        "n_monthly_values": 0,
        "warnings": [],
        "errors": [],
        "aquifer_info": {},
    }

    clean = gwl_series.dropna()
    if len(clean) == 0:
        result["valid"] = False
        result["errors"].append("Aucune donnee valide (toutes NaN).")
        return result

    # Check that values are in physical units (not normalized)
    mean_val = float(clean.mean())
    std_val = float(clean.std())
    if abs(mean_val) < 1.0 and 0.5 < std_val < 2.0:
        result["warnings"].append(
            f"Les donnees semblent normalisees (mu={mean_val:.3f}, sigma={std_val:.3f}). "
            f"L'IPS doit etre calcule sur les valeurs brutes en m NGF, pas sur des z-scores."
        )

    # Aggregate to monthly means for proper validation
    monthly = daily_to_monthly_mean(clean)
    result["n_monthly_values"] = len(monthly)

    if len(monthly) == 0:
        result["valid"] = False
        result["errors"].append("Aucune moyenne mensuelle calculable (pas assez de valeurs journalieres par mois).")
        return result

    n_years = len(monthly.index.year.unique())
    result["n_years"] = n_years

    months_covered = monthly.index.month.unique()
    result["n_months_covered"] = len(months_covered)

    if n_years < MIN_YEARS_HARD:
        result["valid"] = False
        result["errors"].append(
            f"Seulement {n_years} annees de donnees mensuelles. "
            f"Minimum requis: {MIN_YEARS_HARD} ans (BRGM recommande: {BRGM_MIN_YEARS} ans)."
        )
    elif n_years < BRGM_MIN_YEARS:
        result["warnings"].append(
            f"Seulement {n_years} annees de donnees mensuelles. "
            f"Le standard BRGM recommande un minimum de {BRGM_MIN_YEARS} ans "
            f"pour un IPS statistiquement fiable (ref: RP-64147-FR). "
            f"Resultats a interpreter avec prudence."
        )

    if len(months_covered) < 12:
        missing = sorted(set(range(1, 13)) - set(months_covered))
        result["warnings"].append(
            f"Mois calendaires manquants: {missing}. "
            f"L'IPS pour ces mois sera interpole."
        )

    # Check per-calendar-month sample count
    monthly_per_cal = monthly.groupby(monthly.index.month).count()
    low_months = monthly_per_cal[monthly_per_cal < 10]
    if len(low_months) > 0:
        result["warnings"].append(
            f"{len(low_months)} mois calendaires ont moins de 10 annees d'observations. "
            f"Statistiques de reference peu fiables pour ces mois."
        )

    # Aquifer-specific info
    if aquifer_type:
        aq_info = get_aquifer_ips_info(aquifer_type)
        result["aquifer_info"] = aq_info
        result["warnings"].extend(aq_info["warnings"])

    return result


def compute_ips_reference(
    gwl_series: pd.Series,
    ref_start: str = "1981",
    ref_end: str = "2010",
    aggregate_to_monthly: bool = True,
) -> dict[int, tuple[float, float]]:
    """Compute monthly reference statistics (mean, std) for IPS.

    BRGM standard: daily data is first aggregated to monthly means,
    then for each calendar month (1-12), computes mu_m and sigma_m
    over the reference period.

    IMPORTANT: gwl_series must contain RAW values in m NGF, NOT normalized values.

    Args:
        gwl_series: Series with DatetimeIndex and gwl values (m NGF).
        ref_start: Start of reference period (default BRGM 1981).
        ref_end: End of reference period (default BRGM 2010).
        aggregate_to_monthly: If True, aggregate daily data to monthly means first.
            Set to False if data is already monthly.

    Returns:
        Dict {month: (mean, std)} for months 1-12.
    """
    # Step 1: Aggregate to monthly means if daily data
    if aggregate_to_monthly:
        monthly_series = daily_to_monthly_mean(gwl_series)
    else:
        monthly_series = gwl_series.dropna()

    if len(monthly_series) == 0:
        logger.warning("No valid monthly data for IPS reference computation.")
        return {m: (float("nan"), float("nan")) for m in range(1, 13)}

    # Step 2: Select reference period
    ref = monthly_series.loc[ref_start:ref_end]

    # Check if we have enough data in the reference period
    n_years = len(ref.index.year.unique()) if len(ref) > 0 else 0
    if n_years < BRGM_MIN_YEARS:
        logger.info(
            f"Reference period {ref_start}-{ref_end} has only {n_years} years "
            f"(BRGM requires {BRGM_MIN_YEARS}). Using full series "
            f"({len(monthly_series)} monthly values, "
            f"{len(monthly_series.index.year.unique())} years)."
        )
        ref = monthly_series

    # Step 3: Compute per-calendar-month statistics
    monthly_groups = ref.groupby(ref.index.month)
    stats = {}
    for month in range(1, 13):
        if month in monthly_groups.groups:
            vals = monthly_groups.get_group(month)
            mu = float(vals.mean())
            sigma = float(vals.std())
            n_vals = len(vals)
            # Protect against zero/NaN std
            if sigma == 0 or np.isnan(sigma):
                logger.warning(
                    f"Month {month}: std=0 or NaN ({n_vals} values). "
                    f"IPS for this month will be unreliable."
                )
                sigma = 1e-6  # Avoid division by zero
            if n_vals < 10:
                logger.warning(
                    f"Month {month}: only {n_vals} years of data "
                    f"(BRGM recommends >= 15). Statistics may be unreliable."
                )
            stats[month] = (mu, sigma)
        else:
            stats[month] = (float("nan"), float("nan"))

    # Fill NaN months by interpolation from neighbors
    for month in range(1, 13):
        if np.isnan(stats[month][0]):
            prev_m = ((month - 2) % 12) + 1
            next_m = (month % 12) + 1
            if not np.isnan(stats[prev_m][0]) and not np.isnan(stats[next_m][0]):
                mu_interp = (stats[prev_m][0] + stats[next_m][0]) / 2
                sigma_interp = (stats[prev_m][1] + stats[next_m][1]) / 2
                stats[month] = (mu_interp, sigma_interp)
                logger.info(f"Month {month}: interpolated from months {prev_m} and {next_m}")

    return stats


def gwl_to_ips_zscore(gwl: float, month: int, ref_stats: dict) -> float:
    """Convert a gwl value to a z-score using monthly reference stats.

    Note: For BRGM compliance, gwl should be the monthly mean level.
    Using daily values directly is an approximation.
    """
    mu, sigma = ref_stats.get(month, (float("nan"), float("nan")))
    if sigma == 0 or np.isnan(sigma) or np.isnan(mu):
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
    mu, sigma = ref_stats.get(month, (float("nan"), float("nan")))

    if np.isnan(mu) or np.isnan(sigma):
        return (float("nan"), float("nan"))

    lower = mu + z_min * sigma if z_min != -float("inf") else -float("inf")
    upper = mu + z_max * sigma if z_max != float("inf") else float("inf")
    return (lower, upper)


def compute_ips_series(
    gwl_series: pd.Series,
    ref_stats: dict,
    aggregate_to_monthly: bool = True,
) -> pd.DataFrame:
    """Compute IPS z-scores and classes for an entire series.

    Args:
        gwl_series: Daily or monthly gwl series in m NGF.
        ref_stats: Monthly reference statistics from compute_ips_reference.
        aggregate_to_monthly: If True, aggregate to monthly means first (BRGM standard).

    Returns DataFrame with columns [gwl, ips_zscore, ips_class].
    """
    if aggregate_to_monthly:
        monthly = daily_to_monthly_mean(gwl_series)
    else:
        monthly = gwl_series.dropna()

    result = pd.DataFrame({"gwl": monthly})
    result["month"] = result.index.month
    result["ips_zscore"] = result.apply(
        lambda r: gwl_to_ips_zscore(r["gwl"], r["month"], ref_stats)
        if not np.isnan(r["gwl"])
        else np.nan,
        axis=1,
    )
    result["ips_class"] = result.apply(
        lambda r: gwl_to_ips_class(r["gwl"], r["month"], ref_stats)
        if not np.isnan(r["gwl"])
        else None,
        axis=1,
    )
    return result.drop(columns=["month"])


# ---- Darts scaler utilities ----

def extract_scaler_params(
    darts_scalers: dict,
) -> Tuple[Optional[float], Optional[float], dict]:
    """Extract real mu/sigma from Darts TimeSeriesPreprocessor scalers.

    Darts stores data as z-score normalized: x_norm = (x_raw - mu) / sigma.
    The scaler can inverse_transform: x_raw = x_norm * sigma + mu.

    We recover mu and sigma by:
        inverse_transform(0) = mu
        inverse_transform(1) = mu + sigma

    Args:
        darts_scalers: Dict from MLflow artifacts, e.g.
            {'target': TimeSeriesPreprocessor, 'covariates': TimeSeriesPreprocessor}

    Returns:
        (mu_target, sigma_target, covariate_params)
        where covariate_params = {col_name: {'mean': mu, 'std': sigma}}
    """
    from darts import TimeSeries

    mu_target = None
    sigma_target = None
    cov_params = {}

    target_scaler = darts_scalers.get("target")
    if target_scaler is not None:
        try:
            # Create dummy TimeSeries with value 0 and 1
            ts_zero = TimeSeries.from_values(np.array([[0.0]]))
            ts_one = TimeSeries.from_values(np.array([[1.0]]))

            raw_zero = target_scaler.inverse_transform(ts_zero).values().flatten()[0]
            raw_one = target_scaler.inverse_transform(ts_one).values().flatten()[0]

            mu_target = float(raw_zero)  # inverse(0) = mu
            sigma_target = float(raw_one - raw_zero)  # inverse(1) - inverse(0) = sigma

            if sigma_target <= 0:
                logger.warning(
                    f"Target sigma <= 0 ({sigma_target}). Scaler may be invalid."
                )
                sigma_target = abs(sigma_target) if sigma_target != 0 else 1.0

            logger.info(
                f"Extracted target scaler: mu={mu_target:.4f} m NGF, "
                f"sigma={sigma_target:.4f} m"
            )
        except Exception as e:
            logger.warning(f"Could not extract target scaler params: {e}")

    cov_scaler = darts_scalers.get("covariates")
    if cov_scaler is not None:
        try:
            # Determine number of components
            ts_zero_1d = TimeSeries.from_values(np.array([[0.0]]))
            try:
                raw = cov_scaler.inverse_transform(ts_zero_1d)
                n_components = raw.n_components
            except Exception:
                n_components = 1

            # Try multi-component
            ts_zero_nd = TimeSeries.from_values(np.zeros((1, n_components)))
            ts_one_nd = TimeSeries.from_values(np.ones((1, n_components)))

            raw_zero = cov_scaler.inverse_transform(ts_zero_nd).values().flatten()
            raw_one = cov_scaler.inverse_transform(ts_one_nd).values().flatten()

            # Get column names if available
            try:
                col_names = list(
                    cov_scaler.inverse_transform(ts_zero_nd).columns.values
                )
            except Exception:
                col_names = [f"cov_{i}" for i in range(n_components)]

            for i, name in enumerate(col_names):
                mu_c = float(raw_zero[i])
                sigma_c = float(raw_one[i] - raw_zero[i])
                if sigma_c <= 0:
                    sigma_c = 1.0
                cov_params[name] = {"mean": mu_c, "std": sigma_c}

            logger.info(f"Extracted covariate scaler params for {len(cov_params)} columns")

        except Exception as e:
            logger.warning(f"Could not extract covariate scaler params: {e}")

    return mu_target, sigma_target, cov_params


def ref_stats_to_json(ref_stats: dict) -> str:
    """Serialize IPS reference stats to JSON string.

    Converts {int: (float, float)} to JSON-safe format.
    """
    serializable = {str(k): list(v) for k, v in ref_stats.items()}
    return json.dumps(serializable)


def ref_stats_from_json(json_str: str) -> dict:
    """Deserialize IPS reference stats from JSON string."""
    raw = json.loads(json_str)
    return {int(k): tuple(v) for k, v in raw.items()}
