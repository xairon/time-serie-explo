from __future__ import annotations

"""Pre-fit diagnostics for piezometric time series.

Analyses a raw pd.Series before fitting a Pastas model and returns a
``PreFitDiagnostic`` dataclass with status indicators and actionable
recommendations.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class PreFitDiagnostic:
    # Coverage
    coverage_pct: float
    coverage_status: str  # green / orange / red

    # Gaps
    max_gap_days: int
    gaps_status: str
    gap_periods: list[dict]  # [{start, end, days}, ...]

    # Trend
    trend_detected: bool
    trend_pvalue: float | None
    trend_slope: float | None  # m/yr
    trend_status: str

    # Breakpoint
    breakpoint_detected: bool
    breakpoint_date: str | None
    breakpoint_status: str

    # Seasonality
    seasonality_strength: float  # ACF at lag 12 on monthly data
    seasonality_status: str

    # Record length
    record_years: float
    record_status: str

    # Recommended period
    recommended_tmin: str | None
    recommended_tmax: str | None

    # Recommendations
    recommendations: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _coverage_analysis(series: pd.Series) -> tuple[float, int, list[dict]]:
    """Return (coverage_pct, max_gap_days, gap_periods).

    gap_periods contains gaps > 30 days.
    """
    if series.empty:
        return 0.0, 0, []

    full_index = pd.date_range(series.index.min(), series.index.max(), freq="D")
    n_total = len(full_index)
    n_present = len(series.dropna())
    coverage_pct = 100.0 * n_present / n_total if n_total > 0 else 0.0

    # Reindex to daily to find gaps
    daily = series.reindex(full_index)

    gap_periods: list[dict] = []
    max_gap_days = 0
    in_gap = False
    gap_start_idx: int = 0

    for i, val in enumerate(daily.values):
        is_nan = val is None or (isinstance(val, float) and math.isnan(val))
        if is_nan and not in_gap:
            in_gap = True
            gap_start_idx = i
        elif not is_nan and in_gap:
            in_gap = False
            gap_len = i - gap_start_idx
            if gap_len > max_gap_days:
                max_gap_days = gap_len
            if gap_len > 30:
                gap_periods.append(
                    {
                        "start": str(full_index[gap_start_idx].date()),
                        "end": str(full_index[i - 1].date()),
                        "days": gap_len,
                    }
                )
    # Trailing gap
    if in_gap:
        gap_len = len(full_index) - gap_start_idx
        if gap_len > max_gap_days:
            max_gap_days = gap_len
        if gap_len > 30:
            gap_periods.append(
                {
                    "start": str(full_index[gap_start_idx].date()),
                    "end": str(full_index[-1].date()),
                    "days": gap_len,
                }
            )

    return coverage_pct, max_gap_days, gap_periods


def _trend_analysis(series: pd.Series) -> tuple[bool, float | None, float | None]:
    """Return (detected, p_value, slope_m_per_year).

    Uses pymannkendall if available, falls back to scipy kendalltau.
    Slope is estimated via Theil-Sen on monthly resampled data.
    """
    monthly = series.resample("MS").median().dropna()
    if len(monthly) < 12:
        return False, None, None

    try:
        import pymannkendall as mk  # type: ignore

        result = mk.original_test(monthly.values)
        detected = result.h  # True/False
        p_value = float(result.p)
    except ImportError:
        from scipy.stats import kendalltau  # type: ignore

        x = np.arange(len(monthly))
        tau, p_value = kendalltau(x, monthly.values)
        detected = bool(p_value < 0.05)

    # Theil-Sen slope (m per year)
    from scipy.stats import theilslopes  # type: ignore

    x_years = np.arange(len(monthly)) / 12.0  # months → years
    slope_result = theilslopes(monthly.values, x_years)
    slope_m_per_year = float(slope_result.slope)

    return detected, float(p_value), slope_m_per_year


def _pettitt_test(series: pd.Series) -> tuple[bool, str | None]:
    """Pettitt change-point test (rank-based).

    Returns (detected, date_str).  p < 0.05 → detected.
    """
    monthly = series.resample("MS").median().dropna()
    n = len(monthly)
    if n < 12:
        return False, None

    x = monthly.values.astype(float)
    # Rank-based cumulative sum
    # U_t = sum_{i<=t, j>t} sign(x_j - x_i)
    # Efficient O(n^2) implementation
    ranks = np.argsort(np.argsort(x))  # 0-based ranks
    # Build sign matrix incrementally
    # S_t = 2 * sum_{j=1}^{t} rank(x_j) - t*(n+1)
    cumrank = np.cumsum(2 * (ranks + 1) - (n + 1))
    # K = argmax |cumrank|
    abs_cumrank = np.abs(cumrank[:-1])  # exclude last (trivially 0)
    if len(abs_cumrank) == 0:
        return False, None

    K_idx = int(np.argmax(abs_cumrank))
    K = float(abs_cumrank[K_idx])

    # Approximate p-value
    p_value = 2.0 * math.exp(-6.0 * K**2 / (n**3 + n**2))
    p_value = min(p_value, 1.0)

    detected = p_value < 0.05
    if detected:
        date_str = str(monthly.index[K_idx].to_period("M"))
    else:
        date_str = None

    return detected, date_str


def _seasonality_analysis(series: pd.Series) -> float:
    """Return ACF at lag 12 on monthly resampled data (proxy for annual cycle)."""
    monthly = series.resample("MS").median().dropna()
    if len(monthly) < 24:
        return 0.0
    try:
        acf_values = acf(monthly.values, nlags=12, fft=True)
        return float(acf_values[12])
    except Exception:
        return 0.0


def _recommend_period(
    series: pd.Series,
    max_gap: int,
    gap_periods: list[dict],
    breakpoint_date: str | None,
) -> tuple[str | None, str | None]:
    """Recommend tmin/tmax skipping large gaps and pre-breakpoint data.

    Strategy:
    1. Start from series start.
    2. If there is a major gap (>180d), set tmin to just after the last such gap.
    3. If there is a breakpoint, set tmin to just after it.
    4. Ensure at least 5 years remain.
    5. tmax = last observation.
    """
    if series.empty:
        return None, None

    tmax = series.index.max()
    tmin = series.index.min()

    # Move tmin past large gaps (>180 days)
    for gp in gap_periods:
        if gp["days"] > 180:
            end_date = pd.Timestamp(gp["end"])
            candidate = end_date + pd.Timedelta(days=1)
            if candidate > tmin:
                tmin = candidate

    # Move tmin past breakpoint if detected
    if breakpoint_date is not None:
        try:
            bp_ts = pd.Timestamp(breakpoint_date)
            candidate = bp_ts + pd.DateOffset(months=1)
            if candidate > tmin:
                tmin = candidate
        except Exception:
            pass

    # Ensure at least 5 years remain; if not, fall back to original start
    if (tmax - tmin).days < 5 * 365:
        tmin = series.index.min()

    return str(tmin.date()), str(tmax.date())


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

def run_prefit_diagnostics(series: pd.Series) -> PreFitDiagnostic:
    """Analyse a piezometric time series before fitting a Pastas model.

    Parameters
    ----------
    series:
        Daily (or sub-daily) piezometric head series. NaN gaps are allowed.

    Returns
    -------
    PreFitDiagnostic
        Dataclass with status indicators and recommendations.
    """
    series = series.dropna().sort_index()

    # --- Record length ---
    if series.empty:
        record_years = 0.0
    else:
        record_years = (series.index.max() - series.index.min()).days / 365.25

    if record_years >= 15:
        record_status = "green"
    elif record_years >= 5:
        record_status = "orange"
    else:
        record_status = "red"

    # --- Coverage & gaps ---
    coverage_pct, max_gap_days, gap_periods = _coverage_analysis(series)

    if coverage_pct > 80:
        coverage_status = "green"
    elif coverage_pct >= 50:
        coverage_status = "orange"
    else:
        coverage_status = "red"

    if max_gap_days < 30:
        gaps_status = "green"
    elif max_gap_days <= 180:
        gaps_status = "orange"
    else:
        gaps_status = "red"

    # --- Trend ---
    trend_detected, trend_pvalue, trend_slope = _trend_analysis(series)

    if trend_pvalue is None or trend_pvalue > 0.05:
        trend_status = "green"
        trend_detected = False
    elif trend_pvalue < 0.01:
        trend_status = "red"
        trend_detected = True
    else:
        trend_status = "orange"
        trend_detected = True

    # --- Breakpoint ---
    breakpoint_detected, breakpoint_date = _pettitt_test(series)
    breakpoint_status = "orange" if breakpoint_detected else "green"

    # --- Seasonality ---
    seasonality_strength = _seasonality_analysis(series)

    if seasonality_strength > 0.3:
        seasonality_status = "green"
    elif seasonality_strength >= 0.1:
        seasonality_status = "orange"
    else:
        seasonality_status = "red"

    # --- Recommended period ---
    recommended_tmin, recommended_tmax = _recommend_period(
        series, max_gap_days, gap_periods, breakpoint_date
    )

    # --- Recommendations ---
    recommendations: list[dict] = []

    # Gap recommendation
    for gp in gap_periods:
        if gp["days"] > 180:
            recommendations.append(
                {
                    "type": "gap",
                    "message": f"Lacune importante du {gp['start']} au {gp['end']} ({gp['days']} j)",
                    "action": "set_tmin",
                    "params": {"tmin": gp["end"]},
                }
            )

    # Trend recommendation
    if trend_detected and trend_slope is not None:
        recommendations.append(
            {
                "type": "trend",
                "message": f"Tendance détectée (pente {trend_slope:.3f} m/an)",
                "action": "add_linear_trend",
                "params": {"slope_m_per_year": trend_slope},
            }
        )

    # Breakpoint recommendation
    if breakpoint_detected and breakpoint_date is not None:
        recommendations.append(
            {
                "type": "breakpoint",
                "message": f"Rupture détectée en {breakpoint_date}",
                "action": "add_step_model",
                "params": {"date": f"{breakpoint_date}-01"},
            }
        )

    # Period recommendation
    if recommended_tmin is not None and recommended_tmax is not None:
        recommendations.append(
            {
                "type": "period",
                "message": f"Période recommandée : {recommended_tmin[:4]} à {recommended_tmax[:4]}",
                "action": "set_period",
                "params": {"tmin": recommended_tmin, "tmax": recommended_tmax},
            }
        )

    return PreFitDiagnostic(
        coverage_pct=coverage_pct,
        coverage_status=coverage_status,
        max_gap_days=max_gap_days,
        gaps_status=gaps_status,
        gap_periods=gap_periods,
        trend_detected=trend_detected,
        trend_pvalue=trend_pvalue,
        trend_slope=trend_slope,
        trend_status=trend_status,
        breakpoint_detected=breakpoint_detected,
        breakpoint_date=breakpoint_date,
        breakpoint_status=breakpoint_status,
        seasonality_strength=seasonality_strength,
        seasonality_status=seasonality_status,
        record_years=record_years,
        record_status=record_status,
        recommended_tmin=recommended_tmin,
        recommended_tmax=recommended_tmax,
        recommendations=recommendations,
    )
