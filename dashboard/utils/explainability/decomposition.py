"""Decomposition and seasonality analysis for time series.

Includes:
- STL decomposition (actual vs predicted comparison)
- Seasonality detection and pattern analysis
- Trend/seasonal match scoring
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple


class DecompositionAnalyzer:
    """Analyzer for time series decomposition."""

    def __init__(self, default_period: int = 365):
        """
        Initialize decomposition analyzer.

        Args:
            default_period: Default seasonal period for decomposition
        """
        self.default_period = default_period

    def stl_decompose(
        self,
        series: pd.Series,
        seasonal: Optional[int] = None,
        trend: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform STL decomposition on a series.

        Args:
            series: Time series (pandas Series with datetime index)
            seasonal: Seasonal period (default: self.default_period)
            trend: Trend window (default: auto-calculated)

        Returns:
            Dictionary with trend, seasonal, residual components and variance contributions
        """
        from statsmodels.tsa.seasonal import STL

        seasonal = seasonal or self.default_period

        # Calculate trend window
        if trend is None:
            trend = max(seasonal + 1, 3)
            if trend % 2 == 0:
                trend += 1
        else:
            trend = max(trend, seasonal + 1, 3)
            if trend % 2 == 0:
                trend += 1

        # Ensure we have enough data
        if len(series) < 2 * seasonal:
            return {
                "success": False,
                "error": f"Series too short for seasonal={seasonal}",
            }

        try:
            stl = STL(series, period=seasonal, trend=trend)
            result = stl.fit()

            trend_series = result.trend
            seasonal_series = result.seasonal
            residual_series = result.resid

            # Variance contributions
            var_original = series.var()
            var_trend = trend_series.var()
            var_seasonal = seasonal_series.var()
            var_residual = residual_series.var()
            var_total = var_trend + var_seasonal + var_residual

            return {
                "success": True,
                "trend": trend_series,
                "seasonal": seasonal_series,
                "residual": residual_series,
                "variance": {
                    "original": float(var_original),
                    "trend": float(var_trend),
                    "seasonal": float(var_seasonal),
                    "residual": float(var_residual),
                    "total": float(var_total),
                    "trend_pct": 100 * var_trend / var_total if var_total > 0 else 0,
                    "seasonal_pct": 100 * var_seasonal / var_total if var_total > 0 else 0,
                    "residual_pct": 100 * var_residual / var_total if var_total > 0 else 0,
                },
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def detect_seasonality(
        self,
        series: pd.Series,
        periods: Optional[List[int]] = None,
        significance: float = 0.05
    ) -> Dict[str, Any]:
        """
        Detect seasonality patterns in a series.

        Args:
            series: Time series
            periods: Periods to test (default: [7, 30, 365])
            significance: Significance level for detection

        Returns:
            Dictionary with detected patterns per period
        """
        from statsmodels.tsa.stattools import acf

        if periods is None:
            periods = [7, 30, 365]

        results = {}
        series_clean = series.dropna()

        for period in periods:
            period_name = {7: "weekly", 30: "monthly", 365: "annual"}.get(
                period, f"{period}-day"
            )

            try:
                # Need at least 2 full cycles
                if len(series_clean) < 2 * period:
                    results[period_name] = {
                        "period": period,
                        "detected": False,
                        "reason": "insufficient_data",
                    }
                    continue

                # Compute ACF
                max_lag = min(period + 10, len(series_clean) - 1)
                acf_values = acf(series_clean.values, nlags=max_lag)

                # Get ACF at period lag
                acf_at_period = acf_values[period] if period < len(acf_values) else 0

                # Significance threshold (approximate)
                n = len(series_clean)
                threshold = 1.96 / np.sqrt(n)

                detected = abs(acf_at_period) > threshold

                # Find peak ACF near period (allow some tolerance)
                search_range = range(max(1, period - 2), min(len(acf_values), period + 3))
                peak_lag = max(search_range, key=lambda i: abs(acf_values[i]))
                peak_acf = acf_values[peak_lag]

                results[period_name] = {
                    "period": period,
                    "detected": detected,
                    "acf_at_period": float(acf_at_period),
                    "peak_lag": int(peak_lag),
                    "peak_acf": float(peak_acf),
                    "threshold": float(threshold),
                    "strength": "strong" if abs(acf_at_period) > 0.5 else
                               "moderate" if abs(acf_at_period) > 0.3 else "weak",
                }

            except Exception as e:
                results[period_name] = {
                    "period": period,
                    "detected": False,
                    "error": str(e),
                }

        return results


def analyze_prediction_decomposition(
    actual: pd.Series,
    predicted: pd.Series,
    period: int = 365
) -> Dict[str, Any]:
    """
    Compare STL decomposition of actual vs predicted series.

    Args:
        actual: Actual values series
        predicted: Predicted values series
        period: Seasonal period

    Returns:
        Dictionary with decomposition comparison and match metrics
    """
    analyzer = DecompositionAnalyzer(default_period=period)

    actual_decomp = analyzer.stl_decompose(actual, seasonal=period)
    predicted_decomp = analyzer.stl_decompose(predicted, seasonal=period)

    if not actual_decomp["success"] or not predicted_decomp["success"]:
        return {
            "success": False,
            "error": "Decomposition failed",
            "actual_error": actual_decomp.get("error"),
            "predicted_error": predicted_decomp.get("error"),
        }

    # Align series for comparison
    common_idx = actual.index.intersection(predicted.index)

    if len(common_idx) < period:
        return {
            "success": False,
            "error": "Not enough overlapping data",
        }

    # Extract aligned components
    actual_trend = actual_decomp["trend"].loc[common_idx]
    predicted_trend = predicted_decomp["trend"].loc[common_idx]
    actual_seasonal = actual_decomp["seasonal"].loc[common_idx]
    predicted_seasonal = predicted_decomp["seasonal"].loc[common_idx]

    # Compute correlation metrics
    trend_corr = actual_trend.corr(predicted_trend)
    seasonal_corr = actual_seasonal.corr(predicted_seasonal)

    # Compute error metrics per component
    trend_error = actual_trend - predicted_trend
    seasonal_error = actual_seasonal - predicted_seasonal

    return {
        "success": True,
        "actual": actual_decomp,
        "predicted": predicted_decomp,
        "comparison": {
            "trend_correlation": float(trend_corr) if not pd.isna(trend_corr) else 0,
            "seasonal_correlation": float(seasonal_corr) if not pd.isna(seasonal_corr) else 0,
            "trend_mae": float(np.abs(trend_error).mean()),
            "trend_rmse": float(np.sqrt((trend_error ** 2).mean())),
            "seasonal_mae": float(np.abs(seasonal_error).mean()),
            "seasonal_rmse": float(np.sqrt((seasonal_error ** 2).mean())),
        },
        "interpretation": _interpret_decomposition_comparison(
            trend_corr, seasonal_corr,
            actual_decomp["variance"], predicted_decomp["variance"]
        ),
    }


def _interpret_decomposition_comparison(
    trend_corr: float,
    seasonal_corr: float,
    actual_variance: Dict,
    predicted_variance: Dict
) -> Dict[str, str]:
    """Generate interpretation of decomposition comparison."""
    interpretations = {}

    # Trend interpretation
    if trend_corr > 0.9:
        interpretations["trend"] = "Excellent trend capture - model follows long-term patterns well"
    elif trend_corr > 0.7:
        interpretations["trend"] = "Good trend capture - minor deviations from actual trend"
    elif trend_corr > 0.5:
        interpretations["trend"] = "Moderate trend capture - some trend patterns missed"
    else:
        interpretations["trend"] = "Poor trend capture - model misses long-term patterns"

    # Seasonal interpretation
    if seasonal_corr > 0.9:
        interpretations["seasonal"] = "Excellent seasonality capture - model learns recurring patterns well"
    elif seasonal_corr > 0.7:
        interpretations["seasonal"] = "Good seasonality capture - minor timing/amplitude issues"
    elif seasonal_corr > 0.5:
        interpretations["seasonal"] = "Moderate seasonality capture - some patterns missed"
    else:
        interpretations["seasonal"] = "Poor seasonality capture - model struggles with recurring patterns"

    # Variance distribution comparison
    trend_diff = abs(actual_variance["trend_pct"] - predicted_variance["trend_pct"])
    seasonal_diff = abs(actual_variance["seasonal_pct"] - predicted_variance["seasonal_pct"])

    if trend_diff < 10 and seasonal_diff < 10:
        interpretations["variance"] = "Model preserves data structure well"
    elif trend_diff > 20:
        interpretations["variance"] = f"Model {'overestimates' if predicted_variance['trend_pct'] > actual_variance['trend_pct'] else 'underestimates'} trend contribution"
    elif seasonal_diff > 20:
        interpretations["variance"] = f"Model {'overestimates' if predicted_variance['seasonal_pct'] > actual_variance['seasonal_pct'] else 'underestimates'} seasonal contribution"
    else:
        interpretations["variance"] = "Minor differences in variance distribution"

    return interpretations


def detect_seasonality_patterns(
    series: pd.Series,
    periods: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Detect multiple seasonality patterns in a series.

    Convenience function wrapping DecompositionAnalyzer.

    Args:
        series: Time series
        periods: Periods to test

    Returns:
        Dictionary with detected patterns
    """
    analyzer = DecompositionAnalyzer()
    return analyzer.detect_seasonality(series, periods)


def compare_decompositions(
    series1: pd.Series,
    series2: pd.Series,
    period: int = 365,
    label1: str = "Series 1",
    label2: str = "Series 2"
) -> Dict[str, Any]:
    """
    Compare decomposition of two series.

    Useful for comparing different model predictions or different time periods.

    Args:
        series1: First series
        series2: Second series
        period: Seasonal period
        label1: Label for first series
        label2: Label for second series

    Returns:
        Dictionary with comparison results
    """
    analyzer = DecompositionAnalyzer(default_period=period)

    decomp1 = analyzer.stl_decompose(series1, seasonal=period)
    decomp2 = analyzer.stl_decompose(series2, seasonal=period)

    if not decomp1["success"] or not decomp2["success"]:
        return {
            "success": False,
            "error": "Decomposition failed",
        }

    return {
        "success": True,
        label1: decomp1,
        label2: decomp2,
        "variance_comparison": {
            f"{label1}_trend_pct": decomp1["variance"]["trend_pct"],
            f"{label2}_trend_pct": decomp2["variance"]["trend_pct"],
            f"{label1}_seasonal_pct": decomp1["variance"]["seasonal_pct"],
            f"{label2}_seasonal_pct": decomp2["variance"]["seasonal_pct"],
            f"{label1}_residual_pct": decomp1["variance"]["residual_pct"],
            f"{label2}_residual_pct": decomp2["variance"]["residual_pct"],
        },
    }


def generate_seasonality_summary(detection_results: Dict[str, Any]) -> str:
    """
    Generate human-readable summary of seasonality detection.

    Args:
        detection_results: Results from detect_seasonality_patterns

    Returns:
        Summary string
    """
    detected_patterns = []
    not_detected = []

    for period_name, result in detection_results.items():
        if result.get("detected", False):
            strength = result.get("strength", "")
            acf = result.get("acf_at_period", 0)
            detected_patterns.append(f"{period_name.capitalize()} ({strength}, ACF={acf:.2f})")
        else:
            reason = result.get("reason", result.get("error", "not significant"))
            not_detected.append(f"{period_name}: {reason}")

    summary_parts = []

    if detected_patterns:
        summary_parts.append(f"**Detected seasonality**: {', '.join(detected_patterns)}")
    else:
        summary_parts.append("**No significant seasonality detected**")

    if not_detected:
        summary_parts.append(f"*Not detected*: {'; '.join(not_detected)}")

    return "\n\n".join(summary_parts)
