"""Statistical test functions."""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf, grangercausalitytests
from statsmodels.tsa.seasonal import STL
from darts import TimeSeries
from darts.utils.statistics import check_seasonality


def test_stationarity(series: pd.Series, name: str = "Series") -> dict:
    """
    ADF and KPSS Stationarity Tests.

    Args:
        series: Time Series (pandas Series)
        name: Variable name

    Returns:
        dict with test results
    """
    # Remove NaN
    series = series.dropna()

    # ADF test (H0: non-stationary)
    adf_result = adfuller(series, autolag='AIC')
    adf_statistic = adf_result[0]
    adf_pvalue = adf_result[1]
    adf_stationary = adf_pvalue < 0.05

    # KPSS test (H0: stationary)
    kpss_result = kpss(series, regression='ct', nlags='auto')
    kpss_statistic = kpss_result[0]
    kpss_pvalue = kpss_result[1]
    kpss_stationary = kpss_pvalue >= 0.05

    return {
        'Variable': name,
        'ADF Statistic': adf_statistic,
        'ADF p-value': adf_pvalue,
        'ADF Result': 'Stationary' if adf_stationary else 'Non-Stationary',
        'KPSS Statistic': kpss_statistic,
        'KPSS p-value': kpss_pvalue,
        'KPSS Result': 'Stationary' if kpss_stationary else 'Non-Stationary'
    }


def check_seasonality_darts(ts: TimeSeries, periods: list = None, max_lag: int = 400) -> dict:
    """
    Seasonality detection using Darts.

    Args:
        ts: Darts TimeSeries
        periods: List of periods to test (default: [7, 30, 365])
        max_lag: Maximum lag for testing

    Returns:
        dict with results per period
    """
    if periods is None:
        periods = [7, 30, 365]

    results = {}

    for period in periods:
        period_name = {7: 'Weekly', 30: 'Monthly', 365: 'Annual'}.get(period, f'{period}-day')

        try:
            # Adjust max_lag if necessary
            actual_max_lag = min(max_lag, max(period * 2, len(ts) - 1))
            seasonality_result = check_seasonality(ts, m=period, max_lag=actual_max_lag, alpha=0.05)
            # check_seasonality returns a tuple (is_seasonal: bool, period: int)
            is_seasonal = seasonality_result[0] if isinstance(seasonality_result, tuple) else bool(seasonality_result)

            # ACF at period lag
            acf_values = acf(ts.values().flatten(), nlags=min(period + 10, len(ts) - 1))
            acf_at_period = acf_values[period] if period < len(acf_values) else 0

            results[period_name] = {
                'period': period,
                'detected': is_seasonal,
                'acf_at_period': acf_at_period
            }
        except Exception:
            results[period_name] = {
                'period': period,
                'detected': False,
                'acf_at_period': 0
            }

    return results


def stl_decomposition(series: pd.Series, seasonal: int = 365, trend: int = None) -> dict:
    """
    STL Decomposition (Seasonal-Trend-Loess).

    Args:
        series: Time Series (pandas Series with datetime index)
        seasonal: Seasonal period
        trend: Trend period (default: calculated automatically)

    Returns:
        dict with trend, seasonal, residual, and variance contributions
    """
    if trend is None:
        # Calculate odd trend window > seasonal
        trend = max(seasonal + 1, 3)
        # Ensure it is odd
        if trend % 2 == 0:
            trend += 1
    else:
        # Validate and correct trend
        trend = max(trend, seasonal + 1, 3)
        if trend % 2 == 0:
            trend += 1  # Make odd

    # Ensure seasonal is odd (required by STL)
    if seasonal % 2 == 0:
        seasonal += 1
    seasonal = max(seasonal, 7)  # STL requires seasonal >= 7

    # STL decomposition: period= sets the cycle length, seasonal= sets the LOESS smoother window
    # seasonal_deg=1 uses LOESS; seasonal parameter sets the smoother window
    seasonal_smoother = seasonal if seasonal % 2 == 1 else seasonal + 1
    stl = STL(series, period=seasonal, seasonal=seasonal_smoother, trend=trend)
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
        'trend': trend_series,
        'seasonal': seasonal_series,
        'residual': residual_series,
        'variance': {
            'original': var_original,
            'trend': var_trend,
            'seasonal': var_seasonal,
            'residual': var_residual,
            'total': var_total,
            'trend_pct': 100 * var_trend / var_total if var_total > 0 else 0,
            'seasonal_pct': 100 * var_seasonal / var_total if var_total > 0 else 0,
            'residual_pct': 100 * var_residual / var_total if var_total > 0 else 0
        }
    }


def cross_correlation(x: np.ndarray, y: np.ndarray, max_lag: int = 60) -> tuple:
    """
    Cross correlation between two series.
    Positive lag (k) = corr(x[t], y[t+k]) -> x leads y.

    Args:
        x: First series (e.g., Rain)
        y: Second series (e.g., Level)
        max_lag: Maximum lag

    Returns:
        tuple: (lags, correlations)
    """
    correlations = []
    lags = range(-max_lag, max_lag + 1)

    for lag in lags:
        try:
            if lag > 0:
                # Positive lag: x leads y
                if len(x[:-lag]) == len(y[lag:]) and len(x[:-lag]) > 1:
                    corr = np.corrcoef(x[:-lag], y[lag:])[0, 1]
                else:
                    corr = 0.0
            elif lag < 0:
                # Negative lag: y leads x
                if len(x[-lag:]) == len(y[:lag]) and len(x[-lag:]) > 1:
                    corr = np.corrcoef(x[-lag:], y[:lag])[0, 1]
                else:
                    corr = 0.0
            else:
                # Lag 0
                corr = np.corrcoef(x, y)[0, 1]

            # Replace NaN with 0 (happens with constant arrays)
            if np.isnan(corr):
                corr = 0.0
        except Exception:
            corr = 0.0

        correlations.append(corr)

    return list(lags), correlations


def granger_causality_test(df: pd.DataFrame, target_col: str, covariate_col: str, max_lag: int = 30) -> dict:
    """
    Granger Causality Test.

    Args:
        df: DataFrame with columns
        target_col: Target column
        covariate_col: Covariate column
        max_lag: Maximum lag

    Returns:
        dict with lags, p-values, significant_lags
    """
    data = df[[target_col, covariate_col]].dropna()

    try:
        gc_result = grangercausalitytests(data, maxlag=max_lag, verbose=False)

        lags = []
        pvalues = []

        for lag in range(1, max_lag + 1):
            # F-test p-value
            pvalue = gc_result[lag][0]['ssr_ftest'][1]
            lags.append(lag)
            pvalues.append(pvalue)

        # Significant lags (p < 0.05)
        significant_lags = [lag for lag, pval in zip(lags, pvalues) if pval < 0.05]

        return {
            'lags': lags,
            'pvalues': pvalues,
            'significant_lags': significant_lags
        }
    except Exception as e:
        return {
            'lags': [],
            'pvalues': [],
            'significant_lags': [],
            'error': str(e)
        }


def nash_sutcliffe_efficiency(actual, predicted):
    """Calculate Nash-Sutcliffe Efficiency (NSE) - standard hydrology metric.

    NSE = 1 means perfect prediction
    NSE = 0 means prediction is as good as using the mean
    NSE < 0 means prediction is worse than using the mean

    Args:
        actual: Darts TimeSeries of observed values
        predicted: Darts TimeSeries of predicted values

    Returns:
        float: NSE value
    """
    actual_vals = actual.values().flatten()
    pred_vals = predicted.values().flatten()

    min_len = min(len(actual_vals), len(pred_vals))
    actual_vals = actual_vals[:min_len]
    pred_vals = pred_vals[:min_len]

    mean_obs = np.mean(actual_vals)
    ss_res = np.sum((actual_vals - pred_vals) ** 2)
    ss_tot = np.sum((actual_vals - mean_obs) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else -np.inf

    return 1 - (ss_res / ss_tot)


def kling_gupta_efficiency(actual, predicted):
    """Calculate Kling-Gupta Efficiency (KGE) - improved hydrology metric.

    KGE decomposes the error into:
    - Correlation (r)
    - Bias ratio (beta)
    - Variability ratio (gamma)

    KGE = 1 means perfect prediction
    KGE > -0.41 is considered useful (better than mean)

    Args:
        actual: Darts TimeSeries of observed values
        predicted: Darts TimeSeries of predicted values

    Returns:
        float: KGE value
    """
    actual_vals = actual.values().flatten()
    pred_vals = predicted.values().flatten()

    min_len = min(len(actual_vals), len(pred_vals))
    actual_vals = actual_vals[:min_len]
    pred_vals = pred_vals[:min_len]

    # Correlation coefficient
    if np.std(actual_vals) == 0 or np.std(pred_vals) == 0:
        r = 0.0
    else:
        r = np.corrcoef(actual_vals, pred_vals)[0, 1]

    # Bias ratio (mean ratio)
    mean_obs = np.mean(actual_vals)
    mean_pred = np.mean(pred_vals)
    if mean_obs == 0:
        beta = 1.0 if mean_pred == 0 else np.inf
    else:
        beta = mean_pred / mean_obs

    # Variability ratio (std ratio)
    std_obs = np.std(actual_vals)
    std_pred = np.std(pred_vals)
    if std_obs == 0:
        gamma = 1.0 if std_pred == 0 else np.inf
    else:
        gamma = std_pred / std_obs

    # KGE formula
    kge = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)

    return kge


def calculate_lagged_correlations(df: pd.DataFrame, target_col: str, covariate_col: str, max_lag: int = 60) -> tuple:
    """
    Calculates correlation for different lags.

    Args:
        df: DataFrame
        target_col: Target column
        covariate_col: Covariate column
        max_lag: Maximum lag

    Returns:
        tuple: (lags, correlations, optimal_lag, optimal_corr)
    """
    correlations = []
    lags = range(0, max_lag + 1)

    for lag in lags:
        df_temp = df[[target_col, covariate_col]].copy()
        df_temp['covariate_lagged'] = df_temp[covariate_col].shift(lag)
        df_temp = df_temp.dropna()

        if len(df_temp) > 0:
            corr = df_temp[[target_col, 'covariate_lagged']].corr().iloc[0, 1]
            correlations.append(corr)
        else:
            correlations.append(np.nan)

    # Optimal lag
    correlations_abs = [abs(c) if not np.isnan(c) else 0 for c in correlations]
    optimal_lag_idx = np.argmax(correlations_abs)
    optimal_lag = lags[optimal_lag_idx]
    optimal_corr = correlations[optimal_lag_idx]

    return list(lags), correlations, optimal_lag, optimal_corr


def normality_test(series: pd.Series, name: str = "Series") -> dict:
    """
    Normality Test (Shapiro-Wilk).

    Args:
        series: Time Series
        name: Variable name

    Returns:
        dict with results
    """
    series = series.dropna()

    # Subsample if too large (Shapiro-Wilk limit 5000)
    if len(series) > 5000:
        series = series.sample(5000, random_state=42)

    shapiro_stat, shapiro_pvalue = stats.shapiro(series)
    is_normal = shapiro_pvalue > 0.05

    return {
        'Variable': name,
        'Shapiro-Wilk Statistic': shapiro_stat,
        'P-value': shapiro_pvalue,
        'Is Normal': is_normal
    }
