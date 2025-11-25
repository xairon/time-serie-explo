"""Fonctions de tests statistiques."""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf, grangercausalitytests
from statsmodels.tsa.seasonal import STL
from darts import TimeSeries
from darts.utils.statistics import check_seasonality
import streamlit as st


@st.cache_data(ttl=3600)
def test_stationarity(series: pd.Series, name: str = "Series") -> dict:
    """
    Tests de stationnarité ADF et KPSS.

    Args:
        series: Série temporelle (pandas Series)
        name: Nom de la variable

    Returns:
        dict avec résultats des tests
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


@st.cache_data(ttl=3600)
def check_seasonality_darts(ts: TimeSeries, periods: list = None, max_lag: int = 400) -> dict:
    """
    Détection de saisonnalité avec Darts.

    Args:
        ts: TimeSeries Darts
        periods: Liste des périodes à tester (défaut: [7, 30, 365])
        max_lag: Lag maximum pour le test

    Returns:
        dict avec résultats par période
    """
    if periods is None:
        periods = [7, 30, 365]

    results = {}

    for period in periods:
        period_name = {7: 'Weekly', 30: 'Monthly', 365: 'Annual'}.get(period, f'{period}-day')

        try:
            # Ajuster max_lag si nécessaire
            actual_max_lag = min(max_lag, max(period * 2, len(ts) - 1))
            is_seasonal = check_seasonality(ts, m=period, max_lag=actual_max_lag, alpha=0.05)

            # ACF au lag de la période
            acf_values = acf(ts.values().flatten(), nlags=min(period + 10, len(ts) - 1))
            acf_at_period = acf_values[period] if period < len(acf_values) else 0

            results[period_name] = {
                'period': period,
                'detected': is_seasonal,
                'acf_at_period': acf_at_period
            }
        except:
            results[period_name] = {
                'period': period,
                'detected': False,
                'acf_at_period': 0
            }

    return results


@st.cache_data(ttl=3600)
def stl_decomposition(series: pd.Series, seasonal: int = 365, trend: int = None) -> dict:
    """
    Décomposition STL (Seasonal-Trend-Loess).

    Args:
        series: Série temporelle (pandas Series avec index datetime)
        seasonal: Période de saisonnalité
        trend: Période de tendance (défaut: calculé automatiquement)

    Returns:
        dict avec trend, seasonal, residual, et variance contributions
    """
    if trend is None:
        # Calculer un trend impair et > seasonal
        trend = max(seasonal + 1, 3)
        # S'assurer que c'est impair
        if trend % 2 == 0:
            trend += 1
    else:
        # Valider et corriger trend
        trend = max(trend, seasonal + 1, 3)  # trend > seasonal et >= 3
        if trend % 2 == 0:
            trend += 1  # Rendre impair

    # STL decomposition
    stl = STL(series, seasonal=seasonal, trend=trend)
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


@st.cache_data(ttl=3600)
def cross_correlation(x: np.ndarray, y: np.ndarray, max_lag: int = 60) -> tuple:
    """
    Corrélation croisée entre deux séries.
    Lag positif (k) = corr(x[t], y[t+k]) -> x mène y (x leads y).

    Args:
        x: Première série (ex: Pluie)
        y: Deuxième série (ex: Niveau)
        max_lag: Lag maximum

    Returns:
        tuple: (lags, correlations)
    """
    correlations = []
    lags = range(-max_lag, max_lag + 1)

    for lag in lags:
        if lag > 0:
            # Lag positif : x leads y
            # On compare x[0:N-lag] avec y[lag:N]
            # x[t] vs y[t+lag]
            if len(x[:-lag]) == len(y[lag:]):
                corr = np.corrcoef(x[:-lag], y[lag:])[0, 1]
            else:
                corr = 0
        elif lag < 0:
            # Lag négatif : y leads x
            # On compare x[-lag:N] avec y[0:N+lag]
            # x[t-lag] vs y[t]  (où -lag > 0)
            if len(x[-lag:]) == len(y[:lag]):
                corr = np.corrcoef(x[-lag:], y[:lag])[0, 1]
            else:
                corr = 0
        else:
            # Lag 0
            corr = np.corrcoef(x, y)[0, 1]

        correlations.append(corr)

    return list(lags), correlations


@st.cache_data(ttl=3600)
def granger_causality_test(df: pd.DataFrame, target_col: str, covariate_col: str, max_lag: int = 30) -> dict:
    """
    Test de causalité de Granger.

    Args:
        df: DataFrame avec les colonnes
        target_col: Colonne cible
        covariate_col: Colonne covariable
        max_lag: Lag maximum

    Returns:
        dict avec lags, p-values, significant_lags
    """
    # Préparer les données
    data = df[[covariate_col, target_col]].dropna()

    try:
        gc_result = grangercausalitytests(data, maxlag=max_lag, verbose=False)

        lags = []
        pvalues = []

        for lag in range(1, max_lag + 1):
            # F-test p-value
            pvalue = gc_result[lag][0]['ssr_ftest'][1]
            lags.append(lag)
            pvalues.append(pvalue)

        # Lags significatifs (p < 0.05)
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


@st.cache_data(ttl=3600)
def calculate_lagged_correlations(df: pd.DataFrame, target_col: str, covariate_col: str, max_lag: int = 60) -> tuple:
    """
    Calcule la corrélation pour différents lags.

    Args:
        df: DataFrame
        target_col: Colonne cible
        covariate_col: Colonne covariable
        max_lag: Lag maximum

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


@st.cache_data(ttl=3600)
def normality_test(series: pd.Series, name: str = "Series") -> dict:
    """
    Test de normalité (Shapiro-Wilk).

    Args:
        series: Série temporelle
        name: Nom de la variable

    Returns:
        dict avec résultats
    """
    series = series.dropna()

    # Sous-échantillonner si trop grand (Shapiro-Wilk limite à 5000)
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
