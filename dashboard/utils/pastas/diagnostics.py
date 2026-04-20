"""Compute diagnostic statistics on a fitted Pastas model."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson, jarque_bera


def compute_diagnostics(residuals: pd.Series) -> dict[str, Any]:
    """Full diagnostic suite on model residuals."""
    clean = residuals.dropna()
    n = len(clean)

    result: dict[str, Any] = {"n_residuals": n}

    if n < 10:
        return result

    result["mean"] = float(clean.mean())
    result["std"] = float(clean.std())
    result["skewness"] = float(scipy_stats.skew(clean))
    result["kurtosis"] = float(scipy_stats.kurtosis(clean))

    if n >= 20:
        jb_stat, jb_pvalue = jarque_bera(clean)[:2]
        result["jarque_bera_pvalue"] = float(jb_pvalue)

        _, sw_pvalue = scipy_stats.shapiro(clean[:5000])
        result["shapiro_wilk_pvalue"] = float(sw_pvalue)

    result["durbin_watson"] = float(durbin_watson(clean))

    for lag in [5, 10, 20]:
        if n > lag:
            try:
                lb = acorr_ljungbox(clean, lags=[lag], return_df=True)
                result[f"ljung_box_p_lag{lag}"] = float(lb["lb_pvalue"].iloc[0])
            except Exception:
                pass

    median = clean.median()
    runs = ((clean > median).astype(int).diff().abs().sum() / 2) + 1
    result["runs_count"] = int(runs)

    nlags = min(40, n // 2 - 1)
    if nlags >= 2:
        result["acf_values"] = acf(clean, nlags=nlags, fft=True).tolist()
        result["pacf_values"] = pacf(clean, nlags=nlags).tolist()
        result["nlags"] = nlags
        result["confidence_bound"] = float(1.96 / np.sqrt(n))

    sorted_res = np.sort(clean.values)
    theoretical = scipy_stats.norm.ppf(np.linspace(0.01, 0.99, n))
    result["qq_theoretical"] = theoretical.tolist()
    result["qq_sample"] = sorted_res.tolist()

    counts, bin_edges = np.histogram(clean, bins=30)
    result["hist_counts"] = counts.tolist()
    result["hist_bins"] = bin_edges.tolist()

    return result
