"""Layer 1: Pastas TFN residual analysis + ACF/PACF diagnostics."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf

from dashboard.utils.counterfactual.pastas_validation import PastasWrapper

logger = logging.getLogger(__name__)


class PastasAnalyzer:
    """Calibrate Pastas recharge-only model and extract residual diagnostics."""

    def __init__(self, response_function: str = "Gamma", noise_model: bool = True):
        self.response_function = response_function
        self.noise_model = noise_model
        self._wrapper = PastasWrapper()

    def analyze(
        self,
        piezo: pd.Series,
        precip: pd.Series,
        etp: pd.Series,
        max_acf_lag: int = 30,
    ) -> dict[str, Any]:
        """Run full Pastas analysis: fit, extract residuals, compute ACF.

        Returns dict with keys: residuals, acf_stats, pastas_fit_quality, modeled.
        """
        tmin = piezo.index.min()
        tmax = piezo.index.max()

        # Fit Pastas
        self._wrapper.fit(piezo, precip, etp, tmin=tmin, tmax=tmax)

        # Get modeled values as pd.Series with date index
        # PastasWrapper.predict() returns a numpy array; we reconstruct the index
        # directly from the Pastas model simulation to get the correct date alignment.
        sim = self._wrapper.model.simulate(tmin=tmin, tmax=tmax)
        modeled = pd.Series(sim.values.flatten(), index=sim.index, name="modeled")

        common_idx = piezo.index.intersection(modeled.index)
        residuals = piezo.loc[common_idx] - modeled.loc[common_idx]
        residuals.name = "residuals"

        # Fit quality
        ss_res = (residuals ** 2).sum()
        ss_tot = ((piezo.loc[common_idx] - piezo.loc[common_idx].mean()) ** 2).sum()
        evp = (1 - ss_res / ss_tot) * 100 if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(ss_res / len(residuals)))

        # ACF / PACF
        nlags = min(max_acf_lag, len(residuals) // 2 - 1)
        acf_values = acf(residuals.dropna(), nlags=nlags, fft=True)
        pacf_values = pacf(residuals.dropna(), nlags=nlags)

        # Ljung-Box test
        lb_result = acorr_ljungbox(residuals.dropna(), lags=[nlags], return_df=True)
        lb_pvalue = float(lb_result["lb_pvalue"].iloc[0])

        return {
            "residuals": residuals,
            "modeled": modeled.loc[common_idx],
            "acf_stats": {
                "acf_values": acf_values.tolist(),
                "pacf_values": pacf_values.tolist(),
                "ljung_box_pvalue": lb_pvalue,
                "nlags": nlags,
            },
            "pastas_fit_quality": {
                "evp": float(evp),
                "rmse": rmse,
                "n_observations": len(common_idx),
            },
        }
