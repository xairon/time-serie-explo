"""Change point detection on Pastas residuals via PELT and optional BEAST."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import Rbeast
    BEAST_AVAILABLE = True
except ImportError:
    BEAST_AVAILABLE = False


class ChangepointDetector:
    """Detect change points in residual time series."""

    def __init__(self, method: str = "pelt", min_segment_length: int = 90):
        self.method = method
        self.min_segment_length = min_segment_length

    def detect(self, residuals: pd.Series) -> dict[str, Any]:
        """Detect changepoints. Returns dict with 'changepoints' list and 'method_used'."""
        results: dict[str, Any] = {"changepoints": [], "method_used": []}

        if self.method in ("pelt", "both"):
            pelt_cps = self._run_pelt(residuals)
            results["changepoints"].extend(pelt_cps)
            results["method_used"].append("pelt")

        if self.method in ("beast", "both") and BEAST_AVAILABLE:
            beast_cps = self._run_beast(residuals)
            results["changepoints"].extend(beast_cps)
            results["method_used"].append("beast")
        elif self.method == "beast" and not BEAST_AVAILABLE:
            logger.warning("BEAST not available, falling back to PELT")
            pelt_cps = self._run_pelt(residuals)
            results["changepoints"].extend(pelt_cps)
            results["method_used"].append("pelt_fallback")

        return results

    def _run_pelt(self, residuals: pd.Series) -> list[dict[str, Any]]:
        """PELT change point detection via ruptures."""
        import ruptures

        signal = residuals.dropna().values
        algo = ruptures.Pelt(model="rbf", min_size=self.min_segment_length).fit(signal)
        breakpoints = algo.predict(pen=3)

        changepoints = []
        for bp in breakpoints[:-1]:
            date = residuals.dropna().index[min(bp, len(signal) - 1)]
            changepoints.append({
                "index": int(bp),
                "date": str(date.date()),
                "method": "pelt",
                "confidence": None,
            })
        return changepoints

    def _run_beast(self, residuals: pd.Series) -> list[dict[str, Any]]:
        """BEAST Bayesian change point detection."""
        signal = residuals.dropna().values
        result = Rbeast.beast(signal, season="none")

        changepoints = []
        if hasattr(result, "trend") and hasattr(result.trend, "cp"):
            for i, cp_idx in enumerate(result.trend.cp):
                if np.isnan(cp_idx):
                    continue
                idx = int(cp_idx)
                date = residuals.dropna().index[min(idx, len(signal) - 1)]
                prob = float(result.trend.cpPr[i]) if hasattr(result.trend, "cpPr") else None
                changepoints.append({
                    "index": idx,
                    "date": str(date.date()),
                    "method": "beast",
                    "confidence": prob,
                })
        return changepoints
