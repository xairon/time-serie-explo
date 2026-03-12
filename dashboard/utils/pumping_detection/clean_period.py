"""Clean period identification from Pastas residuals.

Algorithm:
1. Compute amplitude threshold T = n_sigma * std(residuals).
2. Rolling Ljung-Box test (180-day window, max lag 30, alpha 0.05).
3. Day is clean if |r(t)| < T AND Ljung-Box p-value > alpha.
4. Merge contiguous clean days, discard windows < min_window_days.
5. If total < min_total_days, relax threshold iteratively.
6. Fallback: seasonal heuristic (Nov 1 - Mar 31).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

logger = logging.getLogger(__name__)


class CleanPeriodSelector:
    """Select temporal windows where Pastas explains the data well."""

    def __init__(
        self,
        n_sigma: float = 2.0,
        rolling_window: int = 180,
        max_lag: int = 30,
        alpha: float = 0.05,
        min_window_days: int = 90,
        min_total_days: int = 365,
    ):
        self.n_sigma = n_sigma
        self.rolling_window = rolling_window
        self.max_lag = max_lag
        self.alpha = alpha
        self.min_window_days = min_window_days
        self.min_total_days = min_total_days

    def select(self, residuals: pd.Series) -> dict[str, Any]:
        """Identify clean periods in residuals.

        Returns dict with: mask (bool Series), n_clean_days, pct_clean,
        windows (list of (start, end) tuples), method.
        """
        residuals_clean = residuals.dropna()

        best_mask = None
        best_windows: list = []
        best_total = 0
        best_sigma = self.n_sigma

        for sigma_mult in [self.n_sigma, 3.0, 4.0]:
            mask = self._compute_mask(residuals_clean, sigma_mult)
            windows = self._merge_windows(mask)
            total_clean = mask.sum()
            if total_clean >= self.min_total_days:
                return self._build_result(residuals, mask, windows, f"auto_{sigma_mult}sigma")
            if windows and total_clean > best_total:
                best_mask = mask
                best_windows = windows
                best_total = total_clean
                best_sigma = sigma_mult

        # Use best auto-detection if it found at least one valid window,
        # otherwise fall back to seasonal heuristic.
        if best_mask is not None:
            return self._build_result(
                residuals, best_mask, best_windows, f"auto_{best_sigma}sigma_partial"
            )

        mask = self._seasonal_heuristic(residuals)
        windows = self._merge_windows(mask)
        return self._build_result(residuals, mask, windows, "seasonal_heuristic")

    def _compute_mask(self, residuals: pd.Series, sigma_mult: float) -> pd.Series:
        """Compute clean mask: amplitude + Ljung-Box criteria."""
        threshold = sigma_mult * residuals.std()
        amp_clean = residuals.abs() < threshold

        lb_clean = pd.Series(False, index=residuals.index)
        half_win = self.rolling_window // 2

        # Step of 10 days provides dense enough sampling to detect clean windows
        # while marking the inner half (±half_win//2) of each passing window.
        step = 10
        spread = half_win // 2
        for i in range(half_win, len(residuals) - half_win, step):
            window = residuals.iloc[max(0, i - half_win):i + half_win]
            if len(window) < self.max_lag + 1:
                continue
            try:
                lb = acorr_ljungbox(window, lags=[self.max_lag], return_df=True)
                pval = lb["lb_pvalue"].iloc[0]
                if pval > self.alpha:
                    # Mark the inner half of the window as clean
                    start_idx = max(0, i - spread)
                    end_idx = min(len(residuals), i + spread)
                    lb_clean.iloc[start_idx:end_idx] = True
            except Exception:
                continue

        return amp_clean & lb_clean

    def _seasonal_heuristic(self, residuals: pd.Series) -> pd.Series:
        """Nov 1 - Mar 31 presumed clean (no agricultural pumping)."""
        months = residuals.index.month
        return pd.Series((months >= 11) | (months <= 3), index=residuals.index)

    def _merge_windows(self, mask: pd.Series) -> list[tuple[str, str]]:
        """Merge contiguous clean days into windows, discard < min_window_days."""
        windows = []
        in_window = False
        start = None

        for i, (date, is_clean) in enumerate(mask.items()):
            if is_clean and not in_window:
                start = date
                in_window = True
            elif not is_clean and in_window:
                duration = (date - start).days
                if duration >= self.min_window_days:
                    windows.append((str(start.date()), str(date.date())))
                in_window = False

        if in_window and start is not None:
            duration = (mask.index[-1] - start).days
            if duration >= self.min_window_days:
                windows.append((str(start.date()), str(mask.index[-1].date())))

        return windows

    def _build_result(
        self, residuals: pd.Series, mask: pd.Series, windows: list, method: str
    ) -> dict[str, Any]:
        full_mask = pd.Series(False, index=residuals.index)
        full_mask.loc[mask.index] = mask
        n_clean = int(full_mask.sum())
        return {
            "mask": full_mask,
            "n_clean_days": n_clean,
            "pct_clean": round(100 * n_clean / len(residuals), 1),
            "windows": windows,
            "method": method,
        }
