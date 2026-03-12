# dashboard/utils/pumping_detection/xai_layer.py
"""Layer 2: XAI attribution drift analysis."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two probability distributions."""
    p = np.abs(p) + 1e-12
    q = np.abs(q) + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    return float(jensenshannon(p, q) ** 2)


def feature_agreement(ranking_a: list[int], ranking_b: list[int], k: int = 3) -> float:
    """Fraction of top-K features that overlap between two rankings."""
    top_a = set(ranking_a[:k])
    top_b = set(ranking_b[:k])
    return len(top_a & top_b) / k


def compute_window_drift(
    ref_attributions: np.ndarray,
    test_attributions: np.ndarray,
    k: int = 3,
) -> dict[str, float]:
    """Compute drift metrics between reference and test attribution matrices."""
    ref_importance = np.abs(ref_attributions).mean(axis=0)
    test_importance = np.abs(test_attributions).mean(axis=0)

    jsd = js_divergence(ref_importance, test_importance)
    corr, _ = spearmanr(ref_importance, test_importance)

    ref_ranking = np.argsort(-ref_importance).tolist()
    test_ranking = np.argsort(-test_importance).tolist()
    k_actual = min(k, len(ref_importance))
    fa = feature_agreement(ref_ranking, test_ranking, k=k_actual)

    return {
        "js_divergence": float(jsd),
        "spearman_corr": float(corr) if not np.isnan(corr) else 0.0,
        "feature_agreement": float(fa),
    }


class XAIDriftAnalyzer:
    """Compute XAI attributions and drift metrics across time windows."""

    def __init__(self, methods: list[str] | None = None, window_size: int = 90, stride: int = 30):
        self.methods = methods or ["integrated_gradients"]
        self.window_size = window_size
        self.stride = stride

    def analyze(
        self,
        model: Any,
        series: Any,
        covariates: Any,
        clean_mask: Any,
        feature_names: list[str],
    ) -> dict[str, Any]:
        """Compute attributions on all windows, then drift from clean baseline."""
        from dashboard.utils.explainability.gradients import compute_integrated_gradients

        all_attributions = []
        window_dates = []
        n_steps = len(series)

        for start in range(0, n_steps - self.window_size, self.stride):
            end = start + self.window_size
            try:
                window_series = series[start:end]
                window_cov = covariates[start:end] if covariates is not None else None
                attrs = compute_integrated_gradients(
                    model, window_series,
                    past_covariates=window_cov,
                    input_chunk_length=min(self.window_size, 30),
                )
                all_attributions.append(attrs)
                mid_date = series.time_index[start + self.window_size // 2]
                window_dates.append(str(mid_date.date()))
            except Exception as e:
                logger.warning(f"IG failed for window {start}-{end}: {e}")
                continue

        if not all_attributions:
            return {"attributions": [], "drift_metrics": [], "feature_names": feature_names}

        attributions = np.array(all_attributions)

        clean_indices = []
        for i, date_str in enumerate(window_dates):
            window_start = i * self.stride
            window_end = window_start + self.window_size
            if window_end <= len(clean_mask):
                pct_clean = clean_mask.iloc[window_start:window_end].mean()
                if pct_clean > 0.7:
                    clean_indices.append(i)

        if not clean_indices:
            logger.warning("No clean windows found for XAI baseline")
            return {
                "attributions": attributions.tolist(),
                "drift_metrics": [],
                "feature_names": feature_names,
                "window_dates": window_dates,
            }

        ref_attrs = attributions[clean_indices]

        drift_metrics = []
        for i in range(len(attributions)):
            drift = compute_window_drift(ref_attrs, attributions[i:i+1])
            drift["window_date"] = window_dates[i]
            drift["is_clean"] = i in clean_indices
            drift_metrics.append(drift)

        return {
            "attributions": attributions.tolist(),
            "drift_metrics": drift_metrics,
            "feature_names": feature_names,
            "window_dates": window_dates,
        }
