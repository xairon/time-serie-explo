"""Layer 2: ML model training on clean periods + full-series prediction."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MLAnalyzer:
    """Train a TFT model on clean periods and predict on the full series."""

    def __init__(
        self,
        model_type: str = "TFTModel",
        input_chunk_length: int = 365,
        output_chunk_length: int = 30,
        max_epochs: int = 100,
    ):
        self.model_type = model_type
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.max_epochs = max_epochs

    def train_and_predict(
        self,
        target: Any,
        covariates: Any,
        clean_mask: pd.Series,
        stop_event: Any = None,
    ) -> dict[str, Any]:
        """Train TFT on clean periods, predict on full series."""
        from darts import TimeSeries
        from darts.metrics import mae, rmse
        from dashboard.utils.model_factory import ModelFactory

        clean_target = self._filter_to_clean(target, clean_mask)
        if clean_target is None or len(clean_target) < self.input_chunk_length + self.output_chunk_length + 100:
            return {"error": "Insufficient clean data for training", "predictions": None}

        clean_covariates = self._filter_to_clean(covariates, clean_mask)

        split_point = int(len(clean_target) * 0.8)
        train_target = clean_target[:split_point]
        val_target = clean_target[split_point:]
        train_cov = clean_covariates[:split_point] if clean_covariates else None
        val_cov = clean_covariates[split_point:] if clean_covariates else None

        model = ModelFactory.create(
            model_type=self.model_type,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            n_epochs=self.max_epochs,
        )

        model.fit(train_target, past_covariates=train_cov, val_series=val_target, val_past_covariates=val_cov)

        predictions = model.historical_forecasts(
            series=target, past_covariates=covariates,
            start=self.input_chunk_length, forecast_horizon=self.output_chunk_length,
            stride=self.output_chunk_length, retrain=False, last_points_only=True,
        )

        pred_values = predictions.pd_series()
        actual_values = target.pd_series().loc[pred_values.index]
        ml_residuals = actual_values - pred_values

        training_metrics = {
            "mae": float(mae(val_target, model.predict(len(val_target), past_covariates=val_cov))),
            "n_clean_train": split_point,
            "n_clean_val": len(clean_target) - split_point,
        }

        return {
            "predictions": predictions,
            "ml_residuals": ml_residuals,
            "training_metrics": training_metrics,
            "model": model,
        }

    def _filter_to_clean(self, ts: Any, mask: pd.Series) -> Any | None:
        """Extract the longest contiguous clean segment from a TimeSeries."""
        groups = mask.astype(int).diff().ne(0).cumsum()
        clean_lengths = mask.groupby(groups).sum()
        clean_lengths = clean_lengths[clean_lengths > 0]

        if clean_lengths.empty:
            return None

        longest = clean_lengths.idxmax()
        segment_mask = (groups == longest) & mask
        start_date = segment_mask.index[segment_mask].min()
        end_date = segment_mask.index[segment_mask].max()

        return ts.slice(pd.Timestamp(start_date), pd.Timestamp(end_date))
