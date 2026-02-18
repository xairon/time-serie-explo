"""Adapter to use Darts models (TFT, GRU, LSTM, TSMixer, etc.) with PhysCF.

Darts models wrap PyTorch Lightning modules. This adapter extracts
the underlying PyTorch nn.Module and provides a unified interface
for gradient-based counterfactual optimization:

    adapter = DartsModelAdapter(darts_model, input_chunk_length, output_chunk_length)
    y_hat = adapter(h_obs, s_obs)  # (batch, H)

The adapter handles:
- Extracting the PyTorch module from Darts
- Converting (h_obs, s_obs) format to Darts internal 3-tuple format
- Freezing model weights while allowing gradient flow through inputs

All Darts PLForecastingModule subclasses (TFT, TSMixer, TiDE, NBEATS,
RNN, GRU, LSTM, DLinear, NLinear, TCN, Transformer...) expect:
    PLModuleInput = (x_past, x_future, x_static)
where x_past = cat([past_target, past_covariates, historic_future_cov], dim=2).

The conversion from raw 5-tuple to 3-tuple is done by _process_input_batch()
which is defined on every Darts PLModule.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
    DARTS_AVAILABLE = True
except ImportError:
    DARTS_AVAILABLE = False


class DartsModelAdapter(nn.Module):
    """Wraps a trained Darts TorchForecastingModel for PhysCF optimization.

    Provides the same interface as PhysCF's GRUForecaster:
        forward(h_obs, s_obs) -> y_hat

    Where:
        h_obs: (batch, L) or (batch, L, 1) - normalized gwl lookback
        s_obs: (batch, L, 3) - normalized stresses [precip, temp, evap]
        y_hat: (batch, H) - predicted gwl

    Args:
        darts_model: Trained Darts model (TFTModel, RNNModel, TSMixerModel, etc.)
        input_chunk_length: L (lookback window)
        output_chunk_length: H (forecast horizon)
    """

    def __init__(
        self,
        darts_model: "TorchForecastingModel",
        input_chunk_length: Optional[int] = None,
        output_chunk_length: Optional[int] = None,
    ) -> None:
        """Initialize the Darts model adapter.

        Args:
            darts_model: Trained Darts TorchForecastingModel instance.
            input_chunk_length: Override for lookback window L (auto-detected if None).
            output_chunk_length: Override for forecast horizon H (auto-detected if None).
        """
        super().__init__()

        if not DARTS_AVAILABLE:
            raise ImportError("Darts is required: pip install darts")

        self.darts_model = darts_model
        self.input_chunk_length = (
            input_chunk_length
            or getattr(darts_model, "input_chunk_length", 365)
        )
        self.output_chunk_length = (
            output_chunk_length
            or getattr(darts_model, "output_chunk_length", 90)
        )

        # Extract the underlying PyTorch module
        self._pytorch_module = self._extract_pytorch_module(darts_model)
        self._model_type = self._detect_model_type(darts_model)
        logger.info(
            f"DartsModelAdapter: {self._model_type} "
            f"(L={self.input_chunk_length}, H={self.output_chunk_length})"
        )

    @staticmethod
    def _extract_pytorch_module(darts_model) -> nn.Module:
        """Extract the raw nn.Module from a Darts model."""
        if hasattr(darts_model, "model") and isinstance(darts_model.model, nn.Module):
            return darts_model.model
        raise ValueError(
            f"Cannot extract PyTorch module from {type(darts_model).__name__}. "
            "Make sure the model is a trained Darts TorchForecastingModel."
        )

    @staticmethod
    def _detect_model_type(darts_model) -> str:
        """Detect the Darts model type for logging."""
        cls_name = type(darts_model).__name__.lower()
        for key in ("tft", "tsmixer", "tide", "nbeats", "nhits", "tcn",
                     "transformer", "dlinear", "nlinear"):
            if key in cls_name:
                return key
        if any(k in cls_name for k in ("rnn", "lstm", "gru")):
            return "rnn"
        return "generic"

    def forward(
        self, h_obs: torch.Tensor, s_obs: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the wrapped Darts model.

        Args:
            h_obs: (batch, L) or (batch, L, 1) - normalized gwl
            s_obs: (batch, L, 3) - normalized stresses

        Returns:
            y_hat: (batch, H) - predicted gwl
        """
        # --- Shape normalization ---
        if h_obs.dim() == 1:
            h_obs = h_obs.unsqueeze(0)   # (L,) -> (1, L)
        if h_obs.dim() == 2:
            h_obs = h_obs.unsqueeze(-1)  # (B, L) -> (B, L, 1)
        if s_obs.dim() == 2:
            s_obs = s_obs.unsqueeze(0)   # (L, 3) -> (1, L, 3)

        pl_module = self._pytorch_module

        # --- Match dtype to model weights ---
        # Darts models may be trained in float32 or float64; we must match.
        model_dtype = next(pl_module.parameters()).dtype
        if h_obs.dtype != model_dtype:
            h_obs = h_obs.to(model_dtype)
        if s_obs.dtype != model_dtype:
            s_obs = s_obs.to(model_dtype)

        # --- Build Darts raw 5-tuple (TorchBatch format) ---
        # (past_target, past_covariates, historic_future_cov, future_cov, static_cov)
        input_batch = (h_obs, s_obs, None, None, None)

        # --- Convert to PLModuleInput 3-tuple via _process_input_batch ---
        # Returns (x_past, x_future, x_static) where
        # x_past = cat([past_target, past_covariates], dim=2)
        # This method is defined on every Darts PLForecastingModule subclass
        # and handles the model-specific concatenation logic.
        try:
            x_past, x_future, x_static = pl_module._process_input_batch(input_batch)
        except (ValueError, TypeError, AttributeError) as e:
            raise RuntimeError(
                f"DartsModelAdapter: _process_input_batch failed for "
                f"{self._model_type}: {e}. "
                f"Input shapes: h_obs={h_obs.shape}, s_obs={s_obs.shape}"
            ) from e

        # --- Forward through the PLModule ---
        try:
            output = pl_module((x_past, x_future, x_static))
        except (ValueError, RuntimeError) as e:
            raise RuntimeError(
                f"DartsModelAdapter: forward failed for {self._model_type}: {e}. "
                f"x_past={x_past.shape}, x_future={x_future}, x_static={x_static}"
            ) from e

        # Darts output can be tuple (e.g. RNN returns (output, hidden_state))
        if isinstance(output, tuple):
            output = output[0]

        # --- Shape normalization of output ---
        # Darts output: (B, H, n_targets, n_samples) or (B, H, n_targets) or (B, H)
        while output.dim() > 2:
            output = output[..., 0]  # Take first target / first sample
        if output.dim() == 1:
            output = output.unsqueeze(0)

        return output[:, :self.output_chunk_length]

    def freeze_weights(self) -> None:
        """Freeze all model weights (for CF optimization)."""
        for p in self._pytorch_module.parameters():
            p.requires_grad_(False)

    def to_train_mode(self) -> None:
        """Set to train mode (required for cuDNN RNN backward) with frozen weights."""
        self._pytorch_module.train()
        self.freeze_weights()


class StandaloneGRUAdapter(nn.Module):
    """Adapter for PhysCF's standalone GRUForecaster (non-Darts).

    This is a pass-through that provides the same interface as DartsModelAdapter
    but wraps the original PhysCF GRUForecaster.
    """

    def __init__(self, gru_model: nn.Module) -> None:
        """Initialize with a standalone GRU forecaster.

        Args:
            gru_model: A trained PhysCF GRUForecaster nn.Module.
        """
        super().__init__()
        self.model = gru_model

    def forward(
        self, h_obs: torch.Tensor, s_obs: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass delegating to the wrapped GRU model.

        Args:
            h_obs: Normalized gwl lookback tensor.
            s_obs: Normalized stresses tensor.

        Returns:
            Predicted gwl tensor.
        """
        return self.model(h_obs, s_obs)

    def freeze_weights(self) -> None:
        """Freeze all model weights (for CF optimization)."""
        for p in self.model.parameters():
            p.requires_grad_(False)

    def to_train_mode(self) -> None:
        """Set to train mode with frozen weights."""
        self.model.train()
        self.freeze_weights()
