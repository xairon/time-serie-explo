"""Adapter to use Darts models (TFT, GRU, LSTM, etc.) with PhysCF.

Darts models wrap PyTorch Lightning modules. This adapter extracts
the underlying PyTorch nn.Module and provides a unified interface
for gradient-based counterfactual optimization:

    adapter = DartsModelAdapter(darts_model, input_chunk_length, output_chunk_length)
    y_hat = adapter(h_obs, s_obs)  # (batch, H)

The adapter handles:
- Extracting the PyTorch module from Darts
- Converting (h_obs, s_obs) format to Darts internal tensor format
- Freezing model weights while allowing gradient flow through inputs
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import numpy as np

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
        darts_model: Trained Darts model (TFTModel, RNNModel, etc.)
        input_chunk_length: L (lookback window)
        output_chunk_length: H (forecast horizon)
    """

    def __init__(
        self,
        darts_model,
        input_chunk_length: Optional[int] = None,
        output_chunk_length: Optional[int] = None,
    ):
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

    @staticmethod
    def _extract_pytorch_module(darts_model) -> nn.Module:
        """Extract the raw nn.Module from a Darts model."""
        # Darts stores the PyTorch Lightning module in .model
        if hasattr(darts_model, "model") and isinstance(darts_model.model, nn.Module):
            pl_module = darts_model.model
        else:
            raise ValueError(
                f"Cannot extract PyTorch module from {type(darts_model).__name__}. "
                "Make sure the model is a trained Darts TorchForecastingModel."
            )

        return pl_module

    @staticmethod
    def _detect_model_type(darts_model) -> str:
        """Detect the Darts model type for format-specific handling."""
        cls_name = type(darts_model).__name__.lower()
        if "tft" in cls_name:
            return "tft"
        elif "rnn" in cls_name or "lstm" in cls_name or "gru" in cls_name:
            return "rnn"
        elif "nbeats" in cls_name or "nhits" in cls_name:
            return "nbeats"
        elif "tcn" in cls_name:
            return "tcn"
        elif "transformer" in cls_name:
            return "transformer"
        elif "tsmixer" in cls_name:
            return "tsmixer"
        elif "tide" in cls_name:
            return "tide"
        elif "dlinear" in cls_name or "nlinear" in cls_name:
            return "linear"
        else:
            return "generic"

    def _prepare_input(
        self, h_obs: torch.Tensor, s_obs: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Convert (h_obs, s_obs) to the format expected by Darts' internal module.

        Darts TorchForecastingModel.model expects:
            For models with past_covariates:
                x = (past_target, past_covariates, ...)
                past_target: (batch, L, 1) - the target variable
                past_covariates: (batch, L, n_cov) - covariates
            Some models concatenate everything into a single tensor.

        We handle each model type specifically.
        """
        batch_size = h_obs.shape[0] if h_obs.dim() > 1 else 1

        if h_obs.dim() == 1:
            h_obs = h_obs.unsqueeze(0)  # (1, L)
        if h_obs.dim() == 2:
            h_obs = h_obs.unsqueeze(-1)  # (batch, L, 1)
        if s_obs.dim() == 2:
            s_obs = s_obs.unsqueeze(0)  # (1, L, 3)

        L = h_obs.shape[1]

        # Concatenate target + covariates for Darts internal format
        # Most Darts models expect: (batch, L, n_features)
        x_past = torch.cat([h_obs, s_obs], dim=-1)  # (batch, L, 4)

        return x_past, batch_size, L

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
        x_past, batch_size, L = self._prepare_input(h_obs, s_obs)

        # Build the input tuple expected by Darts _PLModule.forward()
        # Darts' internal _PLModule.forward() signature depends on model type
        # but generally:
        #   (past_target, past_covariates, historic_future_covariates,
        #    future_covariates, future_past_covariates)
        #
        # For our case (past_covariates only):
        past_target = h_obs if h_obs.dim() == 3 else h_obs.unsqueeze(-1)
        past_covariates = s_obs if s_obs.dim() == 3 else s_obs.unsqueeze(0)

        try:
            # Try the standard Darts PLModule forward
            output = self._forward_darts_module(past_target, past_covariates)
        except Exception:
            # Fallback: concatenate and pass as single tensor
            output = self._forward_concatenated(x_past)

        # Ensure output is (batch, H)
        if output.dim() == 3:
            output = output[:, :, 0]  # (batch, H, 1) -> (batch, H)
        if output.dim() == 1:
            output = output.unsqueeze(0)

        return output[:, : self.output_chunk_length]

    def _forward_darts_module(
        self, past_target: torch.Tensor, past_covariates: torch.Tensor
    ) -> torch.Tensor:
        """Forward through Darts PLModule with proper input tuple."""
        pl_module = self._pytorch_module

        # Darts PLModule forward expects a tuple of tensors
        # The exact format varies by model, but the pattern is:
        # (past_target, past_covariates, historic_future_cov, future_cov, future_past_cov)
        # For models without future covariates, we pass None

        H = self.output_chunk_length
        B = past_target.shape[0]
        device = past_target.device

        # Create dummy static covariates (None for most models)
        none_tensor = None

        # For RNN models (LSTM, GRU)
        if self._model_type == "rnn":
            # RNNModel expects (past_target, past_covariates) concatenated
            # and processes through GRU/LSTM
            x = torch.cat([past_target, past_covariates], dim=-1)
            # Access the actual RNN layers
            if hasattr(pl_module, "rnn"):
                out, _ = pl_module.rnn(x)
                last = out[:, -1, :]
                if hasattr(pl_module, "fc"):
                    return pl_module.fc(last)
                elif hasattr(pl_module, "V"):
                    return pl_module.V(last)

        # For TFT and other complex models, use the internal forward
        # Build the standardized input tuple
        input_tuple = (
            past_target,          # past target values
            past_covariates,      # past covariates
            none_tensor,          # historic future covariates
            none_tensor,          # future covariates
            none_tensor,          # future_past_covariates
        )

        output = pl_module(input_tuple)

        # Darts output is typically (batch, H, n_targets, n_samples)
        # or (batch, H, n_targets)
        if isinstance(output, tuple):
            output = output[0]

        return output

    def _forward_concatenated(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback: pass concatenated [target, covariates] as single tensor."""
        pl_module = self._pytorch_module

        # Try direct forward with concatenated input
        if hasattr(pl_module, "forward"):
            return pl_module(x)

        raise RuntimeError(
            f"Cannot forward through {type(pl_module).__name__}. "
            "Model architecture not supported by DartsModelAdapter."
        )

    def freeze_weights(self):
        """Freeze all model weights (for CF optimization)."""
        for p in self._pytorch_module.parameters():
            p.requires_grad_(False)

    def to_train_mode(self):
        """Set to train mode (required for cuDNN RNN backward)."""
        self._pytorch_module.train()
        self.freeze_weights()


class StandaloneGRUAdapter(nn.Module):
    """Adapter for PhysCF's standalone GRUForecaster (non-Darts).

    This is a pass-through that provides the same interface as DartsModelAdapter
    but wraps the original PhysCF GRUForecaster.
    """

    def __init__(self, gru_model: nn.Module):
        super().__init__()
        self.model = gru_model

    def forward(
        self, h_obs: torch.Tensor, s_obs: torch.Tensor
    ) -> torch.Tensor:
        return self.model(h_obs, s_obs)

    def freeze_weights(self):
        for p in self.model.parameters():
            p.requires_grad_(False)

    def to_train_mode(self):
        self.model.train()
        self.freeze_weights()
