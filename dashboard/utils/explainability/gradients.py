"""Gradient-based explanations using Captum.

Includes:
- Temporal saliency (which timesteps matter)
- Integrated Gradients (stable attributions)
- General gradient attributions

Note: Requires 'captum' package. Install with: pip install captum
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

warnings.filterwarnings("ignore")

# Check if captum is available
CAPTUM_AVAILABLE = False
try:
    import captum
    CAPTUM_AVAILABLE = True
except ImportError:
    pass


def _check_captum():
    """Check if captum is available and raise helpful error if not."""
    if not CAPTUM_AVAILABLE:
        raise ImportError(
            "Captum is not installed. Gradient-based explanations require captum.\n"
            "Install with: pip install captum\n"
            "Or use correlation-based methods instead."
        )


class GradientExplainer:
    """Gradient-based explainer for PyTorch models wrapped in Darts."""

    def __init__(self, model, input_chunk_length: int = 30, output_chunk_length: int = 7):
        """
        Initialize gradient explainer.

        Args:
            model: Darts forecasting model (must be PyTorch-based)
            input_chunk_length: Length of input context window
            output_chunk_length: Prediction horizon
        """
        self.darts_model = model
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self._torch_model = None
        self._device = None

    @property
    def torch_model(self):
        """Get underlying PyTorch model."""
        if self._torch_model is None:
            self._torch_model = self._unwrap_model()
        return self._torch_model

    @property
    def device(self):
        """Get model device."""
        if self._device is None:
            try:
                import torch
                self._device = next(self.torch_model.parameters()).device
            except Exception:
                import torch
                self._device = torch.device("cpu")
        return self._device

    def _unwrap_model(self):
        """Unwrap Darts model to get PyTorch model."""
        try:
            # Darts wraps: model.model (PLModule) -> model.model.model (actual nn.Module)
            if hasattr(self.darts_model, "model"):
                pl_module = self.darts_model.model
                if hasattr(pl_module, "model"):
                    return pl_module.model
                return pl_module
        except Exception as e:
            raise ValueError(f"Could not unwrap PyTorch model: {e}")

        raise ValueError("Model is not a PyTorch model")

    def _prepare_input(
        self,
        series,
        past_covariates=None,
        future_covariates=None
    ) -> "torch.Tensor":
        """
        Prepare input tensor from Darts TimeSeries.

        Args:
            series: Target TimeSeries
            past_covariates: Past covariates TimeSeries
            future_covariates: Future covariates TimeSeries

        Returns:
            PyTorch tensor ready for model input
        """
        import torch

        # Extract values from target series
        target_values = series.values()[-self.input_chunk_length:]

        # Start with target as first feature
        features = [target_values]

        # Add past covariates if available
        if past_covariates is not None:
            cov_values = past_covariates.values()[-self.input_chunk_length:]
            if len(cov_values.shape) == 1:
                cov_values = cov_values.reshape(-1, 1)
            features.append(cov_values)

        # Stack features: (seq_len, n_features)
        input_array = np.concatenate(features, axis=1)

        # Convert to tensor: (batch=1, seq_len, n_features)
        input_tensor = torch.tensor(
            input_array.reshape(1, -1, input_array.shape[-1]),
            dtype=torch.float32,
            device=self.device
        )
        input_tensor.requires_grad = True

        return input_tensor

    def compute_saliency(
        self,
        series,
        past_covariates=None,
        future_covariates=None,
        target_step: int = 0
    ) -> Dict[str, Any]:
        """
        Compute vanilla gradient saliency.

        Args:
            series: Target TimeSeries
            past_covariates: Past covariates
            future_covariates: Future covariates
            target_step: Which forecast step to explain

        Returns:
            Dictionary with saliency map and metadata
        """
        if not CAPTUM_AVAILABLE:
            return {
                "success": False,
                "error": "Captum not installed. Install with: pip install captum",
                "method": "saliency",
            }

        try:
            import torch
            from captum.attr import Saliency

            input_tensor = self._prepare_input(series, past_covariates, future_covariates)

            # Create forward function that returns specific output step
            def forward_fn(x):
                # Model expects specific input format
                output = self.torch_model(x)
                # Select target step
                if output.dim() == 3:
                    return output[:, target_step, 0]
                elif output.dim() == 2:
                    return output[:, target_step]
                return output.squeeze()

            saliency = Saliency(forward_fn)
            attributions = saliency.attribute(input_tensor)

            attr_np = attributions.detach().cpu().numpy().squeeze()

            return {
                "success": True,
                "attributions": attr_np,  # Shape: (seq_len, n_features)
                "temporal_importance": np.abs(attr_np).mean(axis=1),  # Per timestep
                "feature_importance": np.abs(attr_np).mean(axis=0),  # Per feature
                "method": "saliency",
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "method": "saliency",
            }

    def compute_integrated_gradients(
        self,
        series,
        past_covariates=None,
        future_covariates=None,
        target_step: int = 0,
        n_steps: int = 50,
        baseline: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compute Integrated Gradients attributions.

        More stable than vanilla saliency, satisfies completeness axiom.

        Args:
            series: Target TimeSeries
            past_covariates: Past covariates
            future_covariates: Future covariates
            target_step: Which forecast step to explain
            n_steps: Number of integration steps
            baseline: Custom baseline (default: zeros)

        Returns:
            Dictionary with IG attributions and metadata
        """
        if not CAPTUM_AVAILABLE:
            return {
                "success": False,
                "error": "Captum not installed. Install with: pip install captum",
                "method": "integrated_gradients",
            }

        try:
            import torch
            from captum.attr import IntegratedGradients

            input_tensor = self._prepare_input(series, past_covariates, future_covariates)

            # Create baseline if not provided
            if baseline is None:
                baseline_tensor = torch.zeros_like(input_tensor)
            else:
                baseline_tensor = torch.tensor(
                    baseline.reshape(input_tensor.shape),
                    dtype=torch.float32,
                    device=self.device
                )

            def forward_fn(x):
                output = self.torch_model(x)
                if output.dim() == 3:
                    return output[:, target_step, 0]
                elif output.dim() == 2:
                    return output[:, target_step]
                return output.squeeze()

            ig = IntegratedGradients(forward_fn)
            attributions, delta = ig.attribute(
                input_tensor,
                baselines=baseline_tensor,
                n_steps=n_steps,
                return_convergence_delta=True
            )

            attr_np = attributions.detach().cpu().numpy().squeeze()

            return {
                "success": True,
                "attributions": attr_np,
                "temporal_importance": np.abs(attr_np).mean(axis=1),
                "feature_importance": np.abs(attr_np).mean(axis=0),
                "convergence_delta": float(delta.detach().cpu().numpy()),
                "method": "integrated_gradients",
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "method": "integrated_gradients",
            }

    def compute_deeplift(
        self,
        series,
        past_covariates=None,
        future_covariates=None,
        target_step: int = 0
    ) -> Dict[str, Any]:
        """
        Compute DeepLIFT attributions.

        Args:
            series: Target TimeSeries
            past_covariates: Past covariates
            future_covariates: Future covariates
            target_step: Which forecast step to explain

        Returns:
            Dictionary with DeepLIFT attributions
        """
        if not CAPTUM_AVAILABLE:
            return {
                "success": False,
                "error": "Captum not installed. Install with: pip install captum",
                "method": "deeplift",
            }

        try:
            import torch
            from captum.attr import DeepLift

            input_tensor = self._prepare_input(series, past_covariates, future_covariates)
            baseline_tensor = torch.zeros_like(input_tensor)

            def forward_fn(x):
                output = self.torch_model(x)
                if output.dim() == 3:
                    return output[:, target_step, 0]
                elif output.dim() == 2:
                    return output[:, target_step]
                return output.squeeze()

            dl = DeepLift(forward_fn)
            attributions = dl.attribute(input_tensor, baselines=baseline_tensor)

            attr_np = attributions.detach().cpu().numpy().squeeze()

            return {
                "success": True,
                "attributions": attr_np,
                "temporal_importance": np.abs(attr_np).mean(axis=1),
                "feature_importance": np.abs(attr_np).mean(axis=0),
                "method": "deeplift",
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "method": "deeplift",
            }


def compute_temporal_saliency(
    model,
    series,
    past_covariates=None,
    future_covariates=None,
    target_step: int = 0,
    input_chunk_length: int = 30
) -> Optional[np.ndarray]:
    """
    Compute which timesteps influence the prediction most.

    Convenience function wrapping GradientExplainer.

    Args:
        model: Darts forecasting model
        series: Target TimeSeries
        past_covariates: Past covariates
        future_covariates: Future covariates
        target_step: Forecast step to explain
        input_chunk_length: Input window size

    Returns:
        Array of temporal importance (length = input_chunk_length)
    """
    try:
        explainer = GradientExplainer(model, input_chunk_length)
        result = explainer.compute_saliency(
            series, past_covariates, future_covariates, target_step
        )

        if result["success"]:
            return result["temporal_importance"]
        return None

    except Exception as e:
        print(f"Temporal saliency failed: {e}")
        return None


def compute_integrated_gradients(
    model,
    series,
    past_covariates=None,
    future_covariates=None,
    target_step: int = 0,
    input_chunk_length: int = 30,
    n_steps: int = 50
) -> Optional[np.ndarray]:
    """
    Compute Integrated Gradients attributions.

    Convenience function wrapping GradientExplainer.

    Args:
        model: Darts forecasting model
        series: Target TimeSeries
        past_covariates: Past covariates
        future_covariates: Future covariates
        target_step: Forecast step to explain
        input_chunk_length: Input window size
        n_steps: Integration steps

    Returns:
        Attribution matrix (seq_len × n_features)
    """
    try:
        explainer = GradientExplainer(model, input_chunk_length)
        result = explainer.compute_integrated_gradients(
            series, past_covariates, future_covariates, target_step, n_steps
        )

        if result["success"]:
            return result["attributions"]
        return None

    except Exception as e:
        print(f"Integrated Gradients failed: {e}")
        return None


def compute_gradient_attributions(
    model,
    series,
    past_covariates=None,
    future_covariates=None,
    method: str = "integrated_gradients",
    target_step: int = 0,
    input_chunk_length: int = 30,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute gradient-based attributions using specified method.

    Args:
        model: Darts forecasting model
        series: Target TimeSeries
        past_covariates: Past covariates
        future_covariates: Future covariates
        method: 'saliency', 'integrated_gradients', or 'deeplift'
        target_step: Forecast step to explain
        input_chunk_length: Input window size
        **kwargs: Additional method-specific arguments

    Returns:
        Dictionary with attributions and metadata
    """
    try:
        explainer = GradientExplainer(model, input_chunk_length)

        if method == "saliency":
            return explainer.compute_saliency(
                series, past_covariates, future_covariates, target_step
            )
        elif method == "integrated_gradients":
            return explainer.compute_integrated_gradients(
                series, past_covariates, future_covariates, target_step,
                n_steps=kwargs.get("n_steps", 50)
            )
        elif method == "deeplift":
            return explainer.compute_deeplift(
                series, past_covariates, future_covariates, target_step
            )
        else:
            return {
                "success": False,
                "error": f"Unknown method: {method}",
                "method": method,
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "method": method,
        }


def create_gradient_heatmap_data(
    attributions: np.ndarray,
    feature_names: List[str],
    input_chunk_length: int
) -> pd.DataFrame:
    """
    Convert attribution matrix to DataFrame for heatmap visualization.

    Args:
        attributions: Attribution matrix (seq_len × n_features)
        feature_names: List of feature names
        input_chunk_length: Input window size

    Returns:
        DataFrame with columns: timestep, feature, attribution
    """
    rows = []
    seq_len, n_features = attributions.shape

    for t in range(seq_len):
        for f_idx, f_name in enumerate(feature_names[:n_features]):
            rows.append({
                "timestep": f"t-{seq_len - t}",
                "timestep_idx": seq_len - t,
                "feature": f_name,
                "attribution": float(attributions[t, f_idx]),
                "abs_attribution": abs(float(attributions[t, f_idx])),
            })

    return pd.DataFrame(rows)
