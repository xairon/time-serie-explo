"""Attention-based explanations for TFT and Transformer models.

Includes:
- TFT attention extraction using darts.explainability.TFTExplainer
- Variable Selection Network weights
- Attention heatmap generation
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple


class TFTExplainer:
    """Explainer for Temporal Fusion Transformer models."""

    def __init__(self, model, background_series=None, background_past_covariates=None):
        """
        Initialize TFT Explainer.

        Args:
            model: Trained TFT model from Darts
            background_series: Background series for explanation (optional)
            background_past_covariates: Background past covariates (optional)
        """
        self.model = model
        self.background_series = background_series
        self.background_past_covariates = background_past_covariates
        self._darts_explainer = None

    @property
    def darts_explainer(self):
        """Get or create Darts TFTExplainer."""
        if self._darts_explainer is None:
            self._darts_explainer = self._create_darts_explainer()
        return self._darts_explainer

    def _create_darts_explainer(self):
        """Create Darts TFTExplainer instance."""
        try:
            from darts.explainability.tft_explainer import TFTExplainer as DartsTFTExplainer

            return DartsTFTExplainer(
                model=self.model,
                background_series=self.background_series,
                background_past_covariates=self.background_past_covariates,
            )
        except ImportError:
            raise ImportError("darts.explainability.TFTExplainer not available")
        except Exception as e:
            raise ValueError(f"Failed to create TFTExplainer: {e}")

    def explain(
        self,
        series,
        past_covariates=None,
        future_covariates=None
    ) -> Dict[str, Any]:
        """
        Generate TFT explanation for given input.

        Args:
            series: Target TimeSeries
            past_covariates: Past covariates TimeSeries
            future_covariates: Future covariates TimeSeries

        Returns:
            Dictionary with attention weights, encoder/decoder importance
        """
        try:
            # Use Darts TFTExplainer
            explain_kwargs = {"series": series}
            if past_covariates is not None:
                explain_kwargs["past_covariates"] = past_covariates
            if future_covariates is not None:
                explain_kwargs["future_covariates"] = future_covariates

            result = self.darts_explainer.explain(**explain_kwargs)

            return {
                "success": True,
                "attention": self._extract_attention(result),
                "encoder_importance": self._extract_encoder_importance(result),
                "decoder_importance": self._extract_decoder_importance(result),
                "variable_selection": self._extract_variable_selection(result),
                "raw_result": result,
                "method": "tft_explainer",
            }

        except Exception as e:
            # Fall back to manual extraction
            return self._explain_manual(series, past_covariates, future_covariates, str(e))

    def _extract_attention(self, result) -> Optional[np.ndarray]:
        """Extract attention weights from TFT explanation result."""
        try:
            if hasattr(result, "get_attention"):
                attention = result.get_attention()
                if attention is not None:
                    if isinstance(attention, pd.DataFrame):
                        return attention.values
                    elif hasattr(attention, 'values') and callable(attention.values):
                        return np.array(attention.values())
                    else:
                        return np.array(attention)
            return None
        except Exception:
            return None

    def _extract_encoder_importance(self, result) -> Optional[Dict[str, float]]:
        """Extract encoder variable importance from TFT explanation."""
        try:
            if hasattr(result, "get_encoder_importance"):
                importance = result.get_encoder_importance()
                if importance is not None:
                    if isinstance(importance, pd.DataFrame):
                        return importance.mean().to_dict()
                    elif isinstance(importance, dict):
                        return importance
                    return {"importance": float(importance)}
            return None
        except Exception:
            return None

    def _extract_decoder_importance(self, result) -> Optional[Dict[str, float]]:
        """Extract decoder variable importance from TFT explanation."""
        try:
            if hasattr(result, "get_decoder_importance"):
                importance = result.get_decoder_importance()
                if importance is not None:
                    if isinstance(importance, pd.DataFrame):
                        return importance.mean().to_dict()
                    elif isinstance(importance, dict):
                        return importance
                    return {"importance": float(importance)}
            return None
        except Exception:
            return None

    def _extract_variable_selection(self, result) -> Optional[Dict[str, Any]]:
        """Extract Variable Selection Network weights."""
        try:
            if hasattr(result, "get_variable_selection_weights"):
                return result.get_variable_selection_weights()
            return None
        except Exception:
            return None

    def _explain_manual(
        self,
        series,
        past_covariates,
        future_covariates,
        original_error: str
    ) -> Dict[str, Any]:
        """
        Manual TFT explanation when Darts explainer fails.

        Extracts attention from model forward pass.
        """
        try:
            import torch

            # Get PyTorch model
            torch_model = self.model.model.model
            torch_model.eval()

            # Prepare input (simplified)
            input_chunk = getattr(self.model, "input_chunk_length", 30)
            target_values = series.values()[-input_chunk:]

            # This is a simplified fallback - full implementation would need
            # proper input preparation matching TFT's expected format
            return {
                "success": False,
                "error": f"Darts TFTExplainer failed: {original_error}. Manual extraction not fully implemented.",
                "method": "manual_fallback",
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Both Darts and manual extraction failed: {e}",
                "method": "failed",
            }

    def get_attention_heatmap_data(
        self,
        series,
        past_covariates=None,
        future_covariates=None
    ) -> Optional[pd.DataFrame]:
        """
        Get attention weights as DataFrame for heatmap visualization.

        Args:
            series: Target TimeSeries
            past_covariates: Past covariates
            future_covariates: Future covariates

        Returns:
            DataFrame with columns: input_step, horizon_step, attention_weight
        """
        result = self.explain(series, past_covariates, future_covariates)

        if not result["success"] or result["attention"] is None:
            return None

        attention = result["attention"]

        # Convert to DataFrame format
        rows = []
        if attention.ndim == 2:
            # attention shape: (input_steps, horizon_steps)
            for i in range(attention.shape[0]):
                for j in range(attention.shape[1]):
                    rows.append({
                        "input_step": f"t-{attention.shape[0] - i}",
                        "input_idx": attention.shape[0] - i,
                        "horizon_step": f"h+{j + 1}",
                        "horizon_idx": j + 1,
                        "attention": float(attention[i, j]),
                    })
        elif attention.ndim == 3:
            # attention shape: (heads, input_steps, horizon_steps)
            # Average across heads
            attention_avg = attention.mean(axis=0)
            for i in range(attention_avg.shape[0]):
                for j in range(attention_avg.shape[1]):
                    rows.append({
                        "input_step": f"t-{attention_avg.shape[0] - i}",
                        "input_idx": attention_avg.shape[0] - i,
                        "horizon_step": f"h+{j + 1}",
                        "horizon_idx": j + 1,
                        "attention": float(attention_avg[i, j]),
                    })

        return pd.DataFrame(rows) if rows else None


def extract_tft_attention(
    model,
    series,
    past_covariates=None,
    future_covariates=None,
    background_series=None,
    background_past_covariates=None
) -> Dict[str, Any]:
    """
    Extract TFT attention weights and variable importance.

    Convenience function wrapping TFTExplainer.

    Args:
        model: Trained TFT model
        series: Target TimeSeries to explain
        past_covariates: Past covariates
        future_covariates: Future covariates
        background_series: Background series for explainer
        background_past_covariates: Background past covariates

    Returns:
        Dictionary with:
            - attention: Attention weight matrix
            - encoder_importance: Encoder variable importance
            - decoder_importance: Decoder variable importance
            - variable_selection: VSN weights
    """
    explainer = TFTExplainer(
        model=model,
        background_series=background_series,
        background_past_covariates=background_past_covariates,
    )

    return explainer.explain(series, past_covariates, future_covariates)


def compute_attention_summary(attention_weights: np.ndarray) -> Dict[str, Any]:
    """
    Compute summary statistics from attention weights.

    Args:
        attention_weights: Attention matrix (input_steps × horizon_steps)

    Returns:
        Dictionary with summary statistics
    """
    if attention_weights is None or attention_weights.size == 0:
        return {}

    # Ensure 2D
    if attention_weights.ndim == 3:
        attention_weights = attention_weights.mean(axis=0)

    # Per input timestep importance (average attention received)
    input_importance = attention_weights.mean(axis=1)

    # Per horizon step (which input steps are attended)
    horizon_focus = attention_weights.mean(axis=0)

    # Find most attended input steps
    top_inputs = np.argsort(input_importance)[::-1]

    # Attention concentration (entropy-based)
    flat_attention = attention_weights.flatten()
    flat_attention = flat_attention / (flat_attention.sum() + 1e-10)
    flat_attention_safe = np.clip(flat_attention, 1e-10, None)
    entropy = -np.sum(flat_attention_safe * np.log(flat_attention_safe))
    max_entropy = np.log(len(flat_attention))
    concentration = 1 - (entropy / max_entropy)  # 1 = very concentrated, 0 = uniform

    return {
        "input_importance": input_importance,
        "horizon_focus": horizon_focus,
        "top_input_steps": top_inputs[:5].tolist(),
        "attention_concentration": float(concentration),
        "max_attention": float(attention_weights.max()),
        "mean_attention": float(attention_weights.mean()),
    }
