"""Base classes and types for explainability module."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


class ModelType(Enum):
    """Supported model types for explainability."""
    TFT = auto()
    TSMIXER = auto()
    NHITS = auto()
    NBEATS = auto()
    LSTM = auto()
    GRU = auto()
    TRANSFORMER = auto()
    TCN = auto()
    TIDE = auto()
    DLINEAR = auto()
    NLINEAR = auto()
    GENERIC = auto()

    @classmethod
    def from_model(cls, model) -> "ModelType":
        """Detect model type from Darts model instance."""
        model_name = type(model).__name__.upper()

        mapping = {
            "TFTMODEL": cls.TFT,
            "TSMIXERMODEL": cls.TSMIXER,
            "NHITSMODEL": cls.NHITS,
            "NBEATSMODEL": cls.NBEATS,
            "RNNMODEL": cls.LSTM,  # Default RNN
            "BLOCKRNNMODEL": cls.LSTM,
            "LSTMMODEL": cls.LSTM,
            "GRUMODEL": cls.GRU,
            "TRANSFORMERMODEL": cls.TRANSFORMER,
            "TCNMODEL": cls.TCN,
            "TIDEMODEL": cls.TIDE,
            "DLINEARMODEL": cls.DLINEAR,
            "NLINEARMODEL": cls.NLINEAR,
        }

        return mapping.get(model_name, cls.GENERIC)


@dataclass
class ExplainabilityResult:
    """Container for explainability results."""

    # Feature importance
    feature_importance: Optional[Dict[str, float]] = None

    # Temporal importance (per timestep)
    temporal_importance: Optional[np.ndarray] = None

    # Gradient attributions (feature × time matrix)
    gradient_attributions: Optional[np.ndarray] = None

    # Attention weights (for TFT and transformers)
    attention_weights: Optional[np.ndarray] = None
    encoder_importance: Optional[Dict[str, float]] = None
    decoder_importance: Optional[Dict[str, float]] = None

    # SHAP values
    shap_values: Optional[np.ndarray] = None
    shap_base_value: Optional[float] = None

    # Decomposition
    decomposition: Optional[Dict[str, Any]] = None

    # Metadata
    model_type: Optional[ModelType] = None
    feature_names: List[str] = field(default_factory=list)
    method: str = "unknown"
    timestamp: Optional[pd.Timestamp] = None

    # Status
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {}

        if self.feature_importance is not None:
            result["feature_importance"] = self.feature_importance

        if self.temporal_importance is not None:
            result["temporal_importance"] = self.temporal_importance.tolist()

        if self.gradient_attributions is not None:
            result["gradient_attributions"] = self.gradient_attributions.tolist()

        if self.attention_weights is not None:
            result["attention_weights"] = self.attention_weights.tolist()

        if self.encoder_importance is not None:
            result["encoder_importance"] = self.encoder_importance

        if self.decoder_importance is not None:
            result["decoder_importance"] = self.decoder_importance

        if self.shap_values is not None:
            result["shap_values"] = self.shap_values.tolist()
            result["shap_base_value"] = self.shap_base_value

        if self.decomposition is not None:
            result["decomposition"] = self.decomposition

        result["model_type"] = self.model_type.name if self.model_type else None
        result["feature_names"] = self.feature_names
        result["method"] = self.method
        result["success"] = self.success
        result["error_message"] = self.error_message

        return result


class BaseExplainer(ABC):
    """Abstract base class for model explainers."""

    def __init__(self, model, input_chunk_length: int = 30, output_chunk_length: int = 7):
        """
        Initialize explainer.

        Args:
            model: Darts forecasting model
            input_chunk_length: Length of input context window
            output_chunk_length: Length of prediction horizon
        """
        self.model = model
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.model_type = ModelType.from_model(model)

    @property
    def is_pytorch_model(self) -> bool:
        """Check if model is a PyTorch-based model."""
        try:
            return hasattr(self.model, 'model') and hasattr(self.model.model, 'parameters')
        except Exception:
            return False

    @property
    def supports_attention(self) -> bool:
        """Check if model supports attention extraction."""
        return self.model_type in (ModelType.TFT, ModelType.TRANSFORMER)

    @property
    def supports_gradients(self) -> bool:
        """Check if model supports gradient-based explanations."""
        return self.is_pytorch_model

    @abstractmethod
    def explain_local(
        self,
        series,
        past_covariates=None,
        future_covariates=None,
        target_step: int = 0
    ) -> ExplainabilityResult:
        """
        Generate local explanation for a specific prediction.

        Args:
            series: Target time series
            past_covariates: Past covariates
            future_covariates: Future covariates
            target_step: Which forecast step to explain (0 = first step)

        Returns:
            ExplainabilityResult with local explanations
        """
        pass

    @abstractmethod
    def explain_global(
        self,
        series,
        past_covariates=None,
        future_covariates=None,
        n_samples: int = 10
    ) -> ExplainabilityResult:
        """
        Generate global explanation aggregated over multiple windows.

        Args:
            series: Target time series
            past_covariates: Past covariates
            future_covariates: Future covariates
            n_samples: Number of windows to sample

        Returns:
            ExplainabilityResult with global explanations
        """
        pass

    def get_available_methods(self) -> List[str]:
        """Get list of available explanation methods for this model."""
        methods = ["correlation", "permutation"]

        if self.is_pytorch_model:
            methods.extend(["saliency", "integrated_gradients"])

        if self.supports_attention:
            methods.append("attention")

        # SHAP is always available via perturbation
        methods.append("shap")

        return methods

    def _unwrap_torch_model(self):
        """Get the underlying PyTorch model from Darts wrapper."""
        if not self.is_pytorch_model:
            return None

        try:
            # Darts wraps PyTorch models: model.model.model
            torch_model = self.model.model
            if hasattr(torch_model, 'model'):
                return torch_model.model
            return torch_model
        except Exception:
            return None

    def _get_feature_names(self, past_covariates=None, target_col: str = "target") -> List[str]:
        """Extract feature names from covariates."""
        names = [target_col]

        if past_covariates is not None:
            try:
                cov_df = past_covariates.pd_dataframe()
                names.extend(list(cov_df.columns))
            except Exception:
                pass

        return names


class BaseVisualizer(ABC):
    """Abstract base class for explainability visualizations."""

    @abstractmethod
    def plot(self, result: ExplainabilityResult, **kwargs):
        """
        Create visualization from explainability result.

        Args:
            result: ExplainabilityResult to visualize
            **kwargs: Additional plotting options

        Returns:
            Plotly figure
        """
        pass
