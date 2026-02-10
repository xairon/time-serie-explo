"""Model-specific explainers for different architectures.

Provides specialized explainers for:
- TFT (attention + variable selection + gradients)
- TSMixer (gradients focus - no native attention)
- NHiTS (multi-scale decomposition)
- NBEATS (trend/seasonal stacks)
- Generic (correlation/permutation fallback)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Type
import warnings

from .base import BaseExplainer, ExplainabilityResult, ModelType

warnings.filterwarnings("ignore")


class GenericExplainer(BaseExplainer):
    """Generic explainer using correlation and permutation methods."""

    def explain_local(
        self,
        series,
        past_covariates=None,
        future_covariates=None,
        target_step: int = 0
    ) -> ExplainabilityResult:
        """Generate local explanation using correlation-based methods."""
        from .feature_importance import compute_correlation_importance, compute_lag_importance

        result = ExplainabilityResult(
            model_type=self.model_type,
            method="correlation",
        )

        try:
            # Extract feature names
            feature_names = self._get_feature_names(past_covariates)
            result.feature_names = feature_names

            # Build DataFrame for correlation analysis
            target_df = series.to_dataframe()
            target_col = target_df.columns[0]

            if past_covariates is not None:
                cov_df = past_covariates.to_dataframe()
                df = pd.concat([target_df, cov_df], axis=1)
                covariate_cols = list(cov_df.columns)
            else:
                df = target_df
                covariate_cols = []

            # Correlation importance
            if covariate_cols:
                result.feature_importance = compute_correlation_importance(
                    df, target_col, covariate_cols
                )

            # Lag importance
            lag_imp = compute_lag_importance(df, target_col, max_lag=self.input_chunk_length)
            if lag_imp:
                result.temporal_importance = np.array(list(lag_imp.values()))

            result.success = True

        except Exception as e:
            result.success = False
            result.error_message = str(e)

        return result

    def explain_global(
        self,
        series,
        past_covariates=None,
        future_covariates=None,
        n_samples: int = 10
    ) -> ExplainabilityResult:
        """Generate global explanation using permutation importance."""
        from .feature_importance import compute_permutation_importance

        result = ExplainabilityResult(
            model_type=self.model_type,
            method="permutation",
        )

        try:
            feature_names = self._get_feature_names(past_covariates)
            result.feature_names = feature_names

            # Permutation importance
            importance = compute_permutation_importance(
                self.model,
                series,
                past_covariates,
                n_permutations=n_samples,
                output_chunk_length=self.output_chunk_length,
            )

            result.feature_importance = importance
            result.success = True

        except Exception as e:
            result.success = False
            result.error_message = str(e)

        return result


class TFTModelExplainer(BaseExplainer):
    """Explainer for Temporal Fusion Transformer models."""

    def explain_local(
        self,
        series,
        past_covariates=None,
        future_covariates=None,
        target_step: int = 0
    ) -> ExplainabilityResult:
        """Generate local explanation using TFT attention and gradients."""
        from .attention import TFTExplainer
        from .gradients import GradientExplainer

        result = ExplainabilityResult(
            model_type=ModelType.TFT,
            method="tft_attention",
        )

        try:
            feature_names = self._get_feature_names(past_covariates)
            result.feature_names = feature_names

            # TFT-specific attention
            tft_explainer = TFTExplainer(self.model)
            tft_result = tft_explainer.explain(series, past_covariates, future_covariates)

            if tft_result.get("success"):
                result.attention_weights = tft_result.get("attention")
                result.encoder_importance = tft_result.get("encoder_importance")
                result.decoder_importance = tft_result.get("decoder_importance")

            # Also compute gradient-based attributions
            if self.supports_gradients:
                try:
                    grad_explainer = GradientExplainer(
                        self.model, self.input_chunk_length, self.output_chunk_length
                    )
                    grad_result = grad_explainer.compute_integrated_gradients(
                        series, past_covariates, future_covariates, target_step
                    )
                    if grad_result.get("success"):
                        result.gradient_attributions = grad_result.get("attributions")
                        result.temporal_importance = grad_result.get("temporal_importance")

                        # Combine with attention for feature importance
                        feat_imp = grad_result.get("feature_importance")
                        if feat_imp is not None and len(feature_names) == len(feat_imp):
                            result.feature_importance = dict(zip(feature_names, feat_imp))
                except Exception:
                    pass

            result.success = True

        except Exception as e:
            result.success = False
            result.error_message = str(e)

        return result

    def explain_global(
        self,
        series,
        past_covariates=None,
        future_covariates=None,
        n_samples: int = 10
    ) -> ExplainabilityResult:
        """Generate global explanation aggregating attention across samples."""
        from .attention import TFTExplainer

        result = ExplainabilityResult(
            model_type=ModelType.TFT,
            method="tft_global",
        )

        try:
            feature_names = self._get_feature_names(past_covariates)
            result.feature_names = feature_names

            # Aggregate encoder importance across samples
            tft_explainer = TFTExplainer(
                self.model,
                background_series=series,
                background_past_covariates=past_covariates,
            )

            tft_result = tft_explainer.explain(series, past_covariates, future_covariates)

            if tft_result.get("success"):
                result.encoder_importance = tft_result.get("encoder_importance")
                result.decoder_importance = tft_result.get("decoder_importance")

                # Convert encoder importance to feature importance
                if result.encoder_importance:
                    result.feature_importance = result.encoder_importance

            result.success = True

        except Exception as e:
            result.success = False
            result.error_message = str(e)

        return result


class TSMixerModelExplainer(BaseExplainer):
    """Explainer for TSMixer models (gradient-based focus)."""

    def explain_local(
        self,
        series,
        past_covariates=None,
        future_covariates=None,
        target_step: int = 0
    ) -> ExplainabilityResult:
        """Generate local explanation using Integrated Gradients."""
        from .gradients import GradientExplainer

        result = ExplainabilityResult(
            model_type=ModelType.TSMIXER,
            method="integrated_gradients",
        )

        try:
            feature_names = self._get_feature_names(past_covariates)
            result.feature_names = feature_names

            # TSMixer has no native attention - use gradients
            grad_explainer = GradientExplainer(
                self.model, self.input_chunk_length, self.output_chunk_length
            )

            # Primary: Integrated Gradients (more stable)
            ig_result = grad_explainer.compute_integrated_gradients(
                series, past_covariates, future_covariates, target_step
            )

            if ig_result.get("success"):
                result.gradient_attributions = ig_result.get("attributions")
                result.temporal_importance = ig_result.get("temporal_importance")

                feat_imp = ig_result.get("feature_importance")
                if feat_imp is not None and len(feature_names) == len(feat_imp):
                    result.feature_importance = dict(zip(feature_names, feat_imp))

                result.success = True
            else:
                # Fallback to saliency
                sal_result = grad_explainer.compute_saliency(
                    series, past_covariates, future_covariates, target_step
                )
                if sal_result.get("success"):
                    result.gradient_attributions = sal_result.get("attributions")
                    result.temporal_importance = sal_result.get("temporal_importance")
                    result.method = "saliency"
                    result.success = True
                else:
                    raise ValueError(f"Gradient methods failed: {ig_result.get('error')}")

        except Exception as e:
            # Ultimate fallback to correlation
            result.success = False
            result.error_message = str(e)

            try:
                generic = GenericExplainer(
                    self.model, self.input_chunk_length, self.output_chunk_length
                )
                fallback = generic.explain_local(series, past_covariates, future_covariates)
                if fallback.success:
                    result.feature_importance = fallback.feature_importance
                    result.temporal_importance = fallback.temporal_importance
                    result.method = "correlation_fallback"
                    result.success = True
                    result.error_message = None
            except Exception:
                pass

        return result

    def explain_global(
        self,
        series,
        past_covariates=None,
        future_covariates=None,
        n_samples: int = 10
    ) -> ExplainabilityResult:
        """Generate global explanation using SHAP or permutation."""
        from .feature_importance import compute_permutation_importance

        result = ExplainabilityResult(
            model_type=ModelType.TSMIXER,
            method="permutation",
        )

        try:
            feature_names = self._get_feature_names(past_covariates)
            result.feature_names = feature_names

            importance = compute_permutation_importance(
                self.model,
                series,
                past_covariates,
                n_permutations=n_samples,
                output_chunk_length=self.output_chunk_length,
            )

            result.feature_importance = importance
            result.success = True

        except Exception as e:
            result.success = False
            result.error_message = str(e)

        return result


class NHiTSModelExplainer(BaseExplainer):
    """Explainer for NHiTS models (multi-scale decomposition)."""

    def explain_local(
        self,
        series,
        past_covariates=None,
        future_covariates=None,
        target_step: int = 0
    ) -> ExplainabilityResult:
        """Generate local explanation combining gradients and stack analysis."""
        from .gradients import GradientExplainer

        result = ExplainabilityResult(
            model_type=ModelType.NHITS,
            method="nhits_multiscale",
        )

        try:
            feature_names = self._get_feature_names(past_covariates)
            result.feature_names = feature_names

            # Gradient-based attributions
            if self.supports_gradients:
                grad_explainer = GradientExplainer(
                    self.model, self.input_chunk_length, self.output_chunk_length
                )
                grad_result = grad_explainer.compute_integrated_gradients(
                    series, past_covariates, future_covariates, target_step
                )

                if grad_result.get("success"):
                    result.gradient_attributions = grad_result.get("attributions")
                    result.temporal_importance = grad_result.get("temporal_importance")

                    feat_imp = grad_result.get("feature_importance")
                    if feat_imp is not None and len(feature_names) == len(feat_imp):
                        result.feature_importance = dict(zip(feature_names, feat_imp))

            # NHiTS-specific: Extract stack contributions (if accessible)
            stack_info = self._extract_stack_contributions()
            if stack_info:
                result.decomposition = {"stacks": stack_info}

            result.success = True

        except Exception as e:
            result.success = False
            result.error_message = str(e)

        return result

    def _extract_stack_contributions(self) -> Optional[Dict[str, Any]]:
        """Extract NHiTS stack contributions if possible."""
        try:
            # NHiTS uses multiple stacks with different pooling
            torch_model = self._unwrap_torch_model()
            if torch_model is None:
                return None

            # Get stack configuration
            if hasattr(torch_model, "stacks"):
                n_stacks = len(torch_model.stacks)
                return {
                    "n_stacks": n_stacks,
                    "description": f"NHiTS with {n_stacks} hierarchical stacks",
                }
            return None

        except Exception:
            return None

    def explain_global(
        self,
        series,
        past_covariates=None,
        future_covariates=None,
        n_samples: int = 10
    ) -> ExplainabilityResult:
        """Generate global explanation."""
        generic = GenericExplainer(
            self.model, self.input_chunk_length, self.output_chunk_length
        )
        result = generic.explain_global(series, past_covariates, future_covariates, n_samples)
        result.model_type = ModelType.NHITS
        return result


class NBEATSModelExplainer(BaseExplainer):
    """Explainer for NBEATS models (interpretable trend/seasonal stacks)."""

    def explain_local(
        self,
        series,
        past_covariates=None,
        future_covariates=None,
        target_step: int = 0
    ) -> ExplainabilityResult:
        """Generate local explanation with trend/seasonal decomposition."""
        from .decomposition import DecompositionAnalyzer
        from .gradients import GradientExplainer

        result = ExplainabilityResult(
            model_type=ModelType.NBEATS,
            method="nbeats_interpretable",
        )

        try:
            feature_names = self._get_feature_names(past_covariates)
            result.feature_names = feature_names

            # NBEATS-specific: interpretable mode outputs trend/seasonal
            stack_decomp = self._extract_interpretable_outputs(series)
            if stack_decomp:
                result.decomposition = stack_decomp

            # Also add gradient attributions
            if self.supports_gradients:
                grad_explainer = GradientExplainer(
                    self.model, self.input_chunk_length, self.output_chunk_length
                )
                grad_result = grad_explainer.compute_integrated_gradients(
                    series, past_covariates, future_covariates, target_step
                )
                if grad_result.get("success"):
                    result.gradient_attributions = grad_result.get("attributions")
                    result.temporal_importance = grad_result.get("temporal_importance")

            # Feature importance from gradients or correlation
            if past_covariates is not None:
                generic = GenericExplainer(
                    self.model, self.input_chunk_length, self.output_chunk_length
                )
                corr_result = generic.explain_local(series, past_covariates)
                if corr_result.success:
                    result.feature_importance = corr_result.feature_importance

            result.success = True

        except Exception as e:
            result.success = False
            result.error_message = str(e)

        return result

    def _extract_interpretable_outputs(self, series) -> Optional[Dict[str, Any]]:
        """Extract NBEATS interpretable stack outputs."""
        try:
            # NBEATS interpretable mode has trend and seasonal stacks
            # This would require model forward pass with hooks
            torch_model = self._unwrap_torch_model()
            if torch_model is None:
                return None

            if hasattr(torch_model, "stacks"):
                return {
                    "mode": "interpretable" if hasattr(torch_model, "trend_blocks") else "generic",
                    "description": "NBEATS with trend/seasonal decomposition",
                }
            return None

        except Exception:
            return None

    def explain_global(
        self,
        series,
        past_covariates=None,
        future_covariates=None,
        n_samples: int = 10
    ) -> ExplainabilityResult:
        """Generate global explanation."""
        generic = GenericExplainer(
            self.model, self.input_chunk_length, self.output_chunk_length
        )
        result = generic.explain_global(series, past_covariates, future_covariates, n_samples)
        result.model_type = ModelType.NBEATS
        return result


class ModelExplainerFactory:
    """Factory for creating model-specific explainers."""

    EXPLAINER_MAP: Dict[ModelType, Type[BaseExplainer]] = {
        ModelType.TFT: TFTModelExplainer,
        ModelType.TSMIXER: TSMixerModelExplainer,
        ModelType.NHITS: NHiTSModelExplainer,
        ModelType.NBEATS: NBEATSModelExplainer,
        ModelType.GENERIC: GenericExplainer,
    }

    @classmethod
    def get_explainer(
        cls,
        model,
        input_chunk_length: int = 30,
        output_chunk_length: int = 7
    ) -> BaseExplainer:
        """
        Get appropriate explainer for model.

        Args:
            model: Darts forecasting model
            input_chunk_length: Input window size
            output_chunk_length: Prediction horizon

        Returns:
            Model-specific explainer instance
        """
        model_type = ModelType.from_model(model)
        explainer_class = cls.EXPLAINER_MAP.get(model_type, GenericExplainer)

        return explainer_class(model, input_chunk_length, output_chunk_length)

    @classmethod
    def get_available_methods(cls, model) -> List[str]:
        """Get available explanation methods for a model."""
        explainer = cls.get_explainer(model)
        return explainer.get_available_methods()

    @classmethod
    def get_model_type(cls, model) -> ModelType:
        """Get model type enum for a model."""
        return ModelType.from_model(model)
