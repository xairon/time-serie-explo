"""Enhanced Streamlit UI components for explainability.

Provides modular components that integrate with the new explainability package.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from darts import TimeSeries


def render_explainability_tabs(
    model: Any,
    series: TimeSeries,
    past_covariates: Optional[TimeSeries],
    predictions_dates: pd.DatetimeIndex,
    config: Any
):
    """
    Render explainability tabs using new package structure.

    Args:
        model: Trained Darts model
        series: Target TimeSeries
        past_covariates: Past covariates TimeSeries
        predictions_dates: Dates of predictions
        config: Model configuration
    """
    from dashboard.utils.explainability import (
        ModelExplainerFactory,
        ModelType,
        compute_correlation_importance,
        compute_permutation_importance_safe,
        compute_lag_importance,
        plot_feature_importance_bar,
        plot_lag_importance,
        plot_shap_waterfall,
    )

    # Get model info
    model_type = ModelType.from_model(model)
    explainer = ModelExplainerFactory.get_explainer(model)
    available_methods = explainer.get_available_methods()

    covariate_cols = config.columns.get('covariates', [])
    input_chunk = getattr(model, 'input_chunk_length', 30)

    # Create tabs
    tab_overview, tab_importance, tab_temporal, tab_local = st.tabs([
        "Overview",
        "Feature Importance",
        "Temporal",
        "Local Analysis"
    ])

    # Tab 1: Overview
    with tab_overview:
        _render_overview_tab(
            model, series, past_covariates, covariate_cols, model_type, available_methods
        )

    # Tab 2: Feature Importance
    with tab_importance:
        _render_importance_tab(
            model, series, past_covariates, covariate_cols
        )

    # Tab 3: Temporal
    with tab_temporal:
        _render_temporal_tab(
            model, series, past_covariates, input_chunk
        )

    # Tab 4: Local Analysis
    with tab_local:
        _render_local_tab(
            model, series, past_covariates, predictions_dates, covariate_cols
        )


def _render_overview_tab(
    model,
    series,
    past_covariates,
    covariate_cols: List[str],
    model_type,
    available_methods: List[str]
):
    """Render overview tab with summary and top features."""
    from dashboard.utils.explainability import (
        compute_correlation_importance,
        plot_feature_importance_bar,
    )

    st.markdown("### Explainability Overview")

    # Model info
    model_type_names = {
        "TFT": "TFT (Attention + Variable Selection)",
        "TSMIXER": "TSMixer (Gradient-based)",
        "NHITS": "NHiTS (Multi-scale)",
        "NBEATS": "NBEATS (Interpretable)",
    }
    st.info(f"**Model Type:** {model_type_names.get(model_type.name, model_type.name)} | "
            f"**Methods:** {', '.join(available_methods)}")

    if not covariate_cols or past_covariates is None:
        st.warning("No covariates available for analysis.")
        return

    # Top 5 features
    st.markdown("#### Top 5 Features")

    try:
        # Build DataFrame
        target_df = series.to_dataframe()
        target_col = target_df.columns[0]

        cov_df = past_covariates.to_dataframe()
        df = pd.concat([target_df, cov_df], axis=1)

        correlations = compute_correlation_importance(df, target_col, list(cov_df.columns))

        if correlations:
            top_5 = dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:5])
            fig = plot_feature_importance_bar(top_5, title="Top 5 Features", top_k=5)
            st.plotly_chart(fig, use_container_width=True)

            # Summary
            st.markdown("**Key Insights:**")
            for i, (name, val) in enumerate(top_5.items(), 1):
                st.write(f"{i}. **{name}**: {val:.1%} correlation")
        else:
            st.info("Could not compute correlations.")
    except Exception as e:
        st.error(f"Error computing overview: {e}")


def _render_importance_tab(
    model,
    series,
    past_covariates,
    covariate_cols: List[str]
):
    """Render feature importance tab with multiple methods."""
    from dashboard.utils.explainability import (
        compute_correlation_importance,
        compute_permutation_importance_safe,
        plot_feature_importance_bar,
    )

    st.markdown("### Feature Importance")

    if not covariate_cols or past_covariates is None:
        st.warning("No covariates available for analysis.")
        return

    method = st.radio("Method", ["Correlation", "Permutation"], horizontal=True)

    if method == "Correlation":
        st.caption("Fast correlation-based importance")

        try:
            target_df = series.to_dataframe()
            target_col = target_df.columns[0]
            cov_df = past_covariates.to_dataframe()
            df = pd.concat([target_df, cov_df], axis=1)

            correlations = compute_correlation_importance(df, target_col, list(cov_df.columns))

            if correlations:
                fig = plot_feature_importance_bar(correlations, title="Feature Importance (Correlation)")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

    else:  # Permutation
        st.caption("Measures prediction degradation when shuffling features")

        if st.button("Compute Permutation Importance"):
            with st.spinner("Computing (this may take a moment)..."):
                try:
                    output_chunk = getattr(model, 'output_chunk_length', 7)
                    importance = compute_permutation_importance_safe(
                        model, series, past_covariates,
                        n_permutations=3,
                        output_chunk_length=output_chunk
                    )

                    if importance:
                        fig = plot_feature_importance_bar(importance, title="Permutation Importance")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Permutation importance failed.")
                except Exception as e:
                    st.error(f"Error: {e}")


def _render_temporal_tab(
    model,
    series,
    past_covariates,
    input_chunk: int
):
    """Render temporal analysis tab."""
    from dashboard.utils.explainability import (
        compute_lag_importance,
        plot_lag_importance,
    )

    st.markdown("### Temporal Importance")
    st.info("Which past days influence the prediction most?")

    try:
        target_df = series.to_dataframe()
        target_col = target_df.columns[0]

        max_lag = min(input_chunk, len(target_df) // 2)
        lag_imp = compute_lag_importance(target_df, target_col, max_lag=max_lag)

        if lag_imp:
            fig = plot_lag_importance(lag_imp, input_chunk_length=input_chunk)
            st.plotly_chart(fig, use_container_width=True)

            # Insights
            peak_lag = max(lag_imp.keys(), key=lambda k: lag_imp[k])
            st.success(f"Most important lag: **t-{peak_lag}** (correlation: {lag_imp[peak_lag]:.2f})")

            recent = [v for k, v in lag_imp.items() if k <= 7]
            distant = [v for k, v in lag_imp.items() if k > 7]

            if recent and distant:
                if np.mean(recent) > np.mean(distant) * 1.3:
                    st.info("Recent days (t-1 to t-7) are most influential.")
                elif np.mean(distant) > np.mean(recent) * 1.3:
                    st.info("Longer history (>7 days) is more influential.")
        else:
            st.info("Not enough data for lag analysis.")
    except Exception as e:
        st.error(f"Error: {e}")


def _render_local_tab(
    model,
    series,
    past_covariates,
    predictions_dates: pd.DatetimeIndex,
    covariate_cols: List[str]
):
    """Render local explanation tab."""
    from dashboard.utils.explainability import (
        ModelExplainerFactory,
        plot_shap_waterfall,
    )

    st.markdown("### Local Explanation")
    st.info("Analyze a specific prediction.")

    if len(predictions_dates) == 0:
        st.warning("No predictions available.")
        return

    selected_date = st.selectbox(
        "Select Date",
        predictions_dates,
        format_func=lambda x: x.strftime("%Y-%m-%d"),
        key="local_date"
    )

    if st.button("Analyze This Prediction"):
        with st.spinner("Computing local explanation..."):
            try:
                explainer = ModelExplainerFactory.get_explainer(model)
                result = explainer.explain_local(series, past_covariates)

                if result.success and result.feature_importance:
                    fig = plot_shap_waterfall(
                        result.feature_importance,
                        title=f"Contributions ({selected_date})"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Top contributors
                    st.markdown("**Top Contributors:**")
                    sorted_imp = sorted(result.feature_importance.items(),
                                       key=lambda x: abs(x[1]), reverse=True)[:5]
                    for name, val in sorted_imp:
                        sign = "+" if val > 0 else ""
                        st.write(f"- **{name}**: {sign}{val:.4f}")
                else:
                    st.warning(f"Local explanation failed: {result.error_message}")
            except Exception as e:
                st.error(f"Error: {e}")


def render_model_specific_tab(
    model: Any,
    series: TimeSeries,
    past_covariates: Optional[TimeSeries],
    model_type: str
):
    """
    Render model-specific explanations based on model type.

    Args:
        model: Trained model
        series: Target series
        past_covariates: Covariates
        model_type: Type of model (TFT, TSMixer, etc.)
    """
    from dashboard.utils.explainability import (
        ModelExplainerFactory,
        ModelType,
        plot_temporal_saliency_heatmap,
        plot_attention_heatmap,
    )

    st.markdown(f"### {model_type} Internals")

    explainer = ModelExplainerFactory.get_explainer(model)
    detected_type = ModelType.from_model(model)

    if detected_type == ModelType.TFT:
        st.info("TFT provides attention weights and variable selection.")

        if st.button("Extract TFT Attention"):
            with st.spinner("Extracting attention..."):
                try:
                    from dashboard.utils.explainability.attention import TFTExplainer

                    tft_exp = TFTExplainer(model)
                    result = tft_exp.explain(series, past_covariates)

                    if result.get('success') and result.get('attention') is not None:
                        fig = plot_attention_heatmap(result['attention'])
                        st.plotly_chart(fig, use_container_width=True)

                    if result.get('encoder_importance'):
                        st.markdown("**Encoder Importance:**")
                        st.json(result['encoder_importance'])
                except Exception as e:
                    st.error(f"Error: {e}")

    elif detected_type == ModelType.TSMIXER:
        st.info("TSMixer uses Integrated Gradients for attribution.")

        if st.button("Compute TSMixer Gradients"):
            with st.spinner("Computing gradients..."):
                try:
                    result = explainer.explain_local(series, past_covariates)

                    if result.success and result.gradient_attributions is not None:
                        fig = plot_temporal_saliency_heatmap(
                            result.gradient_attributions,
                            result.feature_names,
                            title="TSMixer Attribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")

    else:
        st.info("Using generic gradient-based analysis.")

        if st.button("Run Analysis"):
            with st.spinner("Analyzing..."):
                try:
                    result = explainer.explain_local(series, past_covariates)

                    if result.success:
                        if result.feature_importance:
                            from dashboard.utils.explainability import plot_feature_importance_bar
                            fig = plot_feature_importance_bar(result.feature_importance)
                            st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")
