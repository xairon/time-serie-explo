"""Composants graphiques pour l'explicabilité."""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from darts import TimeSeries


def render_explainability_tabs(
    model: Any,
    series: TimeSeries,
    past_covariates: Optional[TimeSeries],
    predictions_dates: pd.DatetimeIndex,
    config: Any
):
    """
    Affiche les onglets d'explicabilité.
    
    Args:
        model: Modèle Darts entraîné
        series: Série temporelle cible
        past_covariates: Covariables passées
        predictions_dates: Dates des prédictions
        config: Configuration du modèle
    """
    from dashboard.utils.explainability import (
        compute_feature_importance_permutation,
        compute_temporal_importance,
        compute_local_explanation,
        plot_feature_importance,
        plot_temporal_importance,
        plot_waterfall_explanation,
        create_explanation_summary
    )
    
    tab_overview, tab_temporal, tab_local = st.tabs([
        " Vue d'ensemble",
        "⏱️ Temporel", 
        " Analyse locale"
    ])
    
    covariate_cols = config.columns.get('covariates', [])
    input_chunk = getattr(model, 'input_chunk_length', 30)
    
    # Tab 1: Vue d'ensemble
    with tab_overview:
        _render_overview_tab(
            model, series, past_covariates, covariate_cols
        )
    
    # Tab 2: Temporel
    with tab_temporal:
        _render_temporal_tab(
            model, series, past_covariates, input_chunk
        )
    
    # Tab 3: Analyse locale
    with tab_local:
        _render_local_tab(
            model, series, past_covariates, predictions_dates, covariate_cols
        )


def _render_overview_tab(model, series, past_covariates, covariate_cols):
    """Affiche l'onglet Vue d'ensemble."""
    from dashboard.utils.explainability import (
        compute_feature_importance_permutation,
        plot_feature_importance,
        create_explanation_summary
    )
    
    st.markdown("###  Importance des Features")
    st.info("Analyse de l'importance globale des covariables par permutation.")
    
    if not covariate_cols or past_covariates is None:
        st.warning("Pas de covariables disponibles pour l'analyse.")
        return
    
    if st.button(" Calculer l'importance", key="calc_importance"):
        with st.spinner("Calcul en cours (peut prendre quelques minutes)..."):
            try:
                importance = compute_feature_importance_permutation(
                    model, series, past_covariates, covariate_cols
                )
                
                if importance:
                    fig = plot_feature_importance(importance)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    summary = create_explanation_summary(importance)
                    st.markdown(summary)
            except Exception as e:
                st.error(f"Erreur: {e}")


def _render_temporal_tab(model, series, past_covariates, input_chunk):
    """Affiche l'onglet Temporel."""
    from dashboard.utils.explainability import (
        compute_temporal_importance,
        plot_temporal_importance
    )
    
    st.markdown("### ⏱️ Importance Temporelle")
    st.info("Quels jours passés influencent le plus la prédiction ?")
    
    if st.button(" Calculer l'importance temporelle", key="calc_temporal"):
        with st.spinner("Analyse temporelle en cours..."):
            try:
                temporal_imp = compute_temporal_importance(
                    model, series, past_covariates, n_steps=min(input_chunk, 30)
                )
                
                if temporal_imp is not None:
                    fig = plot_temporal_importance(temporal_imp)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Insights
                    recent_imp = temporal_imp[:7].mean() if len(temporal_imp) >= 7 else temporal_imp.mean()
                    distant_imp = temporal_imp[7:].mean() if len(temporal_imp) > 7 else 0
                    
                    if recent_imp > distant_imp * 1.5:
                        st.success(" Les jours récents sont les plus influents.")
                    else:
                        st.info(" L'influence est répartie sur toute la fenêtre.")
            except Exception as e:
                st.error(f"Erreur: {e}")


def _render_local_tab(model, series, past_covariates, predictions_dates, covariate_cols):
    """Affiche l'onglet Analyse locale."""
    from dashboard.utils.explainability import (
        compute_local_explanation,
        plot_waterfall_explanation
    )
    
    st.markdown("###  Explication Locale")
    st.info("Analyse détaillée d'une prédiction spécifique.")
    
    if len(predictions_dates) == 0:
        st.warning("Aucune prédiction disponible.")
        return
    
    selected_date = st.selectbox(
        "Sélectionner une date",
        predictions_dates,
        format_func=lambda x: x.strftime("%Y-%m-%d"),
        key="local_date_select"
    )
    
    if st.button(" Analyser cette prédiction", key="calc_local"):
        with st.spinner("Analyse en cours..."):
            try:
                local_exp = compute_local_explanation(
                    model, series, past_covariates, 
                    selected_date, covariate_cols
                )
                
                if local_exp:
                    fig = plot_waterfall_explanation(local_exp, selected_date)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Top contributeurs
                    st.markdown("**Top contributeurs:**")
                    sorted_exp = sorted(
                        local_exp.items(), 
                        key=lambda x: abs(x[1]), 
                        reverse=True
                    )[:5]
                    
                    for feature, contrib in sorted_exp:
                        sign = "+" if contrib > 0 else ""
                        st.markdown(f"- **{feature}**: {sign}{contrib:.4f}")
            except Exception as e:
                st.error(f"Erreur: {e}")

