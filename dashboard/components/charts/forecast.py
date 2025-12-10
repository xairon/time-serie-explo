"""Composant graphique pour les prédictions."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Optional


def render_forecast_chart(
    dates: pd.DatetimeIndex,
    real_values: np.ndarray,
    autoregressive_pred: Optional[np.ndarray] = None,
    one_step_pred: Optional[np.ndarray] = None,
    target_name: str = "Valeur",
    height: int = 500
):
    """
    Affiche le graphique des prédictions vs valeurs réelles.
    
    Args:
        dates: Index temporel
        real_values: Valeurs réelles
        autoregressive_pred: Prédictions autorégressives (optionnel)
        one_step_pred: Prédictions one-step (optionnel)
        target_name: Nom de la variable cible
        height: Hauteur du graphique
    """
    fig = go.Figure()
    
    # Valeurs réelles
    fig.add_trace(go.Scatter(
        x=dates,
        y=real_values,
        mode='lines',
        name='Réel',
        line=dict(color='#2E86AB', width=2)
    ))
    
    # Prédictions autorégressives
    if autoregressive_pred is not None:
        fig.add_trace(go.Scatter(
            x=dates,
            y=autoregressive_pred,
            mode='lines',
            name='Prédiction (auto)',
            line=dict(color='#F24236', width=2)
        ))
    
    # Prédictions one-step
    if one_step_pred is not None:
        fig.add_trace(go.Scatter(
            x=dates,
            y=one_step_pred,
            mode='lines',
            name='Prédiction (1-step)',
            line=dict(color='#28A745', width=2, dash='dot')
        ))
    
    fig.update_layout(
        title=f" Prédictions - {target_name}",
        xaxis_title="Date",
        yaxis_title=target_name,
        height=height,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_residuals_chart(
    dates: pd.DatetimeIndex,
    residuals: np.ndarray,
    height: int = 300
):
    """
    Affiche le graphique des résidus.
    
    Args:
        dates: Index temporel
        residuals: Résidus (real - predicted)
        height: Hauteur du graphique
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=residuals,
        mode='lines+markers',
        name='Résidus',
        line=dict(color='#6C757D', width=1),
        marker=dict(size=4)
    ))
    
    # Ligne zéro
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title=" Résidus",
        xaxis_title="Date",
        yaxis_title="Erreur",
        height=height,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

