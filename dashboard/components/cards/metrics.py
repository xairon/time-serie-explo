"""Composants de cartes pour afficher les métriques et informations."""

import streamlit as st
from typing import Dict, Any, Optional
import pandas as pd


def render_metrics_cards(metrics: Dict[str, float], columns: int = 4):
    """
    Affiche les métriques dans des cartes.
    
    Args:
        metrics: Dict de métriques {nom: valeur}
        columns: Nombre de colonnes
    """
    cols = st.columns(columns)
    for i, (name, value) in enumerate(metrics.items()):
        with cols[i % columns]:
            if value is not None and not pd.isna(value):
                st.metric(name, f"{value:.4f}")
            else:
                st.metric(name, "N/A")


def render_dataset_card(
    train_size: int, 
    val_size: int, 
    test_size: int,
    total_size: Optional[int] = None
):
    """
    Affiche les informations du dataset.
    
    Args:
        train_size: Taille du jeu d'entraînement
        val_size: Taille du jeu de validation
        test_size: Taille du jeu de test
        total_size: Taille totale (optionnel)
    """
    st.markdown("### 📊 Dataset")
    
    if total_size is None:
        total_size = train_size + val_size + test_size
    
    st.markdown(f"""
| Split | Taille | % |
|-------|--------|---|
| Train | {train_size:,} | {train_size/total_size*100:.1f}% |
| Val | {val_size:,} | {val_size/total_size*100:.1f}% |
| Test | {test_size:,} | {test_size/total_size*100:.1f}% |
| **Total** | **{total_size:,}** | 100% |
""")


def render_model_card(
    model_name: str,
    input_chunk: int,
    output_chunk: int,
    use_covariates: bool,
    covariate_count: int = 0
):
    """
    Affiche les informations du modèle.
    
    Args:
        model_name: Nom du modèle
        input_chunk: Taille de la fenêtre d'entrée
        output_chunk: Horizon de prédiction
        use_covariates: Utilise des covariables
        covariate_count: Nombre de covariables
    """
    st.markdown("### 🤖 Modèle")
    st.markdown(f"""
| Paramètre | Valeur |
|-----------|--------|
| Type | {model_name} |
| Input | {input_chunk} jours |
| Horizon | {output_chunk} jours |
| Covariables | {'✅ ' + str(covariate_count) if use_covariates else '❌'} |
""")


def render_training_metrics_card(metrics: Dict[str, float]):
    """
    Affiche les métriques d'entraînement dans une carte.
    
    Args:
        metrics: Dict avec MAE, RMSE, R2, etc.
    """
    st.markdown("### 📈 Métriques")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'MAE' in metrics:
            st.metric("MAE", f"{metrics['MAE']:.4f}")
        if 'MSE' in metrics:
            st.metric("MSE", f"{metrics['MSE']:.4f}")
    
    with col2:
        if 'RMSE' in metrics:
            st.metric("RMSE", f"{metrics['RMSE']:.4f}")
        if 'R2' in metrics:
            st.metric("R²", f"{metrics['R2']:.4f}")
