"""Composant de sélection de modèle pour la sidebar."""

import streamlit as st
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from dashboard.utils.model_registry import get_registry, ModelInfo
from dashboard.utils.model_config import load_model_with_config, load_scalers


def render_model_selector() -> Optional[Tuple[Any, Any, Dict, Any, ModelInfo]]:
    """
    Affiche le sélecteur de modèle dans la sidebar.
    
    Returns:
        Tuple (model, config, data_dict, scalers, model_info) ou None si erreur
    """
    st.sidebar.header(" Modèle")
    
    registry = get_registry()
    all_models = registry.scan_models()
    models = [m for m in all_models if m.model_path.exists()]
    
    if not models:
        st.sidebar.warning("Aucun modèle trouvé.")
        return None
    
    # Grouper par station
    models_by_station = {}
    for m in models:
        if m.station not in models_by_station:
            models_by_station[m.station] = []
        models_by_station[m.station].append(m)
    
    # Sélection
    selected_station = st.sidebar.selectbox(
        "Station", 
        sorted(models_by_station.keys()),
        key="model_selector_station"
    )
    
    station_models = models_by_station[selected_station]
    model_labels = [f"{m.model_type} ({m.creation_date[:10]})" for m in station_models]
    
    selected_idx = st.sidebar.selectbox(
        "Modèle", 
        range(len(station_models)), 
        format_func=lambda i: model_labels[i],
        key="model_selector_model"
    )
    
    selected_model_info = station_models[selected_idx]
    
    # Chargement avec cache
    cache_key = f"model_cache_{selected_model_info.model_path}"
    
    if cache_key not in st.session_state:
        with st.spinner("Chargement du modèle..."):
            try:
                model, config, data_dict = load_model_with_config(
                    selected_model_info.model_path.parent
                )
                scalers = load_scalers(selected_model_info.model_path.parent)
                st.session_state[cache_key] = {
                    'model': model, 
                    'config': config, 
                    'data_dict': data_dict, 
                    'scalers': scalers
                }
            except Exception as e:
                st.sidebar.error(f"Erreur: {e}")
                return None
    
    cached = st.session_state[cache_key]
    return (
        cached['model'], 
        cached['config'], 
        cached['data_dict'], 
        cached['scalers'],
        selected_model_info
    )

