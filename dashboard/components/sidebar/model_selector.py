"""Composant de sélection de modèle pour la sidebar."""

import streamlit as st
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from dashboard.utils.model_registry import get_registry, ModelEntry


def render_model_selector() -> Optional[Tuple[Any, Any, Dict, Any, ModelEntry]]:
    """
    Affiche le sélecteur de modèle dans la sidebar.
    
    Returns:
        Tuple (model, config, data_dict, scalers, model_info) ou None si erreur
    """
    st.sidebar.header("Modèle")
    
    registry = get_registry()
    all_models = registry.list_all_models()
    
    if not all_models:
        st.sidebar.warning("Aucun modèle trouvé.")
        return None
    
    # Grouper par station
    models_by_station = {}
    for m in all_models:
        # Use stations list or fallback
        station = m.primary_station or (m.stations[0] if m.stations else "Global")
        if station not in models_by_station:
            models_by_station[station] = []
        models_by_station[station].append(m)
    
    # Sélection
    sorted_stations = sorted(models_by_station.keys())
    selected_station = st.sidebar.selectbox(
        "Station", 
        sorted_stations,
        key="model_selector_station"
    )
    
    station_models = models_by_station[selected_station]
    model_labels = [m.display_name for m in station_models]
    
    selected_idx = st.sidebar.selectbox(
        "Modèle", 
        range(len(station_models)), 
        format_func=lambda i: model_labels[i],
        key="model_selector_model"
    )
    
    selected_model_info = station_models[selected_idx]
    
    # Chargement avec cache
    cache_key = f"model_cache_{selected_model_info.model_id}"
    
    if cache_key not in st.session_state:
        with st.spinner("Chargement du modèle..."):
            try:
                # Load components via registry
                model = registry.load_model(selected_model_info)
                scalers = registry.load_scalers(selected_model_info)
                
                # Reconstruct config-like object for compatibility
                config = type('Config', (), {})()
                config.model_name = selected_model_info.model_name
                config.hyperparams = selected_model_info.hyperparams
                config.metrics = selected_model_info.metrics
                config.preprocessing = selected_model_info.preprocessing_config
                
                # Load data (Train/Val/Test)
                data_dict = {}
                for split in ['train', 'val', 'test']:
                    data_dict[split] = registry.load_data(selected_model_info, split)
                
                st.session_state[cache_key] = {
                    'model': model, 
                    'config': config, 
                    'data_dict': data_dict, 
                    'scalers': scalers
                }
            except Exception as e:
                st.sidebar.error(f"Erreur chargement: {e}")
                import traceback
                st.sidebar.code(traceback.format_exc())
                return None
    
    cached = st.session_state[cache_key]
    return (
        cached['model'], 
        cached['config'], 
        cached['data_dict'], 
        cached['scalers'],
        selected_model_info
    )

