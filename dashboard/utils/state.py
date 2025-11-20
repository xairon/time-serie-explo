"""Gestion de l'état de session Streamlit."""

import streamlit as st
from pathlib import Path

def init_session_state():
    """Initialise les variables de session par défaut."""
    if 'selected_station' not in st.session_state:
        st.session_state.selected_station = None
    
    if 'selected_model_path' not in st.session_state:
        st.session_state.selected_model_path = None
        
    if 'last_trained_model' not in st.session_state:
        st.session_state.last_trained_model = None

def save_context(station: str = None, model_path: str = None):
    """Sauvegarde le contexte actuel."""
    if station:
        st.session_state.selected_station = station
    if model_path:
        st.session_state.selected_model_path = str(model_path)
        st.session_state.last_trained_model = str(model_path)

def load_context():
    """Récupère le contexte actuel."""
    return {
        'station': st.session_state.get('selected_station'),
        'model_path': st.session_state.get('selected_model_path')
    }
