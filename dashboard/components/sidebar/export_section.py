"""Composant d'export pour la sidebar."""

import streamlit as st
from pathlib import Path
from dashboard.utils.export import add_download_button


def render_export_section(model_dir: Path, key_suffix: str = ""):
    """
    Affiche la section d'export dans la sidebar.
    
    Args:
        model_dir: Répertoire du modèle
        key_suffix: Suffixe unique pour les clés Streamlit
    """
    st.sidebar.divider()
    st.sidebar.header("📦 Export")
    
    with st.sidebar:
        add_download_button(model_dir, key_suffix=key_suffix)
