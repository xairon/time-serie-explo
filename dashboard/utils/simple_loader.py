"""
Compatibilité rétroactive : ancien chargeur « simple ».

Historiquement, ce module lançait un script externe (`load_model_isolated.py`)
pour contourner les patchs de Streamlit. Désormais, toute cette logique vit
dans `robust_loader.load_model_safe`. Afin de garder la compatibilité avec
les appels existants, on redirige simplement vers le nouveau chargeur.
"""

from pathlib import Path
from typing import Any

from .robust_loader import load_model_safe


def load_model_external(model_path: Path, model_type: str) -> Any:
    """
    Compatibilité : redirige vers load_model_safe.

    Args:
        model_path: Chemin vers le fichier .pkl
        model_type: Nom du modèle (TFT, NBEATS, etc.)

    Returns:
        Modèle Darts chargé.
    """
    return load_model_safe(model_path, model_type)