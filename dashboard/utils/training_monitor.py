"""
Moniteur de progression d'entraînement.

Ce module lit les métriques écrites par MetricsFileCallback.
La partie UI (affichage Streamlit) est dans dashboard.components.training_monitor_ui.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any


class TrainingMonitor:
    """
    Moniteur qui lit un fichier JSON de métriques.

    Cette classe est complètement séparée du processus d'entraînement et peut être
    utilisée pour monitorer n'importe quel entraînement qui écrit dans le format JSON.
    """

    def __init__(self, metrics_file: Path):
        """
        Args:
            metrics_file: Chemin vers le fichier JSON de métriques
        """
        self.metrics_file = Path(metrics_file)

    def read_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Lit les métriques depuis le fichier JSON.

        Returns:
            Dictionnaire de métriques ou None si le fichier n'existe pas ou est invalide
        """
        if not self.metrics_file.exists():
            return None

        try:
            with open(self.metrics_file, 'r') as f:
                metrics = json.load(f)
            return metrics
        except (json.JSONDecodeError, IOError):
            # Fichier en cours d'écriture ou invalide
            return None
