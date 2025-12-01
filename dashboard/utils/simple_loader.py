"""
Chargeur simple de modèles utilisant un script Python externe.

Cette approche évite complètement les conflits Streamlit en utilisant
un script Python qui s'exécute de manière totalement indépendante.
"""

import os
import sys
import json
import pickle
import tempfile
import subprocess
from pathlib import Path
from typing import Any


def load_model_external(model_path: Path, model_type: str) -> Any:
    """
    Charge un modèle en utilisant un script Python externe.

    Cette méthode lance le script standalone_loader.py dans un nouveau
    processus Python, complètement isolé de l'environnement Streamlit.

    Args:
        model_path: Chemin vers le fichier du modèle
        model_type: Type du modèle (TFT, NBEATS, etc.)

    Returns:
        Le modèle chargé

    Raises:
        RuntimeError: Si le chargement échoue
    """
    model_path = Path(model_path).absolute()  # Convertir en chemin absolu
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Créer un fichier temporaire pour le modèle sérialisé
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        output_path = f.name

    try:
        # Utiliser le script isolé à la racine du projet
        root_dir = Path(__file__).parent.parent.parent  # Remonter à la racine
        script_path = root_dir / 'load_model_isolated.py'

        if not script_path.exists():
            # Fallback vers le script dans utils
            script_path = Path(__file__).parent / 'standalone_loader.py'
            if not script_path.exists():
                raise FileNotFoundError(f"No loader script found")

        # Lancer le script dans un nouveau processus Python
        # avec un environnement complètement propre
        cmd = [
            sys.executable,  # Python executable
            str(script_path.absolute()),  # Script path (absolu)
            str(model_path.absolute()),  # Model path (absolu)
            model_type,  # Model type
            output_path  # Output path
        ]

        # Créer un environnement propre sans Streamlit
        clean_env = os.environ.copy()
        # Supprimer toutes les variables Streamlit
        for key in list(clean_env.keys()):
            if 'STREAMLIT' in key:
                del clean_env[key]

        # Forcer la désactivation de Streamlit
        clean_env['_IS_RUNNING_WITH_STREAMLIT'] = 'false'
        clean_env['STREAMLIT_RUNTIME_EXISTS'] = 'false'

        # Exécuter le script depuis un répertoire temporaire
        # pour éviter toute interférence avec le dashboard
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=120,
            cwd=tempfile.gettempdir(),  # Exécuter depuis le dossier temp
            env=clean_env  # Utiliser l'environnement propre
        )

        # Vérifier le résultat
        if result.returncode == 0:
            try:
                response = json.loads(result.stdout.strip())
                if response.get('success'):
                    # Charger le modèle depuis le fichier temporaire
                    with open(output_path, 'rb') as f:
                        model = pickle.load(f)
                    return model
                else:
                    raise RuntimeError(f"Failed to load model: {response.get('error', 'Unknown error')}")
            except json.JSONDecodeError:
                raise RuntimeError(f"Invalid response from loader: {result.stdout}")
        else:
            # Essayer de parser l'erreur
            try:
                response = json.loads(result.stdout.strip())
                raise RuntimeError(f"Loading failed: {response.get('error', 'Unknown error')}")
            except:
                raise RuntimeError(f"Subprocess failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")

    finally:
        # Nettoyer le fichier temporaire
        if os.path.exists(output_path):
            os.unlink(output_path)