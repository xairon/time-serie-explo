"""
Chargeur robuste de modèles Darts pour contourner les conflits Streamlit.

Cette solution utilise plusieurs stratégies pour charger les modèles de manière fiable.
"""

import os
import sys
import pickle
import tempfile
import subprocess
import warnings
from pathlib import Path
from typing import Any, Optional


def load_model_safe(model_path: Path, model_type: str) -> Any:
    """
    Charge un modèle Darts de manière sûre, même dans Streamlit.

    Stratégies utilisées (dans l'ordre):
    1. Chargement direct avec gestion d'exceptions
    2. Chargement dans un subprocess isolé si échec

    Args:
        model_path: Chemin vers le fichier .pkl
        model_type: Type du modèle (TFT, NBEATS, etc.)

    Returns:
        Le modèle chargé

    Raises:
        RuntimeError: Si toutes les méthodes échouent
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Méthode 1: Essayer le chargement direct
    try:
        # Import des classes de modèles
        from darts.models import (
            TFTModel, NBEATSModel, NHiTSModel, TransformerModel,
            RNNModel, BlockRNNModel, TCNModel, TiDEModel,
            DLinearModel, NLinearModel
        )

        model_classes = {
            'TFT': TFTModel,
            'NBEATS': NBEATSModel,
            'NHITS': NHiTSModel,
            'TRANSFORMER': TransformerModel,
            'LSTM': RNNModel,
            'GRU': RNNModel,
            'BLOCKRNN': BlockRNNModel,
            'TCN': TCNModel,
            'TIDE': TiDEModel,
            'DLINEAR': DLinearModel,
            'NLINEAR': NLinearModel,
        }

        model_class = model_classes.get(model_type.upper())
        if not model_class:
            raise ValueError(f"Unknown model type: {model_type}")

        # Supprimer les warnings temporairement
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = model_class.load(str(model_path))

        return model

    except Exception as e:
        error_msg = str(e)

        # Si c'est l'erreur Streamlit, essayer le subprocess
        if "__setstate__" in error_msg or "StreamlitAPIException" in error_msg:
            return _load_in_subprocess(model_path, model_type)
        else:
            # Pour toute autre erreur, la propager
            raise


def _load_in_subprocess(model_path: Path, model_type: str) -> Any:
    """
    Charge un modèle dans un subprocess Python isolé (sans Streamlit).

    Cette méthode est utilisée quand le chargement direct échoue à cause
    de l'interférence de Streamlit avec pickle.
    """

    # Script pour charger le modèle dans un environnement propre
    loader_script = '''# -*- coding: utf-8 -*-
import sys
import os

# ISOLATION TOTALE DE STREAMLIT - TRÈS IMPORTANT !
# Bloquer Streamlit AVANT tout import pour éviter l'erreur __setstate__
os.environ['NO_STREAMLIT'] = '1'

# Supprimer tous les modules Streamlit du cache
modules_to_remove = [m for m in sys.modules.keys() if 'streamlit' in m.lower()]
for module in modules_to_remove:
    del sys.modules[module]

# Créer un mock pour streamlit pour bloquer toute tentative d'import
class StreamlitMock:
    def __getattr__(self, name):
        raise ImportError("Streamlit is disabled in this subprocess")

sys.modules['streamlit'] = StreamlitMock()

# Maintenant on peut importer les autres modules en sécurité
import pickle
from pathlib import Path

# Arguments
model_path = Path(sys.argv[1])
model_type = sys.argv[2]
output_path = sys.argv[3]

try:
    # Import Darts sans Streamlit
    from darts.models import (
        TFTModel, NBEATSModel, NHiTSModel, TransformerModel,
        RNNModel, BlockRNNModel, TCNModel, TiDEModel,
        DLinearModel, NLinearModel
    )

    model_classes = {
        'TFT': TFTModel,
        'NBEATS': NBEATSModel,
        'NHITS': NHiTSModel,
        'TRANSFORMER': TransformerModel,
        'LSTM': RNNModel,
        'GRU': RNNModel,
        'BLOCKRNN': BlockRNNModel,
        'TCN': TCNModel,
        'TIDE': TiDEModel,
        'DLINEAR': DLinearModel,
        'NLINEAR': NLinearModel,
    }

    model_class = model_classes.get(model_type.upper())
    if not model_class:
        raise ValueError(f"Unknown model type: {model_type}")

    # Charger le modèle
    model = model_class.load(str(model_path))

    # Sauvegarder dans un fichier temporaire
    with open(output_path, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("OK")

except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
'''

    try:
        # Créer des fichiers temporaires
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(loader_script)
            script_path = f.name

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            output_path = f.name

        # Exécuter le script dans un environnement PROPRE
        # Créer un nouvel environnement minimal pour éviter toute contamination Streamlit
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['NO_STREAMLIT'] = '1'

        # Convertir le chemin en string avec protection des caractères spéciaux
        model_path_str = str(model_path)

        # Utiliser python -u pour forcer unbuffered output
        result = subprocess.run(
            [sys.executable, '-u', script_path, model_path_str, model_type, output_path],
            capture_output=True,
            text=True,
            encoding='utf-8',
            env=env,
            timeout=120
        )

        # Nettoyer le script
        os.unlink(script_path)

        if result.returncode == 0 and "OK" in result.stdout:
            # Charger le modèle depuis le fichier temporaire
            with open(output_path, 'rb') as f:
                model = pickle.load(f)

            # Nettoyer
            os.unlink(output_path)

            return model
        else:
            # Nettoyer en cas d'erreur
            if os.path.exists(output_path):
                os.unlink(output_path)

            raise RuntimeError(f"Subprocess loading failed:\n{result.stderr}\n{result.stdout}")

    except Exception as e:
        # Nettoyer les fichiers temporaires
        if 'script_path' in locals() and os.path.exists(script_path):
            os.unlink(script_path)
        if 'output_path' in locals() and os.path.exists(output_path):
            os.unlink(output_path)

        raise RuntimeError(f"Failed to load model in subprocess: {e}")