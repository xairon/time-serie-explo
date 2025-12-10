"""
Solution robuste pour charger les modèles Darts dans Streamlit.

Le problème principal est que Streamlit patche certaines méthodes Python globalement,
ce qui interfère avec pickle/torch.load(). Cette solution contourne le problème.
"""

import os
import sys
import pickle
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Optional, Tuple
import json


def load_model_isolated(model_path: Path, model_type: str) -> Tuple[Any, Optional[str]]:
    """
    Charge un modèle dans un processus Python complètement isolé.

    Cette méthode crée un script Python temporaire qui charge le modèle
    dans un environnement sans Streamlit, puis retourne le modèle.
    """

    # Script Python autonome pour charger le modèle
    loader_script = '''
import sys
import os
import pickle
from pathlib import Path

# IMPORTANT: Empêcher toute importation de Streamlit
if 'streamlit' in sys.modules:
    del sys.modules['streamlit']

# Désactiver complètement Streamlit
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

# Paramètres passés
model_path = Path(sys.argv[1])
model_type = sys.argv[2]
temp_output = sys.argv[3]

try:
    # Importer Darts SANS Streamlit dans l'environnement
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

    # Charger le modèle avec Darts
    model = model_class.load(str(model_path))

    # Sauvegarder le modèle dans un fichier temporaire
    with open(temp_output, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("SUCCESS")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''

    try:
        # Créer un fichier temporaire pour le script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
            script_file.write(loader_script)
            script_path = script_file.name

        # Créer un fichier temporaire pour la sortie
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as output_file:
            output_path = output_file.name

        # Exécuter le script dans un subprocess complètement isolé
        env = os.environ.copy()
        # Supprimer toute référence à Streamlit de l'environnement
        env.pop('STREAMLIT_SERVER_HEADLESS', None)
        env.pop('STREAMLIT_BROWSER_GATHER_USAGE_STATS', None)

        result = subprocess.run(
            [sys.executable, script_path, str(model_path), model_type, output_path],
            capture_output=True,
            text=True,
            env=env,
            timeout=120
        )

        # Nettoyer le script temporaire
        os.unlink(script_path)

        if result.returncode == 0 and "SUCCESS" in result.stdout:
            # Charger le modèle depuis le fichier temporaire
            with open(output_path, 'rb') as f:
                model = pickle.load(f)

            # Nettoyer le fichier temporaire
            os.unlink(output_path)

            return model, None
        else:
            # Nettoyer le fichier temporaire si erreur
            if os.path.exists(output_path):
                os.unlink(output_path)

            error_msg = result.stderr + "\n" + result.stdout
            return None, error_msg

    except Exception as e:
        # Nettoyer les fichiers temporaires en cas d'exception
        if 'script_path' in locals() and os.path.exists(script_path):
            os.unlink(script_path)
        if 'output_path' in locals() and os.path.exists(output_path):
            os.unlink(output_path)

        return None, str(e)


def load_darts_model_safe(model_path: Path, model_type: str) -> Any:
    """
    Charge un modèle Darts de manière sûre dans Streamlit.

    Cette fonction essaie plusieurs méthodes pour charger le modèle :
    1. Chargement direct (peut échouer avec Streamlit)
    2. Chargement dans un processus isolé (contourne Streamlit)

    Args:
        model_path: Chemin vers le fichier .pkl du modèle
        model_type: Type du modèle (TFT, NBEATS, etc.)

    Returns:
        Le modèle chargé

    Raises:
        RuntimeError: Si le chargement échoue
    """
    from pathlib import Path

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Méthode 1 : Essayer le chargement direct d'abord
    try:
        # Importer les classes de modèles
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
            raise ValueError(f"Type de modèle inconnu: {model_type}")

        # Tenter le chargement direct
        model = model_class.load(str(model_path))
        print(f" Modèle chargé avec succès (méthode directe)")
        return model

    except Exception as e:
        error_msg = str(e)

        # Si c'est l'erreur Streamlit spécifique, essayer la méthode isolée
        if "__setstate__" in error_msg or "StreamlitAPIException" in error_msg:
            print(f" Conflit Streamlit détecté, utilisation du chargeur isolé...")

            # Méthode 2 : Chargement dans un processus isolé
            model, error = load_model_isolated(model_path, model_type)

            if model is not None:
                print(f" Modèle chargé avec succès (méthode isolée)")
                return model
            else:
                # Si même la méthode isolée échoue, afficher les solutions
                raise RuntimeError(
                    f"Impossible de charger le modèle.\n\n"
                    f"Erreur : {error}\n\n"
                    f"Solutions possibles :\n"
                    f"1. Vérifiez que toutes les dépendances sont installées (statsforecast, xgboost)\n"
                    f"2. Si le problème persiste, supprimez et ré-entraînez le modèle :\n"
                    f"   • Supprimer : rmdir /s /q \"{model_path.parent}\"\n"
                    f"   • Puis ré-entraîner sur la page Train Models"
                )
        else:
            # Autre type d'erreur
            raise RuntimeError(f"Erreur lors du chargement : {e}")
