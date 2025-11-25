"""
Solution ULTIME pour charger des modèles PyTorch dans Streamlit.

Le problème: Streamlit patche object.__setattr__ et __setstate__ globalement,
ce qui interfère avec pickle.Unpickler utilisé par torch.load().

Cette solution:
1. Désactive temporairement les patches Streamlit
2. Charge le modèle avec torch.load
3. Restaure les patches

Si ça échoue, on utilise un subprocess Python séparé pour charger le modèle
HORS du contexte Streamlit.
"""

import subprocess
import sys
import json
import tempfile
from pathlib import Path
from typing import Any, Optional, Tuple
import pickle


def _unpatch_streamlit():
    """
    Supprime temporairement les patches Streamlit sur pickle.
    
    Returns:
        dict avec les références originales pour restauration
    """
    import pickle
    
    saved = {}
    
    # Sauvegarder et restaurer le Unpickler original si patché
    if hasattr(pickle, '_original_Unpickler'):
        saved['Unpickler'] = pickle.Unpickler
        pickle.Unpickler = pickle._original_Unpickler
    
    # Vérifier si object a des méthodes patchées
    # Streamlit peut patcher __setattr__ sur certaines classes
    
    return saved


def _repatch_streamlit(saved: dict):
    """Restaure les patches Streamlit."""
    import pickle
    
    if 'Unpickler' in saved:
        pickle.Unpickler = saved['Unpickler']


def load_with_unpatch(model_path: Path, model_class: Any) -> Any:
    """
    Charge un modèle en désactivant temporairement les patches Streamlit.
    
    Args:
        model_path: Chemin vers le fichier .pkl
        model_class: Classe du modèle Darts
        
    Returns:
        Le modèle chargé
    """
    import torch
    import pickle
    import io
    
    # Essayer de charger avec torch directement mais en contournant Streamlit
    # En créant notre propre Unpickler non-patché
    
    class CleanUnpickler(pickle.Unpickler):
        """Unpickler qui ignore les interférences Streamlit."""
        
        def find_class(self, module, name):
            # Utiliser la méthode standard
            return super().find_class(module, name)
    
    try:
        # Méthode 1: Utiliser torch.load avec un buffer
        # torch.load gère mieux les incompatibilités
        with open(model_path, 'rb') as f:
            # Lire tout le fichier
            data = f.read()
        
        buffer = io.BytesIO(data)
        
        # Essayer torch.load qui a sa propre logique de désérialisation
        model = torch.load(buffer, map_location='cpu', weights_only=False)
        
        if hasattr(model, 'predict'):
            return model
            
    except Exception as e:
        pass  # Continuer vers la méthode 2
    
    # Méthode 2: Utiliser la méthode .load() de Darts
    # avec les patches temporairement désactivés
    saved = _unpatch_streamlit()
    try:
        model = model_class.load(str(model_path))
        return model
    finally:
        _repatch_streamlit(saved)


def load_model_in_subprocess(model_path: Path, model_type: str) -> Tuple[Optional[Any], Optional[str]]:
    """
    Charge un modèle Darts dans un subprocess Python séparé (sans Streamlit).

    Args:
        model_path: Chemin vers le fichier .pkl
        model_type: Type du modèle (TFT, NBEATS, etc.)

    Returns:
        Tuple (model, error_message)
        model est None si échec
    """
    # Ajouter le chemin du projet au PYTHONPATH du subprocess
    project_root = Path(__file__).parent.parent.parent.absolute()
    
    # Script Python qui s'exécute HORS de Streamlit
    loader_script = f"""
import sys
import pickle
from pathlib import Path

# Ajouter le root au path pour trouver 'dashboard'
sys.path.insert(0, r"{project_root}")

# 🛑 BANNIR STREAMLIT DU SUBPROCESS 🛑
# C'est crucial car sinon pickle peut essayer de charger des hooks Streamlit
import sys
sys.modules['streamlit'] = None  # Empêcher l'import de streamlit

# Si streamlit est déjà chargé (peu probable dans un subprocess frais mais on sait jamais)
for key in list(sys.modules.keys()):
    if key.startswith('streamlit'):
        del sys.modules[key]

# Charger le modèle avec torch.load (sans Streamlit)
model_path = Path(r"{model_path}")

try:
    # Import Darts DANS le subprocess (pas de Streamlit ici!)
    from darts.models import (
        TFTModel, NBEATSModel, NHiTSModel, TransformerModel,
        RNNModel, BlockRNNModel, TCNModel, TiDEModel,
        DLinearModel, NLinearModel
    )
    
    # S'assurer que dashboard est importable (pour les classes custom si besoin)
    import dashboard

    model_classes = {{
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
    }}

    model_class = model_classes.get('{model_type}')
    if not model_class:
        raise ValueError(f"Unknown model type: {model_type}")

    # Charger avec la méthode standard (PAS de Streamlit ici!)
    model = model_class.load(str(model_path))

    # Sauvegarder dans un fichier temporaire
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
    with open(temp_file.name, 'wb') as f:
        pickle.dump(model, f)

    # Retourner le chemin du fichier temporaire
    print(temp_file.name)
    sys.exit(0)

except Exception as e:
    print(f"ERROR: {{e}}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"""

    try:
        # Exécuter le script dans un subprocess PROPRE (sans Streamlit)
        result = subprocess.run(
            [sys.executable, "-c", loader_script],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            error_msg = result.stderr
            print(f"Subprocess error: {error_msg}")
            return None, error_msg

        # Récupérer le chemin du fichier temporaire
        temp_path = result.stdout.strip()

        # Charger depuis le fichier temporaire (déjà sérialisé, donc compatible)
        with open(temp_path, 'rb') as f:
            model = pickle.load(f)

        # Nettoyer le fichier temporaire
        Path(temp_path).unlink()

        return model, None

    except Exception as e:
        print(f"Failed to load model in subprocess: {e}")
        return None, str(e)


def load_darts_model_streamlit_safe(model_path: Path, model_type: str) -> Any:
    """
    Charge un modèle Darts de manière compatible avec Streamlit.

    Stratégie de chargement (en ordre):
    1. Chargement avec désactivation temporaire des patches Streamlit
    2. Chargement via subprocess (si erreur __setstate__)
    3. Chargement direct standard (fallback)

    Args:
        model_path: Chemin vers le fichier .pkl
        model_type: Type du modèle (TFT, NBEATS, etc.)

    Returns:
        Le modèle chargé

    Raises:
        RuntimeError: Si toutes les méthodes échouent
    """
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

    model_class = model_classes.get(model_type)
    if not model_class:
        raise ValueError(f"Type de modèle inconnu: {model_type}")

    errors = []
    
    # Méthode 1: Chargement avec désactivation des patches Streamlit
    try:
        model = load_with_unpatch(model_path, model_class)
        if model is not None and hasattr(model, 'predict'):
            return model
    except Exception as e:
        errors.append(f"load_with_unpatch: {e}")

    # Méthode 2: Essayer le chargement direct (parfois ça marche)
    try:
        model = model_class.load(str(model_path))
        return model
    except Exception as e:
        error_msg = str(e)
        errors.append(f"direct_load: {e}")

        # Si c'est l'erreur Streamlit __setstate__, essayer le subprocess
        if "__setstate__" in error_msg or "StreamlitAPIException" in str(type(e)):
            print(f"⚠️ Streamlit conflict detected, trying subprocess loader...")

            # Méthode 3: Subprocess (ultime recours)
            try:
                model, sub_error = load_model_in_subprocess(model_path, model_type)

                if model is not None:
                    print("✅ Model loaded successfully via subprocess!")
                    return model
                else:
                    errors.append(f"subprocess failed: {sub_error}")
            except Exception as e3:
                errors.append(f"subprocess exception: {e3}")

    # Toutes les méthodes ont échoué
    all_errors = "\n".join(f"  - {err}" for err in errors)
    raise RuntimeError(
        f"Impossible de charger le modèle après 3 tentatives.\n\n"
        f"Erreurs rencontrées:\n{all_errors}\n\n"
        f"✅ Solution: Supprimez le dossier du modèle et ré-entraînez:\n"
        f"   rm -rf {model_path.parent}\n"
        f"   Puis ré-entraînez sur la page Train Models"
    )
