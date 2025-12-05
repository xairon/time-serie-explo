"""
Chargeur de modèles PyTorch/Darts utilisant un processus Python complètement isolé.

Cette approche évite totalement les conflits avec Streamlit en utilisant
un nouveau processus Python sans aucune contamination.
"""

import os
import sys
import tempfile
import subprocess
import json
import pickle
from pathlib import Path
from typing import Any


def create_isolated_loader_script():
    """Crée un script Python autonome pour charger les modèles."""

    return '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script autonome pour charger les modèles Darts/PyTorch."""

import sys
import os
import json

def main():
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Usage: script.py <model_path> <model_type>"}))
        sys.exit(1)

    model_path = sys.argv[1]
    model_type = sys.argv[2].upper()

    try:
        # Désactiver les warnings
        import warnings
        warnings.filterwarnings("ignore")

        # Forcer l'encodage UTF-8
        os.environ['PYTHONIOENCODING'] = 'utf-8'

        # Import des modèles Darts
        from darts.models import (
            TFTModel, NBEATSModel, NHiTSModel, TransformerModel,
            RNNModel, BlockRNNModel, TCNModel, TiDEModel,
            DLinearModel, NLinearModel
        )

        # Mapping des modèles
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
            print(json.dumps({"error": f"Unknown model type: {model_type}"}))
            sys.exit(1)

        # Charger le modèle
        model = model_class.load(model_path)

        # Sauvegarder dans un fichier temporaire
        import pickle
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl', mode='wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
            temp_path = f.name

        # Retourner le chemin
        print(json.dumps({"success": True, "temp_path": temp_path}))

    except Exception as e:
        import traceback
        print(json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc()
        }))
        sys.exit(1)

if __name__ == "__main__":
    main()
'''


def load_model_completely_isolated(model_path: Path, model_type: str) -> Any:
    """
    Charge un modèle dans un processus Python COMPLÈTEMENT isolé.

    Utilise un nouveau processus Python avec l'option -I pour ignorer
    complètement l'environnement utilisateur et les variables PYTHON*.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Créer le script
    script_content = create_isolated_loader_script()

    # Écrire dans un fichier temporaire
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(script_content)
        script_path = f.name

    try:
        # Préparer l'environnement avec les variables essentielles
        # On copie seulement ce qui est nécessaire, pas Streamlit
        env = {
            'PATH': os.environ.get('PATH', ''),
            'SYSTEMROOT': os.environ.get('SYSTEMROOT', 'C:\\Windows'),
            'TEMP': tempfile.gettempdir(),
            'TMP': tempfile.gettempdir(),
            'PYTHONIOENCODING': 'utf-8',
            'HOME': os.environ.get('HOME', os.path.expanduser('~')),
            'USERPROFILE': os.environ.get('USERPROFILE', os.path.expanduser('~')),
            'HOMEDRIVE': os.environ.get('HOMEDRIVE', 'C:'),
            'HOMEPATH': os.environ.get('HOMEPATH', '\\'),
        }

        # Ajouter le PYTHONPATH pour que Darts soit trouvé
        # Mais PAS l'environnement Streamlit
        python_path_parts = [str(Path(__file__).parent.parent.parent)]

        # Trouver où Darts est installé
        try:
            import darts
            darts_path = Path(darts.__file__).parent.parent
            python_path_parts.append(str(darts_path))
        except ImportError:
            pass

        # Ajouter les site-packages actuels
        import site
        for sp in site.getsitepackages():
            if os.path.exists(sp):
                python_path_parts.append(sp)

        # Sur Windows, utiliser ; comme séparateur
        env['PYTHONPATH'] = ';'.join(python_path_parts) if sys.platform == 'win32' else ':'.join(python_path_parts)

        # Lancer Python avec -s pour ignorer le site utilisateur
        # mais PAS -E car on a besoin de PYTHONPATH pour trouver Darts
        cmd = [sys.executable, '-s', '-u', script_path, str(model_path), model_type]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            env=env,
            timeout=120,
            cwd=tempfile.gettempdir()  # Changer le répertoire de travail
        )

        # Parser le résultat
        if result.returncode == 0:
            try:
                response = json.loads(result.stdout.strip())

                if response.get('success'):
                    # Charger depuis le fichier temporaire
                    temp_path = response['temp_path']
                    with open(temp_path, 'rb') as f:
                        model = pickle.load(f)

                    # Nettoyer
                    os.unlink(temp_path)

                    return model
                else:
                    error = response.get('error', 'Unknown error')
                    raise RuntimeError(f"Model loading failed: {error}")

            except json.JSONDecodeError:
                raise RuntimeError(f"Invalid response: {result.stdout[:500]}")
        else:
            # Erreur d'exécution
            try:
                if result.stdout:
                    response = json.loads(result.stdout.strip())
                    error = response.get('error', 'Unknown error')
                    trace = response.get('traceback', '')
                    raise RuntimeError(f"Loading failed: {error}\n{trace}")
            except:
                pass

            raise RuntimeError(
                f"Subprocess failed:\n"
                f"STDOUT: {result.stdout[:500]}\n"
                f"STDERR: {result.stderr[:500]}"
            )

    finally:
        # Nettoyer le script
        if os.path.exists(script_path):
            os.unlink(script_path)
