#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script autonome de chargement de modèles Darts avec isolation complète de Streamlit
et gestion des incompatibilités numpy.

Ce script doit être exécuté depuis la racine du projet ou un dossier temporaire,
PAS depuis dashboard/utils pour éviter l'interférence de Streamlit.
"""

import sys
import os
import pickle
import json

# Créer les classes factices au niveau global pour qu'elles soient accessibles partout
class FakeStreamlit:
    def __getattr__(self, name):
        # Retourner une fonction lambda qui ne fait rien pour toute méthode
        return lambda *args, **kwargs: None

    def __setstate__(self, state):
        # Cette méthode est appelée par pickle, on la rend silencieuse
        pass

class FakeDeltaGenerator:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

    def wrapper(self, func):
        # Retourner la fonction sans modification
        return func

def block_streamlit():
    """
    Bloque complètement Streamlit en créant un module factice qui empêche son import.
    Cette fonction doit être appelée AVANT tout autre import.
    """
    # Remplacer streamlit dans sys.modules AVANT qu'il ne soit importé
    fake_streamlit = FakeStreamlit()
    fake_delta_gen = FakeDeltaGenerator()

    sys.modules['streamlit'] = fake_streamlit
    sys.modules['streamlit.delta_generator'] = fake_delta_gen
    sys.modules['streamlit.errors'] = fake_streamlit
    sys.modules['streamlit.components'] = fake_streamlit
    sys.modules['streamlit.components.v1'] = fake_streamlit

    # Désactiver complètement l'environnement Streamlit
    os.environ['_IS_RUNNING_WITH_STREAMLIT'] = 'false'
    os.environ['STREAMLIT_RUNTIME_EXISTS'] = 'false'

    # Supprimer toute variable d'environnement Streamlit
    for key in list(os.environ.keys()):
        if 'STREAMLIT' in key:
            del os.environ[key]


def fix_numpy_compatibility():
    """
    Fix les problèmes de compatibilité entre différentes versions de numpy.
    Doit être appelé APRÈS l'import de numpy mais AVANT torch.load.
    """
    import numpy as np
    import numpy.random
    from numpy.random import MT19937

    # 1. Fix pour numpy._core (numpy 2.x vs 1.x)
    # numpy 2.x utilise _core, numpy 1.x utilise core
    if not hasattr(np, '_core') and hasattr(np, 'core'):
        np._core = np.core
        sys.modules['numpy._core'] = np.core
    elif hasattr(np, '_core') and not hasattr(np, 'core'):
        np.core = np._core
        sys.modules['numpy.core'] = np._core

    # 2. Fix pour numpy.random._mt19937
    # Créer un module factice pour numpy.random._mt19937 si nécessaire
    if 'numpy.random._mt19937' not in sys.modules:
        # Créer un module dynamique
        import types
        mt_module = types.ModuleType('numpy.random._mt19937')
        mt_module.MT19937 = MT19937
        sys.modules['numpy.random._mt19937'] = mt_module
        # Aussi l'ajouter comme attribut
        numpy.random._mt19937 = mt_module

    # 3. Patch le module pickle de numpy pour gérer les anciens formats
    try:
        import numpy.random._pickle as pickle_module

        # Patch la fonction de construction du générateur
        def patched_bit_generator_ctor(bit_generator_name):
            """Version patchée qui gère les références manquantes."""
            # Gestion des différentes références possibles
            if 'MT19937' in str(bit_generator_name):
                return MT19937
            # Si le nom contient _mt19937, retourner MT19937
            if '_mt19937' in str(bit_generator_name):
                return MT19937
            # Fallback pour tout autre cas
            return MT19937

        # Remplacer la fonction
        pickle_module.__bit_generator_ctor = patched_bit_generator_ctor

    except (ImportError, AttributeError):
        pass  # Le module pickle n'existe peut-être pas dans cette version


def create_custom_unpickler():
    """
    Crée un unpickler personnalisé qui gère les problèmes de compatibilité.
    """
    import pickle
    import io

    class CompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Gérer les remappages de modules numpy
            if module == 'numpy._core.multiarray':
                module = 'numpy.core.multiarray'
            elif module == 'numpy.core.multiarray' and 'numpy._core' in sys.modules:
                module = 'numpy._core.multiarray'
            elif module == 'numpy.random._mt19937':
                # Retourner directement la classe MT19937
                import numpy.random
                return numpy.random.MT19937

            # Utiliser la méthode parent
            return super().find_class(module, name)

    return CompatUnpickler


def main():
    if len(sys.argv) != 4:
        print(json.dumps({"error": "Usage: load_model_isolated.py <model_path> <model_type> <output_path>"}))
        sys.exit(1)

    model_path = sys.argv[1]
    model_type = sys.argv[2].upper()
    output_path = sys.argv[3]

    # Convertir en chemin absolu si nécessaire
    from pathlib import Path
    model_path = Path(model_path).absolute()
    output_path = Path(output_path).absolute()

    # Vérifier que le fichier existe
    if not model_path.exists():
        print(json.dumps({
            "error": f"Model file not found: {model_path}",
            "cwd": os.getcwd(),
            "model_path_str": str(model_path)
        }))
        sys.exit(1)

    try:
        # TRÈS IMPORTANT : Bloquer Streamlit AVANT tout autre import
        block_streamlit()

        # Désactiver les warnings
        import warnings
        warnings.filterwarnings("ignore")

        # Importer numpy et fixer les problèmes de compatibilité
        import numpy as np
        fix_numpy_compatibility()

        # Maintenant on peut importer torch en toute sécurité
        import torch
        import io

        # Créer une fonction de chargement personnalisée
        def safe_torch_load(path):
            """
            Charge un modèle PyTorch de manière sécurisée en gérant les problèmes de compatibilité.
            """
            # Utiliser notre unpickler personnalisé
            CompatUnpickler = create_custom_unpickler()

            # Charger le fichier en mémoire
            with open(path, 'rb') as f:
                buffer = io.BytesIO(f.read())

            # Essayer avec l'unpickler personnalisé
            try:
                buffer.seek(0)
                # Utiliser torch.load avec un unpickler personnalisé est compliqué
                # On va plutôt patcher temporairement pickle
                original_unpickler = pickle.Unpickler
                pickle.Unpickler = CompatUnpickler

                try:
                    model = torch.load(buffer, map_location='cpu')
                finally:
                    # Restaurer l'unpickler original
                    pickle.Unpickler = original_unpickler

                return model

            except Exception as e:
                # Si ça ne marche pas, essayer avec torch.load standard
                buffer.seek(0)
                return torch.load(buffer, map_location='cpu')

        # Import des modèles Darts APRÈS avoir bloqué Streamlit et fixé numpy
        from darts.models import (
            TFTModel, NBEATSModel, NHiTSModel, TransformerModel,
            RNNModel, BlockRNNModel, TCNModel, TiDEModel,
            DLinearModel, NLinearModel
        )

        # Remplacer la méthode load de TorchForecastingModel
        from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel

        # Créer une version patchée
        @classmethod
        def patched_load(cls, path, **kwargs):
            """Version patchée de load qui utilise safe_torch_load."""
            path = Path(path)

            # S'assurer que le fichier existe
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {path}")

            # Charger le modèle avec notre méthode sécurisée
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Désactiver temporairement les hooks pickle de Streamlit s'ils existent
                old_modules = {}
                for mod_name in ['streamlit', 'streamlit.delta_generator', 'streamlit.errors']:
                    if mod_name in sys.modules:
                        old_modules[mod_name] = sys.modules[mod_name]
                        sys.modules[mod_name] = FakeStreamlit()

                try:
                    # Charger avec notre fonction safe
                    model = safe_torch_load(str(path))
                finally:
                    # Restaurer les modules
                    for mod_name, mod in old_modules.items():
                        sys.modules[mod_name] = mod

                return model

        # Remplacer la méthode load pour tous les modèles
        TorchForecastingModel.load = patched_load

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

        # Charger le modèle avec notre méthode patchée
        model = model_class.load(model_path)

        # Sauvegarder dans le fichier de sortie
        with open(output_path, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(json.dumps({"success": True}))

    except Exception as e:
        import traceback
        error_msg = str(e)

        # Vérifier si c'est toujours une erreur Streamlit
        if "streamlit" in error_msg.lower() or "__setstate__" in error_msg:
            error_msg = f"Streamlit interference detected despite isolation: {error_msg}"

        print(json.dumps({
            "error": error_msg,
            "traceback": traceback.format_exc()
        }))
        sys.exit(1)


if __name__ == "__main__":
    # S'assurer qu'on n'est pas dans l'environnement Streamlit
    if os.environ.get('STREAMLIT_RUNTIME_EXISTS') == 'true':
        print(json.dumps({"error": "ERROR: Streamlit is already loaded in this environment"}))
        sys.exit(1)

    main()