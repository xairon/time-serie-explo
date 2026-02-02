"""
Chargeur robuste de modèles Darts optimisé pour Streamlit.

⚠️ DEPRECATED: Ce module est conservé pour la compatibilité avec les anciens modèles
qui contiennent des références Streamlit. Les nouveaux modèles ne devraient plus
nécessiter ce patch.

Pour les nouveaux modèles, utilisez le chargement standard de Darts :
    model = ModelClass.load(model_path)

Voir ARCHITECTURE.md pour les bonnes pratiques et la migration.
"""

import sys
import os
import pickle
import warnings
from pathlib import Path
from typing import Any
import torch
import numpy as np
import numpy.random
import types

# =============================================================================
# PATCH CRITIQUE : SafeBitGenerator
# =============================================================================
# Streamlit ou des versions différentes de Numpy peuvent faire planter le 
# chargement de l'état du générateur aléatoire (RNG). On crée une classe 
# "bouclier" qui absorbe ces données sans planter.

class SafeBitGenerator:
    """Une classe factice qui remplace numpy.random.BitGenerator défectueux."""
    def __init__(self, *args, **kwargs):
        pass
    
    def __setstate__(self, state):
        # On ignore silencieusement l'état corrompu
        pass
        
    def __getstate__(self):
        return {}

# On rend cette classe accessible globalement pour pickle
if 'SafeBitGenerator' not in globals():
    globals()['SafeBitGenerator'] = SafeBitGenerator

# =============================================================================
# Streamlit stubs
# =============================================================================


class _FakeCallable:
    def __call__(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        if name == "wrapper":
            return lambda func: func
        return self

    def __setstate__(self, state):
        pass


class _FakeDeltaGenerator:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        if name == "wrapper":
            return lambda func: func
        return self

    def wrapper(self, func):
        return func


def _make_fake_class(name: str):
    """Crée dynamiquement une classe factice."""

    class _FakeClass:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return None

        def __getattr__(self, attr):
            if attr == "wrapper":
                return lambda func: func
            return self

        def __getitem__(self, key):
            return _FakeCallable()

        def __iter__(self):
            return iter([])

        def __len__(self):
            return 0

        def __setstate__(self, state):
            pass

    _FakeClass.__name__ = name
    return _FakeClass


class _FakeStreamlitModule(types.ModuleType):
    """Module factice pour neutraliser les hooks Streamlit pendant le chargement."""

    def __init__(self, name):
        super().__init__(name)
        self.__file__ = "<fake_streamlit>"
        self.__path__ = []
        self.__spec__ = None
        self.__loader__ = None

    def __getattr__(self, name):
        if name in {"__file__", "__path__", "__spec__", "__loader__"}:
            return getattr(self, name)
        if name == "DeltaGenerator":
            return _FakeDeltaGenerator
        if name.endswith("Exception") or (name and name[0].isupper()):
            return _make_fake_class(name)
        return _FakeCallable()


def _disable_streamlit_temporarily():
    """Remplace tous les modules Streamlit par un stub no-op."""
    saved_modules = {}
    fake_module = _FakeStreamlitModule("streamlit")

    for module_name in list(sys.modules.keys()):
        if module_name == "streamlit" or module_name.startswith("streamlit."):
            saved_modules[module_name] = sys.modules[module_name]
            sys.modules[module_name] = fake_module

    # S'assurer que les modules clés existent même s'ils n'étaient pas importés
    sys.modules["streamlit"] = fake_module
    sys.modules["streamlit.delta_generator"] = fake_module
    sys.modules["streamlit.errors"] = fake_module

    # Neutraliser les variables d'environnement
    os.environ["_IS_RUNNING_WITH_STREAMLIT"] = "false"
    os.environ["STREAMLIT_RUNTIME_EXISTS"] = "false"

    return saved_modules


def _restore_streamlit_modules(saved_modules):
    """Restaure les modules Streamlit après le chargement."""
    for module_name in list(sys.modules.keys()):
        if module_name.startswith("streamlit"):
            if module_name not in saved_modules:
                del sys.modules[module_name]

    for module_name, module in saved_modules.items():
        sys.modules[module_name] = module

# =============================================================================
# Unpickler Personnalisé
# =============================================================================

class StreamlitSafeUnpickler(pickle.Unpickler):
    """
    Unpickler qui remplace à la volée les classes problématiques et délègue
    le chargement PyTorch (storage) pour éviter "no persistent_load function".
    """
    def find_class(self, module, name):
        # 0. PyTorch storage (évite "no persistent_load function")
        if module == "torch.storage" and name == "_load_from_bytes":
            import io
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        # 1. Intercepter les générateurs Numpy qui plantent
        if "BitGenerator" in name and "numpy" in module:
            return SafeBitGenerator
        # 2. Gérer les renommages internes de Numpy (1.x vs 2.x)
        if module == "numpy._core.multiarray":
            module = "numpy.core.multiarray"
        elif module == "numpy.core.multiarray" and "numpy._core" in sys.modules:
            module = "numpy._core.multiarray"
        # 3. Fallback standard
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            class Dummy:
                def __setstate__(self, state): pass
                def __init__(self, *args, **kwargs): pass
            return Dummy

# =============================================================================
# Fonction principale
# =============================================================================

def load_model_safe(model_path: Path, model_type: str) -> Any:
    """
    Charge un modèle Darts de manière robuste.
    
    Args:
        model_path: Chemin vers le fichier .pkl
        model_type: Type du modèle (TFT, NBEATS, etc.)
        
    Returns:
        Le modèle chargé et prêt à l'emploi.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 1. Import des classes de modèles
    from darts.models import (
        TFTModel, NBEATSModel, NHiTSModel, TransformerModel,
        RNNModel, BlockRNNModel, TCNModel, TiDEModel,
        DLinearModel, NLinearModel, TSMixerModel,
        GlobalNaiveAggregate, GlobalNaiveDrift, GlobalNaiveSeasonal
    )
    
    model_classes = {
        'TFT': TFTModel, 'NBEATS': NBEATSModel, 'NHITS': NHiTSModel,
        'TRANSFORMER': TransformerModel, 'LSTM': RNNModel, 'GRU': RNNModel,
        'BLOCKRNN': BlockRNNModel, 'TCN': TCNModel, 'TIDE': TiDEModel,
        'DLINEAR': DLinearModel, 'NLINEAR': NLinearModel, 'TSMIXER': TSMixerModel,
        'GLOBALNAIVEAGGREGATE': GlobalNaiveAggregate,
        'GLOBALNAIVEDRIFT': GlobalNaiveDrift,
        'GLOBALNAIVESEASONAL': GlobalNaiveSeasonal,
    }
    
    model_class = model_classes.get(model_type.upper())
    if not model_class:
        raise ValueError(f"Unknown model type: {model_type}")

    # 2. Patching de torch.load pour utiliser notre Unpickler
    # C'est ici que la magie opère : on injecte notre logique dans Torch
    
    original_torch_load = torch.load
    
    def safe_torch_load(f, map_location=None, pickle_module=None, **kwargs):
        """Version patchée de torch.load qui utilise notre StreamlitSafeUnpickler."""
        if pickle_module is None:
            # On crée un module pickle factice qui utilise notre Unpickler
            class CustomPickle:
                def __init__(self):
                    # Copie les attributs de pickle
                    for k, v in pickle.__dict__.items():
                        if not k.startswith('__'):
                            setattr(self, k, v)
                    self.Unpickler = StreamlitSafeUnpickler
                    self.__name__ = "pickle" # Important pour torch
                
                def load(self, file, **kwargs):
                    return self.Unpickler(file, **kwargs).load()
            
            pickle_module = CustomPickle()
            
        return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, **kwargs)

    # 3. Application du patch temporaire sur la méthode .load() du modèle
    # Darts utilise torch.load en interne via TorchForecastingModel.load
    
    # On doit patcher au niveau de la classe Darts ou intercepter l'appel
    # Le plus simple est de charger manuellement si possible, ou de patcher torch.load globalement
    # Patcher torch.load globalement est risqué, mais temporaire c'est ok.
    
    saved_streamlit = _disable_streamlit_temporarily()
    try:
        # Patch temporaire de torch.load
        torch.load = safe_torch_load
        
        # Désactiver les warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Chargement !
            model = model_class.load(str(model_path))
            
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_type} despite patches: {e}")
        
    finally:
        # RESTAURATION IMPORTANTE
        torch.load = original_torch_load
        _restore_streamlit_modules(saved_streamlit)

def _load_in_subprocess(*args, **kwargs):
    """Deprecated: Kept for compatibility but redirects to load_model_safe."""
    return load_model_safe(*args, **kwargs)
