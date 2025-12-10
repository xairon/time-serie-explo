"""Factory pour instancier dynamiquement les modèles Darts.

Modèles supportés:
- Deep Learning: Tous utilisent input_chunk_length/output_chunk_length
- Global Baselines: Utilisent aussi input/output chunk (compatibles)
"""

import torch
from typing import Dict, Any, Optional
from darts.models import (
    # Deep Learning - Tous utilisent input_chunk_length/output_chunk_length
    TFTModel,
    NBEATSModel,
    NHiTSModel,
    TransformerModel,
    RNNModel,
    BlockRNNModel,
    TCNModel,
    TiDEModel,
    DLinearModel,
    NLinearModel,
    TSMixerModel,
    
    # Global Baselines - Compatibles avec input/output chunk
    GlobalNaiveAggregate,
    GlobalNaiveDrift,
    GlobalNaiveSeasonal,
)


class ModelFactory:
    """
    Factory pour créer des modèles Darts de façon dynamique.
    
    Supporte les modèles Deep Learning et Global Baselines qui partagent
    une interface commune (input_chunk_length, output_chunk_length).
    """

    # Mapping des noms vers les classes
    MODEL_CLASSES = {
        # Deep Learning
        'TFT': TFTModel,
        'NBEATS': NBEATSModel,
        'NHiTS': NHiTSModel,
        'Transformer': TransformerModel,
        'LSTM': RNNModel,
        'GRU': RNNModel,
        'BlockRNN': BlockRNNModel,
        'TCN': TCNModel,
        'TiDE': TiDEModel,
        'DLinear': DLinearModel,
        'NLinear': NLinearModel,
        'TSMixer': TSMixerModel,
        
        # Global Baselines
        'GlobalNaiveAggregate': GlobalNaiveAggregate,
        'GlobalNaiveDrift': GlobalNaiveDrift,
        'GlobalNaiveSeasonal': GlobalNaiveSeasonal,
    }

    # Modèles PyTorch (Deep Learning)
    TORCH_MODELS = {
        'TFT', 'NBEATS', 'NHiTS', 'Transformer',
        'LSTM', 'GRU', 'BlockRNN', 'TCN', 'TiDE',
        'DLinear', 'NLinear', 'TSMixer'
    }
    
    # Modèles qui n'ont PAS besoin de PyTorch
    NON_TORCH_MODELS = {
        'GlobalNaiveAggregate', 'GlobalNaiveDrift', 'GlobalNaiveSeasonal'
    }

    @classmethod
    def create_model(
        cls,
        model_name: str,
        hyperparams: Dict[str, Any],
        device: Optional[str] = None,
        pl_trainer_kwargs_override: Optional[Dict[str, Any]] = None
    ):
        """
        Crée une instance de modèle avec les hyperparamètres donnés.

        Args:
            model_name: Nom du modèle (ex: 'TFT', 'NBEATS', etc.)
            hyperparams: Dictionnaire d'hyperparamètres
            device: 'cpu', 'cuda', ou None (auto-détection)
            pl_trainer_kwargs_override: Override pour pl_trainer_kwargs (callbacks, etc.)

        Returns:
            Instance du modèle Darts
        """
        if model_name not in cls.MODEL_CLASSES:
            raise ValueError(f"Modèle inconnu: {model_name}. Disponibles: {list(cls.MODEL_CLASSES.keys())}")

        model_class = cls.MODEL_CLASSES[model_name]
        params = hyperparams.copy()

        # =====================================================================
        # MODÈLES NON-PYTORCH (Global Baselines)
        # =====================================================================
        if model_name in cls.NON_TORCH_MODELS:
            # Ces modèles n'ont besoin que de input/output_chunk_length
            simple_params = {
                'input_chunk_length': params.get('input_chunk_length', 30),
                'output_chunk_length': params.get('output_chunk_length', 7),
            }
            try:
                model = model_class(**simple_params)
                return model
            except Exception as e:
                raise RuntimeError(f"Erreur création {model_name}: {e}")

        # =====================================================================
        # MODÈLES PYTORCH (Deep Learning)
        # =====================================================================
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Extraire les paramètres optimiseur
        learning_rate = params.pop('learning_rate', 1e-3)
        n_epochs = params.pop('n_epochs', 50)
        batch_size = params.pop('batch_size', 32)

        params['n_epochs'] = n_epochs
        params['batch_size'] = batch_size

        # Learning rate dans optimizer_kwargs
        if 'optimizer_kwargs' not in params:
            params['optimizer_kwargs'] = {}
        params['optimizer_kwargs']['lr'] = learning_rate

        # Gestion de la fonction de loss
        loss_fn_name = params.pop('loss_fn', None)
        if loss_fn_name:
            loss_fn = cls._get_loss_function(loss_fn_name, params)
            if loss_fn is not None:
                params['loss_fn'] = loss_fn

        # Configuration PyTorch Lightning
        if 'pl_trainer_kwargs' not in params:
            params['pl_trainer_kwargs'] = {}

        params['pl_trainer_kwargs']['accelerator'] = 'gpu' if device == 'cuda' else 'cpu'

        if device == 'cuda' and torch.cuda.is_available():
            params['pl_trainer_kwargs']['devices'] = [0]  # GPU 0

        # Désactiver certains logs et la progress bar pour compatibilité Streamlit
        params['pl_trainer_kwargs']['enable_progress_bar'] = False
        params['pl_trainer_kwargs']['enable_model_summary'] = False
        params['pl_trainer_kwargs']['enable_checkpointing'] = False
        
        #  Gradient clipping pour éviter l'explosion des gradients
        params['pl_trainer_kwargs']['gradient_clip_val'] = 1.0
        params['pl_trainer_kwargs']['gradient_clip_algorithm'] = 'norm'

        # Override avec les kwargs fournis (pour callbacks notamment)
        if pl_trainer_kwargs_override:
            params['pl_trainer_kwargs'].update(pl_trainer_kwargs_override)

        # Random seed pour reproductibilité
        params['random_state'] = 42

        # Gestion spéciale pour RNNModel (LSTM/GRU)
        if model_name in ['LSTM', 'GRU']:
            params['model'] = model_name  # Spécifier le type de RNN
            # RNNModel requires training_length >= input_chunk_length
            if 'training_length' not in params:
                input_len = params.get('input_chunk_length', 30)
                output_len = params.get('output_chunk_length', 7)
                params['training_length'] = input_len + output_len

        # Gestion spéciale pour TFT (requires future covariates or add_relative_index)
        if model_name == 'TFT':
            if 'add_relative_index' not in params and 'add_encoders' not in params:
                params['add_relative_index'] = True

        # Créer le modèle
        try:
            model = model_class(**params)
            return model
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la création du modèle {model_name}: {e}")

    @classmethod
    def _get_loss_function(cls, loss_name: str, params: Dict[str, Any]):
        """
        Convertit le nom de la fonction de loss en objet PyTorch loss.
        """
        import torch.nn as nn

        if loss_name == 'MAE':
            return None  # Laisser Darts utiliser le défaut

        elif loss_name == 'MSE':
            return nn.MSELoss()

        elif loss_name == 'Huber':
            delta = params.pop('loss_delta', 1.0)
            return nn.HuberLoss(delta=delta)

        elif loss_name == 'Quantile':
            try:
                from darts.utils.losses import QuantileLoss
                quantile = params.pop('loss_quantile', 0.5)
                return QuantileLoss(quantile=quantile)
            except ImportError:
                return None

        elif loss_name == 'RMSE':
            return nn.MSELoss()

        else:
            return None

    @classmethod
    def get_model_class(cls, model_name: str):
        """Retourne la classe du modèle."""
        return cls.MODEL_CLASSES.get(model_name)

    @classmethod
    def is_torch_model(cls, model_name: str) -> bool:
        """Vérifie si le modèle utilise PyTorch."""
        return model_name in cls.TORCH_MODELS

    @classmethod
    def get_available_models(cls):
        """Retourne la liste des modèles disponibles."""
        return list(cls.MODEL_CLASSES.keys())


def get_device() -> str:
    """Détecte le device disponible."""
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def get_device_info() -> Dict[str, Any]:
    """Retourne des informations sur le device."""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    if torch.cuda.is_available():
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda

    info['torch_version'] = torch.__version__

    return info

