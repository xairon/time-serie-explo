"""Factory pour instancier dynamiquement les modèles Darts.

Modèles supportés:
- Deep Learning: Tous utilisent input_chunk_length/output_chunk_length
- Global Baselines: Utilisent aussi input/output chunk (compatibles)
"""

import torch
import logging
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
        logger = logging.getLogger(__name__)

        if model_name not in cls.MODEL_CLASSES:
            raise ValueError(f"Modèle inconnu: {model_name}. Disponibles: {list(cls.MODEL_CLASSES.keys())}")

        model_class = cls.MODEL_CLASSES[model_name]
        params = hyperparams.copy()

        # Validate hyperparameters
        cls._validate_hyperparams(params, model_name, logger)

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
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                device = 'xpu'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

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

        # XPU is handled via custom strategy in pl_trainer_kwargs_override
        # Do NOT set accelerator='xpu' here as it's not natively supported by PL 2.5
        if device == 'xpu':
            # XPU config will come from pl_trainer_kwargs_override (custom strategy)
            # Don't set accelerator here - it conflicts with the strategy
            pass
        elif device == 'cuda':
            params['pl_trainer_kwargs']['accelerator'] = 'gpu'
            if torch.cuda.is_available():
                params['pl_trainer_kwargs']['devices'] = [0]  # GPU 0
        else:
            params['pl_trainer_kwargs']['accelerator'] = 'cpu'

        # Désactiver certains logs et la progress bar pour compatibilité Streamlit
        params['pl_trainer_kwargs']['enable_progress_bar'] = False
        params['pl_trainer_kwargs']['enable_model_summary'] = False
        # IMPORTANT: enable checkpointing so model.save includes weights (.ckpt)
        params['pl_trainer_kwargs']['enable_checkpointing'] = True
        params['save_checkpoints'] = True

        # Disable sanity validation to avoid confusion in logs
        params['pl_trainer_kwargs']['num_sanity_val_steps'] = 0

        # Gradient clipping pour éviter l'explosion des gradients
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
    def _validate_hyperparams(cls, params: Dict[str, Any], model_name: str, logger) -> None:
        """
        Validate hyperparameters and raise ValueError if invalid.

        Args:
            params: Hyperparameters dictionary
            model_name: Name of the model
            logger: Logger instance for warnings
        """
        # Validate learning rate
        lr = params.get('learning_rate', 1e-3)
        if lr is not None and lr <= 0:
            raise ValueError(f"learning_rate must be > 0, got {lr}")

        # Validate epochs
        n_epochs = params.get('n_epochs', 50)
        if n_epochs is not None and n_epochs < 1:
            raise ValueError(f"n_epochs must be >= 1, got {n_epochs}")

        # Validate batch size
        batch_size = params.get('batch_size', 32)
        if batch_size is not None and batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        # Validate chunk lengths
        input_chunk = params.get('input_chunk_length', 30)
        output_chunk = params.get('output_chunk_length', 7)

        if input_chunk is not None and input_chunk < 1:
            raise ValueError(f"input_chunk_length must be >= 1, got {input_chunk}")

        if output_chunk is not None and output_chunk < 1:
            raise ValueError(f"output_chunk_length must be >= 1, got {output_chunk}")

        # Model-specific validation
        if model_name in ['LSTM', 'GRU']:
            hidden_dim = params.get('hidden_dim', 64)
            if hidden_dim is not None and hidden_dim < 1:
                raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")

            n_rnn_layers = params.get('n_rnn_layers', 1)
            if n_rnn_layers is not None and n_rnn_layers < 1:
                raise ValueError(f"n_rnn_layers must be >= 1, got {n_rnn_layers}")

        # Validate dropout if present
        dropout = params.get('dropout')
        if dropout is not None and (dropout < 0 or dropout > 1):
            raise ValueError(f"dropout must be between 0 and 1, got {dropout}")

        # Log warning for potentially problematic values
        if lr and lr > 0.1:
            logger.warning(f"learning_rate={lr} is unusually high, consider values < 0.01")

        if batch_size and batch_size > 512:
            logger.warning(f"batch_size={batch_size} is large, may cause memory issues")

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
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return 'xpu'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def get_device_info() -> Dict[str, Any]:
    """Retourne des informations sur le device."""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'xpu_available': hasattr(torch, 'xpu') and torch.xpu.is_available(),
        'device': get_device(),
    }

    if info['xpu_available']:
        info['gpu_count'] = torch.xpu.device_count()
        try:
            info['gpu_name'] = torch.xpu.get_device_name(0)
        except:
            info['gpu_name'] = "Intel Arc GPU"
        info['xpu_version'] = getattr(torch.version, 'xpu', 'Unknown')

    elif info['cuda_available']:
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda

    info['torch_version'] = torch.__version__

    return info

