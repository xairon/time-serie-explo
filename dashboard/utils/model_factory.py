"""Factory pour instancier dynamiquement les modèles Darts."""

import torch
from typing import Dict, Any, Optional
from darts.models import (
    # Deep Learning
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

    # Statistical
    ARIMA,
    AutoARIMA,
    VARIMA,
    ExponentialSmoothing,
    Theta,
    FourTheta,
    Prophet,

    # ML
    RandomForest,
    LightGBMModel,
    XGBModel,
    CatBoostModel,
    LinearRegressionModel,

    # Ensemble
    NaiveEnsembleModel,

    # Baselines
    NaiveSeasonal,
    NaiveDrift,
    NaiveMean,
)


class ModelFactory:
    """
    Factory pour créer des modèles Darts de façon dynamique.
    """

    # Mapping des noms vers les classes
    MODEL_CLASSES = {
        # Deep Learning
        'TFT': TFTModel,
        'NBEATS': NBEATSModel,
        'NHiTS': NHiTSModel,
        'Transformer': TransformerModel,
        'LSTM': RNNModel,  # RNNModel avec model='LSTM'
        'GRU': RNNModel,   # RNNModel avec model='GRU'
        'BlockRNN': BlockRNNModel,
        'TCN': TCNModel,
        'TiDE': TiDEModel,
        'DLinear': DLinearModel,
        'NLinear': NLinearModel,

        # Statistical
        'ARIMA': ARIMA,
        'AutoARIMA': AutoARIMA,
        'VARIMA': VARIMA,
        'ExponentialSmoothing': ExponentialSmoothing,
        'Theta': Theta,
        'FourTheta': FourTheta,
        'Prophet': Prophet,

        # ML
        'RandomForest': RandomForest,
        'LightGBM': LightGBMModel,
        'XGBoost': XGBModel,
        'CatBoost': CatBoostModel,
        'LinearRegression': LinearRegressionModel,

        # Ensemble
        'NaiveEnsemble': NaiveEnsembleModel,

        # Baselines
        'NaiveSeasonal': NaiveSeasonal,
        'NaiveDrift': NaiveDrift,
        'NaiveMean': NaiveMean,
    }

    # Modèles qui supportent PyTorch Lightning (deep learning)
    TORCH_MODELS = {
        'TFT', 'NBEATS', 'NHiTS', 'Transformer',
        'LSTM', 'GRU', 'BlockRNN', 'TCN', 'TiDE',
        'DLinear', 'NLinear'
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

        # Préparer les paramètres
        params = hyperparams.copy()

        # Gestion device pour modèles PyTorch
        if model_name in cls.TORCH_MODELS:
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # Extraire les paramètres qui ne sont pas du modèle mais de l'optimiseur
            learning_rate = params.pop('learning_rate', 1e-3)
            n_epochs = params.pop('n_epochs', 50)
            batch_size = params.pop('batch_size', 32)

            # Ajouter ces paramètres aux bons endroits
            params['n_epochs'] = n_epochs
            params['batch_size'] = batch_size

            # Learning rate va dans optimizer_kwargs
            if 'optimizer_kwargs' not in params:
                params['optimizer_kwargs'] = {}
            params['optimizer_kwargs']['lr'] = learning_rate

            # Gestion de la fonction de loss
            loss_fn_name = params.pop('loss_fn', None)
            if loss_fn_name:
                loss_fn = cls._get_loss_function(loss_fn_name, params)
                if loss_fn is not None:
                    params['loss_fn'] = loss_fn

            # Ajouter le device et autres configs PyTorch Lightning
            if 'pl_trainer_kwargs' not in params:
                params['pl_trainer_kwargs'] = {}

            params['pl_trainer_kwargs']['accelerator'] = 'gpu' if device == 'cuda' else 'cpu'

            if device == 'cuda' and torch.cuda.is_available():
                params['pl_trainer_kwargs']['devices'] = [0]  # GPU 0

            # Désactiver certains logs et la progress bar pour compatibilité Streamlit
            params['pl_trainer_kwargs']['enable_progress_bar'] = False  # Fix: Évite OSError dans Streamlit
            params['pl_trainer_kwargs']['enable_model_summary'] = False
            params['pl_trainer_kwargs']['enable_checkpointing'] = False
            
            # ⚠️ Gradient clipping pour éviter l'explosion des gradients
            # Ceci est CRITIQUE pour éviter les pertes NaN
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
                # Enable automatic time index generation if no future covariates provided
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

        Args:
            loss_name: Nom de la loss ('MAE', 'MSE', 'Huber', 'Quantile', 'RMSE')
            params: Paramètres du modèle (pour extraire des infos comme quantile)

        Returns:
            Fonction de loss PyTorch ou None pour utiliser le défaut
        """
        import torch.nn as nn
        from darts.utils.losses import MAELoss

        if loss_name == 'MAE':
            # MAE est le défaut pour la plupart des modèles Darts
            return None  # Laisser Darts utiliser le défaut

        elif loss_name == 'MSE':
            return nn.MSELoss()

        elif loss_name == 'Huber':
            # Huber loss avec delta=1.0 par défaut
            delta = params.pop('loss_delta', 1.0)
            return nn.HuberLoss(delta=delta)

        elif loss_name == 'Quantile':
            # Quantile loss nécessite darts.utils.losses
            try:
                from darts.utils.losses import QuantileLoss
                quantile = params.pop('loss_quantile', 0.5)
                return QuantileLoss(quantile=quantile)
            except ImportError:
                # Fallback si QuantileLoss n'est pas disponible
                return None

        elif loss_name == 'RMSE':
            # RMSE = sqrt(MSE), on peut utiliser MSE car l'ordre est préservé
            return nn.MSELoss()

        else:
            # Loss inconnue, utiliser le défaut
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
    """
    Détecte le device disponible.

    Returns:
        'cuda' si GPU disponible, sinon 'cpu'
    """
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def get_device_info() -> Dict[str, Any]:
    """
    Retourne des informations sur le device.

    Returns:
        Dict avec infos GPU/CPU
    """
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
