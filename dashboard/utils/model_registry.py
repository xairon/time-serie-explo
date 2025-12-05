"""
Système de registre et de gestion des modèles entraînés.

Permet de lister, charger et gérer les modèles sauvegardés
de manière uniforme, indépendamment du format (.pkl, .pth.tar, etc.)
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import ForecastingModel


class ModelInfo:
    """Informations sur un modèle sauvegardé."""

    def __init__(
        self,
        model_path: Path,
        model_name: str,
        model_type: str,
        station: str,
        creation_date: Optional[str] = None,
        metrics: Optional[Dict] = None,
        hyperparams: Optional[Dict] = None
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.model_type = model_type
        self.station = station
        self.creation_date = creation_date or datetime.now().isoformat()
        self.metrics = metrics or {}
        self.hyperparams = hyperparams or {}

    def to_dict(self) -> Dict:
        """Convertit en dictionnaire."""
        return {
            'model_path': str(self.model_path),
            'model_name': self.model_name,
            'model_type': self.model_type,
            'station': self.station,
            'creation_date': self.creation_date,
            'metrics': self.metrics,
            'hyperparams': self.hyperparams
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelInfo':
        """Crée depuis un dictionnaire."""
        return cls(
            model_path=Path(data['model_path']),
            model_name=data['model_name'],
            model_type=data['model_type'],
            station=data['station'],
            creation_date=data.get('creation_date'),
            metrics=data.get('metrics', {}),
            hyperparams=data.get('hyperparams', {})
        )

    def __repr__(self) -> str:
        return f"ModelInfo({self.model_name}, {self.model_type}, {self.station})"


class ModelRegistry:
    """
    Registre centralisé des modèles entraînés.

    Scanne le répertoire des checkpoints et maintient un index
    des modèles disponibles avec leurs métadonnées.
    """

    def __init__(self, checkpoints_dir: Path):
        """
        Initialise le registre.

        Args:
            checkpoints_dir: Répertoire racine des checkpoints
        """
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def scan_models(self) -> List[ModelInfo]:
        """
        Scanne le répertoire et retourne tous les modèles trouvés.

        Returns:
            Liste de ModelInfo pour chaque modèle trouvé
        """
        models = []

        # Scanner récursivement tous les fichiers .pkl (nouveau système)
        for pkl_file in self.checkpoints_dir.rglob("*.pkl"):
            # Ignorer les fichiers de checkpoint PyTorch Lightning et les scalers
            if pkl_file.name.endswith('.ckpt') or pkl_file.name == 'scalers.pkl':
                continue

            model_info = self._extract_model_info(pkl_file)
            if model_info:
                models.append(model_info)

        # Scanner aussi les anciens .pth.tar pour rétrocompatibilité
        for pth_file in self.checkpoints_dir.rglob("*.pth.tar"):
            model_info = self._extract_model_info(pth_file)
            if model_info:
                models.append(model_info)

        # Trier par date de création (plus récent en premier)
        models.sort(key=lambda m: m.creation_date, reverse=True)

        return models

    def _extract_model_info(self, model_path: Path) -> Optional[ModelInfo]:
        """
        Extrait les informations d'un fichier de modèle.

        Priorité:
        1. Config YAML (nouveau système)
        2. Metadata JSON (ancien système)
        3. Nom de fichier (fallback)

        Args:
            model_path: Chemin vers le fichier .pkl ou .pth.tar

        Returns:
            ModelInfo si extraction réussie, None sinon
        """
        # 1. Chercher d'abord une config YAML (nouveau système)
        config_yaml_path = model_path.parent / "model_config.yaml"

        if config_yaml_path.exists():
            try:
                from dashboard.utils.model_config import ModelConfig

                config = ModelConfig.load(config_yaml_path)
                
                # Utiliser original_station_id pour le nom complet de la station
                station_display = config.original_station_id or config.station

                return ModelInfo(
                    model_path=model_path,
                    model_name=config.model_name,
                    model_type=config.model_name,
                    station=station_display,
                    creation_date=config.creation_date,
                    metrics=config.metrics,
                    hyperparams=config.hyperparams
                )
            except Exception as e:
                print(f"Warning: Could not parse YAML config for {model_path}: {e}")

        # 2. Fallback: chercher le fichier de métadonnées JSON (ancien système)
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"

        if not metadata_path.exists():
            # Fallback: chercher avec l'ancien format
            metadata_path = model_path.with_suffix('.json')

        # Extraire depuis les métadonnées JSON si disponibles
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                return ModelInfo(
                    model_path=model_path,
                    model_name=metadata.get('model_name', model_path.stem),
                    model_type=metadata.get('model_name', 'Unknown').split('_')[0],
                    station=metadata.get('station', 'Unknown'),
                    creation_date=metadata.get('creation_date'),
                    metrics=metadata.get('metrics', {}),
                    hyperparams=metadata.get('hyperparams', {})
                )
            except Exception as e:
                print(f"Warning: Could not parse JSON metadata for {model_path}: {e}")

        # 3. Fallback: extraire depuis le nom de fichier
        # Format attendu: {MODEL}_{STATION}.pkl ou {MODEL}_{STATION}_{TIMESTAMP}.pkl
        try:
            parts = model_path.stem.split('_')
            model_type = parts[0]
            station = parts[1] if len(parts) > 1 else 'Unknown'

            return ModelInfo(
                model_path=model_path,
                model_name=model_path.stem,
                model_type=model_type,
                station=station,
                creation_date=datetime.fromtimestamp(model_path.stat().st_mtime).isoformat()
            )
        except Exception as e:
            print(f"Warning: Could not extract info from filename {model_path}: {e}")
            return None

    def get_models_by_station(self, station: str) -> List[ModelInfo]:
        """
        Retourne tous les modèles pour une station donnée.

        Args:
            station: Nom de la station

        Returns:
            Liste de ModelInfo pour cette station
        """
        all_models = self.scan_models()
        return [m for m in all_models if m.station == station]

    def get_models_by_type(self, model_type: str) -> List[ModelInfo]:
        """
        Retourne tous les modèles d'un type donné.

        Args:
            model_type: Type de modèle (TFT, NBEATS, etc.)

        Returns:
            Liste de ModelInfo de ce type
        """
        all_models = self.scan_models()
        return [m for m in all_models if m.model_type.upper() == model_type.upper()]

    def load_model(self, model_info: ModelInfo) -> ForecastingModel:
        """
        Charge un modèle depuis le disque.

        Args:
            model_info: Informations sur le modèle à charger

        Returns:
            Modèle Darts chargé
        """
        from darts.models import (
            TFTModel, NBEATSModel, NHiTSModel, TransformerModel,
            RNNModel, BlockRNNModel, TCNModel, TiDEModel,
            DLinearModel, NLinearModel, TSMixerModel,
            GlobalNaiveAggregate, GlobalNaiveDrift, GlobalNaiveSeasonal
        )

        model_path = model_info.model_path

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Mapping des types vers les classes
        MODEL_CLASSES = {
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
            'TSMIXER': TSMixerModel,
            'GLOBALNAIVEAGGREGATE': GlobalNaiveAggregate,
            'GLOBALNAIVEDRIFT': GlobalNaiveDrift,
            'GLOBALNAIVESEASONAL': GlobalNaiveSeasonal,
        }

        model_type = model_info.model_type.upper()
        model_class = MODEL_CLASSES.get(model_type)

        if not model_class:
            raise ValueError(f"Unknown model type: {model_type}")

        try:
            # Charger le modèle
            model = model_class.load(str(model_path))
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_info.model_name}: {e}")

    def delete_model(self, model_info: ModelInfo):
        """
        Supprime un modèle et ses fichiers associés.

        Args:
            model_info: Modèle à supprimer
        """
        # Supprimer le fichier principal
        if model_info.model_path.exists():
            model_info.model_path.unlink()

        # Supprimer le checkpoint PyTorch Lightning si existe
        ckpt_path = model_info.model_path.with_suffix('.pkl.ckpt')
        if ckpt_path.exists():
            ckpt_path.unlink()

        # Supprimer les métadonnées
        metadata_path = model_info.model_path.parent / f"{model_info.model_path.stem}_metadata.json"
        if metadata_path.exists():
            metadata_path.unlink()

        # Supprimer le répertoire si vide
        try:
            if model_info.model_path.parent != self.checkpoints_dir:
                model_info.model_path.parent.rmdir()
        except OSError:
            pass  # Répertoire non vide

    def get_best_model(self, station: str, metric: str = 'MAE') -> Optional[ModelInfo]:
        """
        Retourne le meilleur modèle pour une station selon une métrique.

        Args:
            station: Nom de la station
            metric: Métrique à minimiser (MAE, RMSE, etc.)

        Returns:
            ModelInfo du meilleur modèle, None si aucun
        """
        models = self.get_models_by_station(station)

        # Filtrer les modèles avec cette métrique
        models_with_metric = [m for m in models if metric in m.metrics]

        if not models_with_metric:
            return None

        # Trouver le meilleur (valeur minimale pour MAE, RMSE, etc.)
        lower_is_better = metric in ['MAE', 'RMSE', 'MAPE', 'sMAPE', 'NRMSE']

        if lower_is_better:
            best_model = min(models_with_metric, key=lambda m: m.metrics[metric])
        else:
            best_model = max(models_with_metric, key=lambda m: m.metrics[metric])

        return best_model


def get_registry() -> ModelRegistry:
    """
    Retourne l'instance du registre de modèles.

    Returns:
        ModelRegistry configuré avec le répertoire par défaut
    """
    from dashboard.config import CHECKPOINTS_DIR
    return ModelRegistry(CHECKPOINTS_DIR)
