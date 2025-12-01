"""
Gestion des configurations de modèles avec YAML.

Chaque modèle sauvegardé a une config YAML qui contient :
- Les noms de colonnes réels utilisés
- La source de données ORIGINALE
- Les fichiers de splits sauvegardés (train.csv, val.csv, test.csv)
- Le preprocessing appliqué
- Les hyperparamètres
- Les métriques

Structure d'un dossier modèle :
    checkpoints/darts/{MODEL}_{STATION}/
    ├── model_config.yaml    # Configuration complète
    ├── {STATION}.pkl        # Modèle Darts
    ├── {STATION}.pkl.ckpt   # Checkpoint PyTorch Lightning
    ├── scalers.pkl          # Scalers fittés (target + covariates)
    ├── train.csv            # Données d'entraînement
    ├── val.csv              # Données de validation
    ├── test.csv             # Données de test
    └── full_data.csv        # Données complètes originales
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd


class ModelConfig:
    """Configuration complète d'un modèle entraîné."""

    def __init__(
        self,
        model_name: str,
        station: str,
        columns: Dict[str, Any],
        data_source: Dict[str, str],
        splits: Dict[str, Any],
        preprocessing: Dict[str, Any],
        hyperparams: Dict[str, Any],
        metrics: Dict[str, float],
        original_station_id: Optional[str] = None,
        creation_date: Optional[str] = None,
        use_covariates: bool = True,
        data_files: Optional[Dict[str, str]] = None
    ):
        self.model_name = model_name
        self.station = station
        self.original_station_id = original_station_id or station
        self.creation_date = creation_date or datetime.now().isoformat()

        # Configuration des colonnes (NOMS RÉELS, pas hardcodés)
        self.columns = columns  # {'date': 'date', 'target': 'niveau_nappe_ngf', 'covariates': [...]}

        # Source de données ORIGINALE (pour référence)
        self.data_source = data_source  # {'type': 'file/uploaded', 'original_file': 'nom_original.csv'}

        # Splits utilisés (tailles + dates)
        self.splits = splits  # {'train_size': 5369, 'val_size': 1151, 'test_size': 1151, 'train_end': '...', etc.}

        # Preprocessing
        self.preprocessing = preprocessing  # {'fill_method': '...', 'scaler_type': '...'}

        # Hyperparamètres et métriques
        self.hyperparams = hyperparams
        self.metrics = metrics
        self.use_covariates = use_covariates

        # Fichiers de données sauvegardés (chemins relatifs au dossier du modèle)
        self.data_files = data_files or {
            'train': 'train.csv',
            'val': 'val.csv',
            'test': 'test.csv',
            'full': 'full_data.csv'
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convertit la config en dictionnaire pour YAML."""
        return {
            'model_name': self.model_name,
            'station': self.station,
            'original_station_id': self.original_station_id,
            'creation_date': self.creation_date,
            'columns': self.columns,
            'data_source': self.data_source,
            'data_files': self.data_files,
            'splits': self.splits,
            'preprocessing': self.preprocessing,
            'hyperparams': self.hyperparams,
            'metrics': self.metrics,
            'use_covariates': self.use_covariates
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Crée une config depuis un dictionnaire."""
        return cls(
            model_name=data['model_name'],
            station=data['station'],
            columns=data['columns'],
            data_source=data['data_source'],
            splits=data['splits'],
            preprocessing=data['preprocessing'],
            hyperparams=data['hyperparams'],
            metrics=data['metrics'],
            original_station_id=data.get('original_station_id'),
            creation_date=data.get('creation_date'),
            use_covariates=data.get('use_covariates', True),
            data_files=data.get('data_files', {
                'train': 'train.csv',
                'val': 'val.csv',
                'test': 'test.csv',
                'full': 'full_data.csv'
            })
        )

    def save(self, config_path: Path):
        """Sauvegarde la config en YAML."""
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    @classmethod
    def load(cls, config_path: Path) -> 'ModelConfig':
        """Charge une config depuis YAML."""
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


def save_model_with_data(
    model,
    save_dir: Path,
    model_name: str,
    station_name: str,
    config: ModelConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_df: Optional[pd.DataFrame] = None,
    target_preprocessor=None,
    cov_preprocessor=None
) -> Path:
    """
    Sauvegarde un modèle avec sa configuration YAML et TOUS ses fichiers de données.

    Cette fonction crée une structure complète et autonome :
    - Le modèle peut être rechargé et utilisé sans dépendre d'autres fichiers externes
    - Les splits train/val/test sont sauvegardés séparément
    - Les scalers sont sauvegardés pour permettre les prédictions

    Args:
        model: Modèle Darts à sauvegarder
        save_dir: Répertoire racine des checkpoints (ex: checkpoints/darts)
        model_name: Nom du modèle (TFT, NBEATS, etc.)
        station_name: Nom de la station (nettoyé, ex: "P1")
        config: Configuration du modèle
        train_df: DataFrame du set d'entraînement
        val_df: DataFrame du set de validation
        test_df: DataFrame du set de test
        full_df: DataFrame complet (optionnel, sinon concaténation train+val+test)
        target_preprocessor: Scaler pour la target
        cov_preprocessor: Scaler pour les covariables

    Returns:
        Path du dossier du modèle
    """
    # Créer le dossier du modèle
    model_dir = save_dir / f"{model_name}_{station_name}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sauvegarder le modèle PyTorch
    model_path = model_dir / f"{station_name}.pkl"
    model.save(str(model_path))

    # 2. Sauvegarder les données de splits
    # Les DataFrames ont l'index = date
    train_path = model_dir / "train.csv"
    val_path = model_dir / "val.csv"
    test_path = model_dir / "test.csv"

    train_df.to_csv(train_path)
    val_df.to_csv(val_path)
    test_df.to_csv(test_path)

    # 3. Sauvegarder les données complètes
    if full_df is None:
        full_df = pd.concat([train_df, val_df, test_df])

    full_path = model_dir / "full_data.csv"
    full_df.to_csv(full_path)

    # 4. Mettre à jour les infos de la config
    config.data_files = {
        'train': 'train.csv',
        'val': 'val.csv',
        'test': 'test.csv',
        'full': 'full_data.csv'
    }

    # Ajouter les dates de split
    config.splits['train_start'] = str(train_df.index[0])
    config.splits['train_end'] = str(train_df.index[-1])
    config.splits['val_start'] = str(val_df.index[0])
    config.splits['val_end'] = str(val_df.index[-1])
    config.splits['test_start'] = str(test_df.index[0])
    config.splits['test_end'] = str(test_df.index[-1])

    # 5. Sauvegarder la config YAML
    config_path = model_dir / "model_config.yaml"
    config.save(config_path)

    # 6. Sauvegarder les scalers
    if target_preprocessor is not None or cov_preprocessor is not None:
        import pickle
        scalers_path = model_dir / "scalers.pkl"
        scalers_data = {
            'target_preprocessor': target_preprocessor,
            'cov_preprocessor': cov_preprocessor
        }
        with open(scalers_path, 'wb') as f:
            pickle.dump(scalers_data, f)

    return model_dir


def load_model_with_config(model_dir: Path):
    """
    Charge un modèle avec sa configuration complète et ses données.

    Args:
        model_dir: Dossier du modèle

    Returns:
        tuple: (model, config, data_dict)
        où data_dict contient: {'train': df, 'val': df, 'test': df, 'full': df}
    """
    from darts.models import (
        TFTModel, NBEATSModel, NHiTSModel, TransformerModel,
        RNNModel, BlockRNNModel, TCNModel, TiDEModel,
        DLinearModel, NLinearModel
    )

    # Mapping des types de modèles
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
    }

    model_dir = Path(model_dir)

    # 1. Charger la config YAML
    config_path = model_dir / "model_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config YAML not found: {config_path}")

    config = ModelConfig.load(config_path)

    # 2. Charger le modèle
    model_path = model_dir / f"{config.station}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_class = MODEL_CLASSES.get(config.model_name.upper())
    if not model_class:
        raise ValueError(f"Unknown model type: {config.model_name}")

    # Chargement avec le loader externe pour éviter les conflits Streamlit/pickle
    try:
        # Essayer d'abord le loader simple et externe
        from dashboard.utils.simple_loader import load_model_external
        model = load_model_external(model_path, config.model_name.upper())
    except Exception as e:
        # Si ça échoue, essayer le loader robuste
        try:
            from dashboard.utils.robust_loader import load_model_safe
            model = load_model_safe(model_path, config.model_name.upper())
        except Exception as e2:
            raise RuntimeError(f"Failed to load model: {e} (fallback also failed: {e2})")

    # 3. Charger les données
    date_col = config.columns.get('date', 'date')
    data_dict = {}

    # Nouveau système: fichiers séparés train/val/test
    if hasattr(config, 'data_files') and config.data_files:
        for split_name, file_name in config.data_files.items():
            file_path = model_dir / file_name
            if file_path.exists():
                try:
                    # Essayer d'abord avec la colonne date comme index
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    data_dict[split_name] = df
                except Exception:
                    # Fallback: charger sans index
                    df = pd.read_csv(file_path, parse_dates=[date_col])
                    df = df.set_index(date_col)
                    data_dict[split_name] = df

    # Fallback: ancien système avec station_data.csv
    if not data_dict:
        old_data_path = model_dir / "station_data.csv"
        if old_data_path.exists():
            try:
                df = pd.read_csv(old_data_path, index_col=0, parse_dates=True)
                data_dict['full'] = df

                # Recréer les splits à partir des tailles
                if 'train_size' in config.splits:
                    train_size = config.splits['train_size']
                    val_size = config.splits['val_size']

                    data_dict['train'] = df.iloc[:train_size]
                    data_dict['val'] = df.iloc[train_size:train_size + val_size]
                    data_dict['test'] = df.iloc[train_size + val_size:]
            except Exception as e:
                raise RuntimeError(f"Failed to load data from {old_data_path}: {e}")

    if not data_dict:
        raise FileNotFoundError(f"No data files found in {model_dir}")

    return model, config, data_dict


def load_scalers(model_dir: Path) -> dict:
    """
    Charge les scalers sauvegardés.

    Args:
        model_dir: Dossier du modèle

    Returns:
        dict avec 'target_preprocessor' et 'cov_preprocessor'
    """
    import pickle

    scalers_path = Path(model_dir) / "scalers.pkl"

    if not scalers_path.exists():
        return {'target_preprocessor': None, 'cov_preprocessor': None}

    with open(scalers_path, 'rb') as f:
        return pickle.load(f)


# Fonction de compatibilité avec l'ancien système
def save_model_with_config(
    model,
    save_dir: Path,
    model_name: str,
    station_name: str,
    config: ModelConfig,
    station_data_df: Optional[pd.DataFrame] = None
) -> Path:
    """
    DEPRECATED: Utilisez save_model_with_data() à la place.

    Cette fonction est conservée pour compatibilité avec l'ancien code.
    Elle sauvegarde le modèle avec un seul fichier de données.
    """
    # Créer le dossier du modèle
    model_dir = save_dir / f"{model_name}_{station_name}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sauvegarder le modèle PyTorch
    model_path = model_dir / f"{station_name}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))

    # 2. Sauvegarder les données de la station si fournies
    if station_data_df is not None:
        data_path = model_dir / "station_data.csv"
        station_data_df.to_csv(data_path)

        # Mettre à jour la config avec la source embedded
        config.data_source = {
            'type': 'embedded',
            'file_path': 'station_data.csv'
        }

    # 3. Sauvegarder la config YAML
    config_path = model_dir / "model_config.yaml"
    config.save(config_path)

    return model_dir


def migrate_old_model_to_yaml(model_dir: Path, target_var: str, covariate_vars: List[str]):
    """
    Migre un ancien modèle (JSON) vers le nouveau format (YAML).

    Args:
        model_dir: Dossier du modèle
        target_var: Nom de la colonne target
        covariate_vars: Liste des colonnes covariates
    """
    import json

    # Charger l'ancien metadata.json
    old_metadata_files = list(model_dir.glob("*_metadata.json"))
    if not old_metadata_files:
        raise FileNotFoundError(f"No metadata.json found in {model_dir}")

    metadata_path = old_metadata_files[0]
    with open(metadata_path, 'r') as f:
        old_metadata = json.load(f)

    # Créer la nouvelle config
    config = ModelConfig(
        model_name=old_metadata['model_name'],
        station=old_metadata['station'],
        original_station_id=old_metadata.get('original_station_id', old_metadata['station']),
        creation_date=old_metadata.get('creation_date', datetime.now().isoformat()),
        columns={
            'date': 'date',
            'target': target_var,
            'covariates': covariate_vars
        },
        data_source={
            'type': 'embedded',
            'file_path': 'station_data.csv'
        },
        splits={
            'train_size': old_metadata.get('train_size', 0),
            'val_size': old_metadata.get('val_size', 0),
            'test_size': old_metadata.get('test_size', 0)
        },
        preprocessing={
            'fill_method': 'Unknown',
            'scaler_type': 'StandardScaler'
        },
        hyperparams=old_metadata.get('hyperparams', {}),
        metrics=old_metadata.get('metrics', {}),
        use_covariates=old_metadata.get('use_covariates', True)
    )

    # Sauvegarder la nouvelle config
    config_path = model_dir / "model_config.yaml"
    config.save(config_path)

    print(f"✅ Migrated {model_dir.name} to YAML format")

    return config
