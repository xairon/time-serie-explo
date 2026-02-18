"""
Model Registry module backed by MLflow.

This module replaces the file-based registry with MLflow queries,
providing a unified view of trained models.
"""

import json
import mlflow
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, field
import logging
import tempfile
import shutil

from dashboard.utils.mlflow_client import get_mlflow_manager

logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    """Represents a registered model entry (mapped from MLflow Run)."""
    model_id: str
    model_name: str
    model_type: Literal["single", "global"]
    stations: List[str]
    primary_station: Optional[str]
    created_at: str
    run_id: str
    metrics: Dict[str, float] = field(default_factory=dict)
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    data_source: Optional[str] = None
    preprocessing_config: Dict[str, Any] = field(default_factory=dict)
    
    @property 
    def display_name(self) -> str:
        if self.model_type == "global":
            return f"[Global] {self.model_name} ({len(self.stations)} stations)"
        else:
            return f"[Solo] {self.model_name} ({self.primary_station or 'Unknown'})"
            
    @property
    def dataset_id(self) -> str:
        if self.data_source:
             return self.data_source.replace('.csv', '').replace('.parquet', '')
        return "default"
        
    @property
    def dataset_display_name(self) -> str:
        return self.dataset_id

    @property
    def path(self) -> str:
        """Compatibility with old registry: return run_id as path identifier."""
        return self.run_id


class ModelRegistry:
    """
    Registry implementation using MLflow as backend.
    """
    
    def __init__(self, checkpoints_dir: Optional[Path] = None):
        self.mlflow_manager = get_mlflow_manager()
        # checkpoints_dir is kept for API compatibility but mostly unused for reading
        self.checkpoints_dir = Path(checkpoints_dir) if checkpoints_dir else Path("checkpoints")
    
    def _run_to_entry(self, run: mlflow.entities.Run) -> ModelEntry:
        """Convert MLflow Run to ModelEntry."""
        data = run.data
        tags = data.tags
        params = data.params
        metrics = data.metrics
        
        # Parse stations
        stations_str = tags.get("stations", "")
        if not stations_str:
            # Fallback to station tag if single
            if tags.get("station"):
                stations = [tags["station"]]
            else:
                stations = []
        else:
            try:
                stations = json.loads(stations_str)
                if not isinstance(stations, list):
                    stations = [str(stations)]
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse stations JSON '{stations_str}': {e}")
                stations = [tags.get("station", "unknown")]

        # Preprocessing config reconstruction (partial)
        preproc = {}
        for k, v in params.items():
            if k.startswith("preproc_"):
                 preproc[k.replace("preproc_", "")] = v

        # Add columns info to preprocessing config
        columns_target = params.get("columns_target")
        columns_covariates_str = params.get("columns_covariates")
        if columns_target or columns_covariates_str:
            try:
                covariates = json.loads(columns_covariates_str) if columns_covariates_str else []
            except (json.JSONDecodeError, TypeError):
                covariates = []
            preproc['columns'] = {
                'target': columns_target,
                'covariates': covariates
            }

        # Priorité : tag override > param dataset_name > param original_filename
        data_source = (
            tags.get("dataset_name_override")
            or params.get("dataset_name") 
            or params.get("original_filename") 
            or "unknown"
        )

        return ModelEntry(
            model_id=run.info.run_id,
            run_id=run.info.run_id,
            model_name=tags.get("model", "Unknown"),
            model_type=tags.get("model_type", "single"),
            stations=stations,
            primary_station=tags.get("station"),
            created_at=datetime.fromtimestamp(run.info.start_time/1000).isoformat(),
            metrics=metrics,
            hyperparams={k.replace("hp_", ""): v for k,v in params.items() if k.startswith("hp_")},
            data_source=data_source,
            preprocessing_config=preproc
        )

    def list_all_models(
        self,
        model_type: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> List[ModelEntry]:
        """List models from MLflow."""
        try:
            # Use native MLflow client for proper Run objects
            client = mlflow.MlflowClient()
            experiment = client.get_experiment_by_name(self.mlflow_manager.experiment_name)
            if not experiment:
                logger.warning(f"Experiment '{self.mlflow_manager.experiment_name}' not found")
                return []

            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"]
            )

            entries = []
            for run in runs:
                if run.info.status != "FINISHED":
                    continue

                try:
                    entry = self._run_to_entry(run)

                    if model_type and entry.model_type != model_type:
                        continue
                    if model_name and entry.model_name != model_name:
                        continue

                    entries.append(entry)
                except Exception as e:
                    logger.warning(f"Failed to convert run {run.info.run_id} to ModelEntry: {e}")
                    continue

            return entries
        except Exception as e:
            logger.error(f"Failed to list models from MLflow: {e}")
            return []

    def get_model(self, model_id: str) -> Optional[ModelEntry]:
        """Get model by Run ID."""
        try:
            run = mlflow.get_run(model_id)
            return self._run_to_entry(run)
        except mlflow.exceptions.MlflowException as e:
            logger.warning(f"MLflow run not found: {model_id} - {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get model {model_id}: {e}")
            return None

    def load_model(self, model_entry: ModelEntry) -> Any:
        """Load Darts model from MLflow artifacts (robust loader for PyTorch pickle)."""
        # Ensure tracking URI is set before downloading artifacts
        tracking_uri = mlflow.get_tracking_uri()
        if not tracking_uri or tracking_uri == "":
            from dashboard.config import MLFLOW_TRACKING_URI
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        local_dir = mlflow.artifacts.download_artifacts(
            run_id=model_entry.run_id,
            artifact_path="model",
        )
        from pathlib import Path
        p = Path(local_dir)
        ckpt_path = p / "model.pkl.ckpt"
        if p.is_dir():
            p = p / "model.pkl"
        if not p.exists():
            raise FileNotFoundError(f"Model artifact not found: {p}")
        try:
            from dashboard.utils.robust_loader import load_model_safe
            model = load_model_safe(p, model_entry.model_name)
            if ckpt_path.exists() and getattr(model, "_fit_called", False) is False:
                model._fit_called = True
            return model
        except Exception as e:
            logger.warning(f"Robust load failed ({e}), trying torch.load fallback")
            # Fallback: torch.load with our safe unpickler to handle NumPy
            # BitGenerator version mismatches. We still use torch.load (not
            # pickle.load) because Darts files contain persistent_id entries.
            import torch
            from dashboard.utils.robust_loader import StreamlitSafeUnpickler
            import pickle as _pickle

            class _FallbackPickle:
                def __init__(self):
                    for k, v in _pickle.__dict__.items():
                        if not k.startswith("__"):
                            setattr(self, k, v)
                    self.Unpickler = StreamlitSafeUnpickler
                    self.__name__ = "pickle"
                def load(self, file, **kwargs):
                    return self.Unpickler(file, **kwargs).load()

            with open(str(p), "rb") as fh:
                model = torch.load(
                    fh, map_location="cpu", weights_only=False,
                    pickle_module=_FallbackPickle(),
                )
            if ckpt_path.exists() and getattr(model, "_fit_called", False) is False:
                model._fit_called = True
            return model

    def load_scalers(self, model_entry: ModelEntry) -> Dict[str, Any]:
        """Load scalers from artifacts."""
        try:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=model_entry.run_id,
                artifact_path="model/scalers.pkl"
            )
            import pickle
            with open(local_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            logger.info(f"No scalers found for model {model_entry.model_id}")
            return {}
        except Exception as e:
            logger.warning(f"Failed to load scalers for model {model_entry.model_id}: {e}")
            return {}

    def load_model_config(self, model_entry: ModelEntry) -> Dict[str, Any]:
        """Load model_config.json from artifacts (supports JSON and pickle formats)."""
        try:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=model_entry.run_id,
                artifact_path="model/model_config.json"
            )
            # Try JSON first
            try:
                with open(local_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
            # Fallback to pickle (some configs are pickled despite .json extension)
            import pickle
            with open(local_path, 'rb') as f:
                data = pickle.load(f)
                return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            logger.info(f"No model_config.json found for model {model_entry.model_id}")
            return {}
        except Exception as e:
            logger.warning(f"Failed to load model_config.json for model {model_entry.model_id}: {e}")
            return {}

    def load_data(self, model_entry: ModelEntry, split: str = "train") -> Optional[Any]:
        """
        Load dataset split (train/val/test) from artifacts.
        Returns pandas DataFrame.
        """
        valid_splits = ["train", "val", "test", "train_cov", "val_cov", "test_cov"]
        if split not in valid_splits:
            raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}")
            
        filename = f"{split}.csv"
        try:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=model_entry.run_id,
                artifact_path=f"model/{filename}"
            )
            import pandas as pd
            # Try loading with date index
            try:
                return pd.read_csv(local_path, index_col=0, parse_dates=True)
            except:
                return pd.read_csv(local_path)
        except Exception as e:
            logger.warning(f"Failed to load data '{filename}' for model {model_entry.model_name}: {e}")
            return None
            
    # Compatibility methods
    def get_all_stations(self) -> List[str]:
        models = self.list_all_models()
        stations = set()
        for m in models:
            stations.update(m.stations)
        return sorted(list(stations))
        
    def get_models_for_station(self, station_id: str) -> List[ModelEntry]:
        all_models = self.list_all_models()
        return [m for m in all_models if station_id in m.stations]

    def get_datasets_for_station(self, station_id: str) -> Dict[str, str]:
        """Get unique datasets for a station."""
        datasets = {}
        # Get all models for this station
        models = self.get_models_for_station(station_id)
        for m in models:
            datasets[m.dataset_id] = m.dataset_display_name
        return datasets

    def get_models_for_station_dataset(
        self,
        station_id: str,
        dataset_id: str,
        model_type: Optional[str] = None
    ) -> List[ModelEntry]:
        """Get models for a station and dataset."""
        models = self.get_models_for_station(station_id)
        filtered = []
        for m in models:
            if m.dataset_id == dataset_id:
                if model_type and m.model_type != model_type:
                    continue
                filtered.append(m)
        return filtered

    def delete_model(self, model_id: str) -> bool:
         mlflow.delete_run(model_id)
         return True


# Singleton
_registry_instance = None

def get_registry(checkpoints_dir: Optional[Path] = None) -> ModelRegistry:
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry(checkpoints_dir)
    return _registry_instance
