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
            except:
                stations = [tags.get("station", "unknown")]

        # Preprocessing config reconstruction (partial)
        preproc = {}
        for k, v in params.items():
            if k.startswith("preproc_"):
                 preproc[k.replace("preproc_", "")] = v

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
            data_source=params.get("original_filename"),
            preprocessing_config=preproc
        )

    def list_all_models(
        self,
        model_type: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> List[ModelEntry]:
        """List models from MLflow."""
        runs = self.mlflow_manager.search_runs()
        entries = []
        for index, run in runs.iterrows():
             # Convert pandas row to object-like for compatibility or fetch standard object
             # mlflow.search_runs returns pandas DataFrame by default
             # Better to use search_runs returning objects?
             # manager.search_runs returns simple list if we implemented it that way?
             # My manager wrapper calls standard mlflow.search_runs which returns DataFrame.
             # I should probably change wrapper or handle DF here.
             pass
        
        # Let's use native client for objects
        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(self.mlflow_manager.experiment_name)
        if not experiment:
             return []
             
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        
        for run in runs:
            if run.info.status != "FINISHED":
                continue
                
            entry = self._run_to_entry(run)
            
            if model_type and entry.model_type != model_type:
                continue
            if model_name and entry.model_name != model_name:
                continue
                
            entries.append(entry)
            
        return entries

    def get_model(self, model_id: str) -> Optional[ModelEntry]:
        """Get model by Run ID."""
        try:
            run = mlflow.get_run(model_id)
            return self._run_to_entry(run)
        except:
            return None

    def load_model(self, model_entry: ModelEntry) -> Any:
        """Load Darts model from artifacts."""
        local_path = mlflow.artifacts.download_artifacts(
            run_id=model_entry.run_id, 
            artifact_path="model/model.pkl"
        )
        from darts.models.forecasting.forecasting_model import ForecastingModel
        return ForecastingModel.load(local_path)

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
        except:
            return {}

    def load_data(self, model_entry: ModelEntry, split: str = "train") -> Optional[Any]:
        """
        Load dataset split (train/val/test) from artifacts.
        Returns pandas DataFrame.
        """
        valid_splits = ["train", "val", "test"]
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
