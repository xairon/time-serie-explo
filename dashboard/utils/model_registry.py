"""
Model Registry module for centralized model management.

This module provides a registry system for tracking trained models,
supporting both single-station and global (multi-station) models.
Enables efficient querying of models by station.
"""

import json
import shutil
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal, Union
from dataclasses import dataclass, asdict, field
import logging

from darts.models.forecasting.forecasting_model import ForecastingModel

logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    """Represents a registered model entry."""
    model_id: str
    model_name: str  # e.g., "TFT", "NHiTS"
    model_type: Literal["single", "global"]
    stations: List[str]  # List of station IDs this model applies to
    primary_station: Optional[str]  # For single models, the main station
    created_at: str  # ISO format datetime
    path: str  # Relative path from checkpoints dir
    metrics: Dict[str, float] = field(default_factory=dict)
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    data_source: Optional[str] = None
    preprocessing_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelEntry":
        # Handle missing optional fields for backward compatibility
        return cls(
            model_id=data.get("model_id", "unknown"),
            model_name=data.get("model_name", "Unknown"),
            model_type=data.get("model_type", "single"),
            stations=data.get("stations", []),
            primary_station=data.get("primary_station"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            path=data.get("path", ""),
            metrics=data.get("metrics", {}),
            hyperparams=data.get("hyperparams", {}),
            data_source=data.get("data_source"),
            preprocessing_config=data.get("preprocessing_config", {})
        )
    
    @property 
    def display_name(self) -> str:
        """Human-readable name for UI display."""
        if self.model_type == "global":
            return f"[Global] {self.model_name} ({len(self.stations)} stations)"
        else:
            return f"[Solo] {self.model_name} ({self.primary_station or 'Unknown'})"
    
    @property
    def dataset_id(self) -> str:
        """Generate dataset ID - use data source filename as primary identifier."""
        if self.data_source:
            # Use the actual data source filename (cleaned)
            source = str(self.data_source)
            # Remove extension
            for ext in ['.csv', '.parquet', '.xlsx']:
                source = source.replace(ext, '')
            # Take just filename if it's a path
            if '/' in source or '\\' in source:
                source = source.split('/')[-1].split('\\')[-1]
            return source
        
        # Fallback: use scaler type if no data_source
        if self.preprocessing_config:
            scaler = self.preprocessing_config.get("scaler_type") or self.preprocessing_config.get("normalization")
            if scaler and scaler not in ("None", "none"):
                return scaler
        
        return "default"
    
    @property
    def dataset_display_name(self) -> str:
        """Human-readable dataset name for UI."""
        parts = []
        
        # Target variable from columns, or fallback to data_source filename
        target = None
        if self.preprocessing_config and 'columns' in self.preprocessing_config:
            target = self.preprocessing_config['columns'].get('target')
        
        if target:
            parts.append(target)
        elif self.data_source:
            # Use data source filename (without .csv) as fallback
            source_name = str(self.data_source).replace('.csv', '').replace('.parquet', '')
            # Take last part if it's a path
            if '/' in source_name or '\\' in source_name:
                source_name = source_name.split('/')[-1].split('\\')[-1]
            parts.append(source_name)
        
        # Covariates
        if self.preprocessing_config and 'columns' in self.preprocessing_config:
            covariates = self.preprocessing_config['columns'].get('covariates', [])
            if covariates:
                parts.append(f"+ {len(covariates)} cov")
        
        # Scaler - always show this
        scaler = None
        if self.preprocessing_config:
            scaler = self.preprocessing_config.get("scaler_type") or self.preprocessing_config.get("normalization")
        if scaler and scaler not in ("None", "none"):
            parts.append(f"({scaler})")
        
        if parts:
            return " ".join(parts)
        return "Default dataset"
    
    def __repr__(self) -> str:
        return f"ModelEntry({self.model_id}, type={self.model_type}, stations={self.stations})"


class ModelRegistry:
    """
    Central registry for managing trained models.
    
    Provides:
    - Model registration with rich metadata
    - Query models by station (finds both single and global models)
    - Query models by type (single/global)
    - Auto-scan existing checkpoints for migration
    - Persistent registry.json storage
    """
    
    REGISTRY_VERSION = "1.0"
    REGISTRY_FILENAME = "registry.json"
    
    def __init__(self, checkpoints_dir: Path):
        """
        Initialize the registry.
        
        Args:
            checkpoints_dir: Root directory for checkpoints (e.g., checkpoints/)
        """
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.checkpoints_dir / self.REGISTRY_FILENAME
        self._registry: Dict[str, Any] = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from disk or create empty one."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Validate version
                    if data.get('version') != self.REGISTRY_VERSION:
                        logger.warning(f"Registry version mismatch. Expected {self.REGISTRY_VERSION}, got {data.get('version')}")
                    return data
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                return self._create_empty_registry()
        return self._create_empty_registry()
    
    def _create_empty_registry(self) -> Dict[str, Any]:
        """Create a new empty registry."""
        return {
            "version": self.REGISTRY_VERSION,
            "models": {}
        }
    
    def _save_registry(self) -> None:
        """Persist registry to disk."""
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(self._registry, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def generate_model_id(
        model_name: str,
        model_type: Literal["single", "global"],
        stations: List[str],
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Generate a unique model ID.
        
        Format:
        - Single: single_TFT_00104X0054-P1_20231209_111500
        - Global: global_TFT_3stations_20231209_111500
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
        
        if model_type == "single" and len(stations) == 1:
            # Make station ID filesystem-safe (replace / with -)
            safe_station = stations[0].replace('/', '-').replace('\\', '-')
            return f"single_{model_name}_{safe_station}_{ts_str}"
        else:
            # Global model
            return f"global_{model_name}_{len(stations)}stations_{ts_str}"
    
    @staticmethod
    def station_to_path_safe(station_id: str) -> str:
        """Convert station ID to filesystem-safe format."""
        return station_id.replace('/', '-').replace('\\', '-')
    
    @staticmethod
    def path_safe_to_station(path_safe: str) -> str:
        """Convert filesystem-safe format back to station ID."""
        return path_safe.replace('-', '/')
    
    def register_model(
        self,
        model_id: str,
        model_name: str,
        model_type: Literal["single", "global"],
        stations: List[str],
        path: str,
        metrics: Optional[Dict[str, float]] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
        data_source: Optional[str] = None,
        preprocessing_config: Optional[Dict[str, Any]] = None,
        primary_station: Optional[str] = None
    ) -> ModelEntry:
        """
        Register a new model in the registry.
        
        Args:
            model_id: Unique identifier for the model
            model_name: Model architecture name (TFT, NHiTS, etc.)
            model_type: "single" or "global"
            stations: List of station IDs this model was trained on
            path: Relative path to model directory from checkpoints_dir
            metrics: Performance metrics dict
            hyperparams: Training hyperparameters
            data_source: Original data file name
            preprocessing_config: Preprocessing configuration used
            primary_station: For single models, the main station ID
            
        Returns:
            The created ModelEntry
        """
        entry = ModelEntry(
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            stations=stations,
            primary_station=primary_station or (stations[0] if model_type == "single" and stations else None),
            created_at=datetime.now().isoformat(),
            path=path,
            metrics=metrics or {},
            hyperparams=hyperparams or {},
            data_source=data_source,
            preprocessing_config=preprocessing_config or {}
        )
        
        self._registry["models"][model_id] = entry.to_dict()
        self._save_registry()
        
        logger.info(f"Registered model: {model_id} ({model_type}, stations={stations})")
        return entry
    
    def get_model(self, model_id: str) -> Optional[ModelEntry]:
        """Get a specific model by ID."""
        data = self._registry["models"].get(model_id)
        if data:
            return ModelEntry.from_dict(data)
        return None
    
    def get_models_for_station(
        self,
        station_id: str,
        model_type: Optional[Literal["single", "global"]] = None,
        model_name: Optional[str] = None
    ) -> List[ModelEntry]:
        """
        Find all models applicable to a station.
        
        Args:
            station_id: The station ID to search for (e.g., "00104X0054/P1")
            model_type: Optional filter for "single" or "global"
            model_name: Optional filter for model architecture
            
        Returns:
            List of ModelEntry objects matching the criteria
        """
        results = []
        
        for model_data in self._registry["models"].values():
            # Check if station matches
            if station_id not in model_data.get("stations", []):
                continue
            
            # Apply filters
            if model_type and model_data.get("model_type") != model_type:
                continue
            if model_name and model_data.get("model_name") != model_name:
                continue
            
            results.append(ModelEntry.from_dict(model_data))
        
        # Sort by created_at descending (newest first)
        results.sort(key=lambda x: x.created_at, reverse=True)
        return results
    
    def list_all_models(
        self,
        model_type: Optional[Literal["single", "global"]] = None,
        model_name: Optional[str] = None
    ) -> List[ModelEntry]:
        """
        List all registered models with optional filtering.
        
        Args:
            model_type: Optional filter for "single" or "global"
            model_name: Optional filter for model architecture
            
        Returns:
            List of ModelEntry objects
        """
        results = []
        
        for model_data in self._registry["models"].values():
            if model_type and model_data.get("model_type") != model_type:
                continue
            if model_name and model_data.get("model_name") != model_name:
                continue
            results.append(ModelEntry.from_dict(model_data))
        
        results.sort(key=lambda x: x.created_at, reverse=True)
        return results
    
    def get_all_stations(self) -> List[str]:
        """Get list of all unique station IDs across all models."""
        stations = set()
        for model_data in self._registry["models"].values():
            stations.update(model_data.get("stations", []))
        return sorted(list(stations))
    
    def get_datasets_for_station(self, station_id: str) -> Dict[str, str]:
        """
        Get unique datasets (preprocessing configs) available for a station.
        
        Args:
            station_id: Station ID to filter by
            
        Returns:
            Dict mapping dataset_id to dataset_display_name
        """
        datasets = {}
        for model_data in self._registry["models"].values():
            if station_id in model_data.get("stations", []):
                entry = ModelEntry.from_dict(model_data)
                if entry.dataset_id not in datasets:
                    datasets[entry.dataset_id] = entry.dataset_display_name
        return datasets
    
    def get_models_for_station_dataset(
        self,
        station_id: str,
        dataset_id: str,
        model_type: Optional[Literal["single", "global"]] = None
    ) -> List[ModelEntry]:
        """
        Get models for a specific station AND dataset combination.
        
        Args:
            station_id: Station ID to filter by
            dataset_id: Dataset ID (from preprocessing config)
            model_type: Optional filter for "single" or "global"
            
        Returns:
            List of matching ModelEntry objects, sorted by date (newest first)
        """
        results = []
        
        for model_data in self._registry["models"].values():
            # Check station
            if station_id not in model_data.get("stations", []):
                continue
            
            # Check dataset
            entry = ModelEntry.from_dict(model_data)
            if entry.dataset_id != dataset_id:
                continue
            
            # Check model type if specified
            if model_type and entry.model_type != model_type:
                continue
            
            results.append(entry)
        
        # Sort by created_at descending (newest first)
        results.sort(key=lambda x: x.created_at, reverse=True)
        return results
    
    def delete_model(self, model_id: str, delete_files: bool = True) -> bool:
        """
        Remove a model from the registry.
        
        Args:
            model_id: The model ID to delete
            delete_files: If True, also delete the model files from disk
            
        Returns:
            True if deleted, False if not found
        """
        if model_id not in self._registry["models"]:
            return False
        
        model_data = self._registry["models"][model_id]
        
        if delete_files:
            model_path = self.checkpoints_dir / model_data["path"]
            if model_path.exists():
                try:
                    shutil.rmtree(model_path)
                    logger.info(f"Deleted model files: {model_path}")
                except Exception as e:
                    logger.error(f"Failed to delete model files: {e}")
        
        del self._registry["models"][model_id]
        self._save_registry()
        
        logger.info(f"Removed model from registry: {model_id}")
        return True
    
    def load_model(self, model_entry: ModelEntry) -> ForecastingModel:
        """
        Load a model from disk.
        
        Args:
            model_entry: ModelEntry to load
            
        Returns:
            Loaded Darts ForecastingModel
        """
        model_dir = self.checkpoints_dir / model_entry.path
        
        # Find the .pkl file
        pkl_files = list(model_dir.glob("*.pkl"))
        pkl_files = [f for f in pkl_files if f.name != "scalers.pkl"]
        
        if not pkl_files:
            raise FileNotFoundError(f"No model .pkl file found in {model_dir}")
        
        model_path = pkl_files[0]
        return ForecastingModel.load(str(model_path))
    
    def load_scalers(self, model_entry: ModelEntry) -> Dict[str, Any]:
        """
        Load scalers/preprocessors for a model.
        
        Args:
            model_entry: ModelEntry to load scalers for
            
        Returns:
            Dict with 'target' and 'covariates' preprocessors
        """
        model_dir = self.checkpoints_dir / model_entry.path
        scalers_path = model_dir / "scalers.pkl"
        
        if not scalers_path.exists():
            return {}
        
        with open(scalers_path, 'rb') as f:
            return pickle.load(f)
    
    def scan_existing_checkpoints(self, darts_subdir: str = "darts") -> int:
        """
        Scan existing checkpoint folders and register unregistered models.
        
        This is useful for migrating existing checkpoints to the new registry system.
        
        Args:
            darts_subdir: Subdirectory containing darts models
            
        Returns:
            Number of newly registered models
        """
        darts_dir = self.checkpoints_dir / darts_subdir
        if not darts_dir.exists():
            return 0
        
        registered_paths = {m["path"] for m in self._registry["models"].values()}
        newly_registered = 0
        
        # Scan for model folders (look for .pkl files)
        for model_folder in darts_dir.rglob("*.pkl"):
            # Skip scalers and checkpoints
            if model_folder.name in ("scalers.pkl",) or model_folder.suffix == ".ckpt":
                continue
                
            model_dir = model_folder.parent
            rel_path = str(model_dir.relative_to(self.checkpoints_dir)).replace("\\", "/")
            
            # Skip if already registered
            if rel_path in registered_paths:
                continue
            
            # Try to extract info from folder structure and config
            try:
                entry = self._parse_legacy_checkpoint(model_dir, rel_path)
                if entry:
                    self._registry["models"][entry.model_id] = entry.to_dict()
                    newly_registered += 1
                    logger.info(f"Auto-registered legacy model: {entry.model_id}")
            except Exception as e:
                logger.warning(f"Could not parse checkpoint {model_dir}: {e}")
        
        if newly_registered > 0:
            self._save_registry()
        
        return newly_registered
    
    def _parse_legacy_checkpoint(self, model_dir: Path, rel_path: str) -> Optional[ModelEntry]:
        """
        Parse a legacy or new checkpoint folder and create a ModelEntry.
        
        Expected structures:
        - New: darts/model_id (e.g. single_TFT_station_time)
        - Old: darts/TFT_P1/timestamp/
        """
        # 1. Try to load config from model_config.yaml (New system)
        config_yaml = model_dir / "model_config.yaml"
        config = {}
        
        if config_yaml.exists():
            # Basic YAML parsing without heavy dependencies if possible, or just standard load
            try:
                from dashboard.utils.model_config import ModelConfig
                # We can't easily use ModelConfig.load because it expects full class structure.
                # Let's just do a manual load for metadata extraction
                import yaml
                with open(config_yaml, 'r') as f:
                    # ModelConfig saves as a custom object, but yaml.safe_load might load as dict if simple
                    # Or we need to be careful. ModelConfig uses dump() which might use default yaml dumper.
                    # Let's try unsafe load or just assume it's loadable.
                    # Actually ModelConfig inherits from nothing special, acts like a dataclass-ish.
                    # Let's try safe_load first.
                    try:
                        content = yaml.safe_load(f)
                        if content and isinstance(content, dict):
                            config = content
                    except:
                        pass
            except Exception as e:
                logger.warning(f"Failed to read model_config.yaml: {e}")

        # 2. Fallback to existing JSON configs
        if not config:
            config_path = model_dir / "config.json"
            metadata_path = model_dir / "metadata.json"
            
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            elif config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
        
        # Extract model name from config or folder name
        model_name = config.get("model_name", "Unknown")
        folder_name = model_dir.name
        
        if model_name == "Unknown":
            # Parsing from folder name:
            # Format: single_TFT_... or global_TFT_...
            parts = folder_name.split("_")
            if len(parts) >= 2 and parts[0] in ["single", "global"]:
                # Assume second part is model name
                model_name = parts[1]
            else:
                # Old format fallback
                parts_slash = rel_path.split("/")
                for part in parts_slash:
                    if part.upper() in ["TFT", "NHITS", "NBEATS", "TCNMODEL", "TRANSFORMER", "XGBOOST"]:
                        model_name = part.upper()
                        break
        
        # Extract station info
        stations = config.get("stations", [])
        if not stations:
            # Try original_station_id field
            orig_station = config.get("original_station_id")
            if orig_station:
                stations = [orig_station]
            else:
                # Try station field
                station = config.get("station")
                if station:
                    stations = [station]
                else:
                    # Parse from folder name
                    stations = [self._extract_station_from_path(rel_path)]
        
        # Determine model type
        model_type: Literal["single", "global"] = "global" if len(stations) > 1 or "global" in rel_path.lower() else "single"
        
        # Generate model ID
        timestamp = datetime.now()
        # Try to extract timestamp from folder name (format: YYYYMMDD_HHMMSS)
        parts = rel_path.split("/")
        for part in parts:
            if len(part) == 15 and part[8:9] == "_":
                try:
                    timestamp = datetime.strptime(part, "%Y%m%d_%H%M%S")
                    break
                except ValueError:
                    pass
        
        model_id = self.generate_model_id(model_name, model_type, stations, timestamp)
        
        return ModelEntry(
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            stations=stations,
            primary_station=stations[0] if model_type == "single" and stations else None,
            created_at=timestamp.isoformat(),
            path=rel_path,
            metrics=config.get("metrics", {}),
            hyperparams=config.get("hyperparams", {}),
            data_source=config.get("data_source", {}).get("original_file") if isinstance(config.get("data_source"), dict) else config.get("data_source"),
            preprocessing_config=config.get("preprocessing", {})
        )
    
    def _extract_station_from_path(self, path: str) -> str:
        """Extract station identifier from a path."""
        # Remove 'darts/' prefix and split
        parts = path.replace("darts/", "").split("/")
        
        # Look for station-like patterns
        for part in parts:
            # Skip known model names
            if part.upper() in ["TFT", "NHITS", "NBEATS", "TCNMODEL", "TRANSFORMER"]:
                continue
            # Skip timestamp-like patterns
            if len(part) == 15 and "_" in part:
                continue
            # This is likely the station identifier
            # Convert back from path-safe format
            return part.replace("-", "/")
        
        return "unknown"


# Singleton registry instance
_registry_instance: Optional[ModelRegistry] = None


def get_registry(checkpoints_dir: Optional[Path] = None) -> ModelRegistry:
    """
    Get a ModelRegistry instance (singleton pattern).
    
    Args:
        checkpoints_dir: Optional path to checkpoints directory.
                        Defaults to 'checkpoints' in current directory.
    
    Returns:
        ModelRegistry instance
    """
    global _registry_instance
    
    if checkpoints_dir is None:
        checkpoints_dir = Path("checkpoints")
    
    if _registry_instance is None or _registry_instance.checkpoints_dir != checkpoints_dir:
        _registry_instance = ModelRegistry(checkpoints_dir)
    
    return _registry_instance
