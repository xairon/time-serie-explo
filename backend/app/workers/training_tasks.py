"""
Celery tasks for model training.

These tasks run in the background and broadcast progress via Redis pub/sub.
"""

import logging
from typing import Dict, Any
from celery import Task
from datetime import datetime
import json

from app.workers.celery_app import celery_app
from app.config import get_settings

logger = logging.getLogger(__name__)


class TrainingTask(Task):
    """Base class for training tasks with progress tracking."""
    
    abstract = True
    
    def on_success(self, retval, task_id, args, kwargs):
        """Called when task succeeds."""
        logger.info(f"Training task {task_id} completed successfully")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails."""
        logger.error(f"Training task {task_id} failed: {exc}")


@celery_app.task(bind=True, base=TrainingTask, name="train_model")
def train_model_task(self, job_id: str, training_config: Dict[str, Any]):
    """
    Background task for training a model.
    
    Args:
        job_id: Unique job identifier for progress tracking
        training_config: Training configuration dict from TrainingRequest
    """
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parents[3]
    sys.path.insert(0, str(project_root))
    
    settings = get_settings()
    
    try:
        # Import training utilities
        from dashboard.utils.training import run_training_pipeline
        from dashboard.utils.dataset_registry import get_dataset_registry
        from dashboard.utils.model_factory import ModelFactory
        from dashboard.utils.preprocessing import prepare_dataframe_for_darts
        
        # Publish start event
        self.update_state(state="STARTED", meta={
            "job_id": job_id,
            "status": "running",
            "progress": 0.0,
            "message": "Loading dataset...",
        })
        
        # Load dataset
        dataset_id = training_config.get("dataset_id")
        registry = get_dataset_registry()
        datasets = registry.scan_datasets()
        
        dataset = None
        for ds in datasets:
            if ds.name == dataset_id:
                dataset = ds
                break
        
        if not dataset:
            raise ValueError(f"Dataset '{dataset_id}' not found")
        
        df, config = registry.load_dataset(dataset)
        
        # Update progress
        self.update_state(state="PROGRESS", meta={
            "job_id": job_id,
            "status": "running",
            "progress": 0.1,
            "message": "Preprocessing data...",
        })
        
        # Extract training config
        model_name = training_config["config"]["model_name"]
        stations = training_config["stations"]
        train_ratio = training_config["config"]["train_ratio"]
        val_ratio = training_config["config"]["val_ratio"]
        epochs = training_config["config"]["epochs"]
        
        # Prepare data for training
        target_col = dataset.target_column
        covariate_cols = dataset.covariate_columns or []
        
        # Create model with hyperparameters
        hyperparams = {
            "input_chunk_length": training_config["config"]["input_chunk_length"],
            "output_chunk_length": training_config["config"]["output_chunk_length"],
            "n_epochs": epochs,
            "batch_size": training_config["config"]["batch_size"],
            "learning_rate": training_config["config"]["learning_rate"],
            **(training_config["config"].get("model_hyperparams") or {}),
        }
        
        # Run training pipeline
        # This is a simplified version - full integration would need callbacks
        # for progress updates
        
        from dashboard.utils.preprocessing import (
            split_by_ratio,
            scale_splits,
        )
        from darts import TimeSeries
        
        # Filter to first station for now
        station = stations[0]
        
        if dataset.station_column:
            df_station = df[df[dataset.station_column] == station].copy()
        else:
            df_station = df.copy()
        
        # Create TimeSeries
        target_series = TimeSeries.from_dataframe(
            df_station,
            time_col=dataset.date_column,
            value_cols=[target_col],
            freq="D",
        )
        
        covariates_series = None
        if covariate_cols:
            covariates_series = TimeSeries.from_dataframe(
                df_station,
                time_col=dataset.date_column,
                value_cols=covariate_cols,
                freq="D",
            )
        
        # Split data
        train_size = int(len(target_series) * train_ratio)
        val_size = int(len(target_series) * val_ratio)
        
        train = target_series[:train_size]
        val = target_series[train_size:train_size + val_size]
        test = target_series[train_size + val_size:]
        
        # Create and train model
        from darts.models import TFTModel, NHiTSModel, NBEATSModel
        
        model_class_map = {
            "TFT": TFTModel,
            "NHiTS": NHiTSModel,
            "NBEATS": NBEATSModel,
        }
        
        ModelClass = model_class_map.get(model_name)
        if not ModelClass:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Update progress
        self.update_state(state="PROGRESS", meta={
            "job_id": job_id,
            "status": "running",
            "progress": 0.2,
            "current_epoch": 0,
            "total_epochs": epochs,
            "message": f"Training {model_name}...",
        })
        
        model = ModelClass(
            input_chunk_length=hyperparams["input_chunk_length"],
            output_chunk_length=hyperparams["output_chunk_length"],
            n_epochs=epochs,
            batch_size=hyperparams["batch_size"],
            random_state=42,
        )
        
        # Train model
        model.fit(
            series=train,
            val_series=val,
            verbose=True,
        )
        
        # Evaluate
        from darts.metrics import mae, rmse
        
        predictions = model.predict(n=len(test))
        
        metrics = {
            "mae": float(mae(test, predictions)),
            "rmse": float(rmse(test, predictions)),
        }
        
        # Save model
        from dashboard.utils.model_registry import get_registry
        
        model_registry = get_registry(settings.checkpoints_dir.parent)
        
        model_id = model_registry.generate_model_id(
            model_name=model_name,
            model_type="single",
            stations=[station],
        )
        
        # Save to checkpoints
        save_path = settings.checkpoints_dir / model_id
        save_path.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path / "model.pt"))
        
        # Register
        model_registry.register_model(
            model_id=model_id,
            model_name=model_name,
            model_type="single",
            stations=[station],
            metrics=metrics,
            hyperparams=hyperparams,
            data_source=dataset.source_file,
        )
        
        # Final update
        return {
            "job_id": job_id,
            "status": "completed",
            "progress": 1.0,
            "model_id": model_id,
            "metrics": metrics,
            "message": "Training completed successfully",
        }
        
    except Exception as e:
        logger.exception(f"Training task {job_id} failed")
        
        return {
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
            "message": f"Training failed: {str(e)}",
        }
