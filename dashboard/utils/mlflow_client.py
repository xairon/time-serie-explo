"""
MLflow Client Wrapper for centralized experiment management.
"""

import os
import mlflow
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MLflowManager:
    """
    Wrapper around MLflow to standardize experiment tracking and model logging.
    """
    
    def __init__(self, experiment_name: str = "Junon_TimeSeries", tracking_uri: Optional[str] = None):
        """
        Initialize MLflow manager.
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: Optional custom tracking URI (defaults to mlruns locall)
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        self.experiment_name = experiment_name
        self.experiment = mlflow.set_experiment(experiment_name)
        self.framework = "darts"
        
        logger.info(f"MLflow initialized: Experiment='{experiment_name}', URI='{mlflow.get_tracking_uri()}'")

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run."""
        tags = tags or {}
        tags['framework'] = self.framework
        tags['user'] = os.environ.get('USERNAME', 'unknown')
        
        return mlflow.start_run(run_name=run_name, tags=tags)

    def log_params(self, params: Dict[str, Any], prefix: str = ""):
        """Log parameters with optional prefix."""
        if not mlflow.active_run():
            logger.warning("No active run to log params")
            return
            
        flat_params = {}
        for k, v in params.items():
            key = f"{prefix}{k}" if prefix else k
            # Handle list/dict values by converting to string or flattening if needed
            if isinstance(v, (list, dict)):
                flat_params[key] = str(v)
            else:
                flat_params[key] = v
                
        mlflow.log_params(flat_params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        if not mlflow.active_run():
            return
        
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model: Any, artifact_path: str, custom_artifacts: Optional[Dict] = None):
        """
        Log a model to MLflow.
        
        Args:
            model: The model object (Darts ForecastingModel)
            artifact_path: Directory in artifact store
            custom_artifacts: Additional files to log (config, scalers)
        """
        if not mlflow.active_run():
            return

        # Darts models are not natively supported by mlflow python_function flavor easily without wrapper
        # For simplicity, we log as a generic pytorch model if possible, or just as an artifact folder (pickle).
        # Robust approach: Save locally then log artifacts.
        
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 1. Save Darts model
            model_file = temp_path / f"model.pkl"
            model.save(str(model_file))
            
            # 2. Save custom artifacts (scalers, config, datasets)
            if custom_artifacts:
                for name, content in custom_artifacts.items():
                    # Content is path, dataframe or object?
                    
                    artifact_file = temp_path / name
                    if isinstance(content, (str, Path)):
                        if Path(content).exists():
                            shutil.copy(content, artifact_file)
                    elif hasattr(content, 'to_csv'): # Pandas DataFrame
                        content.to_csv(artifact_file)
                    else:
                        # Pickle object
                        import pickle
                        with open(artifact_file, 'wb') as f:
                            pickle.dump(content, f)
            
            # 3. Log all as artifacts
            mlflow.log_artifacts(str(temp_path), artifact_path=artifact_path)
            
    def search_runs(self, filter_string: str = "", order_by: Optional[List[str]] = None) -> List[Any]:
        """Search runs in current experiment."""
        return mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            filter_string=filter_string,
            order_by=order_by or ["attributes.start_time DESC"]
        )

# Global singleton
_mlflow_manager = None

def get_mlflow_manager(tracking_uri: Optional[str] = None) -> MLflowManager:
    global _mlflow_manager
    if _mlflow_manager is None:
        _mlflow_manager = MLflowManager(tracking_uri=tracking_uri)
    return _mlflow_manager
