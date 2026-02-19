"""
MLflow Client Wrapper for centralized experiment management.

Fonctionnalités:
- Tracking des expériences et runs
- Autolog PyTorch Lightning
- System Metrics (CPU, GPU, mémoire)
- Tracing des fonctions (@mlflow.trace)
- Model Signatures
- Dataset tracking
"""

import os
import functools
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, TypeVar
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Type variable for decorator
F = TypeVar('F', bound=Callable[..., Any])


def _default_tracking_uri() -> str:
    """URI par défaut : sqlite dans le répertoire du projet (aligné avec run_app et mlflow ui)."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    try:
        from dashboard.config import MLFLOW_TRACKING_URI
        return MLFLOW_TRACKING_URI
    except Exception:
        return "sqlite:///mlflow.db"


class MLflowManager:
    """
    Wrapper around MLflow to standardize experiment tracking and model logging.
    
    Fonctionnalités avancées:
    - Autolog PyTorch Lightning (métriques automatiques)
    - System Metrics (CPU, GPU, mémoire)
    - Tracing des fonctions
    - Model Signatures (schémas input/output)
    - Dataset tracking
    """
    
    def __init__(
        self, 
        experiment_name: str = "Junon_TimeSeries", 
        tracking_uri: Optional[str] = None,
        enable_system_metrics: bool = True,
        enable_autolog: bool = False  # Disabled by default - conflicts with Darts internal PyTorch Lightning
    ):
        """
        Initialize MLflow manager.
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: Optional custom tracking URI (defaults to project mlflow.db)
            enable_system_metrics: Log CPU/GPU/memory during training
            enable_autolog: Enable PyTorch Lightning autolog (disabled by default, 
                           can conflict with Darts which uses PL internally)
        """
        uri = tracking_uri or _default_tracking_uri()
        mlflow.set_tracking_uri(uri)
        
        self.experiment_name = experiment_name
        self.experiment = mlflow.set_experiment(experiment_name)
        self.framework = "darts"
        self._autolog_enabled = False
        self._system_metrics_enabled = enable_system_metrics
        
        # Autolog disabled by default - Darts uses PyTorch Lightning internally
        # and autolog can cause duplicate/nested runs or interfere with training
        if enable_autolog:
            self._setup_autolog()
        
        logger.info(f"MLflow initialized: Experiment='{experiment_name}', URI='{mlflow.get_tracking_uri()}'")
    
    def _setup_autolog(self):
        """Configure autolog for PyTorch Lightning."""
        try:
            mlflow.pytorch.autolog(
                log_every_n_epoch=1,
                log_every_n_step=None,
                log_models=False,  # We handle model logging ourselves (Darts format)
                log_datasets=False,
                disable=False,
                exclusive=False,
                disable_for_unsupported_versions=True,
                silent=True,
                registered_model_name=None,
                extra_tags={"autolog": "true"}
            )
            self._autolog_enabled = True
            logger.info("MLflow autolog enabled for PyTorch Lightning")
        except Exception as e:
            logger.warning(f"Failed to enable autolog: {e}")
            self._autolog_enabled = False

    def start_run(
        self, 
        run_name: Optional[str] = None, 
        tags: Optional[Dict] = None,
        log_system_metrics: Optional[bool] = None
    ) -> mlflow.ActiveRun:
        """
        Start a new MLflow run with optional system metrics logging.
        
        Args:
            run_name: Name for the run
            tags: Additional tags
            log_system_metrics: Override system metrics setting for this run
        """
        tags = tags or {}
        tags['framework'] = self.framework
        tags['user'] = os.environ.get('USERNAME', 'unknown')
        
        # Determine if we should log system metrics
        log_sys = log_system_metrics if log_system_metrics is not None else self._system_metrics_enabled
        
        return mlflow.start_run(
            run_name=run_name, 
            tags=tags,
            log_system_metrics=log_sys
        )

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
            logger.warning("No active MLflow run - skipping model logging")
            return

        # Darts models are not natively supported by mlflow python_function flavor easily without wrapper
        # For simplicity, we log as a generic pytorch model if possible, or just as an artifact folder (pickle).
        # Robust approach: Save locally then log artifacts.

        import tempfile
        import shutil

        temp_dir = None
        try:
            temp_dir = tempfile.mkdtemp()
            temp_path = Path(temp_dir)

            # 1. Save Darts model
            model_file = temp_path / "model.pkl"
            try:
                model.save(str(model_file))
            except Exception as e:
                logger.error(f"Failed to save model to {model_file}: {e}")
                raise

            # Validate model was saved
            if not model_file.exists():
                raise RuntimeError(f"Model file was not created at {model_file}")

            # 2. Save custom artifacts (scalers, config, datasets)
            if custom_artifacts:
                for name, content in custom_artifacts.items():
                    artifact_file = temp_path / name
                    try:
                        if isinstance(content, (str, Path)):
                            if Path(content).exists():
                                shutil.copy(content, artifact_file)
                            else:
                                logger.warning(f"Artifact path does not exist: {content}")
                        elif hasattr(content, 'to_csv'):  # Pandas DataFrame
                            content.to_csv(artifact_file)
                        elif isinstance(content, dict) and name.endswith('.json'):
                            # JSON serializable dict
                            with open(artifact_file, 'w') as f:
                                json.dump(content, f, indent=2, default=str)
                        else:
                            # Pickle object
                            import pickle
                            with open(artifact_file, 'wb') as f:
                                pickle.dump(content, f)
                    except Exception as e:
                        logger.warning(f"Failed to save artifact '{name}': {e}")

            # 3. Log all as artifacts
            mlflow.log_artifacts(str(temp_path), artifact_path=artifact_path)
            logger.info(f"Model and artifacts logged to MLflow: {artifact_path}")

        except Exception as e:
            logger.error(f"Failed to log model to MLflow: {e}", exc_info=True)
            raise
        finally:
            # Cleanup temp directory
            if temp_dir and Path(temp_dir).exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")
            
    def search_runs(self, filter_string: str = "", order_by: Optional[List[str]] = None) -> List[Any]:
        """Search runs in current experiment."""
        return mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            filter_string=filter_string,
            order_by=order_by or ["attributes.start_time DESC"]
        )
    
    def log_input_dataset(
        self,
        df: Any,
        name: str = "input_data",
        context: str = "training",
        targets: Optional[str] = None
    ):
        """
        Log an input dataset to MLflow for lineage tracking.

        Args:
            df: Pandas DataFrame or dict of DataFrames to log
            name: Name for the dataset
            context: Context (training, validation, test)
            targets: Target column name
        """
        if not mlflow.active_run():
            return

        try:
            import pandas as pd

            if isinstance(df, dict):
                # Log all DataFrames in the dict (for global models with multiple stations)
                for station_name, station_df in df.items():
                    if isinstance(station_df, pd.DataFrame):
                        try:
                            dataset = mlflow.data.from_pandas(
                                station_df,
                                name=f"{name}_{station_name}",
                                targets=targets
                            )
                            mlflow.log_input(dataset, context=context)
                            logger.debug(f"Logged dataset '{name}_{station_name}' ({len(station_df)} rows)")
                        except Exception as e:
                            logger.warning(f"Failed to log dataset for station {station_name}: {e}")
            elif isinstance(df, pd.DataFrame):
                dataset = mlflow.data.from_pandas(
                    df,
                    name=name,
                    targets=targets
                )
                mlflow.log_input(dataset, context=context)
                logger.debug(f"Logged dataset '{name}' ({len(df)} rows) to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log dataset: {e}")
    
    def log_model_with_signature(
        self, 
        model: Any, 
        artifact_path: str,
        input_example: Optional[Any] = None,
        custom_artifacts: Optional[Dict] = None
    ):
        """
        Log a model with signature inference for documentation.
        
        Args:
            model: The model object (Darts ForecastingModel)
            artifact_path: Directory in artifact store
            input_example: Example input for signature inference
            custom_artifacts: Additional files to log
        """
        if not mlflow.active_run():
            return
        
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 1. Save Darts model
            model_file = temp_path / "model.pkl"
            model.save(str(model_file))
            
            # 2. Save model info/signature as JSON
            model_info = {
                "model_class": model.__class__.__name__,
                "input_chunk_length": getattr(model, "input_chunk_length", None),
                "output_chunk_length": getattr(model, "output_chunk_length", None),
                "supports_past_covariates": getattr(model, "supports_past_covariates", False),
                "supports_future_covariates": getattr(model, "supports_future_covariates", False),
                "supports_multivariate": getattr(model, "supports_multivariate", False),
            }
            
            info_file = temp_path / "model_info.json"
            with open(info_file, 'w') as f:
                json.dump(model_info, f, indent=2, default=str)
            
            # 3. Save custom artifacts
            if custom_artifacts:
                for name, content in custom_artifacts.items():
                    artifact_file = temp_path / name
                    if isinstance(content, (str, Path)):
                        if Path(content).exists():
                            shutil.copy(content, artifact_file)
                    elif hasattr(content, 'to_csv'):
                        content.to_csv(artifact_file)
                    else:
                        import pickle
                        with open(artifact_file, 'wb') as f:
                            pickle.dump(content, f)
            
            # 4. Log all as artifacts
            mlflow.log_artifacts(str(temp_path), artifact_path=artifact_path)
            
            # 5. Log model info as params for easy filtering
            try:
                mlflow.log_params({
                    "model_class": model_info["model_class"],
                    "input_chunk": model_info.get("input_chunk_length"),
                    "output_chunk": model_info.get("output_chunk_length"),
                })
            except Exception as e:
                logging.getLogger(__name__).debug(f"Param logging skipped: {e}")

    def register_model_version(
        self,
        model_name: str,
        artifact_path: str = "model",
        tags: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """
        Register the latest logged model in MLflow Model Registry.

        Args:
            model_name: Registered model name
            artifact_path: Artifact path under the current run
            tags: Optional tags for the model version
        """
        if not mlflow.active_run():
            logger.warning("No active MLflow run - skipping model registration")
            return None

        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/{artifact_path}"
        try:
            registered = mlflow.register_model(model_uri, model_name)
            if tags:
                client = mlflow.MlflowClient()
                for k, v in tags.items():
                    try:
                        client.set_model_version_tag(registered.name, registered.version, k, str(v))
                    except Exception as e:
                        logger.warning(f"Failed to set model version tag {k}: {e}")
            return registered.version
        except Exception as e:
            logger.warning(f"Failed to register model '{model_name}': {e}")
            return None
    
    def set_run_status(self, status: str = "FINISHED"):
        """Set the status of the current run."""
        if mlflow.active_run():
            mlflow.set_tag("run_status", status)


# Global singleton
_mlflow_manager = None

def get_mlflow_manager(
    tracking_uri: Optional[str] = None,
    enable_system_metrics: bool = True,
    enable_autolog: bool = False  # Disabled - conflicts with Darts
) -> MLflowManager:
    """
    Get or create the global MLflow manager instance.
    
    Args:
        tracking_uri: Custom tracking URI
        enable_system_metrics: Log CPU/GPU/memory
        enable_autolog: Enable PyTorch autolog (disabled by default, conflicts with Darts)
    """
    global _mlflow_manager
    if _mlflow_manager is None:
        _mlflow_manager = MLflowManager(
            tracking_uri=tracking_uri,
            enable_system_metrics=enable_system_metrics,
            enable_autolog=enable_autolog
        )
    return _mlflow_manager


# =============================================================================
# TRACING UTILITIES
# =============================================================================

def trace_function(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None
) -> Callable[[F], F]:
    """
    Decorator to trace a function with MLflow Tracing.
    
    Captures inputs, outputs, and execution time.
    
    Usage:
        @trace_function()
        def my_function(x, y):
            return x + y
        
        @trace_function(name="custom_name", attributes={"step": "preprocessing"})
        def preprocess_data(df):
            return df.dropna()
    
    Args:
        name: Custom name for the trace span (defaults to function name)
        attributes: Additional attributes to attach to the span
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Build span name
            span_name = name or func.__name__
            span_attrs = dict(attributes) if attributes else {}
            span_attrs["function"] = func.__name__
            span_attrs["module"] = func.__module__
            
            try:
                # Use MLflow trace if available (MLflow 2.0+)
                if hasattr(mlflow, 'trace'):
                    traced_func = mlflow.trace(
                        func,
                        name=span_name,
                        attributes=span_attrs
                    )
                    return traced_func(*args, **kwargs)
                else:
                    # Fallback: just run the function
                    return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Tracing failed for {func.__name__}: {e}")
                return func(*args, **kwargs)
        
        return wrapper  # type: ignore
    return decorator


def trace_training_step(step_name: str):
    """
    Decorator specifically for training pipeline steps.
    
    Usage:
        @trace_training_step("data_preparation")
        def prepare_data(df):
            ...
    """
    return trace_function(
        name=f"training.{step_name}",
        attributes={"pipeline": "training", "step": step_name}
    )


def trace_prediction_step(step_name: str):
    """
    Decorator for prediction/inference steps.
    
    Usage:
        @trace_prediction_step("model_inference")
        def predict(model, data):
            ...
    """
    return trace_function(
        name=f"prediction.{step_name}",
        attributes={"pipeline": "prediction", "step": step_name}
    )
