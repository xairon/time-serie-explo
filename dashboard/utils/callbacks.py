import logging
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor

logger = logging.getLogger(__name__)


@dataclass
class MetricsState:
    """Internal state for metrics written to JSON."""

    status: str = "initializing"
    start_time: Optional[str] = None
    current_epoch: int = 0
    total_epochs: Optional[int] = None
    train_losses: List[Optional[float]] = field(default_factory=list)
    val_losses: List[Optional[float]] = field(default_factory=list)
    epochs: List[int] = field(default_factory=list)
    last_update: Optional[str] = None
    elapsed_seconds: float = 0.0
    eta_seconds: Optional[float] = None
    total_time_seconds: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "start_time": self.start_time,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "epochs": self.epochs,
            "last_update": self.last_update,
            "elapsed_seconds": self.elapsed_seconds,
            "eta_seconds": self.eta_seconds,
            "total_time_seconds": self.total_time_seconds,
            "error": self.error,
        }


class MetricsFileCallback(Callback):
    """
    PyTorch Lightning callback that writes training metrics to a JSON file.

    Le format du JSON est consommé par `dashboard.utils.training_monitor.TrainingMonitor`.
    """

    def __init__(self, metrics_file: Path, total_epochs: Optional[int] = None) -> None:
        super().__init__()
        self.metrics_file = Path(metrics_file)
        self.state = MetricsState(total_epochs=total_epochs)
        self._epoch_start_time: Optional[float] = None
        self._training_start_time: Optional[float] = None

        # S'assurer que le répertoire existe
        try:
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create metrics directory {self.metrics_file.parent}: {e}")

    # ------------------------------------------------------------------ #
    # Utils
    # ------------------------------------------------------------------ #
    def _write_state(self) -> None:
        """Write current state to the JSON file."""
        try:
            tmp_path = self.metrics_file.with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                json.dump(self.state.to_dict(), f, indent=2)
            tmp_path.replace(self.metrics_file)
        except Exception as e:
            logger.warning(f"Failed to write metrics file {self.metrics_file}: {e}")

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Lightning hooks
    # ------------------------------------------------------------------ #
    def on_train_start(self, trainer, pl_module) -> None:
        self._training_start_time = time.time()
        self.state.status = "training"
        self.state.start_time = datetime.now().isoformat()
        # Si total_epochs n'est pas renseigné, utiliser max_epochs du trainer
        if self.state.total_epochs is None:
            self.state.total_epochs = getattr(trainer, "max_epochs", None)
        self.state.last_update = self.state.start_time
        self._write_state()

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        current_epoch = int(getattr(trainer, "current_epoch", 0)) + 1
        self.state.current_epoch = current_epoch
        self.state.epochs.append(current_epoch)

        # Récupérer les métriques depuis callback_metrics
        metrics = getattr(trainer, "callback_metrics", {}) or {}
        train_loss = (
            metrics.get("train_loss_epoch")
            or metrics.get("train_loss")
            or metrics.get("loss")
        )
        val_loss = metrics.get("val_loss") or metrics.get("val_loss_epoch")

        self.state.train_losses.append(self._to_float(train_loss))
        self.state.val_losses.append(self._to_float(val_loss))

        now = time.time()
        if self._training_start_time is not None:
            self.state.elapsed_seconds = now - self._training_start_time

        # ETA approximative
        if self.state.total_epochs and current_epoch > 0 and self._training_start_time:
            avg_epoch = self.state.elapsed_seconds / current_epoch
            remaining = avg_epoch * max(self.state.total_epochs - current_epoch, 0)
            self.state.eta_seconds = remaining

        self.state.last_update = datetime.now().isoformat()
        self._write_state()

    def on_exception(self, trainer, pl_module, exception: BaseException) -> None:
        self.state.status = "error"
        self.state.error = str(exception)
        if self._training_start_time is not None:
            self.state.total_time_seconds = time.time() - self._training_start_time
        self.state.last_update = datetime.now().isoformat()
        self._write_state()

    def on_train_end(self, trainer, pl_module) -> None:
        # Write "finalizing" instead of "completed" to allow the training pipeline
        # to finish evaluation and MLflow logging before signaling completion.
        # The training thread will write "completed" after all post-training steps.
        if self.state.status != "error":
            self.state.status = "finalizing"
        if self._training_start_time is not None:
            self.state.total_time_seconds = time.time() - self._training_start_time
        self.state.last_update = datetime.now().isoformat()
        self._write_state()


def create_training_callbacks(
    *,
    metrics_file: Optional[Path] = None,
    total_epochs: Optional[int] = None,
    early_stopping_patience: Optional[int] = 10,
    early_stopping_monitor: str = "val_loss",  # Default to val_loss, not train_loss
    early_stopping_mode: str = "min",
    use_mlflow: bool = False,  # Conservé pour compatibilité, géré ailleurs
    verbose: bool = True,
    enable_lr_monitor: bool = False,
) -> List[Callback]:
    """
    Crée les callbacks de training standards (LR monitor, early stopping,
    et éventuellement un callback qui écrit les métriques dans un fichier JSON).

    Cette signature est alignée avec `run_training_pipeline` dans `training.py`.
    """

    callbacks: List[Callback] = []

    # Learning rate monitor (désactivé par défaut car nécessite un logger actif)
    if enable_lr_monitor:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    # Callback de métriques JSON (pour le monitoring Streamlit)
    if metrics_file is not None:
        callbacks.append(MetricsFileCallback(metrics_file=metrics_file, total_epochs=total_epochs))
        if verbose:
            logger.info(f"MetricsFileCallback enabled (file={metrics_file})")

    # Early stopping
    if early_stopping_patience is not None and early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor=early_stopping_monitor,
                patience=early_stopping_patience,
                mode=early_stopping_mode,
                verbose=verbose,
            )
        )
        if verbose:
            logger.info(
                f"EarlyStopping enabled (monitor={early_stopping_monitor}, "
                f"mode={early_stopping_mode}, patience={early_stopping_patience})"
            )

    return callbacks
