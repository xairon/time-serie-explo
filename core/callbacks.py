"""
Callbacks PyTorch Lightning standards pour l'entraînement.

Ces callbacks sont conçus pour être indépendants de Streamlit et de toute interface graphique.
Ils écrivent les métriques dans des fichiers JSON pour permettre un monitoring externe.
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
from pytorch_lightning.callbacks import Callback
import logging

logger = logging.getLogger(__name__)


class MetricsFileCallback(Callback):
    """
    Callback PyTorch Lightning standard qui écrit les métriques dans un fichier JSON.
    
    Ce callback est complètement indépendant de Streamlit et peut être utilisé
    dans n'importe quel contexte (CLI, backend, notebooks, etc.).
    
    Les métriques sont écrites dans un fichier JSON qui peut être lu par n'importe
    quel processus externe pour afficher la progression.
    """
    
    def __init__(
        self,
        metrics_file: Path,
        total_epochs: Optional[int] = None,
        log_interval: int = 1
    ):
        """
        Args:
            metrics_file: Chemin vers le fichier JSON où écrire les métriques
            total_epochs: Nombre total d'epochs (optionnel, pour calculer le pourcentage)
            log_interval: Intervalle d'écriture (écrire toutes les N epochs)
        """
        super().__init__()
        self.metrics_file = Path(metrics_file)
        self.total_epochs = total_epochs
        self.log_interval = log_interval
        
        # Initialiser le fichier avec une structure vide
        self.metrics = {
            'status': 'initializing',
            'start_time': None,
            'current_epoch': 0,
            'total_epochs': total_epochs,
            'train_losses': [],
            'val_losses': [],
            'epochs': [],
            'last_update': None
        }
        
        # Créer le répertoire parent si nécessaire
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        self._write_metrics()
    
    def _write_metrics(self):
        """Écrit les métriques dans le fichier JSON."""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write metrics to {self.metrics_file}: {e}")
    
    def on_train_start(self, trainer, pl_module):
        """Appelé au début de l'entraînement."""
        self.metrics['status'] = 'training'
        self.metrics['start_time'] = time.time()
        self.metrics['current_epoch'] = 0
        self._write_metrics()
        logger.info(f"Training started. Metrics will be written to {self.metrics_file}")
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Appelé au début de chaque epoch."""
        self.metrics['current_epoch'] = trainer.current_epoch + 1
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Appelé à la fin de chaque epoch."""
        # Ne logger que selon l'intervalle
        if self.metrics['current_epoch'] % self.log_interval != 0:
            return
        
        # Récupérer les losses
        train_loss = None
        val_loss = None
        
        # Train loss
        if 'train_loss' in trainer.callback_metrics:
            train_loss = float(trainer.callback_metrics['train_loss'])
        elif 'loss' in trainer.callback_metrics:
            train_loss = float(trainer.callback_metrics['loss'])
        
        # Val loss
        if 'val_loss' in trainer.callback_metrics:
            val_loss = float(trainer.callback_metrics['val_loss'])
        
        # Sauvegarder l'historique
        self.metrics['epochs'].append(self.metrics['current_epoch'])
        if train_loss is not None:
            self.metrics['train_losses'].append(train_loss)
        else:
            self.metrics['train_losses'].append(None)
        
        if val_loss is not None:
            self.metrics['val_losses'].append(val_loss)
        else:
            self.metrics['val_losses'].append(None)
        
        # Calculer le temps écoulé et estimé
        elapsed = time.time() - self.metrics['start_time']
        if self.metrics['current_epoch'] > 0:
            avg_epoch_time = elapsed / self.metrics['current_epoch']
            if self.total_epochs:
                remaining_epochs = self.total_epochs - self.metrics['current_epoch']
                eta = avg_epoch_time * remaining_epochs
                self.metrics['eta_seconds'] = eta
            else:
                self.metrics['eta_seconds'] = None
        else:
            self.metrics['eta_seconds'] = None
        
        self.metrics['elapsed_seconds'] = elapsed
        self.metrics['last_update'] = time.time()
        
        # Écrire les métriques
        self._write_metrics()
    
    def on_train_end(self, trainer, pl_module):
        """Appelé à la fin de l'entraînement."""
        total_time = time.time() - self.metrics['start_time']
        self.metrics['status'] = 'completed'
        self.metrics['total_time_seconds'] = total_time
        self.metrics['last_update'] = time.time()
        self._write_metrics()
        logger.info(f"Training completed in {total_time:.1f}s. Final metrics saved to {self.metrics_file}")
    
    def on_train_error(self, trainer, pl_module, exception):
        """Appelé en cas d'erreur pendant l'entraînement."""
        self.metrics['status'] = 'error'
        self.metrics['error'] = str(exception)
        self.metrics['last_update'] = time.time()
        self._write_metrics()
        logger.error(f"Training error: {exception}")


def create_training_callbacks(
    metrics_file: Optional[Path] = None,
    total_epochs: Optional[int] = None,
    early_stopping_patience: Optional[int] = None,
    early_stopping_monitor: str = "val_loss",
    early_stopping_mode: str = "min"
) -> list:
    """
    Crée une liste de callbacks standards pour l'entraînement.
    
    Args:
        metrics_file: Chemin vers le fichier JSON pour les métriques (optionnel)
        total_epochs: Nombre total d'epochs (pour le callback de métriques)
        early_stopping_patience: Patience pour EarlyStopping (None pour désactiver)
        early_stopping_monitor: Métrique à monitorer pour EarlyStopping
        early_stopping_mode: Mode ('min' ou 'max')
    
    Returns:
        Liste de callbacks PyTorch Lightning
    """
    from pytorch_lightning.callbacks import EarlyStopping
    
    callbacks = []
    
    # Callback de métriques (si un fichier est fourni)
    if metrics_file:
        callbacks.append(MetricsFileCallback(
            metrics_file=metrics_file,
            total_epochs=total_epochs
        ))
    
    # Early Stopping (si activé)
    if early_stopping_patience is not None and early_stopping_patience > 0:
        callbacks.append(EarlyStopping(
            monitor=early_stopping_monitor,
            patience=early_stopping_patience,
            mode=early_stopping_mode,
            verbose=True
        ))
    
    return callbacks
