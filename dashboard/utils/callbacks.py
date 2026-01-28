import logging
from typing import List, Optional
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor

logger = logging.getLogger(__name__)

def create_training_callbacks(
    early_stopping_patience: Optional[int] = 10,
    verbose: bool = True
) -> List[Callback]:
    """
    Create standard PyTorch Lightning callbacks for training.
    
    Args:
        early_stopping_patience: Number of epochs with no improvement to wait
        verbose: Whether to print status updates
    
    Returns:
        List of PL Callbacks
    """
    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
    ]
    
    if early_stopping_patience is not None and early_stopping_patience > 0:
        callbacks.append(EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            mode="min",
            verbose=verbose
        ))
        if verbose:
            logger.info(f"EarlyStopping enabled (patience={early_stopping_patience})")
            
    return callbacks
