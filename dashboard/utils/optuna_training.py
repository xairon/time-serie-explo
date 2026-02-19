"""Utilitaires pour l'optimisation Optuna adaptés à la nouvelle architecture."""

import optuna
import copy
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from darts import TimeSeries, concatenate
from darts.metrics import mae, rmse, r2_score
from pathlib import Path

from dashboard.utils.model_factory import ModelFactory
from dashboard.utils.training import train_model, calculate_metrics

logger = logging.getLogger(__name__)


def get_hyperparam_search_space(model_name: str, trial: optuna.Trial, base_hyperparams: Dict[str, Any]) -> Dict[str, Any]:
    """
    Définit l'espace de recherche Optuna pour un modèle donné.
    
    Args:
        model_name: Nom du modèle
        trial: Optuna trial
        base_hyperparams: Hyperparamètres de base (fixes)
    
    Returns:
        Dict des hyperparamètres suggérés
    """
    # On garde les params de base (input/output chunk length fixés par l'utilisateur, pas optimisés par Optuna)
    hyperparams = copy.deepcopy(base_hyperparams)
    
    # Hyperparamètres communs à optimiser (sans input/output chunk length)
    hyperparams['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64])
    hyperparams['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    
    # Hyperparamètres spécifiques au modèle
    model_upper = model_name.upper()
    
    if model_upper == 'NBEATS':
        hyperparams['num_stacks'] = trial.suggest_categorical('num_stacks', [2, 3, 4])
        hyperparams['num_blocks'] = trial.suggest_int('num_blocks', 1, 3)
        hyperparams['num_layers'] = trial.suggest_int('num_layers', 2, 4)
        hyperparams['layer_widths'] = trial.suggest_categorical('layer_widths', [128, 256, 512])
        
    elif model_upper == 'NHITS':
        hyperparams['num_stacks'] = trial.suggest_categorical('num_stacks', [2, 3, 4])
        hyperparams['num_blocks'] = trial.suggest_int('num_blocks', 1, 3)
        hyperparams['num_layers'] = trial.suggest_int('num_layers', 2, 4)
        hyperparams['layer_widths'] = trial.suggest_categorical('layer_widths', [256, 512])
        
    elif model_upper == 'TFT':
        hyperparams['hidden_size'] = trial.suggest_categorical('hidden_size', [32, 64, 128])
        hyperparams['lstm_layers'] = trial.suggest_int('lstm_layers', 1, 3)
        hyperparams['num_attention_heads'] = trial.suggest_categorical('num_attention_heads', [2, 4, 8])
        hyperparams['dropout'] = trial.suggest_float('dropout', 0.0, 0.3)
        hyperparams['hidden_continuous_size'] = trial.suggest_categorical('hidden_continuous_size', [8, 16, 32])
        
    elif model_upper == 'TCN':
        hyperparams['num_filters'] = trial.suggest_categorical('num_filters', [32, 64, 128])
        hyperparams['kernel_size'] = trial.suggest_categorical('kernel_size', [3, 5, 7])
        hyperparams['num_layers'] = trial.suggest_int('num_layers', 2, 5)
        hyperparams['dilation_base'] = trial.suggest_int('dilation_base', 2, 4)
        hyperparams['dropout'] = trial.suggest_float('dropout', 0.0, 0.3)
        
    elif model_upper in ['RNNMODEL', 'LSTM', 'GRU']:
        hyperparams['hidden_dim'] = trial.suggest_categorical('hidden_dim', [64, 128, 256])
        hyperparams['n_rnn_layers'] = trial.suggest_int('n_rnn_layers', 1, 3)
        hyperparams['dropout'] = trial.suggest_float('dropout', 0.0, 0.3)
        
    elif model_upper == 'TRANSFORMER':
        hyperparams['d_model'] = trial.suggest_categorical('d_model', [32, 64, 128])
        hyperparams['nhead'] = trial.suggest_categorical('nhead', [2, 4, 8])
        hyperparams['num_encoder_layers'] = trial.suggest_int('num_encoder_layers', 2, 4)
        hyperparams['num_decoder_layers'] = trial.suggest_int('num_decoder_layers', 2, 4)
        hyperparams['dropout'] = trial.suggest_float('dropout', 0.0, 0.3)
        
    elif model_upper == 'TIDE':
        hyperparams['num_encoder_layers'] = trial.suggest_int('num_encoder_layers', 1, 3)
        hyperparams['num_decoder_layers'] = trial.suggest_int('num_decoder_layers', 1, 3)
        hyperparams['decoder_output_dim'] = trial.suggest_categorical('decoder_output_dim', [8, 16, 32])
        hyperparams['hidden_size'] = trial.suggest_categorical('hidden_size', [128, 256, 512])
        hyperparams['dropout'] = trial.suggest_float('dropout', 0.0, 0.3)
    
    return hyperparams


def create_optuna_objective(
    model_name: str,
    train: TimeSeries,
    val: TimeSeries,
    train_cov: Optional[TimeSeries],
    val_cov: Optional[TimeSeries],
    full_cov: Optional[TimeSeries],
    use_covariates: bool,
    metric: str = 'MAE',
    n_epochs: int = 30,
    early_stopping: bool = True,
    early_stopping_patience: int = 5,
    pl_trainer_kwargs: Optional[Dict] = None,
    input_chunk_length: int = 30,
    output_chunk_length: int = 7,
):
    """
    Crée la fonction objective pour Optuna.
    
    Args:
        model_name: Nom du modèle
        train, val: Séries d'entraînement et validation
        train_cov, val_cov, full_cov: Covariables
        use_covariates: Utiliser les covariables
        metric: Métrique à optimiser ('MAE', 'RMSE')
        n_epochs: Nombre d'epochs (réduit pour Optuna)
        early_stopping: Activer early stopping
        early_stopping_patience: Patience pour early stopping
        pl_trainer_kwargs: Arguments PyTorch Lightning
        input_chunk_length: Input chunk fixé (non optimisé par Optuna)
        output_chunk_length: Output chunk fixé (non optimisé par Optuna)
    """
    
    def objective(trial: optuna.Trial) -> float:
        # Base hyperparams (input/output chunk fixés par l'UI, pas dans l'espace Optuna)
        base_params = {
            'n_epochs': n_epochs,
            'loss_fn': 'MAE',
            'input_chunk_length': input_chunk_length,
            'output_chunk_length': output_chunk_length,
        }
        
        # Suggérer les hyperparamètres (sans input/output_chunk_length)
        hyperparams = get_hyperparam_search_space(model_name, trial, base_params)
        
        try:
            # Créer le modèle avec early stopping si demandé
            trainer_kwargs = copy.deepcopy(pl_trainer_kwargs) if pl_trainer_kwargs else {}
            
            if early_stopping:
                from pytorch_lightning.callbacks import EarlyStopping
                es_callback = EarlyStopping(
                    monitor='val_loss',  # Must be val_loss, not train_loss
                    patience=early_stopping_patience,
                    mode='min',
                    verbose=False
                )
                if 'callbacks' in trainer_kwargs:
                    trainer_kwargs['callbacks'].append(es_callback)
                else:
                    trainer_kwargs['callbacks'] = [es_callback]
            
            model = ModelFactory.create_model(
                model_name,
                hyperparams,
                pl_trainer_kwargs_override=trainer_kwargs
            )
            
            # Préparer les covariables selon ce que le modèle supporte (only past_covariates)
            train_past_cov = None
            val_past_cov = None
            
            if use_covariates and train_cov is not None:
                # Only use past_covariates to avoid prediction bias
                supports_past = getattr(model, "supports_past_covariates", False)
                
                if supports_past:
                    train_past_cov = train_cov
                    val_past_cov = val_cov
            
            # Entraîner
            model = train_model(
                model=model,
                train_series=train,
                val_series=val,
                train_past_covariates=train_past_cov,
                val_past_covariates=val_past_cov,
                verbose=False
            )
            
            # Prédire sur validation
            output_chunk = hyperparams['output_chunk_length']
            pred_kwargs = {
                'n': min(output_chunk, len(val)),
                'series': train
            }
            
            if use_covariates and train_cov is not None and val_cov is not None:
                if getattr(model, "supports_past_covariates", False):
                    pred_kwargs['past_covariates'] = concatenate([train_cov, val_cov], axis=0)
            
            predictions = model.predict(**pred_kwargs)

            # Calculer la métrique avec alignement temporel correct
            # Les prédictions commencent à la fin de train, donc on aligne avec le début de val
            # Utiliser slice_intersect pour un alignement temporel propre
            val_aligned = val.slice_intersect(predictions)
            pred_aligned = predictions.slice_intersect(val)

            if len(val_aligned) == 0 or len(pred_aligned) == 0:
                logger.warning(f"Trial {trial.number}: No overlap between predictions and validation")
                return float('inf')

            if metric == 'MAE':
                score = float(mae(val_aligned, pred_aligned))
            elif metric == 'RMSE':
                score = float(rmse(val_aligned, pred_aligned))
            else:
                score = float(mae(val_aligned, pred_aligned))

            # Cleanup GPU memory between trials
            del model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                torch.xpu.empty_cache()

            return score

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
            # Cleanup GPU memory between trials
            try:
                del model
            except NameError:
                pass
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                torch.xpu.empty_cache()
            return float('inf')
    
    return objective


def run_optuna_study(
    objective,
    n_trials: int = 20,
    timeout: int = 1800,
    study_name: str = None,
    direction: str = 'minimize',
    progress_callback=None
) -> optuna.Study:
    """
    Lance une étude Optuna.
    
    Args:
        objective: Fonction objective
        n_trials: Nombre de trials
        timeout: Timeout en secondes
        study_name: Nom de l'étude
        direction: 'minimize' ou 'maximize'
        progress_callback: Callback pour la progression (n_trials_done, n_trials_total)
    
    Returns:
        Étude Optuna
    """
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    def callback(study, trial):
        if progress_callback:
            progress_callback(len(study.trials), n_trials)
    
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        callbacks=[callback] if progress_callback else None,
        show_progress_bar=False
    )
    
    return study


def get_early_stopping_callback(patience: int = 10, min_delta: float = 0.0):
    """
    Crée un callback EarlyStopping pour PyTorch Lightning.

    Args:
        patience: Nombre d'epochs sans amélioration avant d'arrêter
        min_delta: Amélioration minimale pour considérer une amélioration

    Returns:
        EarlyStopping callback
    """
    from pytorch_lightning.callbacks import EarlyStopping

    return EarlyStopping(
        monitor='val_loss',  # Must be val_loss, not train_loss
        patience=patience,
        min_delta=min_delta,
        mode='min',
        verbose=True
    )
