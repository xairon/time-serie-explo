"""Utilitaires pour l'optimisation Optuna adaptés à la nouvelle architecture."""

import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import numpy as np
from typing import Dict, Any, Optional, Tuple
from darts import TimeSeries
from darts.metrics import mae, rmse, mape, r2_score
from pathlib import Path

from dashboard.utils.model_factory import ModelFactory
from dashboard.utils.training import train_model, calculate_metrics


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
    # On garde les params de base
    hyperparams = base_hyperparams.copy()
    
    # Hyperparamètres communs à optimiser
    hyperparams['input_chunk_length'] = trial.suggest_categorical('input_chunk_length', [14, 30, 60, 90, 180])
    hyperparams['output_chunk_length'] = trial.suggest_categorical('output_chunk_length', [7, 14, 30])
    hyperparams['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64])
    hyperparams['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    
    # Hyperparamètres spécifiques au modèle
    model_upper = model_name.upper()
    
    if model_upper == 'NBEATS':
        hyperparams['num_stacks'] = trial.suggest_categorical('num_stacks', [10, 20, 30])
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
        
    elif model_upper in ['RNNMODEL', 'LSTM']:
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
    pl_trainer_kwargs: Optional[Dict] = None
):
    """
    Crée la fonction objective pour Optuna.
    
    Args:
        model_name: Nom du modèle
        train, val: Séries d'entraînement et validation
        train_cov, val_cov, full_cov: Covariables
        use_covariates: Utiliser les covariables
        metric: Métrique à optimiser ('MAE', 'RMSE', 'MAPE')
        n_epochs: Nombre d'epochs (réduit pour Optuna)
        early_stopping: Activer early stopping
        early_stopping_patience: Patience pour early stopping
        pl_trainer_kwargs: Arguments PyTorch Lightning
    """
    
    def objective(trial: optuna.Trial) -> float:
        # Base hyperparams
        base_params = {
            'n_epochs': n_epochs,
            'loss_fn': 'MAE'
        }
        
        # Suggérer les hyperparamètres
        hyperparams = get_hyperparam_search_space(model_name, trial, base_params)
        
        try:
            # Créer le modèle avec early stopping si demandé
            trainer_kwargs = pl_trainer_kwargs.copy() if pl_trainer_kwargs else {}
            
            if early_stopping:
                from pytorch_lightning.callbacks import EarlyStopping
                es_callback = EarlyStopping(
                    monitor='val_loss',
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
            
            # Préparer les covariables selon ce que le modèle supporte
            train_past_cov = None
            val_past_cov = None
            train_future_cov = None
            val_future_cov = None
            
            if use_covariates and train_cov is not None:
                # Vérifier quel type de covariables le modèle supporte
                supports_past = getattr(model, "supports_past_covariates", False)
                supports_future = getattr(model, "supports_future_covariates", False)
                
                if supports_past:
                    train_past_cov = train_cov
                    val_past_cov = val_cov
                elif supports_future:
                    train_future_cov = train_cov
                    val_future_cov = val_cov
            
            # Entraîner
            model = train_model(
                model=model,
                train_series=train,
                val_series=val,
                train_past_covariates=train_past_cov,
                val_past_covariates=val_past_cov,
                train_future_covariates=train_future_cov,
                val_future_covariates=val_future_cov,
                verbose=False
            )
            
            # Prédire sur validation
            output_chunk = hyperparams['output_chunk_length']
            pred_kwargs = {
                'n': min(output_chunk, len(val)),
                'series': train
            }
            
            if use_covariates and full_cov is not None:
                if getattr(model, "uses_past_covariates", False):
                    pred_kwargs['past_covariates'] = full_cov
                if getattr(model, "uses_future_covariates", False):
                    pred_kwargs['future_covariates'] = full_cov
            
            predictions = model.predict(**pred_kwargs)
            
            # Calculer la métrique
            val_trimmed = val[:len(predictions)]
            
            if metric == 'MAE':
                score = float(mae(val_trimmed, predictions))
            elif metric == 'RMSE':
                score = float(rmse(val_trimmed, predictions))
            elif metric == 'MAPE':
                score = float(mape(val_trimmed, predictions))
            else:
                score = float(mae(val_trimmed, predictions))
            
            return score
            
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
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
        monitor='val_loss',
        patience=patience,
        min_delta=min_delta,
        mode='min',
        verbose=True
    )
