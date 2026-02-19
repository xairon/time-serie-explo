"""Fonctions utilitaires pour la visualisation et l'exploration des résultats Optuna.

NOTE: L'optimisation elle-même est gérée par optuna_training.py.
Ce module fournit uniquement des fonctions de visualisation et d'analyse.
"""

import logging
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def suggest_hyperparameters(trial: optuna.Trial, model_type: str) -> dict:
    """
    Définit l'espace de recherche des hyperparamètres.

    Args:
        trial: Optuna trial
        model_type: Type de modèle ('NBEATS', 'TFT', 'TCN', 'LSTM')

    Returns:
        dict avec les hyperparamètres suggérés
    """
    # Hyperparamètres communs
    hyperparams = {
        'input_chunk': trial.suggest_categorical('input_chunk', [20, 30, 60, 90]),
        'output_chunk': trial.suggest_categorical('output_chunk', [7, 14, 30]),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'n_epochs': 30,  # Fixe pour accélérer
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    }

    # Hyperparamètres spécifiques au modèle
    if model_type == 'NBEATS':
        hyperparams.update({
            'num_stacks': trial.suggest_categorical('num_stacks', [2, 3, 4]),
            'num_blocks': trial.suggest_int('num_blocks', 1, 3),
            'num_layers': trial.suggest_int('num_layers', 2, 5),
            'layer_widths': trial.suggest_categorical('layer_widths', [128, 256, 512])
        })

    elif model_type == 'TFT':
        hyperparams.update({
            'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128]),
            'lstm_layers': trial.suggest_int('lstm_layers', 1, 3),
            'num_attention_heads': trial.suggest_categorical('num_attention_heads', [2, 4, 8]),
            'dropout': trial.suggest_float('dropout', 0.0, 0.3)
        })

    elif model_type == 'TCN':
        hyperparams.update({
            'num_filters': trial.suggest_categorical('num_filters', [32, 64, 128]),
            'kernel_size': trial.suggest_categorical('kernel_size', [3, 5, 7]),
            'num_layers': trial.suggest_int('num_layers', 2, 5),
            'dilation_base': trial.suggest_int('dilation_base', 2, 4),
            'dropout': trial.suggest_float('dropout', 0.0, 0.3)
        })

    elif model_type in ('LSTM', 'GRU'):
        hyperparams.update({
            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
            'n_rnn_layers': trial.suggest_int('n_rnn_layers', 1, 3),
            'dropout': trial.suggest_float('dropout', 0.0, 0.3)
        })

    return hyperparams


def run_optuna_study(objective, n_trials: int = 50, timeout: int = 3600,
                     study_name: str = None, direction: str = 'minimize'):
    """
    Lance une étude Optuna.

    Args:
        objective: Fonction objective
        n_trials: Nombre d'essais
        timeout: Temps maximum (secondes)
        study_name: Nom de l'étude
        direction: 'minimize' ou 'maximize'

    Returns:
        optuna.Study
    """
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False  # Géré par Streamlit
    )

    return study


def get_best_params(study: optuna.Study) -> dict:
    """Retourne les meilleurs hyperparamètres."""
    return study.best_params


def get_best_value(study: optuna.Study) -> float:
    """Retourne la meilleure valeur."""
    return study.best_value


def plot_optuna_optimization_history(study: optuna.Study) -> go.Figure:
    """Graphique de l'historique d'optimisation."""
    fig = plot_optimization_history(study)
    fig.update_layout(template='plotly_white', height=400)
    return fig


def plot_optuna_param_importances(study: optuna.Study) -> go.Figure:
    """Graphique de l'importance des paramètres."""
    try:
        fig = plot_param_importances(study)
        fig.update_layout(template='plotly_white', height=400)
        return fig
    except Exception:
        # Pas assez de trials
        return None


def plot_optuna_contour(study: optuna.Study, params: list = None) -> go.Figure:
    """Graphique de contour."""
    try:
        if params is None:
            # Sélectionner les 2 paramètres les plus importants
            importances = optuna.importance.get_param_importances(study)
            params = list(importances.keys())[:2]

        fig = plot_contour(study, params=params)
        fig.update_layout(template='plotly_white', height=500)
        return fig
    except Exception:
        return None


def get_trials_dataframe(study: optuna.Study):
    """Retourne un DataFrame avec les trials."""
    return study.trials_dataframe()
