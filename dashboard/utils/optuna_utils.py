"""Fonctions pour l'optimisation d'hyperparamètres avec Optuna."""

import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour
import plotly.graph_objects as go
from pathlib import Path
import sys
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.utils.forecasting import create_model, train_model, predict, compute_metrics
from dashboard.config import METRICS_INFO


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
            'num_stacks': trial.suggest_categorical('num_stacks', [10, 20, 30]),
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

    elif model_type == 'LSTM':
        hyperparams.update({
            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
            'n_rnn_layers': trial.suggest_int('n_rnn_layers', 1, 3),
            'dropout': trial.suggest_float('dropout', 0.0, 0.3)
        })

    return hyperparams


def create_optuna_objective(model_type: str, train_data: dict, val_data: dict,
                            work_dir: Path, metric: str = 'MAE'):
    """
    Crée la fonction objective pour Optuna.

    Args:
        model_type: Type de modèle
        train_data: Données d'entraînement
        val_data: Données de validation
        work_dir: Répertoire de travail
        metric: Métrique à optimiser

    Returns:
        Fonction objective
    """
    def objective(trial):
        # Suggérer les hyperparamètres
        hyperparams = suggest_hyperparameters(trial, model_type)

        # Créer le modèle
        model_name = f"optuna_trial_{trial.number}"
        try:
            model = create_model(model_type, hyperparams, work_dir, model_name)

            # Entraîner
            use_covariates = model_type in ['TFT', 'TCN', 'LSTM']
            train_model(model, train_data, val_data, use_covariates=use_covariates)

            # Prédire sur validation
            n_pred = len(val_data['target'])
            if use_covariates:
                pred = predict(
                    model,
                    train_data['target'],
                    train_data['covariates'],
                    n=n_pred
                )
            else:
                pred = predict(model, train_data['target'], n=n_pred)

            # Calculer la métrique
            metrics = compute_metrics(val_data['target'], pred)
            score = metrics[metric]

            return score

        except Exception as e:
            # En cas d'erreur, retourner une valeur pénalisante
            print(f"Trial {trial.number} failed: {e}")
            return float('inf') if METRICS_INFO[metric]['lower_is_better'] else float('-inf')

    return objective


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
    """
    Retourne les meilleurs hyperparamètres.

    Args:
        study: Étude Optuna

    Returns:
        dict avec les meilleurs paramètres
    """
    return study.best_params


def get_best_value(study: optuna.Study) -> float:
    """
    Retourne la meilleure valeur.

    Args:
        study: Étude Optuna

    Returns:
        Meilleure valeur
    """
    return study.best_value


@st.cache_data(ttl=3600)
def plot_optuna_optimization_history(study: optuna.Study) -> go.Figure:
    """
    Graphique de l'historique d'optimisation.

    Args:
        study: Étude Optuna

    Returns:
        Figure Plotly
    """
    fig = plot_optimization_history(study)
    fig.update_layout(template='plotly_white', height=400)
    return fig


@st.cache_data(ttl=3600)
def plot_optuna_param_importances(study: optuna.Study) -> go.Figure:
    """
    Graphique de l'importance des paramètres.

    Args:
        study: Étude Optuna

    Returns:
        Figure Plotly
    """
    try:
        fig = plot_param_importances(study)
        fig.update_layout(template='plotly_white', height=400)
        return fig
    except:
        # Pas assez de trials
        return None


@st.cache_data(ttl=3600)
def plot_optuna_contour(study: optuna.Study, params: list = None) -> go.Figure:
    """
    Graphique de contour.

    Args:
        study: Étude Optuna
        params: Liste des paramètres à afficher (None = auto)

    Returns:
        Figure Plotly
    """
    try:
        if params is None:
            # Sélectionner les 2 paramètres les plus importants
            importances = optuna.importance.get_param_importances(study)
            params = list(importances.keys())[:2]

        fig = plot_contour(study, params=params)
        fig.update_layout(template='plotly_white', height=500)
        return fig
    except:
        return None


@st.cache_data(ttl=3600)
def get_trials_dataframe(study: optuna.Study):
    """
    Retourne un DataFrame avec les trials.

    Args:
        study: Étude Optuna

    Returns:
        DataFrame avec tous les trials
    """
    return study.trials_dataframe()
