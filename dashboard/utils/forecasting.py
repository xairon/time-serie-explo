"""Wrapper pour entraînement et prédiction avec Darts."""

import torch
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime

from darts.models import NBEATSModel, TFTModel, TCNModel, RNNModel
from darts.metrics import mae, rmse, mape, r2_score, smape
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.config import DEVICE, RANDOM_SEED, LOGS_DIR, RESULTS_DIR, CHECKPOINTS_DIR
from dashboard.utils.data_loader import prepare_data_for_training
from dashboard.utils.state import save_context
import pandas as pd

def update_leaderboard(model_name: str, station: str, model_type: str, metrics: dict):
    """
    Met à jour le fichier CSV des résultats.
    """
    results_path = RESULTS_DIR / 'darts_comparison_complete.csv'
    
    # Créer une ligne de résultat
    new_row = {
        'model_name': model_name,
        'station': station,
        'model': model_type,
        **metrics
    }
    
    df_new = pd.DataFrame([new_row])
    
    if results_path.exists():
        try:
            df_old = pd.read_csv(results_path)
            # Concaténer et garder le plus récent si doublon exact (optionnel, ici on ajoute tout)
            df_updated = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            # Si le fichier est corrompu, on écrase
            df_updated = df_new
    else:
        df_updated = df_new
        
    # Sauvegarder
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df_updated.to_csv(results_path, index=False)
    return True


def create_model(model_type: str, hyperparams: dict, work_dir: Path, model_name: str):
    """
    Crée un modèle Darts avec les hyperparamètres fournis.

    Args:
        model_type: 'NBEATS', 'TFT', 'TCN', 'LSTM'
        hyperparams: dict avec les hyperparamètres
        work_dir: Répertoire pour sauvegarder les checkpoints
        model_name: Nom du modèle

    Returns:
        Instance du modèle Darts
    """
    # Hyperparamètres communs
    input_chunk = hyperparams.get('input_chunk', 30)
    output_chunk = hyperparams.get('output_chunk', 7)
    batch_size = hyperparams.get('batch_size', 32)
    n_epochs = hyperparams.get('n_epochs', 50)
    learning_rate = hyperparams.get('learning_rate', 1e-3)

    # Callbacks communs
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        min_delta=0.0001,
        mode='min'
    )

    logger = TensorBoardLogger(str(LOGS_DIR), name=model_name)

    pl_trainer_kwargs = {
        'callbacks': [early_stopping],
        'logger': logger,
        'accelerator': 'gpu' if DEVICE == 'cuda' else 'cpu',
        'devices': 1,
        'enable_model_summary': False,
    }

    # Créer le modèle selon le type
    if model_type == 'NBEATS':
        num_stacks = hyperparams.get('num_stacks', 30)
        num_blocks = hyperparams.get('num_blocks', 1)
        num_layers = hyperparams.get('num_layers', 4)
        layer_widths = hyperparams.get('layer_widths', 256)

        model = NBEATSModel(
            input_chunk_length=input_chunk,
            output_chunk_length=output_chunk,
            num_stacks=num_stacks,
            num_blocks=num_blocks,
            num_layers=num_layers,
            layer_widths=layer_widths,
            batch_size=batch_size,
            n_epochs=n_epochs,
            optimizer_kwargs={'lr': learning_rate},
            pl_trainer_kwargs=pl_trainer_kwargs,
            loss_fn=torch.nn.L1Loss(),
            random_state=RANDOM_SEED,
            force_reset=True,
            save_checkpoints=True,
            work_dir=str(work_dir),
            model_name=model_name
        )

    elif model_type == 'TFT':
        hidden_size = hyperparams.get('hidden_size', 64)
        lstm_layers = hyperparams.get('lstm_layers', 1)
        num_attention_heads = hyperparams.get('num_attention_heads', 4)
        dropout = hyperparams.get('dropout', 0.1)

        model = TFTModel(
            input_chunk_length=input_chunk,
            output_chunk_length=output_chunk,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            batch_size=batch_size,
            n_epochs=n_epochs,
            optimizer_kwargs={'lr': learning_rate},
            pl_trainer_kwargs=pl_trainer_kwargs,
            loss_fn=torch.nn.L1Loss(),
            random_state=RANDOM_SEED,
            force_reset=True,
            save_checkpoints=True,
            work_dir=str(work_dir),
            model_name=model_name
        )

    elif model_type == 'TCN':
        num_filters = hyperparams.get('num_filters', 64)
        kernel_size = hyperparams.get('kernel_size', 3)
        num_layers = hyperparams.get('num_layers', 3)
        dilation_base = hyperparams.get('dilation_base', 2)
        dropout = hyperparams.get('dropout', 0.1)

        model = TCNModel(
            input_chunk_length=input_chunk,
            output_chunk_length=output_chunk,
            num_filters=num_filters,
            kernel_size=kernel_size,
            num_layers=num_layers,
            dilation_base=dilation_base,
            dropout=dropout,
            batch_size=batch_size,
            n_epochs=n_epochs,
            optimizer_kwargs={'lr': learning_rate},
            pl_trainer_kwargs=pl_trainer_kwargs,
            loss_fn=torch.nn.L1Loss(),
            random_state=RANDOM_SEED,
            force_reset=True,
            save_checkpoints=True,
            work_dir=str(work_dir),
            model_name=model_name
        )

    elif model_type == 'LSTM':
        hidden_dim = hyperparams.get('hidden_dim', 128)
        n_rnn_layers = hyperparams.get('n_rnn_layers', 2)
        dropout = hyperparams.get('dropout', 0.1)

        model = RNNModel(
            model='LSTM',
            input_chunk_length=input_chunk,
            training_length=input_chunk + output_chunk,
            hidden_dim=hidden_dim,
            n_rnn_layers=n_rnn_layers,
            dropout=dropout,
            batch_size=batch_size,
            n_epochs=n_epochs,
            optimizer_kwargs={'lr': learning_rate},
            pl_trainer_kwargs=pl_trainer_kwargs,
            loss_fn=torch.nn.L1Loss(),
            random_state=RANDOM_SEED,
            force_reset=True,
            save_checkpoints=True,
            work_dir=str(work_dir),
            model_name=model_name
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def train_model(model, train_data: dict, val_data: dict, use_covariates: bool = True):
    """
    Entraîne un modèle Darts.

    Args:
        model: Instance du modèle Darts
        train_data: dict avec 'target' et 'covariates'
        val_data: dict avec 'target' et 'covariates'
        use_covariates: Utiliser les covariates (True pour TFT, TCN, LSTM)
    """
    if use_covariates and isinstance(model, (TFTModel, TCNModel, RNNModel)):
        model.fit(
            series=train_data['target'],
            past_covariates=train_data['covariates'],
            val_series=val_data['target'],
            val_past_covariates=val_data['covariates'],
            verbose=False
        )
    else:
        # NBEATS ne supporte pas les covariates
        model.fit(
            series=train_data['target'],
            val_series=val_data['target'],
            verbose=False
        )


def predict(model, series, past_covariates=None, n: int = None):
    """
    Génère des prédictions.

    Args:
        model: Modèle Darts entraîné
        series: Série historique
        past_covariates: Covariates passées (optionnel)
        n: Nombre de pas à prédire

    Returns:
        TimeSeries avec les prédictions
    """
    if past_covariates is not None and isinstance(model, (TFTModel, TCNModel, RNNModel)):
        pred = model.predict(
            n=n,
            series=series,
            past_covariates=past_covariates,
            verbose=False
        )
    else:
        pred = model.predict(
            n=n,
            series=series,
            verbose=False
        )

    return pred


@st.cache_data
def calculate_nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency."""
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)


@st.cache_data
def calculate_kge(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Kling-Gupta Efficiency."""
    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = np.std(y_pred) / np.std(y_true)
    beta = np.mean(y_pred) / np.mean(y_true)
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


@st.cache_data
def compute_metrics(y_true, y_pred) -> dict:
    """
    Calcule toutes les métriques.

    Args:
        y_true: TimeSeries vraies valeurs
        y_pred: TimeSeries prédictions

    Returns:
        dict avec toutes les métriques
    """
    # Métriques Darts
    mae_val = mae(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    r2_val = r2_score(y_true, y_pred)
    smape_val = smape(y_true, y_pred)

    # NRMSE (normalisé par la plage)
    y_range = y_true.values().max() - y_true.values().min()
    nrmse_val = rmse_val / y_range if y_range > 0 else 0

    # Directional Accuracy
    y_true_diff = np.diff(y_true.values().flatten())
    y_pred_diff = np.diff(y_pred.values().flatten())
    dir_acc = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff)) * 100

    # Hydrological Metrics
    y_true_np = y_true.values().flatten()
    y_pred_np = y_pred.values().flatten()
    
    nse_val = calculate_nse(y_true_np, y_pred_np)
    kge_val = calculate_kge(y_true_np, y_pred_np)

    return {
        'MAE': float(mae_val),
        'RMSE': float(rmse_val),
        'MAPE': float(mape_val),
        'R2': float(r2_val),
        'sMAPE': float(smape_val),
        'NRMSE': float(nrmse_val),
        'Dir_Acc': float(dir_acc),
        'NSE': float(nse_val),
        'KGE': float(kge_val)
    }


def save_model_checkpoint(model, save_path: Path):
    """
    Sauvegarde un modèle avec métadonnées.

    Args:
        model: Modèle Darts
        save_path: Chemin de sauvegarde (doit finir par .pth.tar)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder le modèle Darts
    model.save(str(save_path))
    
    # Sauvegarder les métadonnées à côté
    metadata_path = save_path.with_suffix('.json')
    
    # Déterminer le type de modèle
    model_type = type(model).__name__.replace('Model', '').upper()
    if model_type == 'RNN': model_type = 'LSTM' # Cas particulier pour RNNModel(model='LSTM')
    
    metadata = {
        'model_type': model_type,
        'creation_date': datetime.now().isoformat(),
        'model_name': model.model_name
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)


@st.cache_resource
def load_model_from_checkpoint(checkpoint_path: str):
    """
    Charge un modèle depuis un checkpoint de manière robuste.

    Args:
        checkpoint_path: Chemin vers le checkpoint

    Returns:
        Modèle chargé
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # 1. Essayer de lire les métadonnées JSON
    metadata_path = checkpoint_path.with_suffix('.json')
    model_type = None
    
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                model_type = metadata.get('model_type')
        except Exception:
            pass # Ignorer si JSON corrompu
            
    # 2. Si pas de métadonnées, deviner via le nom de fichier (fallback)
    if not model_type:
        path_str = str(checkpoint_path).lower()
        if 'nbeats' in path_str: model_type = 'NBEATS'
        elif 'tft' in path_str: model_type = 'TFT'
        elif 'tcn' in path_str: model_type = 'TCN'
        elif 'lstm' in path_str: model_type = 'LSTM'
        else:
            raise ValueError(f"Cannot determine model type from {checkpoint_path}. No metadata found and filename is ambiguous.")

    try:
        if model_type == 'NBEATS':
            model = NBEATSModel.load(str(checkpoint_path))
        elif model_type == 'TFT':
            model = TFTModel.load(str(checkpoint_path))
        elif model_type == 'TCN':
            model = TCNModel.load(str(checkpoint_path))
        elif model_type == 'LSTM':
            model = RNNModel.load(str(checkpoint_path))
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        model._fit_called = True
        return model

    except Exception as e:
        raise RuntimeError(f"Failed to load model ({model_type}): {e}")


def run_training_pipeline(station: str, model_type: str, hyperparams: dict, fill_missing: bool, use_covariates: bool, progress_callback=None):
    """
    Exécute le pipeline complet d'entraînement.
    
    Args:
        station: Nom de la station
        model_type: Type de modèle
        hyperparams: Dictionnaire d'hyperparamètres
        fill_missing: Interpoler les données manquantes
        use_covariates: Utiliser les covariables
        progress_callback: Fonction(int, str) pour mettre à jour l'UI
        
    Returns:
        dict: {
            'metrics': dict,
            'pred': TimeSeries,
            'test_data': TimeSeries,
            'scalers': dict,
            'save_path': Path
        }
    """
    def report(prog, msg):
        if progress_callback:
            progress_callback(prog, msg)
            
    # 1. Charger et préparer
    report(10, "📥 Chargement des données...")
    data = prepare_data_for_training(station, fill_missing=fill_missing)
    train_data = data['train_scaled']
    val_data = data['val_scaled']
    test_data = data['test_scaled']
    scalers = data['scalers']

    # 2. Créer le modèle
    report(20, "🔧 Création du modèle...")
    
    # Nommage standardisé: {MODEL}_{STATION}_{TIMESTAMP}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{model_type}_{station}_{timestamp}"
    
    model = create_model(model_type, hyperparams, CHECKPOINTS_DIR, model_name)

    # 3. Entraîner
    report(30, "🏋️ Entraînement en cours...")
    use_cov = use_covariates and model_type in ['TFT', 'TCN', 'LSTM']
    train_model(model, train_data, val_data, use_covariates=use_cov)

    # 4. Évaluer
    report(70, "📊 Évaluation...")
    n_pred = len(test_data['target'])
    if use_cov:
        pred = predict(model, train_data['target'], train_data['covariates'], n=n_pred)
    else:
        pred = predict(model, train_data['target'], n=n_pred)

    metrics = compute_metrics(test_data['target'], pred)

    # 5. Sauvegarder
    report(90, "💾 Sauvegarde...")
    save_path = CHECKPOINTS_DIR / f"{model_name}.pth.tar"
    save_model_checkpoint(model, save_path)
    
    # Sauvegarder le contexte
    save_context(station=station, model_path=save_path)
    
    # 6. Mettre à jour le leaderboard
    update_leaderboard(model_name, station, model_type, metrics)

    report(100, "✅ Terminé !")
    
    return {
        'metrics': metrics,
        'pred': pred,
        'test_data': test_data,
        'scalers': scalers,
        'save_path': save_path
    }
