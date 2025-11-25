"""Module d'entraînement pour modèles Darts."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from darts import TimeSeries
from darts.metrics import mae, rmse, mape, r2_score, smape
from darts.models.forecasting.forecasting_model import ForecastingModel

from dashboard.utils.model_factory import ModelFactory


def train_model(
    model: ForecastingModel,
    train_series: TimeSeries,
    val_series: Optional[TimeSeries] = None,
    train_past_covariates: Optional[TimeSeries] = None,
    val_past_covariates: Optional[TimeSeries] = None,
    train_future_covariates: Optional[TimeSeries] = None,
    val_future_covariates: Optional[TimeSeries] = None,
    verbose: bool = True
) -> ForecastingModel:
    """
    Entraîne un modèle Darts.

    Args:
        model: Instance du modèle Darts
        train_series: Série d'entraînement
        val_series: Série de validation (optionnel)
        train_past_covariates: Covariables passées pour train
        val_past_covariates: Covariables passées pour validation
        train_future_covariates: Covariables futures pour train
        val_future_covariates: Covariables futures pour validation
        verbose: Afficher les logs

    Returns:
        Modèle entraîné
    """
    # Préparer les arguments pour fit()
    fit_kwargs = {
        'series': train_series,
        'verbose': verbose
    }

    # Ajouter validation si fournie
    if val_series is not None:
        fit_kwargs['val_series'] = val_series

    # Ajouter covariates pour train
    if train_past_covariates is not None:
        fit_kwargs['past_covariates'] = train_past_covariates

        # Si validation fournie, il FAUT aussi les covariates de validation
        if val_series is not None and val_past_covariates is not None:
            fit_kwargs['val_past_covariates'] = val_past_covariates

    if train_future_covariates is not None:
        fit_kwargs['future_covariates'] = train_future_covariates

        # Si validation fournie, il FAUT aussi les covariates de validation
        if val_series is not None and val_future_covariates is not None:
            fit_kwargs['val_future_covariates'] = val_future_covariates

    # Entraîner
    model.fit(**fit_kwargs)

    return model


def evaluate_model(
    model: ForecastingModel,
    train_series: TimeSeries,
    test_series: TimeSeries,
    horizon: int,
    past_covariates: Optional[TimeSeries] = None,
    future_covariates: Optional[TimeSeries] = None,
    num_samples: int = 1
) -> Tuple[TimeSeries, Dict[str, float]]:
    """
    Évalue un modèle sur les données de test.

    Args:
        model: Modèle entraîné
        train_series: Série d'entraînement (nécessaire pour certains modèles)
        test_series: Série de test
        horizon: Horizon de prédiction
        past_covariates: Covariables passées (TOUT le dataset)
        future_covariates: Covariables futures (TOUT le dataset)
        num_samples: Nombre d'échantillons pour prédiction probabiliste

    Returns:
        Tuple (prédictions, métriques)
    """
    # Faire les prédictions depuis la fin de train_series
    pred_kwargs = {
        'n': min(horizon, len(test_series)),
        'series': train_series,
        'num_samples': num_samples
    }

    # Les covariates doivent couvrir TOUT le range (train + test)
    if past_covariates is not None:
        pred_kwargs['past_covariates'] = past_covariates

    if future_covariates is not None:
        pred_kwargs['future_covariates'] = future_covariates

    predictions = model.predict(**pred_kwargs)

    # Calculer les métriques
    metrics = calculate_metrics(test_series, predictions)

    return predictions, metrics


def calculate_metrics(
    actual: TimeSeries,
    predicted: TimeSeries,
    metrics_list: Optional[list] = None
) -> Dict[str, float]:
    """
    Calcule les métriques d'évaluation.

    Args:
        actual: Série réelle
        predicted: Série prédite
        metrics_list: Liste des métriques à calculer

    Returns:
        Dict avec les métriques
    """
    if metrics_list is None:
        metrics_list = ['MAE', 'RMSE', 'MAPE', 'R2', 'sMAPE']

    results = {}

    # Aligner les séries (même longueur)
    min_len = min(len(actual), len(predicted))
    actual_aligned = actual[:min_len]
    predicted_aligned = predicted[:min_len]

    try:
        if 'MAE' in metrics_list:
            results['MAE'] = mae(actual_aligned, predicted_aligned)
    except Exception:
        results['MAE'] = np.nan

    try:
        if 'RMSE' in metrics_list:
            results['RMSE'] = rmse(actual_aligned, predicted_aligned)
    except Exception:
        results['RMSE'] = np.nan

    try:
        if 'MAPE' in metrics_list:
            results['MAPE'] = mape(actual_aligned, predicted_aligned)
    except Exception:
        results['MAPE'] = np.nan

    try:
        if 'R2' in metrics_list:
            results['R2'] = r2_score(actual_aligned, predicted_aligned)
    except Exception:
        results['R2'] = np.nan

    try:
        if 'sMAPE' in metrics_list:
            results['sMAPE'] = smape(actual_aligned, predicted_aligned)
    except Exception:
        results['sMAPE'] = np.nan

    # Direction accuracy (% de fois où la direction du changement est correcte)
    try:
        actual_diff = np.diff(actual_aligned.values().flatten())
        pred_diff = np.diff(predicted_aligned.values().flatten())
        correct_direction = np.sum((actual_diff > 0) == (pred_diff > 0))
        results['Dir_Acc'] = correct_direction / len(actual_diff) * 100
    except Exception:
        results['Dir_Acc'] = np.nan

    return results


def save_model(
    model: ForecastingModel,
    save_dir: Path,
    model_name: str,
    station: str,
    metadata: Optional[Dict] = None
) -> Path:
    """
    Sauvegarde un modèle entraîné (ancienne méthode).

    DEPRECATED: Utilisez run_training_pipeline avec save_dir à la place.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Nom du fichier
    filename = f"{model_name}_{station}.pkl"
    filepath = save_dir / filename

    # S'assurer que le répertoire parent du fichier existe
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Sauvegarder le modèle
    model.save(str(filepath))

    # Sauvegarder les métadonnées si fournies
    if metadata:
        import json
        metadata_path = save_dir / f"{model_name}_{station}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    return filepath


def load_model(filepath: Path) -> ForecastingModel:
    """
    Charge un modèle sauvegardé.

    Args:
        filepath: Chemin vers le modèle

    Returns:
        Modèle chargé
    """
    from darts.models.forecasting.forecasting_model import ForecastingModel

    model = ForecastingModel.load(str(filepath))
    return model


def run_training_pipeline(
    model_name: str,
    hyperparams: Dict[str, Any],
    train: TimeSeries,
    val: TimeSeries,
    test: TimeSeries,
    train_cov: Optional[TimeSeries] = None,
    val_cov: Optional[TimeSeries] = None,
    test_cov: Optional[TimeSeries] = None,
    full_cov: Optional[TimeSeries] = None,
    use_covariates: bool = True,
    save_dir: Optional[Path] = None,
    station_name: str = 'default',
    verbose: bool = True,
    progress_callback: Optional[Any] = None,
    pl_trainer_kwargs: Optional[Dict[str, Any]] = None,
    station_data_df: Optional[pd.DataFrame] = None,
    column_mapping: Optional[Dict[str, str]] = None,
    target_preprocessor: Optional[Any] = None,
    cov_preprocessor: Optional[Any] = None,
    original_filename: Optional[str] = None,
    preprocessing_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Pipeline d'entraînement complet avec sauvegarde des splits.

    Args:
        model_name: Nom du modèle
        hyperparams: Hyperparamètres
        train, val, test: Données train/val/test (TimeSeries)
        train_cov, val_cov, test_cov: Covariables splittées
        full_cov: Covariables complètes (train+val+test) pour prediction
        use_covariates: Utiliser les covariables
        save_dir: Répertoire de sauvegarde
        station_name: Nom de la station (peut être un path comme "00104X0054/P1")
        verbose: Afficher les logs
        progress_callback: Callback pour progression
        pl_trainer_kwargs: Kwargs pour PyTorch Lightning Trainer
        station_data_df: DataFrame complet de la station (OBLIGATOIRE pour sauvegarde)
        column_mapping: Mapping des colonnes {'target_var': '...', 'covariate_vars': [...]}
        target_preprocessor: Scaler fitté sur train pour la target
        cov_preprocessor: Scaler fitté sur train pour les covariables
        original_filename: Nom du fichier original (pour référence)
        preprocessing_config: Configuration du preprocessing appliqué

    Returns:
        Dict avec résultats (model, metrics, predictions, saved_path)
    """
    results = {
        'model_name': model_name,
        'station': station_name,
        'status': 'success'
    }

    try:
        # 1. Créer le modèle
        model = ModelFactory.create_model(
            model_name,
            hyperparams,
            pl_trainer_kwargs_override=pl_trainer_kwargs
        )

        # 2. Préparer les covariables
        train_past_cov = None
        val_past_cov = None

        if use_covariates:
            if train_cov is not None:
                train_past_cov = train_cov
            if val_cov is not None:
                val_past_cov = val_cov

        # 3. Entraîner
        model = train_model(
            model=model,
            train_series=train,
            val_series=val,
            train_past_covariates=train_past_cov,
            val_past_covariates=val_past_cov,
            verbose=verbose
        )

        results['model'] = model

        # 4. Évaluer sur test
        output_chunk = hyperparams.get('output_chunk_length', 7)

        # Pour la prédiction, on a besoin de la série complète jusqu'au test
        full_train = train.append(val)

        # Covariables complètes pour prédiction
        full_past_cov = None
        if use_covariates and full_cov is not None:
            full_past_cov = full_cov

        predictions, metrics = evaluate_model(
            model=model,
            train_series=full_train,
            test_series=test,
            horizon=min(output_chunk, len(test)),
            past_covariates=full_past_cov,
            num_samples=1
        )

        results['predictions'] = predictions
        results['metrics'] = metrics

        # 5. Sauvegarder si demandé (NOUVEAU SYSTÈME avec splits)
        if save_dir and station_data_df is not None:
            # Nettoyer le nom de station
            clean_station_name = station_name.split('/')[-1] if '/' in station_name else station_name

            # Créer les DataFrames pour chaque split
            # On doit reconstruire les DataFrames à partir des tailles
            train_size = len(train)
            val_size = len(val)
            test_size = len(test)

            train_df = station_data_df.iloc[:train_size].copy()
            val_df = station_data_df.iloc[train_size:train_size + val_size].copy()
            test_df = station_data_df.iloc[train_size + val_size:].copy()

            # Construire la config des colonnes
            if column_mapping:
                columns_config = {
                    'date': 'date',
                    'target': column_mapping['target_var'],
                    'covariates': column_mapping['covariate_vars']
                }
            else:
                # Fallback
                columns_config = {
                    'date': 'date',
                    'target': 'level',
                    'covariates': ['PRELIQ_Q', 'T_Q', 'ETP_Q']
                }

            # Construire la config du preprocessing
            if preprocessing_config:
                preproc = {
                    'fill_method': preprocessing_config.get('fill_method', 'Unknown'),
                    'scaler_type': preprocessing_config.get('scaler_type', 'StandardScaler')
                }
            else:
                preproc = {
                    'fill_method': 'Unknown',
                    'scaler_type': 'StandardScaler'
                }

            # Créer la config du modèle
            from dashboard.utils.model_config import ModelConfig, save_model_with_data

            config = ModelConfig(
                model_name=model_name,
                station=clean_station_name,
                original_station_id=station_name,
                columns=columns_config,
                data_source={
                    'type': 'embedded',
                    'original_file': original_filename or 'unknown'
                },
                splits={
                    'train_size': train_size,
                    'val_size': val_size,
                    'test_size': test_size
                },
                preprocessing=preproc,
                hyperparams={k: str(v) for k, v in hyperparams.items()},
                metrics={k: float(v) if not np.isnan(v) else None for k, v in metrics.items()},
                use_covariates=use_covariates
            )

            # Sauvegarder avec le nouveau système (modèle + config + splits)
            model_dir = save_model_with_data(
                model=model,
                save_dir=save_dir,
                model_name=model_name,
                station_name=clean_station_name,
                config=config,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                full_df=station_data_df,
                target_preprocessor=target_preprocessor,
                cov_preprocessor=cov_preprocessor
            )

            results['saved_path'] = str(model_dir)
            results['scalers_saved'] = str(model_dir / "scalers.pkl")

    except Exception as e:
        results['status'] = 'error'
        results['error'] = str(e)
        import traceback
        results['traceback'] = traceback.format_exc()

    return results
