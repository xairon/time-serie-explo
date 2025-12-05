"""Module pour la génération de prévisions avec les modèles Darts."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import ForecastingModel
from dashboard.utils.preprocessing import prepare_dataframe_for_darts


def generate_single_window_forecast(
    model: ForecastingModel,
    full_df: pd.DataFrame,
    target_col: str,
    covariate_cols: Optional[List[str]],
    preprocessing_config: Dict[str, Any],
    scalers: Dict[str, Any],
    start_date: pd.Timestamp,
    use_covariates: bool = True
) -> Tuple[TimeSeries, TimeSeries, TimeSeries, Dict[str, float], int]:
    """
    Génère une prévision pour une seule fenêtre à partir d'une date donnée.
    
    Retourne à la fois la prédiction auto-régressive ET la prédiction one-step.
    
    Args:
        model: Modèle entraîné
        full_df: DataFrame complet (train + val + test)
        target_col: Nom de la colonne cible
        covariate_cols: Liste des colonnes covariables
        preprocessing_config: Configuration du preprocessing
        scalers: Dictionnaire des scalers
        start_date: Date de début de la prédiction
        use_covariates: Si True, utilise les covariables
        
    Returns:
        Tuple (pred_autoregressive, pred_onestep, target_series, metrics, horizon)
    """
    fill_method = preprocessing_config.get('fill_method', 'Interpolation linéaire')
    
    # Récupérer l'horizon du modèle
    horizon = getattr(model, 'output_chunk_length', 30)
    
    # Préparer la série complète
    full_series, covariates = prepare_dataframe_for_darts(
        full_df,
        target_col=target_col,
        covariate_cols=covariate_cols if use_covariates else None,
        freq='D',
        fill_method=fill_method
    )
    
    # Scaling
    target_preprocessor = scalers.get('target_preprocessor')
    cov_preprocessor = scalers.get('cov_preprocessor')
    
    full_series_scaled = full_series
    covariates_scaled = covariates
    
    if target_preprocessor:
        full_series_scaled = target_preprocessor.transform(full_series)
        
    if cov_preprocessor and covariates is not None:
        covariates_scaled = cov_preprocessor.transform(covariates)
        
    # Couper l'historique juste avant start_date
    history_cutoff = start_date - pd.Timedelta(days=1)
    history_series_scaled = full_series_scaled.split_after(history_cutoff)[0]
    
    # =========================================================================
    # 1. PRÉDICTION AUTO-RÉGRESSIVE (le modèle prédit sur ses propres prédictions)
    # =========================================================================
    predict_kwargs = {
        'n': horizon,
        'series': history_series_scaled
    }
    
    if covariates_scaled is not None:
        if getattr(model, "uses_past_covariates", False):
            predict_kwargs['past_covariates'] = covariates_scaled
        if getattr(model, "uses_future_covariates", False):
            predict_kwargs['future_covariates'] = covariates_scaled
    
    pred_series_scaled = model.predict(**predict_kwargs)
    
    # =========================================================================
    # 2. PRÉDICTION ONE-STEP (chaque prédiction utilise les vraies valeurs passées)
    # =========================================================================
    end_date = start_date + pd.Timedelta(days=horizon - 1)
    
    try:
        # Utiliser historical_forecasts avec forecast_horizon=1 pour du one-step
        onestep_kwargs = {
            'series': full_series_scaled,
            'start': start_date,
            'forecast_horizon': 1,
            'stride': 1,
            'retrain': False,
            'verbose': False
        }
        
        if covariates_scaled is not None:
            if getattr(model, "uses_past_covariates", False):
                onestep_kwargs['past_covariates'] = covariates_scaled
            if getattr(model, "uses_future_covariates", False):
                onestep_kwargs['future_covariates'] = covariates_scaled
        
        onestep_forecasts = model.historical_forecasts(**onestep_kwargs)
        
        # Limiter à l'horizon
        if isinstance(onestep_forecasts, list):
            # Concatener les prédictions one-step
            from darts import concatenate
            pred_onestep_scaled = concatenate(onestep_forecasts[:horizon])
        else:
            pred_onestep_scaled = onestep_forecasts[:horizon]
            
    except Exception as e:
        # Fallback: utiliser la même prédiction auto-régressive
        print(f"One-step prediction failed: {e}")
        pred_onestep_scaled = pred_series_scaled
    
    # Inverse scaling
    if target_preprocessor:
        pred_series = target_preprocessor.inverse_transform(pred_series_scaled)
        pred_onestep = target_preprocessor.inverse_transform(pred_onestep_scaled)
        full_series_unscaled = full_series
    else:
        pred_series = pred_series_scaled
        pred_onestep = pred_onestep_scaled
        full_series_unscaled = full_series
        
    # Extraire la tranche réelle correspondante
    target_series = full_series_unscaled.slice(start_date, end_date)
    
    # Aligner les longueurs
    min_len = min(len(pred_series), len(target_series))
    pred_series = pred_series[:min_len]
    pred_onestep = pred_onestep[:min_len] if len(pred_onestep) >= min_len else pred_onestep
    target_series = target_series[:min_len]
    
    # Calculer les métriques (sur autoregressive)
    from darts.metrics import mae, rmse, mape, r2_score, smape
    
    metrics = {
        'MAE': float(mae(target_series, pred_series)),
        'RMSE': float(rmse(target_series, pred_series)),
        'MAPE': float(mape(target_series, pred_series)),
        'R2': float(r2_score(target_series, pred_series)),
        'sMAPE': float(smape(target_series, pred_series))
    }
    
    return pred_series, pred_onestep, target_series, metrics, horizon

def generate_global_forecast(
    model: ForecastingModel,
    history_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target_col: str,
    covariate_cols: Optional[List[str]],
    preprocessing_config: Dict[str, Any],
    scalers: Dict[str, Any],
    use_covariates: bool = True
) -> Tuple[TimeSeries, TimeSeries, Dict[str, float]]:
    """
    Génère une prévision globale sur l'ensemble du jeu de test.
    
    Args:
        model: Modèle entraîné
        history_df: DataFrame d'historique (train + val)
        target_df: DataFrame cible (test)
        target_col: Nom de la colonne cible
        covariate_cols: Liste des colonnes covariables
        preprocessing_config: Configuration du preprocessing (pour fill_method)
        scalers: Dictionnaire des scalers (target_preprocessor, cov_preprocessor)
        use_covariates: Si True, utilise les covariables
        
    Returns:
        Tuple (pred_series, target_series_original, metrics)
    """
    # 1. Récupérer la méthode de remplissage de la config
    fill_method = preprocessing_config.get('fill_method', 'Interpolation linéaire')
    
    # 2. Préparer les séries Darts
    # Historique (Contexte)
    history_series, _ = prepare_dataframe_for_darts(
        history_df,
        target_col=target_col,
        covariate_cols=covariate_cols if use_covariates else None,
        freq='D',
        fill_method=fill_method
    )
    
    # Cible (Ground Truth pour métriques)
    target_series, _ = prepare_dataframe_for_darts(
        target_df,
        target_col=target_col,
        covariate_cols=covariate_cols if use_covariates else None,
        freq='D',
        fill_method=fill_method
    )
    
    # Covariables pour la prédiction (doivent couvrir le futur)
    covariates_future = None
    if use_covariates and covariate_cols:
        # On a besoin des covariables sur history + target
        full_cov_df = pd.concat([history_df, target_df])
        _, covariates_future = prepare_dataframe_for_darts(
            full_cov_df,
            target_col=target_col,
            covariate_cols=covariate_cols,
            freq='D',
            fill_method=fill_method
        )
    
    # 3. Appliquer le scaling
    target_preprocessor = scalers.get('target_preprocessor')
    cov_preprocessor = scalers.get('cov_preprocessor')
    
    history_series_scaled = history_series
    target_series_scaled = target_series
    covariates_scaled = covariates_future
    
    if target_preprocessor:
        history_series_scaled = target_preprocessor.transform(history_series)
        target_series_scaled = target_preprocessor.transform(target_series)
        
    if cov_preprocessor and covariates_future is not None:
        covariates_scaled = cov_preprocessor.transform(covariates_future)
        
    # 4. Prédire
    n_pred = len(target_series)
    predict_kwargs = {
        'n': n_pred,
        'series': history_series_scaled
    }
    
    if covariates_scaled is not None:
        if getattr(model, "uses_past_covariates", False):
            predict_kwargs['past_covariates'] = covariates_scaled
        if getattr(model, "uses_future_covariates", False):
            predict_kwargs['future_covariates'] = covariates_scaled

    pred_series_scaled = model.predict(**predict_kwargs)
    
    # 5. Inverse scaling
    if target_preprocessor:
        pred_series = target_preprocessor.inverse_transform(pred_series_scaled)
    else:
        pred_series = pred_series_scaled
        
    # 6. Calculer les métriques
    from darts.metrics import mae, rmse, mape, r2_score, smape
    
    # Aligner les longueurs
    min_len = min(len(pred_series), len(target_series))
    pred_series = pred_series[:min_len]
    target_series = target_series[:min_len]
    
    metrics = {
        'MAE': float(mae(target_series, pred_series)),
        'RMSE': float(rmse(target_series, pred_series)),
        'MAPE': float(mape(target_series, pred_series)),
        'R2': float(r2_score(target_series, pred_series)),
        'sMAPE': float(smape(target_series, pred_series))
    }
    
    return pred_series, target_series, metrics

def generate_rolling_forecast(
    model: ForecastingModel,
    full_df: pd.DataFrame,
    target_col: str,
    covariate_cols: Optional[List[str]],
    preprocessing_config: Dict[str, Any],
    scalers: Dict[str, Any],
    start_date: pd.Timestamp,
    forecast_horizon: int,
    stride: int,
    use_covariates: bool = True
) -> Tuple[List[TimeSeries], TimeSeries]:
    """
    Génère des prévisions glissantes (historical forecasts).
    
    Args:
        model: Modèle entraîné
        full_df: DataFrame complet (train + val + test)
        target_col: Nom de la colonne cible
        covariate_cols: Liste des colonnes covariables
        preprocessing_config: Configuration du preprocessing
        scalers: Dictionnaire des scalers
        start_date: Date de début des prévisions (généralement début du test set)
        forecast_horizon: Horizon de prévision à chaque pas
        stride: Pas de déplacement de la fenêtre
        use_covariates: Si True, utilise les covariables
        
    Returns:
        Tuple (liste des fenêtres de prévision, série cible complète)
    """
    fill_method = preprocessing_config.get('fill_method', 'Interpolation linéaire')
    
    # Préparer la série complète
    full_series, covariates = prepare_dataframe_for_darts(
        full_df,
        target_col=target_col,
        covariate_cols=covariate_cols if use_covariates else None,
        freq='D',
        fill_method=fill_method
    )
    
    # Scaling
    target_preprocessor = scalers.get('target_preprocessor')
    cov_preprocessor = scalers.get('cov_preprocessor')
    
    full_series_scaled = full_series
    covariates_scaled = covariates
    
    if target_preprocessor:
        full_series_scaled = target_preprocessor.transform(full_series)
        
    if cov_preprocessor and covariates is not None:
        covariates_scaled = cov_preprocessor.transform(covariates)
        
    # Historical Forecasts
    forecast_kwargs = {
        'series': full_series_scaled,
        'start': start_date,
        'forecast_horizon': forecast_horizon,
        'stride': stride,
        'retrain': False,
        'last_points_only': False, # On veut toute la trajectoire
        'verbose': False
    }
    
    if covariates_scaled is not None:
        if getattr(model, "uses_past_covariates", False):
            forecast_kwargs['past_covariates'] = covariates_scaled
        if getattr(model, "uses_future_covariates", False):
            forecast_kwargs['future_covariates'] = covariates_scaled
            
    forecasts_scaled = model.historical_forecasts(**forecast_kwargs)
    
    # Inverse scaling pour chaque fenêtre
    forecasts = []
    if isinstance(forecasts_scaled, TimeSeries):
        forecasts_scaled = [forecasts_scaled]
        
    for ts in forecasts_scaled:
        if target_preprocessor:
            forecasts.append(target_preprocessor.inverse_transform(ts))
        else:
            forecasts.append(ts)
            
    return forecasts, full_series

def generate_comparison_forecast(
    model: ForecastingModel,
    full_df: pd.DataFrame,
    target_col: str,
    covariate_cols: Optional[List[str]],
    preprocessing_config: Dict[str, Any],
    scalers: Dict[str, Any],
    start_date: pd.Timestamp,
    forecast_horizon: int,
    use_covariates: bool = True
) -> Tuple[TimeSeries, TimeSeries, TimeSeries, Dict[str, float], Dict[str, float]]:
    """
    Génère une comparaison entre prévision auto-régressive et fenêtre exacte (teacher forcing).
    
    Args:
        model: Modèle entraîné
        full_df: DataFrame complet
        target_col: Nom de la colonne cible
        covariate_cols: Liste des colonnes covariables
        preprocessing_config: Configuration du preprocessing
        scalers: Dictionnaire des scalers
        start_date: Date de début de la fenêtre de comparaison
        forecast_horizon: Durée de la fenêtre de comparaison
        use_covariates: Si True, utilise les covariables
        
    Returns:
        Tuple (target_slice, autoregressive_forecast, exact_window_forecast, metrics_auto, metrics_exact)
    """
    fill_method = preprocessing_config.get('fill_method', 'Interpolation linéaire')
    
    # 1. Préparer les données
    full_series, covariates = prepare_dataframe_for_darts(
        full_df,
        target_col=target_col,
        covariate_cols=covariate_cols if use_covariates else None,
        freq='D',
        fill_method=fill_method
    )
    
    # 2. Scaling
    target_preprocessor = scalers.get('target_preprocessor')
    cov_preprocessor = scalers.get('cov_preprocessor')
    
    full_series_scaled = full_series
    covariates_scaled = covariates
    
    if target_preprocessor:
        full_series_scaled = target_preprocessor.transform(full_series)
        
    if cov_preprocessor and covariates is not None:
        covariates_scaled = cov_preprocessor.transform(covariates)
        
    # 3. Prévision Auto-régressive (Drift)
    # On coupe l'historique juste avant start_date
    history_cutoff = start_date - pd.Timedelta(days=1)
    history_series_scaled = full_series_scaled.split_after(history_cutoff)[0]
    
    predict_kwargs = {
        'n': forecast_horizon,
        'series': history_series_scaled
    }
    
    if covariates_scaled is not None:
        if getattr(model, "uses_past_covariates", False):
            predict_kwargs['past_covariates'] = covariates_scaled
        if getattr(model, "uses_future_covariates", False):
            predict_kwargs['future_covariates'] = covariates_scaled
            
    autoregressive_scaled = model.predict(**predict_kwargs)
    
    # 4. Prévision Exacte (Historical Forecast / Teacher Forcing)
    # On demande une prévision qui commence exactement à start_date
    # historical_forecasts utilise les vraies valeurs passées pour chaque point si horizon=1
    # Ici on veut simuler une fenêtre "idéale" où on aurait prédit ce bloc
    # C'est en fait équivalent à autoregressive si on part du même point.
    # Ce que l'utilisateur veut probablement dire par "fenêtre de prédiction exact",
    # c'est "si on avait prédit ce jour là avec les données de la veille".
    # Donc une série de prédictions à horizon 1 (ou horizon court) concaténées.
    
    hist_kwargs = {
        'series': full_series_scaled,
        'start': start_date,
        'forecast_horizon': 1, # Horizon 1 = prédiction "exacte" step-by-step
        'stride': 1,
        'retrain': False,
        'last_points_only': True,
        'verbose': False
    }
    
    if covariates_scaled is not None:
        if getattr(model, "uses_past_covariates", False):
            hist_kwargs['past_covariates'] = covariates_scaled
        if getattr(model, "uses_future_covariates", False):
            hist_kwargs['future_covariates'] = covariates_scaled
            
    # On génère sur la même durée que l'horizon
    # Note: historical_forecasts va générer jusqu'à la fin de la série si on ne limite pas
    # On va slicer après
    exact_window_scaled = model.historical_forecasts(**hist_kwargs)
    
    # Slicer pour garder juste la fenêtre demandée
    end_date = start_date + pd.Timedelta(days=forecast_horizon - 1)
    exact_window_scaled = exact_window_scaled.slice(start_date, end_date)
    
    # 5. Inverse Scaling
    if target_preprocessor:
        autoregressive = target_preprocessor.inverse_transform(autoregressive_scaled)
        exact_window = target_preprocessor.inverse_transform(exact_window_scaled)
        target_slice = target_preprocessor.inverse_transform(full_series_scaled.slice(start_date, end_date))
    else:
        autoregressive = autoregressive_scaled
        exact_window = exact_window_scaled
        target_slice = full_series_scaled.slice(start_date, end_date)
        
    # 6. Métriques
    from darts.metrics import mae, rmse, mape, r2_score
    
    def calc_metrics(true, pred):
        return {
            'MAE': float(mae(true, pred)),
            'RMSE': float(rmse(true, pred)),
            'MAPE': float(mape(true, pred)),
            'R2': float(r2_score(true, pred))
        }
        
    metrics_auto = calc_metrics(target_slice, autoregressive)
    metrics_exact = calc_metrics(target_slice, exact_window)
    
    return target_slice, autoregressive, exact_window, metrics_auto, metrics_exact
