"""Fonctions de chargement et préparation des données."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
import streamlit as st


def load_data_flexible(
    file_path: Path,
    date_col: str = 'date',
    target_col: Optional[str] = None,
    covariate_cols: Optional[list] = None,
    fill_method: str = 'interpolate'
) -> pd.DataFrame:
    """
    Charge les données avec N'IMPORTE QUELS noms de colonnes.
    
    Args:
        file_path: Chemin vers le fichier CSV
        date_col: Nom de la colonne de date
        target_col: Nom de la colonne target (optionnel)
        covariate_cols: Liste des colonnes covariates (optionnel)
        fill_method: 'drop', 'interpolate', 'ffill', 'bfill'
    
    Returns:
        DataFrame avec les colonnes demandées, date en index
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Charger le CSV avec gestion de l'encodage
    try:
        df = pd.read_csv(file_path, parse_dates=[date_col], encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, parse_dates=[date_col], encoding='latin1')
    
    df = df.set_index(date_col).sort_index()
    
    # Sélectionner les colonnes demandées
    if target_col or covariate_cols:
        cols_to_keep = []
        if target_col:
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")
            cols_to_keep.append(target_col)
        
        if covariate_cols:
            for col in covariate_cols:
                if col not in df.columns:
                    raise ValueError(f"Covariate column '{col}' not found. Available: {list(df.columns)}")
                cols_to_keep.append(col)
        
        df = df[cols_to_keep]
    
    # Gérer les NaN
    if fill_method == 'drop':
        df = df.dropna()
    elif fill_method == 'interpolate':
        df = df.interpolate(method='time')
        df = df.dropna()  # Drop remaining NaN at edges
    elif fill_method == 'ffill':
        df = df.ffill()
        df = df.dropna()
    elif fill_method == 'bfill':
        df = df.bfill()
        df = df.dropna()
    
    return df


def prepare_for_darts(
    df: pd.DataFrame,
    target_col: str,
    covariate_cols: Optional[List[str]] = None,
    freq: str = 'D'
) -> Tuple[TimeSeries, Optional[TimeSeries]]:
    """
    Convertit un DataFrame en TimeSeries Darts.
    
    Args:
        df: DataFrame avec index temporel
        target_col: Colonne cible
        covariate_cols: Colonnes covariables
        freq: Fréquence ('D', 'H', etc.)
    
    Returns:
        Tuple (target_series, covariates_series ou None)
    """
    # Créer la série cible
    target_series = TimeSeries.from_dataframe(
        df,
        value_cols=target_col,
        freq=freq
    )
    
    # Créer les covariables si spécifiées
    covariates_series = None
    if covariate_cols and len(covariate_cols) > 0:
        covariates_series = TimeSeries.from_dataframe(
            df,
            value_cols=covariate_cols,
            freq=freq
        )
    
    return target_series, covariates_series


def split_timeseries(
    target: TimeSeries,
    covariates: Optional[TimeSeries],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Dict:
    """
    Split les séries en train/val/test.
    
    Args:
        target: TimeSeries cible
        covariates: TimeSeries covariables (ou None)
        train_ratio: Proportion du train
        val_ratio: Proportion de la validation
    
    Returns:
        Dict avec train, val, test pour target et covariates
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must be < 1.0")

    n = len(target)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    result = {
        'train_target': target[:train_end],
        'val_target': target[train_end:val_end],
        'test_target': target[val_end:],
    }
    
    if covariates is not None:
        result['train_cov'] = covariates[:train_end]
        result['val_cov'] = covariates[train_end:val_end]
        result['test_cov'] = covariates[val_end:]
        result['full_cov'] = covariates
    else:
        result['train_cov'] = None
        result['val_cov'] = None
        result['test_cov'] = None
        result['full_cov'] = None
    
    return result


def scale_data(
    splits: Dict,
    scaler_type: str = 'minmax'
) -> Tuple[Dict, Dict]:
    """
    Normalise les données (fit sur train uniquement).
    
    Args:
        splits: Dict du split avec train/val/test
        scaler_type: 'minmax', 'standard', 'robust', ou 'none'
    
    Returns:
        Tuple (scaled_splits, scalers)
    """
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
    
    if scaler_type == 'none':
        return splits, {'target': None, 'covariates': None}
    
    # Choisir le scaler
    if scaler_type == 'minmax':
        sklearn_scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        sklearn_scaler = StandardScaler()
    elif scaler_type == 'robust':
        sklearn_scaler = RobustScaler()
    else:
        sklearn_scaler = MinMaxScaler()
    
    # Scaler pour target
    target_scaler = Scaler(scaler=sklearn_scaler)
    target_scaler.fit(splits['train_target'])
    
    scaled = {
        'train_target': target_scaler.transform(splits['train_target']),
        'val_target': target_scaler.transform(splits['val_target']),
        'test_target': target_scaler.transform(splits['test_target']),
    }
    
    scalers = {'target': target_scaler, 'covariates': None}
    
    # Scaler pour covariates
    if splits['train_cov'] is not None:
        if scaler_type == 'minmax':
            sklearn_scaler_cov = MinMaxScaler()
        elif scaler_type == 'standard':
            sklearn_scaler_cov = StandardScaler()
        elif scaler_type == 'robust':
            sklearn_scaler_cov = RobustScaler()
        else:
            sklearn_scaler_cov = MinMaxScaler()
        
        cov_scaler = Scaler(scaler=sklearn_scaler_cov)
        cov_scaler.fit(splits['train_cov'])
        
        scaled['train_cov'] = cov_scaler.transform(splits['train_cov'])
        scaled['val_cov'] = cov_scaler.transform(splits['val_cov'])
        scaled['test_cov'] = cov_scaler.transform(splits['test_cov'])
        scaled['full_cov'] = cov_scaler.transform(splits['full_cov'])
        scalers['covariates'] = cov_scaler
    else:
        scaled['train_cov'] = None
        scaled['val_cov'] = None
        scaled['test_cov'] = None
        scaled['full_cov'] = None
    
    return scaled, scalers


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute des features temporelles au DataFrame.
    
    Args:
        df: DataFrame avec index DatetimeIndex
    
    Returns:
        DataFrame avec features temporelles ajoutées
    """
    df = df.copy()
    
    # Features de base
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['day_of_year'] = df.index.dayofyear
    df['month'] = df.index.month
    df['week'] = df.index.isocalendar().week.values
    df['quarter'] = df.index.quarter
    
    # Features cycliques (captures la périodicité)
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    
    return df


def add_lag_features(df: pd.DataFrame, target_col: str, lags: List[int]) -> pd.DataFrame:
    """
    Ajoute des lags de la colonne cible.
    
    Args:
        df: DataFrame
        target_col: Colonne sur laquelle créer les lags
        lags: Liste des lags (ex: [1, 7, 30])
    
    Returns:
        DataFrame avec lags ajoutés
    """
    df = df.copy()
    
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

    # Drop NaN créés par les shifts (only in lag columns, preserve other data)
    lag_cols = [f'{target_col}_lag_{lag}' for lag in lags]
    df = df.dropna(subset=lag_cols)
    
    return df


@st.cache_data(ttl=3600)
def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Calcule un résumé des données.
    
    Args:
        df: DataFrame
    
    Returns:
        Dict avec statistiques
    """
    if len(df) == 0:
        return {'rows': 0, 'columns': len(df.columns), 'missing_pct': 0.0}

    return {
        'n_samples': len(df),
        'n_columns': len(df.columns),
        'columns': list(df.columns),
        'start_date': str(df.index[0]),
        'end_date': str(df.index[-1]),
        'duration_days': (df.index[-1] - df.index[0]).days,
        'missing_values': int(df.isnull().sum().sum()),
        'missing_pct': float(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
    }
