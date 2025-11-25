"""Fonctions de chargement et préparation des données."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
import streamlit as st
import sys

# Ajouter le parent au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.config import DATA_DIR, STATIONS


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_station_data(station_name: str, fill_missing: bool = False) -> pd.DataFrame:
    """
    Charge les données d'une station (ancienne fonction, hardcodée).

    ⚠️ DEPRECATED: Utilisez load_station_data_flexible() à la place.
    Cette fonction est conservée pour compatibilité avec l'ancien code.

    Args:
        station_name: Nom de la station (ex: 'piezo1')
        fill_missing: Si True, interpole les valeurs manquantes. Sinon, supprime les lignes avec NaN.

    Returns:
        DataFrame avec colonnes: date (index), level, PRELIQ_Q, T_Q, ETP_Q
    """
    file_path = DATA_DIR / f"{station_name}.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Station {station_name} not found at {file_path}")

    df = pd.read_csv(file_path, parse_dates=['date'])
    df = df.set_index('date').sort_index()

    # Gérer les NaN
    if fill_missing:
        # Interpolation linéaire pour le niveau (target)
        if 'level' in df.columns:
            df['level'] = df['level'].interpolate(method='time')

        # Interpolation linéaire ou ffill pour les covariables météo
        # (On suppose que la météo est continue)
        for col in ['PRELIQ_Q', 'T_Q', 'ETP_Q']:
            if col in df.columns:
                df[col] = df[col].interpolate(method='time')

        # S'il reste des NaN au début ou à la fin, on drop
        df = df.dropna()
    else:
        # Supprimer les NaN
        df = df.dropna()

    # Valider les colonnes
    required_cols = ['level', 'PRELIQ_Q', 'T_Q', 'ETP_Q']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    return df


def load_station_data_flexible(
    file_path: Path,
    date_col: str = 'date',
    target_col: Optional[str] = None,
    covariate_cols: Optional[list] = None,
    fill_missing: bool = False
) -> pd.DataFrame:
    """
    Charge les données d'une station avec N'IMPORTE QUELS noms de colonnes.

    Pas de hardcoding de noms de colonnes !

    Args:
        file_path: Chemin vers le fichier CSV
        date_col: Nom de la colonne de date
        target_col: Nom de la colonne target (optionnel, si None retourne toutes les colonnes)
        covariate_cols: Liste des noms des colonnes covariates (optionnel)
        fill_missing: Si True, interpole les valeurs manquantes

    Returns:
        DataFrame avec les colonnes demandées, date en index
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Charger le CSV
    df = pd.read_csv(file_path, parse_dates=[date_col])
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
    if fill_missing:
        # Interpolation linéaire pour toutes les colonnes
        for col in df.columns:
            df[col] = df[col].interpolate(method='time')

        # Drop les NaN restants (début/fin)
        df = df.dropna()
    else:
        # Supprimer les NaN
        df = df.dropna()

    return df


def load_data_from_model_config(model_config) -> pd.DataFrame:
    """
    Charge les données à partir d'une ModelConfig.

    Utilise les VRAIS noms de colonnes du modèle (pas de hardcoding).

    Args:
        model_config: Instance de ModelConfig

    Returns:
        DataFrame avec les données
    """
    from dashboard.utils.model_config import ModelConfig

    if not isinstance(model_config, ModelConfig):
        raise TypeError("model_config must be a ModelConfig instance")

    # Déterminer le chemin du fichier
    if model_config.data_source['type'] == 'embedded':
        # Données embarquées dans le dossier du modèle
        # Le chemin est stocké comme relatif, on doit le reconstruire
        # Cette fonction sera appelée avec le bon contexte
        raise ValueError("Use load_model_with_config() instead for embedded data")
    elif model_config.data_source['type'] == 'file':
        file_path = Path(model_config.data_source['file_path'])
    else:
        raise ValueError(f"Unknown data source type: {model_config.data_source['type']}")

    # Charger avec les noms de colonnes du modèle
    df = load_station_data_flexible(
        file_path=file_path,
        date_col=model_config.columns['date'],
        target_col=model_config.columns['target'],
        covariate_cols=model_config.columns['covariates'],
        fill_missing=(model_config.preprocessing.get('fill_method', 'Supprimer les lignes') != 'Supprimer les lignes')
    )

    return df


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_all_stations_summary() -> pd.DataFrame:
    """
    Retourne un résumé de toutes les stations.

    Returns:
        DataFrame avec: Station, Samples, Start Date, End Date, Duration (years)
    """
    summaries = []

    for station in STATIONS:
        try:
            file_path = DATA_DIR / f"{station}.csv"
            if not file_path.exists():
                continue
                
            # Optimisation: lire seulement la colonne date
            df = pd.read_csv(file_path, usecols=['date'], parse_dates=['date'])
            df = df.sort_values('date')
            
            if len(df) == 0:
                continue
                
            start_date = df['date'].iloc[0]
            end_date = df['date'].iloc[-1]
            duration = (end_date - start_date).days / 365.25
            
            summaries.append({
                'Station': station,
                'Samples': len(df),
                'Start Date': start_date.strftime('%Y-%m-%d'),
                'End Date': end_date.strftime('%Y-%m-%d'),
                'Duration (years)': f"{duration:.1f}"
            })
        except Exception:
            continue

    return pd.DataFrame(summaries)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_timeseries(station_name: str, fill_missing: bool = False) -> dict:
    """
    Convertit les données en TimeSeries Darts.

    Args:
        station_name: Nom de la station
        fill_missing: Interpoler les valeurs manquantes

    Returns:
        dict avec 'target' (level) et 'covariates' (PRELIQ_Q, T_Q, ETP_Q)
    """
    df = load_station_data(station_name, fill_missing=fill_missing)

    # Target: level
    ts_level = TimeSeries.from_dataframe(
        df[['level']],
        value_cols='level',
        fill_missing_dates=True,
        freq='D'
    )

    # Covariates: PRELIQ_Q, T_Q, ETP_Q
    ts_cov = TimeSeries.from_dataframe(
        df[['PRELIQ_Q', 'T_Q', 'ETP_Q']],
        fill_missing_dates=True,
        freq='D'
    )

    return {
        'target': ts_level,
        'covariates': ts_cov
    }


def split_data(ts_target, ts_cov, train_ratio=0.5, val_ratio=0.1):
    """
    Split temporel train/val/test.

    Args:
        ts_target: TimeSeries de la target
        ts_cov: TimeSeries des covariates
        train_ratio: Proportion du train set (défaut: 50%)
        val_ratio: Proportion du val set (défaut: 10%)

    Returns:
        tuple: (train_data, val_data, test_data)
        Chaque élément est un dict avec 'target' et 'covariates'
    """
    n = len(ts_target)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    # Target splits
    train_target = ts_target[:train_end]
    val_target = ts_target[train_end:val_end]
    test_target = ts_target[val_end:]

    # Covariates splits
    train_cov = ts_cov[:train_end]
    val_cov = ts_cov[train_end:val_end]
    test_cov = ts_cov[val_end:]

    train_data = {'target': train_target, 'covariates': train_cov}
    val_data = {'target': val_target, 'covariates': val_cov}
    test_data = {'target': test_target, 'covariates': test_cov}

    return train_data, val_data, test_data


def normalize_data(train_data, val_data, test_data):
    """
    Normalise les données avec StandardScaler.

    Args:
        train_data: dict avec 'target' et 'covariates'
        val_data: dict avec 'target' et 'covariates'
        test_data: dict avec 'target' et 'covariates'

    Returns:
        tuple: (train_scaled, val_scaled, test_scaled, scalers)
    """
    # Scaler pour target
    scaler_target = Scaler()
    scaler_target.fit(train_data['target'])

    # Scaler pour covariates
    scaler_cov = Scaler()
    scaler_cov.fit(train_data['covariates'])

    # Transform
    train_scaled = {
        'target': scaler_target.transform(train_data['target']),
        'covariates': scaler_cov.transform(train_data['covariates'])
    }
    val_scaled = {
        'target': scaler_target.transform(val_data['target']),
        'covariates': scaler_cov.transform(val_data['covariates'])
    }
    test_scaled = {
        'target': scaler_target.transform(test_data['target']),
        'covariates': scaler_cov.transform(test_data['covariates'])
    }

    scalers = {
        'target': scaler_target,
        'covariates': scaler_cov
    }

    return train_scaled, val_scaled, test_scaled, scalers


@st.cache_data(ttl=1800)  # Cache for 30 minutes
def prepare_data_for_training(station_name: str, train_ratio=0.5, val_ratio=0.1, fill_missing: bool = False):
    """
    Pipeline complet de préparation des données.

    Args:
        station_name: Nom de la station
        train_ratio: Proportion du train set
        val_ratio: Proportion du val set
        fill_missing: Interpoler les valeurs manquantes

    Returns:
        dict avec train_scaled, val_scaled, test_scaled, scalers, raw_data
    """
    # 1. Charger
    ts_data = load_timeseries(station_name, fill_missing=fill_missing)

    # 2. Split
    train_data, val_data, test_data = split_data(
        ts_data['target'],
        ts_data['covariates'],
        train_ratio,
        val_ratio
    )

    # 3. Normaliser
    train_scaled, val_scaled, test_scaled, scalers = normalize_data(
        train_data, val_data, test_data
    )

    return {
        'train_scaled': train_scaled,
        'val_scaled': val_scaled,
        'test_scaled': test_scaled,
        'scalers': scalers,
        'train_raw': train_data,
        'val_raw': val_data,
        'test_raw': test_data
    }


# ============================================================================
# Fonctions pour gérer les données uploadées
# ============================================================================

def get_uploaded_or_default_data(station_name: str = None, fill_missing: bool = False):
    """
    Retourne les données uploadées si disponibles, sinon charge la station par défaut.
    
    Args:
        station_name: Nom de la station par défaut (si pas de données uploadées)
        fill_missing: Interpoler les valeurs manquantes
    
    Returns:
        tuple: (df, source) où source est 'uploaded' ou 'station'
    """
    # Vérifier s'il y a des données uploadées
    if 'uploaded_data' in st.session_state:
        return st.session_state['uploaded_data'], 'uploaded'
    
    # Sinon charger la station par défaut
    if station_name is None:
        raise ValueError("Aucune donnée uploadée et aucune station spécifiée")
    
    df = load_station_data(station_name, fill_missing=fill_missing)
    return df, 'station'


def get_available_data_sources():
    """
    Retourne la liste des sources de données disponibles.
    
    Returns:
        dict avec 'has_uploaded' (bool), 'uploaded_filename' (str ou None), 'default_stations' (list)
    """
    has_uploaded = 'uploaded_data' in st.session_state
    uploaded_filename = st.session_state.get('uploaded_filename', None)
    
    return {
        'has_uploaded': has_uploaded,
        'uploaded_filename': uploaded_filename,
        'default_stations': STATIONS
    }
