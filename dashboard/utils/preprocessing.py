"""Module de preprocessing pour séries temporelles avec Darts."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from darts import TimeSeries
from darts.dataprocessing.transformers import (
    Scaler,
    MissingValuesFiller,
    InvertibleMapper,
    StaticCovariatesTransformer
)
from darts.dataprocessing.transformers.boxcox import BoxCox
from darts.dataprocessing.transformers.diff import Diff


class TimeSeriesPreprocessor:
    """
    Classe pour appliquer une chaîne de preprocessing sur des séries temporelles.
    Utilise les transformers Darts pour garantir la compatibilité.
    """

    def __init__(self, config: Dict):
        """
        Initialise le preprocessor avec une configuration.

        Args:
            config: Dictionnaire avec les clés:
                - fill_method: str
                - normalization: str
                - transformation: str
                - datetime_features: bool
                - lags: List[int]
        """
        self.config = config
        self.transformers = []
        self.fitted = False

        # Construire la pipeline
        self._build_pipeline()

    def _build_pipeline(self):
        """Construit la pipeline de transformers selon la config."""

        # 1. Gestion des valeurs manquantes
        fill_method = self.config.get('fill_method', 'Supprimer les lignes')

        # MissingValuesFiller n'accepte que 'auto' pour fill
        # On gérera les valeurs manquantes AVANT la conversion en TimeSeries
        # dans prepare_dataframe_for_darts() ou en appelant dropna() sur le TimeSeries

        # Si des valeurs manquantes subsistent, utiliser auto
        if fill_method != 'Supprimer les lignes':
            self.transformers.append(
                ('filler', MissingValuesFiller(fill='auto', name='Missing filler'))
            )
        # Si "Supprimer les lignes", on le fera avec series.dropna() avant preprocessing

        # 2. Transformation (avant normalisation pour stabiliser variance)
        transformation = self.config.get('transformation', 'Aucune')

        if transformation == 'Log':
            # Log transform avec mapper custom
            self.transformers.append(
                ('log', InvertibleMapper(
                    fn=lambda x: np.log(x + 1),  # +1 pour éviter log(0)
                    inverse_fn=lambda x: np.exp(x) - 1,
                    name='Log transform'
                ))
            )
        elif transformation == 'BoxCox':
            self.transformers.append(
                ('boxcox', BoxCox(lmbda=None, name='BoxCox'))  # lmbda=None pour auto
            )
        elif transformation == 'Différenciation (order 1)':
            self.transformers.append(
                ('diff', Diff(lags=1, dropna=True, name='Differencing'))
            )

        # 3. Normalisation (après transformation)
        normalization = self.config.get('normalization', 'Aucune')

        if normalization == 'MinMax (0-1)':
            self.transformers.append(
                ('scaler', Scaler(name='MinMax scaler'))
            )
        elif normalization == 'StandardScaler (z-score)':
            self.transformers.append(
                ('scaler', Scaler(scaler=None, name='Standard scaler'))  # Default = StandardScaler
            )
        elif normalization == 'RobustScaler (médiane+IQR)':
            from sklearn.preprocessing import RobustScaler as SklearnRobustScaler
            self.transformers.append(
                ('scaler', Scaler(scaler=SklearnRobustScaler(), name='Robust scaler'))
            )

    def fit_transform(self, series: TimeSeries) -> TimeSeries:
        """
        Fit les transformers et transforme la série.

        Args:
            series: TimeSeries Darts

        Returns:
            TimeSeries transformée
        """
        transformed = series

        for name, transformer in self.transformers:
            # Certains transformers (MissingValuesFiller, InvertibleMapper)
            # ne supportent que transform(), pas fit_transform()
            if hasattr(transformer, 'fit_transform'):
                transformed = transformer.fit_transform(transformed)
            else:
                # Pas de fitting nécessaire, juste transform
                transformed = transformer.transform(transformed)

        self.fitted = True
        return transformed

    def transform(self, series: TimeSeries) -> TimeSeries:
        """
        Applique les transformers (déjà fitted) à une nouvelle série.

        Args:
            series: TimeSeries Darts

        Returns:
            TimeSeries transformée
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before calling transform()")

        transformed = series

        for name, transformer in self.transformers:
            transformed = transformer.transform(transformed)

        return transformed

    def inverse_transform(self, series: TimeSeries) -> TimeSeries:
        """
        Inverse la transformation (pour récupérer les valeurs originales).

        Args:
            series: TimeSeries transformée

        Returns:
            TimeSeries dans l'échelle originale
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before calling inverse_transform()")

        inversed = series

        # Appliquer les inverses dans l'ordre inverse
        for name, transformer in reversed(self.transformers):
            # Certains transformers ne sont pas réversibles (MissingValuesFiller)
            if hasattr(transformer, 'inverse_transform'):
                inversed = transformer.inverse_transform(inversed)
            # Sinon on skip (transformation non-réversible)

        return inversed

    def get_scaler(self, name='scaler'):
        """Retourne un transformer spécifique par nom."""
        for tf_name, transformer in self.transformers:
            if tf_name == name:
                return transformer
        return None


def prepare_dataframe_for_darts(
    df: pd.DataFrame,
    target_col: str,
    covariate_cols: Optional[List[str]] = None,
    freq: Optional[str] = None,
    fill_method: str = 'Supprimer les lignes'
) -> Tuple[TimeSeries, Optional[TimeSeries]]:
    """
    Convertit un DataFrame pandas en TimeSeries Darts.

    Args:
        df: DataFrame avec index temporel
        target_col: Colonne cible
        covariate_cols: Colonnes covariables
        freq: Fréquence ('D', 'H', etc.)
        fill_method: Méthode pour valeurs manquantes

    Returns:
        Tuple (target_series, covariates_series ou None)
    """
    # Gestion valeurs manquantes AVANT conversion Darts
    if fill_method == 'Supprimer les lignes':
        df = df.dropna()
    elif fill_method == 'Interpolation linéaire':
        df = df.interpolate(method='linear')
    elif fill_method == 'Forward fill':
        df = df.fillna(method='ffill')
    elif fill_method == 'Backward fill':
        df = df.fillna(method='bfill')

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


def add_datetime_features(series: TimeSeries) -> TimeSeries:
    """
    Ajoute des features temporelles (jour, mois, etc.) comme covariables futures.

    Args:
        series: TimeSeries

    Returns:
        TimeSeries avec features temporelles ajoutées
    """
    df = series.to_dataframe()

    # Extraire features datetime
    df['day_of_month'] = df.index.day
    df['day_of_week'] = df.index.dayofweek
    df['day_of_year'] = df.index.dayofyear
    df['week_of_year'] = df.index.isocalendar().week
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

    # Créer cyclical features pour capturer la périodicité
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

    # Convertir en TimeSeries
    datetime_features = TimeSeries.from_dataframe(
        df.drop(columns=series.columns),  # Garder seulement les nouvelles features
        freq=series.freq_str
    )

    return datetime_features


def add_lag_features(
    series: TimeSeries,
    lags: List[int]
) -> TimeSeries:
    """
    Ajoute des lags de la série cible comme features.

    Args:
        series: TimeSeries cible
        lags: Liste des lags à ajouter (ex: [1, 7, 30])

    Returns:
        TimeSeries avec lags ajoutés
    """
    df = series.to_dataframe()
    lag_df = pd.DataFrame(index=df.index)

    for lag in lags:
        lag_df[f'lag_{lag}'] = df[series.columns[0]].shift(lag)

    # Supprimer les NaN créés par les shifts
    lag_df = lag_df.dropna()

    # Convertir en TimeSeries
    lag_series = TimeSeries.from_dataframe(
        lag_df,
        freq=series.freq_str
    )

    return lag_series


def split_train_val_test(
    series: TimeSeries,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[TimeSeries, TimeSeries, TimeSeries]:
    """
    Split une série temporelle en train/val/test.

    Args:
        series: TimeSeries à splitter
        train_ratio: Ratio du train
        val_ratio: Ratio de la validation
        test_ratio: Ratio du test

    Returns:
        Tuple (train, val, test)
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Les ratios doivent sommer à 1.0"

    n = len(series)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = series[:train_end]
    val = series[train_end:val_end]
    test = series[val_end:]

    return train, val, test


def compute_data_statistics(series: TimeSeries) -> Dict:
    """
    Calcule des statistiques descriptives sur une série.

    Args:
        series: TimeSeries

    Returns:
        Dict avec stats
    """
    df = series.to_dataframe()

    stats = {
        'n_samples': len(df),
        'start_date': str(df.index[0]),
        'end_date': str(df.index[-1]),
        'duration_days': (df.index[-1] - df.index[0]).days,
        'mean': float(df.mean().iloc[0]),
        'std': float(df.std().iloc[0]),
        'min': float(df.min().iloc[0]),
        'max': float(df.max().iloc[0]),
        'missing_values': int(df.isnull().sum().sum()),
        'missing_pct': float(df.isnull().sum().sum() / len(df) * 100)
    }

    return stats


def detect_frequency(series: TimeSeries) -> str:
    """
    Détecte automatiquement la fréquence d'une série.

    Args:
        series: TimeSeries

    Returns:
        Fréquence détectée ('D', 'H', 'W', etc.)
    """
    if series.freq_str:
        return series.freq_str

    # Inférer depuis le DataFrame
    df = series.to_dataframe()
    if len(df) < 2:
        return 'D'  # Default

    inferred_freq = pd.infer_freq(df.index)
    if inferred_freq:
        return inferred_freq

    # Fallback: calculer la différence médiane
    diffs = df.index.to_series().diff().dropna()
    median_diff = diffs.median()

    if median_diff <= pd.Timedelta(hours=1):
        return 'H'
    elif median_diff <= pd.Timedelta(days=1):
        return 'D'
    elif median_diff <= pd.Timedelta(weeks=1):
        return 'W'
    elif median_diff <= pd.Timedelta(days=31):
        return 'M'
    else:
        return 'D'  # Default


def get_preprocessing_summary(config: Dict) -> str:
    """
    Génère un résumé textuel de la configuration preprocessing.

    Args:
        config: Dict de configuration

    Returns:
        Résumé en markdown
    """
    summary = "### Configuration du Preprocessing\n\n"

    # Valeurs manquantes
    fill_method = config.get('fill_method', 'Supprimer les lignes')
    summary += f"**Valeurs manquantes** : {fill_method}\n\n"

    # Transformation
    transformation = config.get('transformation', 'Aucune')
    summary += f"**Transformation** : {transformation}\n\n"

    # Normalisation
    normalization = config.get('normalization', 'Aucune')
    summary += f"**Normalisation** : {normalization}\n\n"

    # Features temporelles
    datetime_features = config.get('datetime_features', False)
    summary += f"**Features temporelles** : {'Oui' if datetime_features else 'Non'}\n\n"

    # Lags
    lags = config.get('lags', [])
    if lags:
        summary += f"**Lags ajoutés** : {lags}\n\n"
    else:
        summary += f"**Lags ajoutés** : Aucun\n\n"

    return summary
