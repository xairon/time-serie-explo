"""Preprocessing module for time series with Darts."""

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
    Class to apply a preprocessing chain on time series.
    Uses Darts transformers to guarantee compatibility.
    """

    def __init__(self, config: Dict):
        """
        Initializes the preprocessor with a configuration.

        Args:
            config: Dictionary with keys:
                - fill_method: str
                - normalization: str
                - transformation: str
                - datetime_features: bool
                - lags: List[int]
        """
        self.config = config
        self.transformers = []
        self.fitted = False

        # Build pipeline
        self._build_pipeline()

    def _build_pipeline(self):
        """Builds the transformer pipeline according to config."""

        # 1. Missing values management
        fill_method = self.config.get('fill_method', 'Supprimer les lignes')

        # MissingValuesFiller only accepts 'auto' for fill
        # We handle missing values BEFORE TimeSeries conversion
        # in prepare_dataframe_for_darts() or by calling dropna() on the TimeSeries

        # If missing values remain, use auto
        if fill_method != 'Supprimer les lignes':
            self.transformers.append(
                ('filler', MissingValuesFiller(fill='auto', name='Missing filler'))
            )
        # If "Supprimer les lignes", we do it with series.dropna() before preprocessing

        # 2. Transformation (before normalization to stabilize variance)
        transformation = self.config.get('transformation', 'Aucune')

        if transformation == 'Log':
            # Log transform with custom mapper
            self.transformers.append(
                ('log', InvertibleMapper(
                    fn=lambda x: np.log(x + 1),  # +1 to avoid log(0)
                    inverse_fn=lambda x: np.exp(x) - 1,
                    name='Log transform'
                ))
            )
        elif transformation == 'BoxCox':
            self.transformers.append(
                ('boxcox', BoxCox(lmbda=None, name='BoxCox'))  # lmbda=None for auto
            )
        elif transformation == 'Différenciation (order 1)':
            self.transformers.append(
                ('diff', Diff(lags=1, dropna=True, name='Differencing'))
            )

        # 3. Normalization (after transformation)
        normalization = self.config.get('normalization', 'Aucune')

        if normalization == 'MinMax (0-1)':
            self.transformers.append(
                ('scaler', Scaler(name='MinMax scaler'))  # Default is MinMaxScaler
            )
        elif normalization == 'StandardScaler (z-score)':
            from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
            self.transformers.append(
                ('scaler', Scaler(scaler=SklearnStandardScaler(), name='Standard scaler'))
            )
        elif normalization == 'RobustScaler (médiane+IQR)':
            from sklearn.preprocessing import RobustScaler as SklearnRobustScaler
            self.transformers.append(
                ('scaler', Scaler(scaler=SklearnRobustScaler(), name='Robust scaler'))
            )

    def fit_transform(self, series: TimeSeries) -> TimeSeries:
        """
        Fits transformers and transforms the series.

        Args:
            series: Darts TimeSeries

        Returns:
            Transformed TimeSeries
        """
        transformed = series

        for name, transformer in self.transformers:
            # Some transformers (MissingValuesFiller, InvertibleMapper)
            # only support transform(), not fit_transform()
            if hasattr(transformer, 'fit_transform'):
                transformed = transformer.fit_transform(transformed)
            else:
                # No fitting needed, just transform
                transformed = transformer.transform(transformed)

        self.fitted = True
        return transformed

    def transform(self, series: TimeSeries) -> TimeSeries:
        """
        Applies transformers (already fitted) to a new series.

        Args:
            series: Darts TimeSeries

        Returns:
            Transformed TimeSeries
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before calling transform()")

        transformed = series

        for name, transformer in self.transformers:
            transformed = transformer.transform(transformed)

        return transformed

    def inverse_transform(self, series: TimeSeries) -> TimeSeries:
        """
        Inverses the transformation (to recover original values).

        Args:
            series: Transformed TimeSeries

        Returns:
            TimeSeries in original scale
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before calling inverse_transform()")

        inversed = series

        # Apply inverses in reverse order
        for name, transformer in reversed(self.transformers):
            # Some transformers are not reversible (MissingValuesFiller)
            if hasattr(transformer, 'inverse_transform'):
                inversed = transformer.inverse_transform(inversed)
            # Else skip (non-reversible transformation)

        return inversed

    def get_scaler(self, name='scaler'):
        """Returns a specific transformer by name."""
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
    Converts a pandas DataFrame to Darts TimeSeries.

    Args:
        df: DataFrame with datetime index
        target_col: Target column
        covariate_cols: Covariate columns
        freq: Frequency ('D', 'H', etc.)
        fill_method: Method for missing values

    Returns:
        Tuple (target_series, covariates_series or None)
    """
    # Manage missing values BEFORE Darts conversion
    if fill_method == 'Supprimer les lignes':
        df = df.dropna()
    elif fill_method == 'Interpolation linéaire':
        df = df.interpolate(method='linear')
    elif fill_method == 'Forward fill':
        df = df.fillna(method='ffill')
    elif fill_method == 'Backward fill':
        df = df.fillna(method='bfill')

    # Create target series with fill_missing_dates to handle gaps in RAW data
    target_series = TimeSeries.from_dataframe(
        df,
        value_cols=target_col,
        freq=freq,
        fill_missing_dates=True
    )

    # Create covariates if specified
    covariates_series = None
    if covariate_cols and len(covariate_cols) > 0:
        covariates_series = TimeSeries.from_dataframe(
            df,
            value_cols=covariate_cols,
            freq=freq,
            fill_missing_dates=True
        )

    return target_series, covariates_series


def add_datetime_features(series: TimeSeries) -> TimeSeries:
    """
    Adds temporal features (day, month, etc.) as future covariates.

    Args:
        series: TimeSeries

    Returns:
        TimeSeries with added temporal features
    """
    df = series.to_dataframe()

    # Extract datetime features
    df['day_of_month'] = df.index.day
    df['day_of_week'] = df.index.dayofweek
    df['day_of_year'] = df.index.dayofyear
    df['week_of_year'] = df.index.isocalendar().week
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

    # Create cyclical features to capture periodicity
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

    # Convert to TimeSeries
    datetime_features = TimeSeries.from_dataframe(
        df.drop(columns=series.columns),  # Keep only new features
        freq=series.freq_str
    )

    return datetime_features


def add_lag_features(
    series: TimeSeries,
    lags: List[int]
) -> TimeSeries:
    """
    Adds lags of the target series as features.

    Args:
        series: Target TimeSeries
        lags: List of lags to add (e.g., [1, 7, 30])

    Returns:
        TimeSeries with added lags
    """
    df = series.to_dataframe()
    lag_df = pd.DataFrame(index=df.index)

    for lag in lags:
        lag_df[f'lag_{lag}'] = df[series.columns[0]].shift(lag)

    # Remove NaNs created by shifts
    lag_df = lag_df.dropna()

    # Convert to TimeSeries
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
    Splits a time series into train/val/test.

    Args:
        series: TimeSeries to split
        train_ratio: Train ratio
        val_ratio: Validation ratio
        test_ratio: Test ratio

    Returns:
        Tuple (train, val, test)
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

    n = len(series)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = series[:train_end]
    val = series[train_end:val_end]
    test = series[val_end:]

    return train, val, test


def compute_data_statistics(series: TimeSeries) -> Dict:
    """
    Calculates descriptive statistics on a series.

    Args:
        series: TimeSeries

    Returns:
        Dict with stats
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
    Automatically detects the frequency of a series.

    Args:
        series: TimeSeries

    Returns:
        Detected frequency ('D', 'H', 'W', etc.)
    """
    if series.freq_str:
        return series.freq_str

    # Infer from DataFrame
    df = series.to_dataframe()
    if len(df) < 2:
        return 'D'  # Default

    inferred_freq = pd.infer_freq(df.index)
    if inferred_freq:
        return inferred_freq

    # Fallback: calculate median difference
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
    Generates a textual summary of the preprocessing configuration.

    Args:
        config: Configuration dict

    Returns:
        Summary in markdown
    """
    summary = "### Preprocessing Configuration\n\n"

    # Missing values
    fill_method = config.get('fill_method', 'Supprimer les lignes')
    summary += f"**Missing Values**: {fill_method}\n\n"

    # Transformation
    transformation = config.get('transformation', 'Aucune')
    summary += f"**Transformation**: {transformation}\n\n"

    # Normalization
    normalization = config.get('normalization', 'Aucune')
    summary += f"**Normalization**: {normalization}\n\n"

    # Temporal features
    datetime_features = config.get('datetime_features', False)
    summary += f"**Temporal Features**: {'Yes' if datetime_features else 'No'}\n\n"

    # Lags
    lags = config.get('lags', [])
    if lags:
        summary += f"**Added Lags**: {lags}\n\n"
    else:
        summary += f"**Added Lags**: None\n\n"

    return summary
