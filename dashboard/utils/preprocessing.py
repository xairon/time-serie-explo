"""Preprocessing module for time series with Darts.

MLflow Tracing:
- Key functions are traced via @trace_training_step for visibility
- Traces appear in MLflow UI under the Traces tab
"""

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

from dashboard.utils.mlflow_client import trace_training_step


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
                    fn=lambda x: np.sign(x) * np.log1p(np.abs(x)),
                    inverse_fn=lambda x: np.sign(x) * (np.exp(np.abs(x)) - 1),
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

        Raises:
            ValueError: If series is empty or contains only NaN values
        """
        # Validate input
        if len(series) == 0:
            raise ValueError("Cannot fit_transform on empty series")

        values = series.values()
        if np.isnan(values).all():
            raise ValueError("Series contains only NaN values - cannot fit transformers")

        nan_count = np.isnan(values).sum()
        if nan_count > 0:
            import logging
            logging.getLogger(__name__).warning(
                f"Series contains {nan_count} NaN values before preprocessing"
            )

        transformed = series

        for name, transformer in self.transformers:
            # Some transformers (MissingValuesFiller, InvertibleMapper)
            # only support transform(), not fit_transform()
            if hasattr(transformer, 'fit_transform'):
                transformed = transformer.fit_transform(transformed)
            else:
                # No fitting needed, just transform
                transformed = transformer.transform(transformed)

        # Validate output
        out_values = transformed.values()
        if np.isnan(out_values).any():
            import logging
            nan_after = np.isnan(out_values).sum()
            logging.getLogger(__name__).warning(
                f"Preprocessing produced {nan_after} NaN values"
            )

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
        df = df.ffill()
    elif fill_method == 'Backward fill':
        df = df.bfill()

    # Create target series with fill_missing_dates to handle gaps in RAW data
    target_series = TimeSeries.from_dataframe(
        df,
        value_cols=target_col,
        freq=freq,
        fill_missing_dates=True
    )

    # fill_missing_dates can introduce NaN for dates not present in the original df.
    # Interpolate those NaN values so training doesn't produce NaN losses.
    if target_series.to_dataframe().isna().any().any():
        ts_df = target_series.to_dataframe().interpolate(method='linear').bfill().ffill()
        target_series = TimeSeries.from_dataframe(ts_df, freq=freq)

    # Create covariates if specified
    covariates_series = None
    if covariate_cols and len(covariate_cols) > 0:
        covariates_series = TimeSeries.from_dataframe(
            df,
            value_cols=covariate_cols,
            freq=freq,
            fill_missing_dates=True
        )
        # Same NaN fix for covariates
        if covariates_series.to_dataframe().isna().any().any():
            cov_df = covariates_series.to_dataframe().interpolate(method='linear').bfill().ffill()
            covariates_series = TimeSeries.from_dataframe(cov_df, freq=freq)

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


@trace_training_step("split_data")
def split_train_val_test(
    series: TimeSeries,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    min_train_size: int = 10,
    min_val_size: int = 2,
    min_test_size: int = 1
) -> Tuple[TimeSeries, TimeSeries, TimeSeries]:
    """
    Split temporel strict train/val/test (aucun shuffle, ordre chronologique conservé).

    Args:
        series: TimeSeries à découper
        train_ratio: Part train (ex. 0.7)
        val_ratio: Part validation (ex. 0.15)
        test_ratio: Part test (ex. 0.15)
        min_train_size: Minimum number of samples in train set
        min_val_size: Minimum number of samples in validation set
        min_test_size: Minimum number of samples in test set

    Returns:
        (train, val, test) — tranches contiguës dans le temps

    Raises:
        ValueError: If ratios don't sum to 1.0 or splits are too small
    """
    # Validate ratios
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not (0.99 <= ratio_sum <= 1.01):  # Allow small floating point errors
        raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum:.4f}")

    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("All ratios must be positive")

    n = len(series)

    # Pre-validate that we have enough data
    min_required = min_train_size + min_val_size + min_test_size
    if n < min_required:
        raise ValueError(
            f"Series too short ({n} samples) for split. "
            f"Minimum required: {min_required} (train={min_train_size}, val={min_val_size}, test={min_test_size})"
        )

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = series[:train_end]
    val = series[train_end:val_end]
    test = series[val_end:]

    # Validate split sizes
    if len(train) < min_train_size:
        raise ValueError(
            f"Train split too small ({len(train)} < {min_train_size}). "
            f"Increase data or adjust ratios."
        )
    if len(val) < min_val_size:
        raise ValueError(
            f"Validation split too small ({len(val)} < {min_val_size}). "
            f"Increase data or adjust ratios."
        )
    if len(test) < min_test_size:
        raise ValueError(
            f"Test split too small ({len(test)} < {min_test_size}). "
            f"Increase data or adjust ratios."
        )

    return train, val, test


def compute_data_statistics(series: TimeSeries) -> Dict:
    """
    Calculates descriptive statistics on a series.

    Args:
        series: TimeSeries

    Returns:
        Dict with stats
    """
    if series is None or len(series) == 0:
        return {}

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


def detect_columns_from_config(model_config, data_dict: dict) -> tuple:
    """Detect target column and covariate list from model config and data.

    Args:
        model_config: Model configuration (dict or object with .columns attribute)
        data_dict: Dictionary with 'train', 'val', 'test', 'train_cov', etc.

    Returns:
        Tuple of (target_col: str, covariate_cols: list[str])
    """
    import json as _json

    target_col = None
    covariate_cols = []

    if model_config:
        if isinstance(model_config, dict):
            cols = model_config.get("columns", {})
            if not cols:
                cols = model_config.get("preprocessing", {}).get("columns", {})
        elif hasattr(model_config, "columns"):
            cols = model_config.columns if isinstance(model_config.columns, dict) else {}
        else:
            cols = {}
        target_col = cols.get("target")
        covariate_cols = cols.get("covariates", [])
        if isinstance(covariate_cols, str):
            try:
                covariate_cols = _json.loads(covariate_cols)
            except Exception:
                covariate_cols = [covariate_cols]

    if not target_col and "train" in data_dict:
        df_train = data_dict["train"]
        for c in ["gwl", "Water_Level", "water_level", "niveau_nappe_eau", "piezo", "level"]:
            if c in df_train.columns:
                target_col = c
                break
        if not target_col:
            target_col = df_train.columns[0]

    if not covariate_cols and "train_cov" in data_dict:
        covariate_cols = list(data_dict["train_cov"].columns)

    return target_col, covariate_cols


def build_complete_dataframe(data_dict: dict, target_col: str, covariate_cols: list) -> tuple:
    """Merge train/val/test + covariates into a single DataFrame (NORMALIZED).

    Args:
        data_dict: Dictionary with 'train', 'val', 'test', 'train_cov', etc.
        target_col: Name of the target column
        covariate_cols: List of covariate column names

    Returns:
        Tuple of (full_df: pd.DataFrame or None, available_covariates: list[str])
    """
    parts = []
    for split in ["train", "val", "test"]:
        if split in data_dict and target_col in data_dict[split].columns:
            parts.append(data_dict[split][[target_col]])
    if not parts:
        return None, []

    df_target = pd.concat(parts).sort_index()
    df_target = df_target[~df_target.index.duplicated(keep="first")]

    cov_parts = []
    for split in ["train_cov", "val_cov", "test_cov"]:
        if split in data_dict:
            cov_parts.append(data_dict[split])

    if cov_parts:
        df_cov = pd.concat(cov_parts).sort_index()
        df_cov = df_cov[~df_cov.index.duplicated(keep="first")]
        available_covs = [c for c in covariate_cols if c in df_cov.columns]
        if not available_covs:
            available_covs = list(df_cov.columns)[:3]
        df_full = df_target.join(df_cov[available_covs], how="inner")
        return df_full, available_covs
    else:
        return df_target, []


def denormalize_data(processed_df: pd.DataFrame, scalers: dict, target_col: str) -> pd.DataFrame:
    """Generate raw (de-normalized) data from processed (normalized) data.

    Uses the target_preprocessor from scalers to inverse-transform normalized
    data back to physical units.

    Args:
        processed_df: DataFrame with normalized data
        scalers: Dictionary containing 'target_preprocessor' (a Darts transformer)
        target_col: Name of the target column

    Returns:
        DataFrame with data in original (physical) units
    """
    target_preprocessor = scalers.get('target_preprocessor')
    if target_preprocessor is not None:
        # Convert processed DataFrame to TimeSeries, inverse_transform, convert back
        processed_series, _ = prepare_dataframe_for_darts(
            processed_df, target_col, []
        )
        raw_series = target_preprocessor.inverse_transform(processed_series)
        raw_df = raw_series.to_dataframe()
        raw_df.index = processed_df.index  # Ensure same index
        return raw_df
    else:
        # No scaler = data wasn't normalized, processed is raw
        return processed_df.copy()


def merge_covariates_with_splits(data_dict_in: dict) -> dict:
    """Merge covariate DataFrames into base split DataFrames.

    For each split (train, val, test), joins the corresponding covariate
    DataFrame (train_cov, val_cov, test_cov) into the base DataFrame.

    Args:
        data_dict_in: Dictionary with split DataFrames and optional covariate DataFrames

    Returns:
        New dictionary with covariates merged into base splits
    """
    data_dict_out = data_dict_in.copy()
    for split in ['train', 'val', 'test']:
        cov_key = f"{split}_cov"
        base_df = data_dict_out.get(split)
        cov_df = data_dict_out.get(cov_key)
        if isinstance(base_df, pd.DataFrame) and isinstance(cov_df, pd.DataFrame):
            cov_df = cov_df.drop(columns=['station'], errors='ignore')
            data_dict_out[split] = base_df.join(cov_df, how='left')
    return data_dict_out
