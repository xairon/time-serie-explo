"""Pastas dual validation for counterfactual analysis.

Provides a Pastas TFN (Transfer Function Noise) model wrapper and
dual validation logic for checking TFT-Pastas agreement on CF stresses.

The dual validation principle (from PhysCF paper):
    1. Fit Pastas on training data (independent model from TFT)
    2. After CF generation, simulate Pastas with CF stresses
    3. Compute RMSE(y_cf_tft, y_cf_pastas) = rmse_cf
    4. Compute baseline RMSE(y_factual_tft, y_factual_pastas) = rmse_0
    5. Accept CF if rmse_cf < gamma * rmse_0

Self-contained module: no dependency on PhysCF library.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

try:
    import pastas as ps

    PASTAS_AVAILABLE = True
except ImportError:
    PASTAS_AVAILABLE = False


# ---------------------------------------------------------------------------
# PastasWrapper
# ---------------------------------------------------------------------------

class PastasWrapper:
    """Wrapper for Pastas TFN model with recharge stress.

    Pastas works in physical units (mm/d, m NGF) -- never pass normalized data.

    Methods:
        fit(gwl, precip, evap, tmin, tmax): Calibrate model.
        predict(tmin, tmax): Simulate GWL with calibrated stresses.
        simulate_with_stresses(precip, evap, tmin, tmax): Simulate with new
            stresses using already-calibrated parameters.
        get_response_params(): Get calibrated parameters dict.
    """

    def __init__(self) -> None:
        """Initialize the Pastas wrapper."""
        if not PASTAS_AVAILABLE:
            raise ImportError("pastas is required. Install with: pip install pastas>=1.7")
        self.model: Optional[ps.Model] = None
        self._fitted: bool = False

    def fit(
        self,
        gwl_series: pd.Series,
        precip_series: pd.Series,
        evap_series: pd.Series,
        tmin: Optional[str] = None,
        tmax: Optional[str] = None,
    ) -> None:
        """Calibrate Pastas TFN model.

        Args:
            gwl_series: Groundwater levels (m NGF), index=date.
            precip_series: Precipitation (mm/d), index=date.
            evap_series: Evaporation (mm/d), index=date.
            tmin: Start of calibration window.
            tmax: End of calibration window.
        """
        self.model = ps.Model(gwl_series, name="gwl")

        recharge = ps.RechargeModel(
            precip_series,
            evap_series,
            rfunc=ps.Gamma(),
            name="recharge",
            recharge=ps.rch.FlexModel(),
        )
        self.model.add_stressmodel(recharge)

        self.model.solve(
            tmin=tmin,
            tmax=tmax,
            solver=ps.LeastSquares(),
            report=False,
        )
        self._fitted = True

    def predict(
        self,
        tmin: Optional[str] = None,
        tmax: Optional[str] = None,
    ) -> np.ndarray:
        """Simulate GWL with the calibrated model (factual stresses).

        Args:
            tmin: Start of simulation window.
            tmax: End of simulation window.

        Returns:
            Predicted GWL values (m NGF) as 1-D numpy array.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        sim = self.model.simulate(tmin=tmin, tmax=tmax)
        return sim.values.flatten()

    def simulate_with_stresses(
        self,
        precip_series: pd.Series,
        evap_series: pd.Series,
        tmin: Optional[str] = None,
        tmax: Optional[str] = None,
    ) -> np.ndarray:
        """Simulate with counterfactual stresses without recalibrating.

        Creates a copy of the calibrated model with new stress data and
        applies the same optimal parameters.

        Args:
            precip_series: Counterfactual precipitation (mm/d).
            evap_series: Counterfactual evapotranspiration (mm/d).
            tmin: Start of simulation window.
            tmax: End of simulation window.

        Returns:
            Simulated GWL values (m NGF) as 1-D numpy array.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        params = self.model.parameters["optimal"].copy()

        gwl_series = self.model.oseries.series
        cf_model = ps.Model(gwl_series, name="gwl_cf")
        recharge = ps.RechargeModel(
            precip_series,
            evap_series,
            rfunc=ps.Gamma(),
            name="recharge",
            recharge=ps.rch.FlexModel(),
        )
        cf_model.add_stressmodel(recharge)
        cf_model.parameters["optimal"] = params

        sim = cf_model.simulate(tmin=tmin, tmax=tmax)
        return sim.values.flatten()

    def get_response_params(self) -> dict:
        """Get calibrated response function parameters.

        Returns:
            Dict of parameter name to optimal value.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        return self.model.parameters["optimal"].to_dict()


# ---------------------------------------------------------------------------
# Dual validation core
# ---------------------------------------------------------------------------

def compute_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Compute RMSE between two arrays.

    Args:
        a: First array.
        b: Second array.

    Returns:
        Root mean squared error as float.
    """
    return float(np.sqrt(np.mean((a - b) ** 2)))


def validate_with_pastas(
    s_cf_phys: dict,
    pastas_model: PastasWrapper,
    y_cf_tft: np.ndarray,
    y_factual_tft: np.ndarray,
    y_factual_pastas: np.ndarray,
    gamma: float = 1.5,
) -> dict:
    """Validate counterfactual using dual TFT + Pastas check.

    The CF is accepted if RMSE(y_cf_tft, y_cf_pastas) < gamma * RMSE_baseline,
    where RMSE_baseline = RMSE(y_factual_tft, y_factual_pastas).

    Args:
        s_cf_phys: Dict with 'precip' and 'evap' pd.Series for Pastas simulation.
        pastas_model: Fitted PastasWrapper instance.
        y_cf_tft: TFT CF prediction (denormalized, m NGF).
        y_factual_tft: TFT factual prediction (denormalized, m NGF).
        y_factual_pastas: Pastas factual prediction (m NGF).
        gamma: Tolerance multiplier for baseline disagreement (default 1.5).

    Returns:
        Dict with keys: accepted, rmse_cf, rmse_0, epsilon, y_cf_pastas.
    """
    rmse_0 = compute_rmse(y_factual_tft, y_factual_pastas)
    epsilon = gamma * rmse_0

    try:
        y_cf_pastas = pastas_model.simulate_with_stresses(
            precip_series=s_cf_phys["precip"],
            evap_series=s_cf_phys["evap"],
        )
        min_len = min(len(y_cf_tft), len(y_cf_pastas))
        y_cf_tft_aligned = y_cf_tft[:min_len]
        y_cf_pastas_aligned = y_cf_pastas[:min_len]

        rmse_cf = compute_rmse(y_cf_tft_aligned, y_cf_pastas_aligned)
        accepted = rmse_cf < epsilon
    except Exception as e:
        logger.warning(f"Pastas simulation failed: {e}")
        rmse_cf = float("inf")
        y_cf_pastas = None
        accepted = False

    return {
        "accepted": accepted,
        "rmse_cf": rmse_cf,
        "rmse_0": rmse_0,
        "epsilon": epsilon,
        "y_cf_pastas": y_cf_pastas,
    }


# ---------------------------------------------------------------------------
# TSE data helpers
# ---------------------------------------------------------------------------

def _identify_column(covariate_cols: list[str], keywords: list[str]) -> Optional[str]:
    """Find a covariate column matching any of the given keywords.

    Args:
        covariate_cols: List of column names.
        keywords: Substrings to search for (lowercased comparison).

    Returns:
        The first matching column name, or None.
    """
    for col in covariate_cols:
        cl = col.lower()
        if any(kw in cl for kw in keywords):
            return col
    return None


def build_pastas_series_from_data(
    data_dict: dict,
    target_col: str,
    covariate_cols: list[str],
    mu_target: float,
    sigma_target: float,
    physcf_scaler: dict,
) -> tuple[pd.Series, pd.Series, pd.Series, str]:
    """Extract date-indexed pd.Series for Pastas from TSE data_dict.

    Handles merging train/val/test splits, denormalizing to physical units,
    and identifying precipitation and evaporation columns.

    Args:
        data_dict: TSE data dict with "train", "val", "test", "train_cov", etc.
        target_col: Target variable column name.
        covariate_cols: List of covariate column names.
        mu_target: Target variable mean (for denormalization).
        sigma_target: Target variable standard deviation.
        physcf_scaler: Dict mapping column names to {mean, std}.

    Returns:
        Tuple of (gwl_series, precip_series, evap_series, train_end_date_str).

    Raises:
        ValueError: If precipitation or evaporation columns cannot be identified.
    """
    # --- Merge target splits → GWL in physical units ---
    target_parts = []
    for split in ["train", "val", "test"]:
        if split in data_dict and target_col in data_dict[split].columns:
            target_parts.append(data_dict[split][[target_col]])
    if not target_parts:
        raise ValueError("No target data found in data_dict")
    gwl_norm = pd.concat(target_parts).sort_index()
    gwl_norm = gwl_norm[~gwl_norm.index.duplicated(keep="first")]
    gwl_raw = gwl_norm[target_col] * sigma_target + mu_target
    gwl_raw.name = "gwl"

    # --- Merge covariate splits ---
    cov_parts = []
    for split in ["train_cov", "val_cov", "test_cov"]:
        if split in data_dict:
            cov_parts.append(data_dict[split])
    if not cov_parts:
        raise ValueError("No covariate data found in data_dict")
    df_cov = pd.concat(cov_parts).sort_index()
    df_cov = df_cov[~df_cov.index.duplicated(keep="first")]

    # --- Identify precip and evap columns ---
    precip_col = _identify_column(covariate_cols, ["precip", "rain", "pluie"])
    evap_col = _identify_column(covariate_cols, ["evap", "etp", "pet"])

    if precip_col is None:
        raise ValueError(
            f"Cannot identify precipitation column from {covariate_cols}. "
            "Expected column name containing 'precip', 'rain', or 'pluie'."
        )
    if evap_col is None:
        raise ValueError(
            f"Cannot identify evaporation column from {covariate_cols}. "
            "Expected column name containing 'evap', 'etp', or 'pet'."
        )

    # --- Denormalize covariates to physical units ---
    precip_norm = df_cov[precip_col].copy()
    evap_norm = df_cov[evap_col].copy()

    # Try column-specific scaler first, then canonical name
    for col, series_ref in [(precip_col, "precip_series"), (evap_col, "evap_series")]:
        scaler_entry = physcf_scaler.get(col)
        if scaler_entry is None:
            # Try canonical name mapping
            cl = col.lower()
            if any(kw in cl for kw in ["precip", "rain"]):
                scaler_entry = physcf_scaler.get("precip")
            elif any(kw in cl for kw in ["evap", "etp"]):
                scaler_entry = physcf_scaler.get("evap")

        if scaler_entry and scaler_entry.get("std", 0) > 0:
            if col == precip_col:
                precip_norm = precip_norm * scaler_entry["std"] + scaler_entry["mean"]
            else:
                evap_norm = evap_norm * scaler_entry["std"] + scaler_entry["mean"]

    precip_raw = precip_norm.clip(lower=0)
    precip_raw.name = "precip"
    evap_raw = evap_norm.clip(lower=0)
    evap_raw.name = "evap"

    # --- Determine training end date ---
    train_end = str(data_dict["train"].index.max().date()) if "train" in data_dict else None

    return gwl_raw, precip_raw, evap_raw, train_end


def fit_pastas_for_station(
    gwl_series: pd.Series,
    precip_series: pd.Series,
    evap_series: pd.Series,
    train_end: Optional[str] = None,
) -> Optional[PastasWrapper]:
    """Fit a Pastas model on training data for a single station.

    Args:
        gwl_series: Full GWL series in m NGF, date index.
        precip_series: Precipitation in mm/d, date index.
        evap_series: Evapotranspiration in mm/d, date index.
        train_end: End date of training period (inclusive).

    Returns:
        Fitted PastasWrapper or None if fitting fails.
    """
    if not PASTAS_AVAILABLE:
        logger.warning("pastas is not installed")
        return None

    wrapper = PastasWrapper()
    try:
        wrapper.fit(gwl_series, precip_series, evap_series, tmax=train_end)
        logger.info(
            f"Pastas model fitted successfully "
            f"(tmax={train_end}, {len(gwl_series)} obs)"
        )
        return wrapper
    except Exception as e:
        logger.warning(f"Pastas fitting failed: {e}")
        return None


def cf_stresses_to_pastas_series(
    s_cf_phys,
    lookback_dates: pd.DatetimeIndex,
) -> dict[str, pd.Series]:
    """Convert CF result s_cf_phys to named pd.Series for Pastas.

    Column ordering follows PerturbationLayer.STRESS_COLUMNS:
    [precip, temp, evap] = columns [0, 1, 2].

    Args:
        s_cf_phys: (L, 3) tensor or ndarray in physical units.
        lookback_dates: Date index for the lookback window.

    Returns:
        Dict with 'precip' and 'evap' keys, each a pd.Series with date index.
    """
    if isinstance(s_cf_phys, torch.Tensor):
        arr = s_cf_phys.detach().cpu().numpy()
    else:
        arr = np.asarray(s_cf_phys)

    precip = pd.Series(arr[:, 0], index=lookback_dates, name="precip")
    evap = pd.Series(arr[:, 2], index=lookback_dates, name="evap")

    return {"precip": precip.clip(lower=0), "evap": evap.clip(lower=0)}


def run_dual_validation_for_results(
    results_dict: dict,
    pastas_model: PastasWrapper,
    lookback_dates: pd.DatetimeIndex,
    y_factual_tft_raw: np.ndarray,
    horizon_start: str,
    horizon_end: str,
    mu_target: float,
    sigma_target: float,
    gamma: float = 1.5,
) -> dict:
    """Run dual validation on all CF results.

    For each method that has s_cf_phys, validates TFT-Pastas agreement.

    Args:
        results_dict: {method_name: CF result dict}.
        pastas_model: Fitted PastasWrapper.
        lookback_dates: Date index of the lookback window.
        y_factual_tft_raw: TFT factual prediction in physical units (m NGF).
        horizon_start: Start date string for Pastas simulation.
        horizon_end: End date string for Pastas simulation.
        mu_target: Target mean for denormalization.
        sigma_target: Target std for denormalization.
        gamma: Tolerance multiplier (default 1.5).

    Returns:
        {method_name: validation_dict} with accepted, rmse_cf, rmse_0,
        epsilon, y_cf_pastas.
    """
    # Factual Pastas prediction over the horizon
    try:
        y_factual_pastas = pastas_model.predict(
            tmin=horizon_start, tmax=horizon_end
        )
    except Exception as e:
        logger.warning(f"Pastas factual prediction failed: {e}")
        return {}

    # Align factual predictions
    min_factual = min(len(y_factual_tft_raw), len(y_factual_pastas))
    y_factual_tft_aligned = y_factual_tft_raw[:min_factual]
    y_factual_pastas_aligned = y_factual_pastas[:min_factual]

    validation_results = {}

    for method_name, result in results_dict.items():
        s_cf_phys = result.get("s_cf_phys")
        y_cf = result.get("y_cf")

        if s_cf_phys is None or y_cf is None:
            logger.info(f"Skipping Pastas validation for {method_name}: no s_cf_phys")
            continue

        # Convert CF stresses to pd.Series for Pastas
        cf_series = cf_stresses_to_pastas_series(s_cf_phys, lookback_dates)

        # Denormalize TFT CF prediction to physical units
        if isinstance(y_cf, torch.Tensor):
            y_cf_np = y_cf.detach().cpu().numpy()
        else:
            y_cf_np = np.asarray(y_cf)
        y_cf_raw = y_cf_np * sigma_target + mu_target

        # Validate
        validation = validate_with_pastas(
            s_cf_phys=cf_series,
            pastas_model=pastas_model,
            y_cf_tft=y_cf_raw,
            y_factual_tft=y_factual_tft_aligned,
            y_factual_pastas=y_factual_pastas_aligned,
            gamma=gamma,
        )
        validation_results[method_name] = validation

    return validation_results
