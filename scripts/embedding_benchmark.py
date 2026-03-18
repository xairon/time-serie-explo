#!/usr/bin/env python3
"""Embedding Benchmark — Evaluate and compare embedding methods.

Evaluates the 4 embedding spaces (piezo×{uni,multi}, hydro×{uni,multi})
stored in PostgreSQL using intrinsic, supervised, and domain-specific metrics.

Modes:
    evaluate  — Evaluate existing DB embeddings (Phase 1)
    compare   — Multi-method comparison: DB embeddings + Catch22, PCA brut, Random

Usage:
    python scripts/embedding_benchmark.py --mode evaluate
    python scripts/embedding_benchmark.py --mode compare --spaces piezo/uni
    python scripts/embedding_benchmark.py --mode compare --output reports/my_run
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.spatial.distance
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    r2_score,
)
from sklearn.metrics.pairwise import cosine_distances, haversine_distances
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_URL = "postgresql://postgres:postgres_default_pass_2024@brgm-postgres:5432/postgres"

# BDLISA code → label (reproduced from dashboard/utils/latent_space.py)
_MILIEU_EH: dict[str, str] = {
    "1": "Poreux",
    "2": "Fissuré",
    "3": "Karstique",
    "4": "Double porosité fissuré et poreux",
    "5": "Double porosité karstique et poreux",
    "6": "Double porosité karstique et fissuré",
    "8": "Milieu composite",
    "9": "Milieu non applicable",
    "X": "Indéterminé",
}

_THEME_EH: dict[str, str] = {
    "0": "Indifférencié",
    "1": "Sédimentaire",
    "2": "Sédimentaire",
    "3": "Socle",
    "4": "Volcanique",
    "5": "Alluvial",
}


def _decode_eh(code: str | None, mapping: dict[str, str]) -> str | None:
    if code is None:
        return None
    return mapping.get(str(code), str(code))


def parse_pgvector(raw: str) -> np.ndarray:
    """Convert pgvector text representation '[0.1,0.2,...]' to numpy array."""
    return np.array([float(x) for x in raw.strip("[]").split(",")], dtype=np.float32)


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------


def load_embeddings(
    engine, domain: str, space: str
) -> tuple[np.ndarray, list[str], dict]:
    """Load station embeddings + metadata from DB.

    Returns
    -------
    embeddings : (N, D) float32 matrix
    station_ids : list of station codes
    metadata : dict of lists aligned with station_ids
    """
    if domain == "piezo":
        sql = text("""
            SELECT
                e.code_bss            AS id,
                e.embedding::text     AS embedding_raw,
                tme.milieu_eh,
                tme.theme_eh,
                tme.libelle_eh,
                s.code_departement    AS departement,
                s.altitude_station    AS altitude,
                s.latitude,
                s.longitude,
                s.profondeur_moyenne_globale  AS depth,
                s.niveau_stddev_global       AS stddev,
                s.amplitude_totale           AS amplitude,
                s.nb_mois_total              AS record_months,
                e.n_windows,
                e.last_date
            FROM ml.piezo_station_embeddings e
            JOIN gold.dim_piezo_stations s ON e.code_bss = s.code_bss
            LEFT JOIN (
                SELECT DISTINCT ON (m.code_bss)
                    m.code_bss, m.libelle_eh, m.milieu_eh, m.theme_eh,
                    m.etat_eh, m.nature_eh
                FROM gold.int_station_era5_mapping m
                ORDER BY m.code_bss
            ) tme ON tme.code_bss = e.code_bss
            WHERE e.space = :space
        """)
    else:  # hydro
        sql = text("""
            SELECT
                e.code_station        AS id,
                e.embedding::text     AS embedding_raw,
                s.code_departement    AS departement,
                s.nom_cours_eau,
                e.n_windows,
                e.last_date
            FROM ml.hydro_station_embeddings e
            JOIN gold.dim_hydro_stations s ON e.code_station = s.code_station
            WHERE e.space = :space
        """)

    with engine.connect() as conn:
        rows = conn.execute(sql, {"space": space}).fetchall()

    if not rows:
        raise ValueError(f"No embeddings found for {domain}/{space}")

    station_ids: list[str] = []
    embeddings_list: list[np.ndarray] = []
    meta_keys = [
        "milieu_eh", "theme_eh", "departement", "altitude",
        "latitude", "longitude", "depth", "stddev", "amplitude",
    ]
    metadata: dict[str, list] = {k: [] for k in meta_keys}

    for row in rows:
        r = row._mapping
        station_ids.append(r["id"])
        embeddings_list.append(parse_pgvector(r["embedding_raw"]))

        if domain == "piezo":
            metadata["milieu_eh"].append(
                _decode_eh(r.get("milieu_eh"), _MILIEU_EH)
            )
            metadata["theme_eh"].append(
                _decode_eh(r.get("theme_eh"), _THEME_EH)
            )
            for fld in ("altitude", "latitude", "longitude", "depth", "stddev", "amplitude"):
                metadata[fld].append(
                    float(r[fld]) if r.get(fld) is not None else None
                )
        else:
            for fld in ("milieu_eh", "theme_eh", "altitude", "latitude",
                        "longitude", "depth", "stddev", "amplitude"):
                metadata[fld].append(None)

        metadata["departement"].append(r.get("departement"))

    embeddings = np.stack(embeddings_list)
    return embeddings, station_ids, metadata


# ---------------------------------------------------------------------------
# 1b. Raw time series loading (for on-the-fly embedding computation)
# ---------------------------------------------------------------------------


def load_raw_series(
    engine, domain: str, station_ids: list[str], max_stations: int = 500
) -> dict[str, np.ndarray]:
    """Load raw daily time series for a subset of stations.

    Returns {station_id: (T,) array of daily values}.
    NaN-fills gaps, then interpolates linearly.
    Subsamples to max_stations for speed.
    """
    if not station_ids:
        return {}

    # Subsample if needed (0 = no limit)
    if max_stations > 0 and len(station_ids) > max_stations:
        rng = np.random.RandomState(42)
        station_ids = list(rng.choice(station_ids, max_stations, replace=False))

    if domain == "piezo":
        sql = text("""
            SELECT code_bss AS id, date AS date, niveau_nappe_eau AS value
            FROM gold.hubeau_daily_chroniques
            WHERE code_bss = ANY(:ids)
            ORDER BY code_bss, date
        """)
    else:
        # Try hydro table -- name may vary
        sql = text("""
            SELECT code_station AS id, date_obs AS date, resultat_obs_elab AS value
            FROM gold.hydro_daily_chroniques
            WHERE code_station = ANY(:ids)
            ORDER BY code_station, date_obs
        """)

    try:
        with engine.connect() as conn:
            rows = conn.execute(sql, {"ids": station_ids}).fetchall()
    except Exception as e:
        print(f"  WARNING: Could not load raw series for {domain}: {e}")
        return {}

    if not rows:
        print(f"  WARNING: No raw series data returned for {domain}")
        return {}

    # Build DataFrame and pivot
    df = pd.DataFrame(
        [(r[0], r[1], r[2]) for r in rows], columns=["id", "date", "value"]
    )
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    result: dict[str, np.ndarray] = {}
    for sid, group in df.groupby("id"):
        ts = group.set_index("date")["value"].sort_index()
        # Resample to daily, fill gaps, interpolate
        ts = ts.resample("D").mean()
        ts = ts.interpolate(method="linear", limit_direction="both")
        if len(ts) < 365:
            continue
        result[str(sid)] = ts.values.astype(np.float64)

    print(f"  Loaded raw series for {len(result)}/{len(station_ids)} stations")
    return result


def load_raw_series_multi(
    engine, domain: str, station_ids: list[str], max_stations: int = 0
) -> dict[str, np.ndarray]:
    """Load multivariate daily series (target + 3 ERA5 covariates).

    Returns {station_id: (T, 4) array} with columns:
      [niveau_nappe_eau, temperature_2m, total_precipitation, potential_evaporation]
    """
    if not station_ids:
        return {}
    if max_stations > 0 and len(station_ids) > max_stations:
        rng = np.random.RandomState(42)
        station_ids = list(rng.choice(station_ids, max_stations, replace=False))

    if domain != "piezo":
        print("  WARNING: multivariate loading only implemented for piezo")
        return {}

    sql = text("""
        SELECT code_bss AS id, date,
               niveau_nappe_eau, temperature_2m,
               total_precipitation, potential_evaporation
        FROM gold.hubeau_daily_chroniques
        WHERE code_bss = ANY(:ids)
        ORDER BY code_bss, date
    """)

    try:
        with engine.connect() as conn:
            rows = conn.execute(sql, {"ids": station_ids}).fetchall()
    except Exception as e:
        print(f"  WARNING: Could not load multi series: {e}")
        return {}

    if not rows:
        return {}

    df = pd.DataFrame(
        [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows],
        columns=["id", "date", "target", "temp", "precip", "evap"],
    )
    df["date"] = pd.to_datetime(df["date"])
    for col in ("target", "temp", "precip", "evap"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    result: dict[str, np.ndarray] = {}
    for sid, group in df.groupby("id"):
        g = group.set_index("date").sort_index()
        g = g[["target", "temp", "precip", "evap"]].resample("D").mean()
        g = g.interpolate(method="linear", limit_direction="both")
        if len(g) < 365 or g.isna().any(axis=None):
            # Drop stations with any remaining NaN in covariates
            g = g.dropna()
            if len(g) < 365:
                continue
        result[str(sid)] = g.values.astype(np.float64)

    print(f"  Loaded multi series for {len(result)}/{len(station_ids)} stations")
    return result


# ---------------------------------------------------------------------------
# 1c. On-the-fly embedding methods
# ---------------------------------------------------------------------------


def compute_catch22_embeddings(
    raw_series: dict[str, np.ndarray],
    window_size: int = 365,
    stride: int = 90,
) -> tuple[np.ndarray, list[str]]:
    """Compute Catch22 features from raw series windows.

    For each station: extract 365-day windows, compute 22 features per window,
    mean pool across windows -> 22D embedding per station.
    Z-score normalize each series before feature extraction.
    """
    try:
        import pycatch22
    except ImportError:
        print("  WARNING: pycatch22 not installed -- skipping Catch22")
        return np.array([]), []

    embeddings = []
    valid_ids = []
    total = len(raw_series)
    for i, (sid, series) in enumerate(raw_series.items()):
        if (i + 1) % 100 == 0:
            print(f"    Catch22: {i + 1}/{total} stations...")
        # Z-score normalize
        s = (series - np.nanmean(series)) / (np.nanstd(series) + 1e-8)
        # Extract windows
        window_features = []
        T = len(s)
        for start in range(0, T - window_size + 1, stride):
            window = s[start : start + window_size]
            if np.isnan(window).mean() > 0.1:
                continue
            window = np.nan_to_num(window, nan=0.0)
            catch_result = pycatch22.catch22_all(window.tolist())
            window_features.append(catch_result["values"])

        if window_features:
            mean_feat = np.mean(window_features, axis=0)
            # Replace NaN/inf from pycatch22 edge cases
            mean_feat = np.nan_to_num(mean_feat, nan=0.0, posinf=0.0, neginf=0.0)
            embeddings.append(mean_feat)
            valid_ids.append(sid)

    if not embeddings:
        return np.array([]), []
    return np.array(embeddings, dtype=np.float32), valid_ids



def compute_pca_brut_embeddings(
    raw_series: dict[str, np.ndarray],
    window_size: int = 365,
    stride: int = 90,
    target_dim: int = 64,
) -> tuple[np.ndarray, list[str]]:
    """PCA directly on raw windows (no encoder). Baseline.

    Handles both univariate (T,) and multivariate (T, C) series.
    Multivariate windows are flattened to (T*C,) before PCA.
    """
    windows = []
    window_ids = []

    for sid, series in raw_series.items():
        if series.ndim == 2:
            s = (series - np.nanmean(series, axis=0)) / (np.nanstd(series, axis=0) + 1e-8)
        else:
            s = (series - np.nanmean(series)) / (np.nanstd(series) + 1e-8)
        T = len(s)
        for start in range(0, T - window_size + 1, stride):
            window = s[start : start + window_size]
            if np.isnan(window).mean() > 0.1:
                continue
            window = np.nan_to_num(window, nan=0.0)
            # Flatten multivariate windows for PCA
            windows.append(window.flatten())
            window_ids.append(sid)

    if not windows:
        return np.array([]), []

    X = np.array(windows, dtype=np.float32)
    n_components = min(target_dim, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    # Mean pool by station
    station_embs: dict[str, list[np.ndarray]] = defaultdict(list)
    for emb, sid in zip(X_pca, window_ids):
        station_embs[sid].append(emb)

    valid_ids = sorted(station_embs.keys())
    embeddings = np.array(
        [np.mean(station_embs[sid], axis=0) for sid in valid_ids], dtype=np.float32
    )

    return embeddings, valid_ids


def compute_minirocket_embeddings(
    raw_series: dict[str, np.ndarray],
    multivariate: bool = False,
    window_size: int = 365,
    stride: int = 90,
    target_dim: int = 320,
) -> tuple[np.ndarray, list[str]]:
    """MiniRocket embeddings (aeon) — deterministic random convolutional features.

    Works for both univariate and multivariate input.
    For uni: raw_series values are (T,) arrays.
    For multi: raw_series values are (T, C) arrays.

    Steps: extract windows → MiniRocket 9996D features → PCA to target_dim → mean pool.
    """
    from aeon.transformations.collection.convolution_based import MiniRocket

    windows = []
    window_ids = []
    total = len(raw_series)

    for i, (sid, series) in enumerate(raw_series.items()):
        if (i + 1) % 500 == 0:
            print(f"    MiniRocket: windowing {i + 1}/{total}...")

        if multivariate and series.ndim == 2:
            # Z-score each channel independently
            s = (series - np.nanmean(series, axis=0)) / (np.nanstd(series, axis=0) + 1e-8)
        else:
            s = (series - np.nanmean(series)) / (np.nanstd(series) + 1e-8)
            if s.ndim == 1:
                s = s[:, np.newaxis]  # (T,) -> (T, 1)

        T = len(s)
        for start in range(0, T - window_size + 1, stride):
            window = s[start : start + window_size]
            if np.isnan(window).any():
                nan_frac = np.isnan(window).mean()
                if nan_frac > 0.1:
                    continue
                window = np.nan_to_num(window, nan=0.0)
            windows.append(window)
            window_ids.append(sid)

    if not windows:
        return np.array([]), []

    # MiniRocket expects (n_instances, n_channels, n_timepoints)
    n_windows = len(windows)
    n_channels = windows[0].shape[1] if windows[0].ndim == 2 else 1
    print(f"    MiniRocket: {n_windows} windows ({n_channels} channels)...")

    # Fit on a sample (MiniRocket only needs to see the shape, not all data)
    sample_size = min(5000, n_windows)
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(n_windows, sample_size, replace=False)
    X_sample = np.array([windows[i] for i in sample_idx], dtype=np.float64)
    X_sample = np.transpose(X_sample, (0, 2, 1)) if X_sample.ndim == 3 else X_sample[:, np.newaxis, :]

    mr = MiniRocket(random_state=42)
    mr.fit(X_sample)

    # Transform in batches to avoid OOM, then mean-pool per station directly
    # Instead of storing all 9996D features, we transform + aggregate on the fly
    batch_size = 10000
    station_feature_sums: dict[str, np.ndarray] = {}
    station_feature_counts: dict[str, int] = defaultdict(int)

    for batch_start in range(0, n_windows, batch_size):
        batch_end = min(batch_start + batch_size, n_windows)
        X_batch = np.array(windows[batch_start:batch_end], dtype=np.float64)
        if X_batch.ndim == 2:
            X_batch = X_batch[:, np.newaxis, :]  # (N, 1, T) for univariate
        else:
            X_batch = np.transpose(X_batch, (0, 2, 1))  # (N, C, T)

        feat_batch = mr.transform(X_batch)  # (batch, 9996)

        # Accumulate per station
        for i, feat in enumerate(feat_batch):
            sid = window_ids[batch_start + i]
            if sid not in station_feature_sums:
                station_feature_sums[sid] = np.zeros(feat.shape[0], dtype=np.float64)
            station_feature_sums[sid] += feat
            station_feature_counts[sid] += 1

        if batch_end % 50000 == 0 or batch_end == n_windows:
            print(f"    MiniRocket: transformed {batch_end}/{n_windows} windows")

    # Mean pool → per-station 9996D features
    valid_ids = sorted(station_feature_sums.keys())
    station_features = np.array(
        [station_feature_sums[sid] / station_feature_counts[sid] for sid in valid_ids],
        dtype=np.float32,
    )

    # PCA to target_dim
    n_comp = min(target_dim, station_features.shape[1], station_features.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=42)
    embeddings = pca.fit_transform(station_features).astype(np.float32)

    return embeddings, valid_ids


def compute_ts2vec_embeddings(
    raw_series: dict[str, np.ndarray],
    multivariate: bool = False,
    window_size: int = 365,
    stride: int = 90,
    output_dims: int = 320,
    n_epochs: int = 50,
) -> tuple[np.ndarray, list[str]]:
    """TS2Vec contrastive embeddings — same family as SoftCLT.

    Trains a contrastive encoder on the provided windows, then encodes.
    For uni: input_dims=1. For multi: input_dims=C.
    """
    import torch
    from ts2vec import TS2Vec

    device = "cuda" if torch.cuda.is_available() else "cpu"

    windows = []
    window_ids = []
    total = len(raw_series)

    for i, (sid, series) in enumerate(raw_series.items()):
        if (i + 1) % 500 == 0:
            print(f"    TS2Vec: windowing {i + 1}/{total}...")

        if multivariate and series.ndim == 2:
            s = (series - np.nanmean(series, axis=0)) / (np.nanstd(series, axis=0) + 1e-8)
        else:
            s = (series - np.nanmean(series)) / (np.nanstd(series) + 1e-8)
            if s.ndim == 1:
                s = s[:, np.newaxis]

        T = len(s)
        for start in range(0, T - window_size + 1, stride):
            window = s[start : start + window_size]
            if np.isnan(window).mean() > 0.1:
                continue
            window = np.nan_to_num(window, nan=0.0)
            windows.append(window)
            window_ids.append(sid)

    if not windows:
        return np.array([]), []

    n_windows = len(windows)
    input_dims = windows[0].shape[1] if windows[0].ndim == 2 else 1

    # Train on a sample to avoid OOM (TS2Vec only needs representative data)
    max_train = min(20000, n_windows)
    rng = np.random.RandomState(42)
    train_idx = rng.choice(n_windows, max_train, replace=False)
    X_train = np.array([windows[i] for i in train_idx], dtype=np.float32)

    print(f"    TS2Vec: training on {max_train}/{n_windows} windows "
          f"({input_dims} dims, {n_epochs} epochs, {device})...")
    model = TS2Vec(
        input_dims=input_dims,
        output_dims=output_dims,
        device=device,
    )
    model.fit(X_train, n_epochs=n_epochs, verbose=False)

    # Encode in batches and aggregate per station on the fly
    print(f"    TS2Vec: encoding {n_windows} windows in batches...")
    station_emb_sums: dict[str, np.ndarray] = {}
    station_emb_counts: dict[str, int] = defaultdict(int)
    batch_size = 5000

    for batch_start in range(0, n_windows, batch_size):
        batch_end = min(batch_start + batch_size, n_windows)
        X_batch = np.array(windows[batch_start:batch_end], dtype=np.float32)

        raw_emb = model.encode(X_batch)  # (batch, T, D)
        emb_pooled = raw_emb.mean(axis=1)  # (batch, D) — mean over time

        for i, emb in enumerate(emb_pooled):
            sid = window_ids[batch_start + i]
            if sid not in station_emb_sums:
                station_emb_sums[sid] = np.zeros(output_dims, dtype=np.float64)
            station_emb_sums[sid] += emb
            station_emb_counts[sid] += 1

        if batch_end % 20000 == 0 or batch_end == n_windows:
            print(f"    TS2Vec: encoded {batch_end}/{n_windows}")

    valid_ids = sorted(station_emb_sums.keys())
    embeddings = np.array(
        [station_emb_sums[sid] / station_emb_counts[sid] for sid in valid_ids],
        dtype=np.float32,
    )
    return embeddings, valid_ids


def compute_random_embeddings(
    station_ids: list[str], dim: int = 64
) -> tuple[np.ndarray, list[str]]:
    """Random Gaussian embeddings. Lower bound baseline."""
    rng = np.random.RandomState(42)
    return rng.randn(len(station_ids), dim).astype(np.float32), list(station_ids)


def compute_dynamic_typology(
    raw_series: dict[str, np.ndarray],
) -> dict[str, str]:
    """Compute a 3-class dynamic typology from raw series: inertial / annual / mixed.

    Inspired by Baulon et al. classification of groundwater hydrographs:
    - inertial: slow response, high autocorrelation (lag-365 ACF > 0.7)
    - annual: strong yearly cycle (spectral peak at 1/365, lag-365 ACF 0.3–0.7)
    - mixed/reactive: fast response, low autocorrelation (lag-365 ACF < 0.3)

    This is computed directly from the data, no external labels needed.
    """
    typology: dict[str, str] = {}
    for sid, series in raw_series.items():
        s = series.copy()
        s = (s - np.nanmean(s)) / (np.nanstd(s) + 1e-8)
        s = np.nan_to_num(s, nan=0.0)
        T = len(s)
        if T < 730:  # need at least 2 years
            continue

        # Lag-365 autocorrelation
        lag = 365
        if T > lag:
            acf_365 = float(np.corrcoef(s[:-lag], s[lag:])[0, 1])
        else:
            acf_365 = 0.0

        if acf_365 > 0.7:
            typology[sid] = "inertial"
        elif acf_365 > 0.3:
            typology[sid] = "annual"
        else:
            typology[sid] = "reactive"

    return typology


# ---------------------------------------------------------------------------
# 2. Intrinsic metrics
# ---------------------------------------------------------------------------


def compute_participation_ratio(embeddings: np.ndarray) -> float:
    """PR = (sum lambda)^2 / sum(lambda^2), where lambda are covariance eigenvalues."""
    try:
        # Remove constant features (zero variance) to avoid singular covariance
        variances = np.var(embeddings, axis=0)
        active = variances > 1e-12
        if active.sum() < 2:
            return 1.0
        emb = embeddings[:, active]
        cov = np.cov(emb, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        if len(eigenvalues) == 0:
            return 1.0
        pr = float(np.sum(eigenvalues) ** 2 / np.sum(eigenvalues**2))
        return round(pr, 2)
    except Exception:
        return float("nan")


def compute_isotropy(embeddings: np.ndarray) -> float:
    """lambda_min / lambda_max of covariance matrix."""
    try:
        variances = np.var(embeddings, axis=0)
        active = variances > 1e-12
        if active.sum() < 2:
            return 0.0
        emb = embeddings[:, active]
        cov = np.cov(emb, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        if len(eigenvalues) == 0:
            return 0.0
        return float(eigenvalues.min() / eigenvalues.max())
    except Exception:
        return float("nan")


def compute_alignment_uniformity(
    embeddings: np.ndarray,
    station_ids: list[str],
    window_station_map: dict | None = None,
) -> dict:
    """Alignment & Uniformity (Wang & Isola 2020).

    For station-level embeddings (no window info), only uniformity is computed.
    True alignment requires window-level embeddings.

    Returns {'alignment': float | None, 'uniformity': float}
    """
    # Normalize to unit sphere
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norm = embeddings / (norms + 1e-8)

    # Uniformity: log E[e^{-2||x-y||^2}]
    n = len(emb_norm)
    idx = np.random.RandomState(42).choice(n, size=min(2000, n), replace=False)
    subset = emb_norm[idx]
    sq_pdist = scipy.spatial.distance.pdist(subset, "sqeuclidean")
    uniformity = float(np.log(np.exp(-2 * sq_pdist).mean()))

    return {"alignment": None, "uniformity": uniformity}


def compute_pca1_amplitude_correlation(
    embeddings: np.ndarray,
    engine,
    domain: str,
    space: str,
    station_ids: list[str],
) -> float:
    """Correlation between 1st PCA component and mean amplitude of raw series.

    This is the KEY diagnostic for normalization issues.
    """
    pca = PCA(n_components=1)
    pca1 = pca.fit_transform(embeddings).flatten()

    if domain == "piezo":
        sql = text("""
            SELECT code_bss AS id, AVG(niveau_nappe_eau) AS mean_val
            FROM gold.hubeau_daily_chroniques
            WHERE code_bss = ANY(:ids)
            GROUP BY code_bss
        """)
    else:
        sql = text("""
            SELECT code_station AS id, AVG(resultat_obs_elab) AS mean_val
            FROM gold.hydro_daily_chroniques
            WHERE code_station = ANY(:ids)
            GROUP BY code_station
        """)

    with engine.connect() as conn:
        result = conn.execute(sql, {"ids": station_ids})
        mean_map = {
            row.id: float(row.mean_val)
            for row in result
            if row.mean_val is not None
        }

    paired_pca1, paired_mean = [], []
    for i, sid in enumerate(station_ids):
        if sid in mean_map:
            paired_pca1.append(pca1[i])
            paired_mean.append(mean_map[sid])

    if len(paired_pca1) < 10:
        return float("nan")

    r, _ = pearsonr(paired_pca1, paired_mean)
    return round(float(r), 4)


# ---------------------------------------------------------------------------
# 3. Supervised metrics (piezo only)
# ---------------------------------------------------------------------------


def _prepare_classification_data(
    embeddings: np.ndarray,
    labels: list[str | None],
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Filter, encode, and drop rare classes. Returns (X, y, label_names) or None."""
    mask = np.array([l is not None and l != "" for l in labels])
    X = embeddings[mask]
    y_raw = np.array([l for l, m in zip(labels, mask) if m])

    if len(set(y_raw)) < 2:
        return None

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Drop classes with fewer samples than n_splits
    class_counts = np.bincount(y)
    valid_classes = set(np.where(class_counts >= n_splits)[0])
    if len(valid_classes) < 2:
        return None
    keep = np.array([yi in valid_classes for yi in y])
    X = X[keep]
    le2 = LabelEncoder()
    y = le2.fit_transform(y_raw[keep])
    return X, y, le2.classes_


_EMPTY_CLF_RESULT: dict = {
    "accuracy": float("nan"),
    "macro_f1": float("nan"),
    "balanced_accuracy": float("nan"),
}


def compute_linear_probe(
    embeddings: np.ndarray, labels: list[str | None], n_splits: int = 5
) -> dict:
    """Linear probe (LogisticRegression) — the gold standard for embedding eval.

    Tests if class information is **linearly separable** in the embedding space.
    Returns {'accuracy', 'macro_f1', 'balanced_accuracy', 'confusion_matrix', ...}
    """
    prepared = _prepare_classification_data(embeddings, labels, n_splits)
    if prepared is None:
        return _EMPTY_CLF_RESULT
    X, y, class_names = prepared

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s, baccs = [], [], []
    all_preds, all_true = [], []

    for train_idx, test_idx in skf.split(X, y):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
            ),
        )
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred, average="macro", zero_division=0))
        baccs.append(balanced_accuracy_score(y[test_idx], pred))
        all_preds.extend(pred)
        all_true.extend(y[test_idx])

    le_classes = sorted(set(all_true))
    cm = confusion_matrix(all_true, all_preds, labels=le_classes)

    return {
        "accuracy": round(float(np.mean(accs)), 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "balanced_accuracy": round(float(np.mean(baccs)), 4),
        "confusion_matrix": cm.tolist(),
        "class_labels": [str(class_names[c]) for c in le_classes],
    }


def compute_fisher_criterion(
    embeddings: np.ndarray, labels: list[str | None]
) -> dict:
    """Fisher's class separability ratio: trace(S_b) / trace(S_w).

    Higher = classes are more separated in embedding space.
    No model fitting — pure geometric measure.
    """
    mask = np.array([l is not None and l != "" for l in labels])
    X = embeddings[mask]
    y = np.array([l for l, m in zip(labels, mask) if m])

    if len(set(y)) < 2 or len(X) < 10:
        return {"ratio": float("nan")}

    grand_mean = X.mean(axis=0)
    classes = np.unique(y)

    # Between-class scatter
    S_b = np.zeros((X.shape[1], X.shape[1]))
    # Within-class scatter
    S_w = np.zeros((X.shape[1], X.shape[1]))

    for c in classes:
        X_c = X[y == c]
        n_c = len(X_c)
        if n_c < 2:
            continue
        mean_c = X_c.mean(axis=0)
        diff = (mean_c - grand_mean).reshape(-1, 1)
        S_b += n_c * (diff @ diff.T)
        S_w += np.cov(X_c, rowvar=False) * (n_c - 1)

    tr_sw = np.trace(S_w)
    tr_sb = np.trace(S_b)

    if tr_sw < 1e-12:
        return {"ratio": float("nan"), "trace_Sb": round(float(tr_sb), 4)}

    return {
        "ratio": round(float(tr_sb / tr_sw), 4),
        "trace_Sb": round(float(tr_sb), 4),
        "trace_Sw": round(float(tr_sw), 4),
    }


def compute_clustering_metrics(embeddings: np.ndarray, labels: list[str]) -> dict:
    """AMI and ARI: HDBSCAN clusters vs ground truth labels.

    Returns {'ami': float, 'ari': float}
    """
    mask = [l is not None and l != "" for l in labels]
    X = embeddings[mask]
    y_true = [l for l, m in zip(labels, mask) if m]

    pred = HDBSCAN(min_cluster_size=25, min_samples=5).fit_predict(X)

    # Filter noise for AMI/ARI
    non_noise = pred != -1
    if non_noise.sum() < 10:
        return {"ami": float("nan"), "ari": float("nan")}

    ami = adjusted_mutual_info_score(
        [y for y, nn in zip(y_true, non_noise) if nn], pred[non_noise]
    )
    ari = adjusted_rand_score(
        [y for y, nn in zip(y_true, non_noise) if nn], pred[non_noise]
    )

    return {"ami": round(float(ami), 4), "ari": round(float(ari), 4)}


# ---------------------------------------------------------------------------
# 4. Domain-specific metrics
# ---------------------------------------------------------------------------


def compute_mantel_geo(
    embeddings: np.ndarray, lats: list[float], lons: list[float]
) -> dict:
    """Mantel test: geographic distance vs embedding distance.

    Returns {'r': float, 'p': float}
    """
    n = len(embeddings)
    rng = np.random.RandomState(42)
    idx = rng.choice(n, min(500, n), replace=False)
    emb_sub = embeddings[idx]
    lats_sub = [lats[i] for i in idx]
    lons_sub = [lons[i] for i in idx]

    # Geographic distance (haversine → km)
    coords_rad = np.radians(np.column_stack([lats_sub, lons_sub]))
    geo_dist = haversine_distances(coords_rad) * 6371

    # Embedding distance (cosine)
    emb_dist = cosine_distances(emb_sub)

    # Condensed distance vectors
    geo_condensed = scipy.spatial.distance.squareform(geo_dist)
    emb_condensed = scipy.spatial.distance.squareform(emb_dist)

    r_obs, _ = spearmanr(geo_condensed, emb_condensed)

    # Permutation test (999 permutations)
    n_perm = 999
    r_perms = []
    for _ in range(n_perm):
        perm = rng.permutation(len(emb_sub))
        emb_perm = emb_dist[perm][:, perm]
        r_p, _ = spearmanr(
            geo_condensed, scipy.spatial.distance.squareform(emb_perm)
        )
        r_perms.append(r_p)

    p_value = (np.sum(np.array(r_perms) >= r_obs) + 1) / (n_perm + 1)

    return {"r": round(float(r_obs), 4), "p": round(float(p_value), 4)}


# ---------------------------------------------------------------------------
# 4b. k-NN retrieval
# ---------------------------------------------------------------------------


def compute_knn_retrieval(
    embeddings: np.ndarray,
    labels: list[str | None],
    ks: tuple[int, ...] = (1, 5, 10, 20),
) -> dict:
    """k-NN label retrieval: what fraction of k nearest neighbors share the same label?

    Returns {'precision@1': float, 'precision@5': float, ...}
    """
    mask = np.array([l is not None and l != "" for l in labels])
    if mask.sum() < 30:
        return {f"precision@{k}": float("nan") for k in ks}

    X = embeddings[mask]
    y = np.array([l for l, m in zip(labels, mask) if m])
    max_k = max(ks)

    # Cosine distance matrix
    dists = cosine_distances(X)
    np.fill_diagonal(dists, np.inf)

    results = {}
    for k in ks:
        if k >= len(X):
            results[f"precision@{k}"] = float("nan")
            continue
        precisions = []
        for i in range(len(X)):
            neighbors = np.argsort(dists[i])[:k]
            match_frac = np.mean(y[neighbors] == y[i])
            precisions.append(match_frac)
        results[f"precision@{k}"] = round(float(np.mean(precisions)), 4)

    # Random baseline: class frequency squared (expected P@k for random)
    _, counts = np.unique(y, return_counts=True)
    freqs = counts / counts.sum()
    random_baseline = float(np.sum(freqs ** 2))
    results["random_baseline"] = round(random_baseline, 4)

    return results


# ---------------------------------------------------------------------------
# 4c. Regression on continuous properties
# ---------------------------------------------------------------------------


def compute_regression_metrics(
    embeddings: np.ndarray,
    targets: dict[str, list[float | None]],
    n_splits: int = 5,
) -> dict:
    """Ridge linear regression from embeddings to continuous station properties.

    Linear probe for regression — tests if continuous properties are
    linearly decodable from the embedding space.

    Returns {'target_name': {'r2': float, 'spearman': float}, ...}
    """
    results = {}
    for target_name, values in targets.items():
        y = np.array([v if v is not None else np.nan for v in values], dtype=float)
        valid = ~np.isnan(y)
        if valid.sum() < 50:
            results[target_name] = {"r2": float("nan"), "spearman": float("nan")}
            continue

        X = embeddings[valid]
        y_valid = y[valid]

        y_mean, y_std = y_valid.mean(), y_valid.std()
        if y_std < 1e-8:
            results[target_name] = {"r2": float("nan"), "spearman": float("nan")}
            continue
        y_norm = (y_valid - y_mean) / y_std

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        r2s, rhos = [], []
        for train_idx, test_idx in kf.split(X):
            pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            pipe.fit(X[train_idx], y_norm[train_idx])
            pred = pipe.predict(X[test_idx])
            r2s.append(r2_score(y_norm[test_idx], pred))
            rho, _ = spearmanr(y_norm[test_idx], pred)
            rhos.append(rho if not np.isnan(rho) else 0.0)

        results[target_name] = {
            "r2": round(float(np.mean(r2s)), 4),
            "spearman": round(float(np.mean(rhos)), 4),
        }

    return results


# ---------------------------------------------------------------------------
# 4d. Post-hoc whitening transform
# ---------------------------------------------------------------------------


def whiten_embeddings(embeddings: np.ndarray, n_components: int = 64) -> np.ndarray:
    """ZCA-like whitening: PCA + standardize each component.

    Fixes representation collapse by spreading embeddings in all directions.
    """
    n_comp = min(n_components, embeddings.shape[1], embeddings.shape[0] - 1)
    pca = PCA(n_components=n_comp, whiten=True)
    return pca.fit_transform(embeddings)


# ---------------------------------------------------------------------------
# 4e. Temporal consistency (window-level)
# ---------------------------------------------------------------------------


def compute_temporal_consistency(
    engine,
    domain: str,
    station_ids: list[str],
    max_stations: int = 200,
) -> dict:
    """Measure if window embeddings from the same station cluster together.

    For each station with multiple windows, compute mean intra-station cosine distance
    vs mean inter-station distance. Ratio < 1 means windows of same station are closer.
    """
    table = f"{domain}_window_embeddings"

    # Sample stations
    rng = np.random.RandomState(42)
    sample_ids = rng.choice(station_ids, min(max_stations, len(station_ids)), replace=False)
    placeholders = ", ".join(f":id_{i}" for i in range(len(sample_ids)))
    params = {f"id_{i}": sid for i, sid in enumerate(sample_ids)}

    query = text(f"""
        SELECT code_bss, embedding::text
        FROM ml.{table}
        WHERE code_bss IN ({placeholders})
        ORDER BY code_bss
    """)

    with engine.connect() as conn:
        rows = conn.execute(query, params).fetchall()

    if len(rows) < 20:
        return {"intra_dist": float("nan"), "inter_dist": float("nan"), "ratio": float("nan")}

    # Parse embeddings
    from collections import defaultdict as _dd
    station_windows: dict[str, list[np.ndarray]] = _dd(list)
    for code_bss, emb_str in rows:
        vec = np.array([float(x) for x in emb_str.strip("[]{}").split(",")])
        station_windows[code_bss].append(vec)

    # Keep stations with >= 2 windows
    multi = {k: v for k, v in station_windows.items() if len(v) >= 2}
    if len(multi) < 10:
        return {"intra_dist": float("nan"), "inter_dist": float("nan"), "ratio": float("nan")}

    # Compute intra-station distances
    intra_dists = []
    for sid, windows in multi.items():
        mat = np.array(windows)
        dists = cosine_distances(mat)
        n = len(mat)
        for i in range(n):
            for j in range(i + 1, n):
                intra_dists.append(dists[i, j])

    # Compute inter-station distances (sample pairs)
    all_means = {sid: np.mean(ws, axis=0) for sid, ws in multi.items()}
    sids = list(all_means.keys())
    mean_mat = np.array([all_means[s] for s in sids])
    inter_mat = cosine_distances(mean_mat)
    inter_dists = inter_mat[np.triu_indices(len(sids), k=1)]

    intra = float(np.mean(intra_dists))
    inter = float(np.mean(inter_dists))
    ratio = intra / inter if inter > 1e-8 else float("nan")

    return {
        "intra_dist": round(intra, 4),
        "inter_dist": round(inter, 4),
        "ratio": round(ratio, 4),
        "n_stations": len(multi),
    }


# ---------------------------------------------------------------------------
# 5. PCA diagnostics
# ---------------------------------------------------------------------------


def compute_pca_diagnostics(embeddings: np.ndarray) -> dict:
    """PCA cumulative variance curve and key thresholds."""
    n_components = min(100, embeddings.shape[1], embeddings.shape[0] - 1)
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    return {
        "n_80": int(np.searchsorted(cumvar, 0.80) + 1),
        "n_90": int(np.searchsorted(cumvar, 0.90) + 1),
        "n_95": int(np.searchsorted(cumvar, 0.95) + 1),
        "n_99": int(np.searchsorted(cumvar, 0.99) + 1),
        "cumvar_curve": [round(float(v), 4) for v in cumvar[:50]],
    }


# ---------------------------------------------------------------------------
# 6. Main evaluation pipeline
# ---------------------------------------------------------------------------


def evaluate_space(engine, domain: str, space: str) -> dict:
    """Run full evaluation for one domain/space combo."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating {domain}/{space}")
    print(f"{'=' * 60}")

    embeddings, station_ids, metadata = load_embeddings(engine, domain, space)
    print(f"  Loaded {len(station_ids)} stations, {embeddings.shape[1]}D embeddings")

    results: dict = {
        "domain": domain,
        "space": space,
        "n_stations": len(station_ids),
        "embedding_dim": embeddings.shape[1],
    }

    # --- Intrinsic ---
    print("  Computing intrinsic metrics...")
    results["participation_ratio"] = compute_participation_ratio(embeddings)
    results["isotropy"] = compute_isotropy(embeddings)
    results["pca"] = compute_pca_diagnostics(embeddings)
    au = compute_alignment_uniformity(embeddings, station_ids)
    results["uniformity"] = au["uniformity"]

    # --- PCA1 vs amplitude (normalization diagnostic) ---
    print("  Computing PCA1 vs amplitude correlation...")
    try:
        results["pca1_amplitude_corr"] = compute_pca1_amplitude_correlation(
            embeddings, engine, domain, space, station_ids
        )
    except Exception as e:
        print(f"    WARNING: PCA1 vs amplitude failed: {e}")
        results["pca1_amplitude_corr"] = float("nan")

    # --- Supervised (piezo only) ---
    if domain == "piezo":
        milieu = metadata.get("milieu_eh", [])
        theme = metadata.get("theme_eh", [])
        compound = [
            f"{m}_{t}" if m and t else None for m, t in zip(milieu, theme)
        ]

        print("  Computing Linear Probe (milieu_eh)...")
        results["linear_milieu"] = compute_linear_probe(embeddings, milieu)

        print("  Computing Linear Probe (milieu_eh x theme_eh)...")
        results["linear_compound"] = compute_linear_probe(embeddings, compound)

        print("  Computing Fisher criterion (milieu_eh)...")
        results["fisher_milieu"] = compute_fisher_criterion(embeddings, milieu)

        print("  Computing clustering AMI/ARI (milieu_eh)...")
        results["clustering_milieu"] = compute_clustering_metrics(
            embeddings, milieu
        )

    # --- Mantel test (if lat/lon available) ---
    lats = metadata.get("latitude", [])
    lons = metadata.get("longitude", [])
    if lats and lons and not all(v is None for v in lats):
        valid = [
            (i, la, lo)
            for i, (la, lo) in enumerate(zip(lats, lons))
            if la is not None and lo is not None
        ]
        if len(valid) > 50:
            print("  Computing Mantel test (geographic)...")
            idx_v, lats_v, lons_v = zip(*valid)
            results["mantel_geo"] = compute_mantel_geo(
                embeddings[list(idx_v)], list(lats_v), list(lons_v)
            )

    return results


# ---------------------------------------------------------------------------
# 6b. Generic evaluation (for any embedding matrix)
# ---------------------------------------------------------------------------


def evaluate_embeddings_generic(
    embeddings: np.ndarray,
    station_ids: list[str],
    metadata: dict,
    method_name: str,
    domain: str,
    engine=None,
) -> dict:
    """Run evaluation metrics on an arbitrary embedding matrix.

    Used by compare mode to evaluate Catch22, PCA brut, Random, etc.
    """
    print(f"\n  --- Evaluating {method_name} ({len(station_ids)} stations, "
          f"{embeddings.shape[1]}D) ---")

    results: dict = {
        "domain": domain,
        "space": method_name,
        "n_stations": len(station_ids),
        "embedding_dim": embeddings.shape[1],
    }

    if len(station_ids) < 10 or embeddings.shape[0] < 10:
        print(f"    Too few stations ({len(station_ids)}) -- skipping metrics")
        return results

    # --- Intrinsic ---
    print("    Computing intrinsic metrics...")
    results["participation_ratio"] = compute_participation_ratio(embeddings)
    results["isotropy"] = compute_isotropy(embeddings)
    results["pca"] = compute_pca_diagnostics(embeddings)
    au = compute_alignment_uniformity(embeddings, station_ids)
    results["uniformity"] = au["uniformity"]

    # --- PCA1 vs amplitude ---
    if engine is not None:
        try:
            results["pca1_amplitude_corr"] = compute_pca1_amplitude_correlation(
                embeddings, engine, domain, method_name, station_ids
            )
        except Exception as e:
            print(f"    WARNING: PCA1 vs amplitude failed: {e}")
            results["pca1_amplitude_corr"] = float("nan")

    # --- Supervised (piezo only) ---
    if domain == "piezo":
        milieu = metadata.get("milieu_eh", [])
        theme = metadata.get("theme_eh", [])
        compound = [
            f"{m}_{t}" if m and t else None for m, t in zip(milieu, theme)
        ]

        if milieu:
            print("    Computing Linear Probe (milieu_eh)...")
            results["linear_milieu"] = compute_linear_probe(embeddings, milieu)

            print("    Computing Linear Probe (milieu_eh x theme_eh)...")
            results["linear_compound"] = compute_linear_probe(embeddings, compound)

            print("    Computing Fisher criterion (milieu_eh)...")
            results["fisher_milieu"] = compute_fisher_criterion(embeddings, milieu)

            print("    Computing clustering AMI/ARI (milieu_eh)...")
            results["clustering_milieu"] = compute_clustering_metrics(
                embeddings, milieu
            )

            print("    Computing k-NN retrieval (milieu_eh)...")
            results["knn_retrieval"] = compute_knn_retrieval(embeddings, milieu)

    # --- Dynamic typology (data-driven label) ---
    dyn_typo = metadata.get("dynamic_typology", [])
    if dyn_typo and any(v is not None for v in dyn_typo):
        print("    Computing Linear Probe (dynamic typology)...")
        results["linear_typology"] = compute_linear_probe(embeddings, dyn_typo)
        print("    Computing k-NN retrieval (dynamic typology)...")
        results["knn_typology"] = compute_knn_retrieval(embeddings, dyn_typo)
        print("    Computing Fisher criterion (dynamic typology)...")
        results["fisher_typology"] = compute_fisher_criterion(embeddings, dyn_typo)

    # --- Regression on exogenous properties only (no leakage) ---
    # altitude = topographic, fully exogenous to the time series
    # depth/stddev/amplitude are DERIVED from the series → data leakage, excluded
    regression_targets = {}
    for target in ("altitude",):
        vals = metadata.get(target, [])
        if vals and any(v is not None for v in vals):
            regression_targets[target] = vals
    if regression_targets:
        print("    Computing Ridge regression (altitude — exogenous only)...")
        results["regression"] = compute_regression_metrics(embeddings, regression_targets)

    # --- Mantel test (if lat/lon available) ---
    lats = metadata.get("latitude", [])
    lons = metadata.get("longitude", [])
    if lats and lons and not all(v is None for v in lats):
        valid = [
            (i, la, lo)
            for i, (la, lo) in enumerate(zip(lats, lons))
            if la is not None and lo is not None
        ]
        if len(valid) > 50:
            print("    Computing Mantel test (geographic)...")
            idx_v, lats_v, lons_v = zip(*valid)
            results["mantel_geo"] = compute_mantel_geo(
                embeddings[list(idx_v)], list(lats_v), list(lons_v)
            )

    return results


def _align_to_common_ids(
    methods: dict[str, tuple[np.ndarray, list[str]]],
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Find intersection of station IDs across all methods and align matrices.

    Returns (common_ids, {method_name: aligned_embedding_matrix})
    """
    all_id_sets = [set(ids) for _, ids in methods.values()]
    common = sorted(set.intersection(*all_id_sets)) if all_id_sets else []

    if not common:
        return [], {}

    aligned: dict[str, np.ndarray] = {}
    for name, (emb, ids) in methods.items():
        id_to_idx = {sid: i for i, sid in enumerate(ids)}
        idx = [id_to_idx[sid] for sid in common]
        aligned[name] = emb[idx]

    return common, aligned


def _build_metadata_for_ids(
    station_ids: list[str],
    reference_metadata: dict,
    dynamic_typo: dict[str, str],
) -> dict[str, list]:
    """Build metadata dict aligned with station_ids from reference metadata."""
    meta_keys = [
        "milieu_eh", "theme_eh", "departement", "altitude",
        "latitude", "longitude", "depth", "stddev", "amplitude",
    ]
    meta: dict[str, list] = {k: [] for k in meta_keys}

    if reference_metadata and "_id_list" in reference_metadata:
        ref_id_to_idx = {
            sid: i for i, sid in enumerate(reference_metadata["_id_list"])
        }
        for sid in station_ids:
            idx = ref_id_to_idx.get(sid)
            for key in meta_keys:
                if idx is not None and key in reference_metadata:
                    meta[key].append(reference_metadata[key][idx])
                else:
                    meta[key].append(None)
    else:
        meta = {k: [None] * len(station_ids) for k in meta_keys}

    meta["dynamic_typology"] = [dynamic_typo.get(sid) for sid in station_ids]
    return meta


def evaluate_all_methods(
    engine,
    domain: str,
    db_spaces: list[str],
    max_stations: int = 0,
    stride: int = 365,
) -> list[dict]:
    """Per-space multi-encoder comparison.

    For each input space (uni, multi):
      - Compute embeddings with ALL encoders on the SAME data
      - Align on common stations
      - Evaluate with identical protocol

    This ensures a fair comparison: within each space, only the encoder differs.
    stride=365 → 1 window/year (low memory). stride=90 → 4 windows/year (high memory).
    """
    print(f"\n{'=' * 60}")
    print(f"Multi-method comparison for {domain}")
    print(f"{'=' * 60}")

    all_results: list[dict] = []

    # --- 1. Load DB embeddings for reference + metadata ---
    db_data: dict[str, tuple[np.ndarray, list[str], dict]] = {}
    all_station_ids: set[str] = set()
    reference_metadata: dict = {}

    for space in db_spaces:
        try:
            emb, ids, meta = load_embeddings(engine, domain, space)
            db_data[space] = (emb, ids, meta)
            all_station_ids.update(ids)
            if not reference_metadata:
                reference_metadata = meta
                reference_metadata["_id_list"] = ids
            print(f"  DB {space}: {len(ids)} stations, {emb.shape[1]}D")
        except Exception as e:
            print(f"  WARNING: Could not load DB embeddings for {domain}/{space}: {e}")

    if not all_station_ids:
        print(f"  No station IDs available for {domain} -- aborting")
        return []

    # --- 2. Load raw series (univariate + multivariate) ---
    all_ids_list = sorted(all_station_ids)
    n_desc = "all" if max_stations == 0 else f"up to {max_stations}"

    print(f"\n  Loading univariate raw series for {n_desc} stations...")
    raw_uni = load_raw_series(engine, domain, all_ids_list, max_stations=max_stations)

    print(f"  Loading multivariate raw series for {n_desc} stations...")
    raw_multi = load_raw_series_multi(engine, domain, all_ids_list, max_stations=max_stations)

    # Keep only stations present in BOTH uni and multi
    common_raw_ids = sorted(set(raw_uni.keys()) & set(raw_multi.keys()))
    print(f"  Stations with both uni and multi data: {len(common_raw_ids)}")

    # Dynamic typology (from univariate)
    print("\n  Computing dynamic typology (inertial/annual/reactive)...")
    dynamic_typo = compute_dynamic_typology(raw_uni)
    n_typo = len(dynamic_typo)
    print(f"    {n_typo} stations: "
          f"{sum(1 for v in dynamic_typo.values() if v == 'inertial')} inertial, "
          f"{sum(1 for v in dynamic_typo.values() if v == 'annual')} annual, "
          f"{sum(1 for v in dynamic_typo.values() if v == 'reactive')} reactive")

    # =====================================================================
    # EVALUATE EACH INPUT SPACE INDEPENDENTLY
    # =====================================================================
    for input_space in ("uni", "multi"):
        print(f"\n{'=' * 60}")
        print(f"  INPUT SPACE: {input_space}")
        print(f"{'=' * 60}")

        methods: dict[str, tuple[np.ndarray, list[str]]] = {}

        # --- DB reference (if available for this space) ---
        if input_space in db_data:
            emb, ids, _meta = db_data[input_space]
            methods[f"DB {input_space}"] = (emb, ids)

        if not common_raw_ids:
            print("  No raw series -- only evaluating DB embeddings")
            if not methods:
                continue
        else:
            if input_space == "uni":
                raw_data = {sid: raw_uni[sid] for sid in common_raw_ids}
            else:
                raw_data = {sid: raw_multi[sid] for sid in common_raw_ids}
            raw_ids = sorted(raw_data.keys())
            is_multi = input_space == "multi"

            # --- MiniRocket ---
            print(f"\n  Computing MiniRocket ({input_space})...")
            try:
                mr_emb, mr_ids = compute_minirocket_embeddings(
                    raw_data, multivariate=is_multi
                )
                if len(mr_ids) > 0:
                    methods[f"MiniRocket ({input_space})"] = (mr_emb, mr_ids)
                    print(f"    MiniRocket: {len(mr_ids)} stations, {mr_emb.shape[1]}D")
            except Exception as e:
                print(f"    WARNING: MiniRocket failed: {e}")

            # --- TS2Vec (contrastive, same family as SoftCLT) ---
            print(f"\n  Computing TS2Vec ({input_space})...")
            try:
                ts_emb, ts_ids = compute_ts2vec_embeddings(
                    raw_data, multivariate=is_multi
                )
                if len(ts_ids) > 0:
                    methods[f"TS2Vec ({input_space})"] = (ts_emb, ts_ids)
                    print(f"    TS2Vec: {len(ts_ids)} stations, {ts_emb.shape[1]}D")
            except Exception as e:
                print(f"    WARNING: TS2Vec failed: {e}")

            # --- Catch22 (univariate only) ---
            if input_space == "uni":
                print(f"\n  Computing Catch22 ({input_space})...")
                c22_emb, c22_ids = compute_catch22_embeddings(raw_data)
                if len(c22_ids) > 0:
                    methods[f"Catch22"] = (c22_emb, c22_ids)
                    print(f"    Catch22: {len(c22_ids)} stations, {c22_emb.shape[1]}D")

            # --- PCA brut ---
            print(f"\n  Computing PCA brut ({input_space})...")
            pca_emb, pca_ids = compute_pca_brut_embeddings(
                {sid: raw_data[sid] for sid in raw_ids}
            )
            if len(pca_ids) > 0:
                methods[f"PCA brut ({input_space})"] = (pca_emb, pca_ids)
                print(f"    PCA brut: {len(pca_ids)} stations, {pca_emb.shape[1]}D")

            # --- Random baseline ---
            rand_emb, rand_ids = compute_random_embeddings(raw_ids)
            methods["Random"] = (rand_emb, rand_ids)

        # --- Align all methods on common station IDs ---
        common_ids, aligned = _align_to_common_ids(methods)
        n_common = len(common_ids)
        print(f"\n  Common stations ({input_space}): {n_common} "
              f"across {len(methods)} methods")

        if n_common < 30:
            print(f"  WARNING: Too few common stations ({n_common}), skipping {input_space}")
            continue

        # Build metadata
        common_meta = _build_metadata_for_ids(
            common_ids, reference_metadata, dynamic_typo
        )

        # --- Whitened variants ---
        whitened: dict[str, np.ndarray] = {}
        for name in list(aligned.keys()):
            if name in ("Random",) or "PCA brut" in name:
                continue
            try:
                whitened[f"{name} +W"] = whiten_embeddings(aligned[name])
            except Exception as e:
                print(f"    WARNING: whitening {name} failed: {e}")
        aligned.update(whitened)

        # --- Evaluate all methods ---
        for method_name, emb_matrix in aligned.items():
            result = evaluate_embeddings_generic(
                emb_matrix, common_ids, common_meta,
                method_name, domain, engine=engine,
            )
            result["input_space"] = input_space
            all_results.append(result)

    return all_results


# ---------------------------------------------------------------------------
# 7. Report generation
# ---------------------------------------------------------------------------


def _fmt(val, fmt_str: str = ".4f") -> str:
    """Format a value for the report, handling NaN and None."""
    if val is None:
        return "N/A"
    try:
        if isinstance(val, float) and np.isnan(val):
            return "N/A"
        return f"{val:{fmt_str}}"
    except (TypeError, ValueError):
        return str(val)


def generate_report(results: list[dict], output_dir: Path) -> None:
    """Generate markdown report with summary tables."""
    lines = ["# Embedding Benchmark Report\n"]
    lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append("## Summary\n")

    # Main table
    lines.append(
        "| Space | N | Dim | PR | Isotropy | Uniformity "
        "| PCA1-Ampl | PCA 80% | PCA 95% | PCA 99% |"
    )
    lines.append(
        "|-------|---|-----|-----|----------|------------|"
        "-----------|---------|---------|---------|"
    )
    for r in results:
        pca = r.get("pca", {})
        lines.append(
            f"| {r['domain']}/{r['space']} "
            f"| {r['n_stations']} "
            f"| {r['embedding_dim']} "
            f"| {_fmt(r.get('participation_ratio'), '.2f')} "
            f"| {_fmt(r.get('isotropy'), '.2e')} "
            f"| {_fmt(r.get('uniformity'), '.2f')} "
            f"| {_fmt(r.get('pca1_amplitude_corr'), '.4f')} "
            f"| {pca.get('n_80', 'N/A')} "
            f"| {pca.get('n_95', 'N/A')} "
            f"| {pca.get('n_99', 'N/A')} |"
        )

    # Supervised table (piezo only)
    piezo_results = [r for r in results if r["domain"] == "piezo"]
    if piezo_results:
        lines.append("\n## Classification (Piezo)\n")
        lines.append(
            "| Space | LP milieu BalAcc | LP milieu F1 "
            "| LP compound F1 | Fisher | AMI | ARI |"
        )
        lines.append(
            "|-------|-----------------|-------------"
            "|----------------|--------|-----|-----|"
        )
        for r in piezo_results:
            lm = r.get("linear_milieu", {})
            lc = r.get("linear_compound", {})
            fi = r.get("fisher_milieu", {})
            cl = r.get("clustering_milieu", {})
            lines.append(
                f"| {r['space']} "
                f"| {_fmt(lm.get('balanced_accuracy'))} "
                f"| {_fmt(lm.get('macro_f1'))} "
                f"| {_fmt(lc.get('macro_f1'))} "
                f"| {_fmt(fi.get('ratio'))} "
                f"| {_fmt(cl.get('ami'))} "
                f"| {_fmt(cl.get('ari'))} |"
            )

    # Mantel
    mantel_results = [r for r in results if "mantel_geo" in r]
    if mantel_results:
        lines.append("\n## Spatial Coherence\n")
        lines.append("| Space | Mantel r | p-value |")
        lines.append("|-------|----------|---------|")
        for r in mantel_results:
            mg = r["mantel_geo"]
            lines.append(
                f"| {r['domain']}/{r['space']} | {mg['r']} | {mg['p']} |"
            )

    # Normalization diagnostic
    lines.append("\n## Normalization Diagnostic\n")
    for r in results:
        corr = r.get("pca1_amplitude_corr", float("nan"))
        if corr is None or (isinstance(corr, float) and np.isnan(corr)):
            status = "N/A"
        elif abs(corr) > 0.8:
            status = (
                f"**CRITICAL** r={corr} — embedding encodes amplitude, "
                "not shape. Normalization likely missing."
            )
        elif abs(corr) > 0.5:
            status = f"WARNING r={corr} — moderate amplitude encoding."
        else:
            status = f"OK r={corr}"
        lines.append(f"- **{r['domain']}/{r['space']}**: {status}")

    # PCA cumulative variance details
    lines.append("\n## PCA Cumulative Variance (first 20 components)\n")
    lines.append("| Space | PC1 | PC2 | PC3 | PC5 | PC10 | PC20 |")
    lines.append("|-------|-----|-----|-----|-----|------|------|")
    for r in results:
        curve = r.get("pca", {}).get("cumvar_curve", [])
        vals = [
            _fmt(curve[i], ".3f") if i < len(curve) else "N/A"
            for i in [0, 1, 2, 4, 9, 19]
        ]
        lines.append(
            f"| {r['domain']}/{r['space']} "
            f"| {vals[0]} | {vals[1]} | {vals[2]} "
            f"| {vals[3]} | {vals[4]} | {vals[5]} |"
        )

    with open(output_dir / "report.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def generate_comparison_report(results: list[dict], output_dir: Path) -> None:
    """Generate markdown report with multi-method comparison table."""
    lines = ["# Embedding Benchmark Comparison Report\n"]
    lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # Group by domain × input_space
    groups = sorted(set(
        (r["domain"], r.get("input_space", "?")) for r in results
    ))

    for domain, input_space in groups:
        domain_results = [
            r for r in results
            if r["domain"] == domain and r.get("input_space", "?") == input_space
        ]
        lines.append(f"\n## {domain.capitalize()} — Input Space: {input_space}\n")

        # --- 1. Classification table ---
        lines.append("### Classification (milieu_eh)\n")
        lines.append(
            "| Method | Dim | N | LP BalAcc | LP F1 | Fisher | P@1 | P@5 "
            "| AMI | Mantel r |"
        )
        lines.append(
            "|--------|-----|---|----------|-------|--------|-----|-----"
            "|-----|----------|"
        )
        for r in domain_results:
            lm = r.get("linear_milieu", {})
            fi = r.get("fisher_milieu", {})
            knn = r.get("knn_retrieval", {})
            cl = r.get("clustering_milieu", {})
            mg = r.get("mantel_geo", {})
            lines.append(
                f"| {r['space']} "
                f"| {r.get('embedding_dim', 'N/A')} "
                f"| {r.get('n_stations', 'N/A')} "
                f"| {_fmt(lm.get('balanced_accuracy'))} "
                f"| {_fmt(lm.get('macro_f1'))} "
                f"| {_fmt(fi.get('ratio'))} "
                f"| {_fmt(knn.get('precision@1'))} "
                f"| {_fmt(knn.get('precision@5'))} "
                f"| {_fmt(cl.get('ami'))} "
                f"| {_fmt(mg.get('r'))} |"
            )
        # Add random baseline note
        sample_knn = next(
            (r.get("knn_retrieval", {}) for r in domain_results
             if r.get("knn_retrieval", {}).get("random_baseline")), {}
        )
        if sample_knn.get("random_baseline"):
            lines.append(
                f"\n> k-NN random baseline (class frequency²): "
                f"{sample_knn['random_baseline']:.4f}"
            )

        # --- 1b. Dynamic typology table ---
        typo_results = [r for r in domain_results if r.get("linear_typology")]
        if typo_results:
            lines.append("\n### Dynamic Typology (inertial / annual / reactive)\n")
            lines.append(
                "> Data-driven labels from lag-365 autocorrelation. "
                "Tests if embeddings capture temporal dynamics. "
                "**Note**: labels derived from same series — this is a consistency "
                "check, not an independent evaluation.\n"
            )
            lines.append(
                "| Method | LP BalAcc | LP F1 | Fisher | P@1 | P@5 |"
            )
            lines.append(
                "|--------|----------|-------|--------|-----|-----|"
            )
            for r in typo_results:
                lt = r.get("linear_typology", {})
                ft = r.get("fisher_typology", {})
                kt = r.get("knn_typology", {})
                lines.append(
                    f"| {r['space']} "
                    f"| {_fmt(lt.get('balanced_accuracy'))} "
                    f"| {_fmt(lt.get('macro_f1'))} "
                    f"| {_fmt(ft.get('ratio'))} "
                    f"| {_fmt(kt.get('precision@1'))} "
                    f"| {_fmt(kt.get('precision@5'))} |"
                )

        # --- 2. Regression table ---
        reg_results = [r for r in domain_results if r.get("regression")]
        if reg_results:
            lines.append("\n### Regression (continuous properties)\n")
            # Collect all targets
            all_targets = sorted(set(
                t for r in reg_results for t in r.get("regression", {}).keys()
            ))
            header = "| Method |"
            sep = "|--------|"
            for t in all_targets:
                header += f" {t} R² | {t} ρ |"
                sep += "------|------|"
            lines.append(header)
            lines.append(sep)
            for r in reg_results:
                reg = r.get("regression", {})
                row = f"| {r['space']} |"
                for t in all_targets:
                    tr = reg.get(t, {})
                    row += f" {_fmt(tr.get('r2'))} | {_fmt(tr.get('spearman'))} |"
                lines.append(row)

        # --- 3. Intrinsic quality table ---
        lines.append("\n### Intrinsic Quality\n")
        lines.append("| Method | PR | Uniformity | PCA 80% | PCA 95% |")
        lines.append("|--------|----|------------|---------|---------|")
        for r in domain_results:
            pca = r.get("pca", {})
            lines.append(
                f"| {r['space']} "
                f"| {_fmt(r.get('participation_ratio'), '.2f')} "
                f"| {_fmt(r.get('uniformity'), '.2f')} "
                f"| {pca.get('n_80', 'N/A')} "
                f"| {pca.get('n_95', 'N/A')} |"
            )

        # --- 4. Temporal consistency ---
        tc_results = [r for r in domain_results if r.get("temporal_consistency")]
        if tc_results:
            lines.append("\n### Temporal Consistency (window embeddings)\n")
            lines.append("| Method | Intra-dist | Inter-dist | Ratio | N stations |")
            lines.append("|--------|-----------|-----------|-------|------------|")
            for r in tc_results:
                tc = r["temporal_consistency"]
                lines.append(
                    f"| {r['space']} "
                    f"| {_fmt(tc.get('intra_dist'))} "
                    f"| {_fmt(tc.get('inter_dist'))} "
                    f"| {_fmt(tc.get('ratio'))} "
                    f"| {tc.get('n_stations', 'N/A')} |"
                )
            lines.append("\n> Ratio < 1 means windows from the same station "
                         "are closer than windows from different stations (good).")

        # --- 5. Ranking ---
        lines.append(f"\n### Overall Ranking ({domain}/{input_space})\n")

        rank_metrics = [
            ("BalAcc", lambda r: r.get("linear_milieu", {}).get("balanced_accuracy")),
            ("F1", lambda r: r.get("linear_milieu", {}).get("macro_f1")),
            ("Fisher", lambda r: r.get("fisher_milieu", {}).get("ratio")),
            ("P@5", lambda r: r.get("knn_retrieval", {}).get("precision@5")),
            ("AMI", lambda r: r.get("clustering_milieu", {}).get("ami")),
            ("Mantel", lambda r: r.get("mantel_geo", {}).get("r")),
            ("PR", lambda r: r.get("participation_ratio")),
        ]

        method_ranks: dict[str, list[float]] = defaultdict(list)
        for metric_name, extractor in rank_metrics:
            vals = []
            for r in domain_results:
                v = extractor(r)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals.append((r["space"], v))
            if len(vals) < 2:
                continue
            sorted_vals = sorted(vals, key=lambda x: x[1], reverse=True)
            for rank, (name, _) in enumerate(sorted_vals, 1):
                method_ranks[name].append(rank)

        if method_ranks:
            lines.append("| Method | Mean Rank | Ranks (BalAcc,F1,Fisher,P@5,AMI,Mantel,PR) |")
            lines.append("|--------|-----------|---------------------------------------------|")
            sorted_methods = sorted(
                method_ranks.items(), key=lambda x: np.mean(x[1])
            )
            for name, ranks in sorted_methods:
                mean_rank = np.mean(ranks)
                ranks_str = ", ".join(str(r) for r in ranks)
                lines.append(f"| {name} | {mean_rank:.2f} | {ranks_str} |")

    # Normalization diagnostic
    lines.append("\n## Normalization Diagnostic\n")
    for r in results:
        corr = r.get("pca1_amplitude_corr", float("nan"))
        if corr is None or (isinstance(corr, float) and np.isnan(corr)):
            status = "N/A"
        elif abs(corr) > 0.8:
            status = (
                f"**CRITICAL** r={corr} -- embedding encodes amplitude, "
                "not shape. Normalization likely missing."
            )
        elif abs(corr) > 0.5:
            status = f"WARNING r={corr} -- moderate amplitude encoding."
        else:
            status = f"OK r={corr}"
        lines.append(f"- **{r['domain']}/{r['space']}**: {status}")

    with open(output_dir / "comparison_report.md", "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)

    parser = argparse.ArgumentParser(
        description="Embedding Benchmark -- Evaluate and compare embedding methods"
    )
    parser.add_argument(
        "--mode",
        choices=["evaluate", "compare"],
        default="evaluate",
        help="Run mode: evaluate (DB only) or compare (multi-method)",
    )
    parser.add_argument(
        "--output",
        default="reports/embedding_benchmark",
        help="Output directory prefix",
    )
    parser.add_argument(
        "--spaces",
        nargs="*",
        default=None,
        help="Subset of spaces to evaluate, e.g. piezo/uni hydro/multi",
    )
    parser.add_argument(
        "--max-stations",
        type=int,
        default=0,
        help="Max stations for raw series loading (0 = all available, compare mode)",
    )
    args = parser.parse_args()

    engine = create_engine(DB_URL)

    # Determine which spaces to evaluate
    all_combos = [
        ("piezo", "uni"),
        ("piezo", "multi"),
        ("hydro", "uni"),
        ("hydro", "multi"),
    ]
    if args.spaces:
        combos = []
        for s in args.spaces:
            domain, space = s.split("/")
            combos.append((domain, space))
    else:
        combos = all_combos

    all_results = []

    if args.mode == "evaluate":
        # Original behavior: evaluate DB embeddings only
        for domain, space in combos:
            try:
                result = evaluate_space(engine, domain, space)
                all_results.append(result)
            except Exception as e:
                print(f"\n  ERROR evaluating {domain}/{space}: {e}")

    elif args.mode == "compare":
        # Multi-method comparison
        # Group combos by domain
        domain_spaces: dict[str, list[str]] = defaultdict(list)
        for domain, space in combos:
            domain_spaces[domain].append(space)

        for domain, spaces in domain_spaces.items():
            try:
                domain_results = evaluate_all_methods(
                    engine, domain, spaces, max_stations=args.max_stations
                )
                all_results.extend(domain_results)
            except Exception as e:
                print(f"\n  ERROR in compare for {domain}: {e}")

    if not all_results:
        print("\nNo results produced. Check DB connectivity.")
        return

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path(args.output + f"_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)

    # JSON metrics (handle NaN serialization)
    class _NaNEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, float) and np.isnan(obj):
                return None
            return super().default(obj)

    with open(output_dir / "data" / "metrics.json", "w") as f:
        json.dump(all_results, f, indent=2, cls=_NaNEncoder, default=str)

    if args.mode == "evaluate":
        generate_report(all_results, output_dir)
        report_file = "report.md"
    else:
        generate_comparison_report(all_results, output_dir)
        report_file = "comparison_report.md"

    print(f"\n{'=' * 60}")
    print(f"Report saved to {output_dir}")
    print(f"  - {output_dir / report_file}")
    print(f"  - {output_dir / 'data' / 'metrics.json'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
