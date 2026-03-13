"""Unit tests for dashboard/utils/latent_space.py — pure math, no DB required."""
from __future__ import annotations

import sys
import types
import unittest.mock as mock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Bootstrap: inject a minimal sqlalchemy stub if the real package is absent.
# dashboard/utils/latent_space.py imports `from sqlalchemy import text` at the
# top level.  On the bare CI/dev host the full dependency set is not installed,
# so we provide a lightweight stand-in that is good enough for collection and
# for the tests that only exercise pure-Python logic.
# ---------------------------------------------------------------------------

_SQLALCHEMY_MISSING = "sqlalchemy" not in sys.modules
try:
    import sqlalchemy  # noqa: F401
except ImportError:
    _stub = types.ModuleType("sqlalchemy")

    def _text_stub(sql_str):
        """Return a simple object whose str() is the SQL string."""
        obj = mock.MagicMock()
        obj.__str__ = lambda self: sql_str
        return obj

    _stub.text = _text_stub
    sys.modules["sqlalchemy"] = _stub
    _SQLALCHEMY_MISSING = True

from dashboard.utils.latent_space import (  # noqa: E402
    build_station_query,
    compute_clustering,
    compute_umap,
    parse_pgvector,
    subsample_stratified,
)

# Mark for tests that need the real sqlalchemy (build_station_query assertions
# on SQL text are valid even with the stub since str() returns the raw string).
_needs_sqlalchemy = pytest.mark.skipif(
    False,  # build_station_query works with the stub — no skip needed
    reason="sqlalchemy not installed",
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class _SimpleFilters:
    """Minimal stand-in for EmbeddingFilters with all attributes falsy by default."""

    def __init__(self, **kwargs):
        defaults = {
            "station_ids": None,
            "libelle_eh": None,
            "milieu_eh": None,
            "theme_eh": None,
            "etat_eh": None,
            "nature_eh": None,
            "departement": None,
            "region": None,
            "cluster_id": None,
        }
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


class _SimpleClusteringParams:
    """Minimal ClusteringParams-like object."""

    class _HDBSCAN:
        min_cluster_size = 5
        min_samples = 3

    class _KMeans:
        n_clusters = 3

    hdbscan = _HDBSCAN()
    kmeans = _KMeans()


# ---------------------------------------------------------------------------
# parse_pgvector
# ---------------------------------------------------------------------------


def test_parse_pgvector():
    raw = "[0.1,0.2,0.3]"
    result = parse_pgvector(raw)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == (3,)
    np.testing.assert_allclose(result, [0.1, 0.2, 0.3], rtol=1e-5)


def test_parse_pgvector_single_value():
    result = parse_pgvector("[42.0]")
    assert result.shape == (1,)
    assert result[0] == pytest.approx(42.0)


def test_parse_pgvector_whitespace():
    # The implementation strips surrounding whitespace; float() handles spaces
    # around values when split by comma.
    result = parse_pgvector("  [1.0, 2.0, 3.0]  ")
    assert result.shape == (3,)


# ---------------------------------------------------------------------------
# compute_umap  (skipped if umap-learn is not installed)
# ---------------------------------------------------------------------------


def test_compute_umap_2d():
    pytest.importorskip("umap", reason="umap-learn not installed")
    rng = np.random.RandomState(42)
    X = rng.randn(100, 320).astype(np.float32)
    result = compute_umap(X, n_components=2)
    assert result.shape == (100, 2)


def test_compute_umap_3d():
    pytest.importorskip("umap", reason="umap-learn not installed")
    rng = np.random.RandomState(42)
    X = rng.randn(100, 320).astype(np.float32)
    result = compute_umap(X, n_components=3)
    assert result.shape == (100, 3)


# ---------------------------------------------------------------------------
# compute_clustering
# ---------------------------------------------------------------------------


def _make_clustered_data(seed: int = 0) -> np.ndarray:
    """Return 200 points in 10-d space from 3 well-separated Gaussian clusters."""
    rng = np.random.RandomState(seed)
    centres = np.array([
        [10.0] * 10,
        [-10.0] * 10,
        [0.0, 0.0, 10.0, 10.0, -10.0, -10.0, 0.0, 0.0, 10.0, -10.0],
    ])
    parts = []
    for c in centres:
        parts.append(rng.randn(67, 10) * 0.5 + c)
    # 67 * 3 = 201; trim to exactly 200
    X = np.vstack(parts)[:200]
    return X.astype(np.float32)


def test_compute_clustering_hdbscan():
    pytest.importorskip("umap", reason="umap-learn not installed — HDBSCAN path needs UMAP pre-reduction")
    X = _make_clustered_data()
    params = _SimpleClusteringParams()
    labels, metrics = compute_clustering(X, method="hdbscan", params=params)
    assert labels.shape == (200,)
    non_noise = set(labels[labels != -1])
    assert len(non_noise) >= 2, (
        f"Expected at least 2 distinct non-noise clusters, got: {set(labels)}"
    )


def test_compute_clustering_kmeans():
    pytest.importorskip("sklearn", reason="scikit-learn not installed")
    X = _make_clustered_data()
    params = _SimpleClusteringParams()
    labels, metrics = compute_clustering(X, method="kmeans", params=params)
    assert labels.shape == (200,)
    assert set(labels) == {0, 1, 2}


def test_compute_clustering_invalid_method():
    pytest.importorskip("sklearn", reason="scikit-learn not installed")
    X = _make_clustered_data()
    params = _SimpleClusteringParams()
    with pytest.raises(ValueError, match="Unknown clustering method"):
        compute_clustering(X, method="dbscan", params=params)


# ---------------------------------------------------------------------------
# subsample_stratified
# ---------------------------------------------------------------------------


def _make_station_data(n_stations: int = 10, points_per_station: int = 100):
    """Build ids, embeddings, and metadata for stratified subsampling tests."""
    n = n_stations * points_per_station
    rng = np.random.RandomState(0)
    embeddings = rng.randn(n, 8).astype(np.float32)
    ids = [f"point_{i}" for i in range(n)]
    metadata = [
        {"station_id": f"station_{i // points_per_station}"}
        for i in range(n)
    ]
    return ids, embeddings, metadata


def test_subsample_stratified():
    ids, embeddings, metadata = _make_station_data(n_stations=10, points_per_station=100)
    ids_sub, emb_sub, meta_sub, was_subsampled, original_count = subsample_stratified(
        ids, embeddings, metadata, max_points=100
    )
    assert len(ids_sub) == 100
    assert emb_sub.shape == (100, 8)
    assert len(meta_sub) == 100
    assert was_subsampled is True
    assert original_count == 1000
    # All 10 stations must be represented
    station_ids_present = {m["station_id"] for m in meta_sub}
    assert len(station_ids_present) == 10, (
        f"Expected all 10 stations, got: {station_ids_present}"
    )


def test_subsample_no_op():
    ids, embeddings, metadata = _make_station_data(n_stations=5, points_per_station=10)
    # 50 points, max=100 → no subsampling
    ids_sub, emb_sub, meta_sub, was_subsampled, original_count = subsample_stratified(
        ids, embeddings, metadata, max_points=100
    )
    assert len(ids_sub) == 50
    assert emb_sub.shape == (50, 8)
    assert was_subsampled is False
    assert original_count == 50


# ---------------------------------------------------------------------------
# build_station_query
# ---------------------------------------------------------------------------


def test_build_station_query_piezo():
    sql, params = build_station_query("piezo", _SimpleFilters())
    sql_str = str(sql)
    assert "ml.piezo_station_embeddings" in sql_str
    assert isinstance(params, dict)


def test_build_station_query_hydro():
    sql, params = build_station_query("hydro", _SimpleFilters())
    sql_str = str(sql)
    assert "ml.hydro_station_embeddings" in sql_str
    assert isinstance(params, dict)


def test_build_station_query_piezo_with_filters():
    filters = _SimpleFilters(
        milieu_eh="Souterrain",
        departement="75",
        cluster_id=2,
    )
    sql, params = build_station_query("piezo", filters)
    sql_str = str(sql)
    assert "milieu_eh" in sql_str
    assert "departement" in sql_str
    assert "cluster_id" in sql_str
    assert params["milieu_eh"] == "Souterrain"
    assert params["departement"] == "75"
    assert params["cluster_id"] == 2


def test_build_station_query_invalid_domain():
    with pytest.raises(ValueError, match="Invalid domain"):
        build_station_query("invalid", _SimpleFilters())
