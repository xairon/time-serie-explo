"""Unit tests for dashboard/utils/cluster_profiling.py — pure math, no DB."""
from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Bootstrap: stub sqlalchemy if absent (same pattern as test_utils.py)
# ---------------------------------------------------------------------------
import sys
import types
import unittest.mock as mock

try:
    import sqlalchemy  # noqa: F401
except ImportError:
    _stub = types.ModuleType("sqlalchemy")
    _stub.text = lambda s: mock.MagicMock(__str__=lambda self: s)
    sys.modules["sqlalchemy"] = _stub

from dashboard.utils.cluster_profiling import (
    compute_metadata_distributions,
    compute_concordance,
    find_medoids,
    build_prototypes,
    compute_feature_fingerprints,
    compute_cluster_shap,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_stations(n=100):
    """Build a list of station dicts with 3 clusters and 2 metadata keys."""
    rng = np.random.RandomState(42)
    stations = []
    for i in range(n):
        cluster = i % 3  # 0, 1, 2 cycling
        stations.append({
            "id": f"BSS{i:04d}",
            "cluster_id": cluster,
            "metadata": {
                "milieu_eh": ["Poreux", "Fissuré", "Karstique"][cluster],
                "departement": rng.choice(["75", "69", "13"]),
            },
        })
    return stations


# ---------------------------------------------------------------------------
# compute_metadata_distributions
# ---------------------------------------------------------------------------

class TestMetadataDistributions:
    def test_basic_counting(self):
        stations = _make_stations(30)
        result = compute_metadata_distributions(stations, ["milieu_eh"])
        assert "milieu_eh" in result
        assert len(result["milieu_eh"]) == 3
        assert result["milieu_eh"]["0"]["Poreux"] == 10

    def test_multiple_keys(self):
        stations = _make_stations(30)
        result = compute_metadata_distributions(stations, ["milieu_eh", "departement"])
        assert "milieu_eh" in result
        assert "departement" in result

    def test_empty_stations(self):
        result = compute_metadata_distributions([], ["milieu_eh"])
        assert result == {"milieu_eh": {}}

    def test_null_metadata_skipped(self):
        stations = [
            {"id": "A", "cluster_id": 0, "metadata": {"key": "val"}},
            {"id": "B", "cluster_id": 0, "metadata": {"key": None}},
            {"id": "C", "cluster_id": 0, "metadata": {}},
        ]
        result = compute_metadata_distributions(stations, ["key"])
        assert result["key"]["0"] == {"val": 1}


# ---------------------------------------------------------------------------
# compute_concordance
# ---------------------------------------------------------------------------

class TestConcordance:
    def test_perfect_agreement(self):
        stations = _make_stations(30)
        result = compute_concordance(stations, ["milieu_eh"])
        assert result["milieu_eh"]["ari"] == pytest.approx(1.0)
        assert result["milieu_eh"]["nmi"] == pytest.approx(1.0)
        assert result["milieu_eh"]["cramers_v"] == pytest.approx(1.0)

    def test_random_agreement(self):
        rng = np.random.RandomState(0)
        stations = [
            {"id": f"S{i}", "cluster_id": i % 5,
             "metadata": {"random": rng.choice(["A", "B", "C", "D", "E"])}}
            for i in range(500)
        ]
        result = compute_concordance(stations, ["random"])
        assert result["random"]["ari"] < 0.1
        assert result["random"]["nmi"] < 0.15

    def test_single_cluster(self):
        stations = [
            {"id": f"S{i}", "cluster_id": 0,
             "metadata": {"key": ["A", "B"][i % 2]}}
            for i in range(20)
        ]
        result = compute_concordance(stations, ["key"])
        assert result["key"]["ari"] == 0.0
        assert result["key"]["nmi"] == 0.0
        assert result["key"]["cramers_v"] == 0.0

    def test_noise_stations_excluded(self):
        stations = [
            {"id": "A", "cluster_id": 0, "metadata": {"k": "X"}},
            {"id": "B", "cluster_id": 1, "metadata": {"k": "Y"}},
            {"id": "C", "cluster_id": -1, "metadata": {"k": "Z"}},
        ] * 10
        result = compute_concordance(stations, ["k"])
        assert "k" in result


# ---------------------------------------------------------------------------
# find_medoids
# ---------------------------------------------------------------------------

class TestFindMedoids:
    def test_basic(self):
        embeddings = {
            "A": np.array([1.0, 0.0]),
            "B": np.array([1.1, 0.1]),
            "C": np.array([-1.0, 0.0]),
            "D": np.array([-0.9, 0.1]),
        }
        cluster_labels = {"A": 0, "B": 0, "C": 1, "D": 1}
        medoids = find_medoids(embeddings, cluster_labels)
        assert medoids[0] in ("A", "B")
        assert medoids[1] in ("C", "D")

    def test_single_station_cluster(self):
        embeddings = {"X": np.array([0.0, 0.0])}
        cluster_labels = {"X": 0}
        medoids = find_medoids(embeddings, cluster_labels)
        assert medoids[0] == "X"

    def test_noise_excluded(self):
        embeddings = {
            "A": np.array([1.0, 0.0]),
            "N": np.array([99.0, 99.0]),
        }
        cluster_labels = {"A": 0, "N": -1}
        medoids = find_medoids(embeddings, cluster_labels)
        assert 0 in medoids
        assert -1 not in medoids


# ---------------------------------------------------------------------------
# build_prototypes
# ---------------------------------------------------------------------------

class TestBuildPrototypes:
    def test_basic(self):
        dates = [f"2024-01-{d:02d}" for d in range(1, 32)]
        series_map = {
            "A": np.sin(np.linspace(0, 2 * np.pi, 31)),
            "B": np.sin(np.linspace(0, 2 * np.pi, 31)) + 0.1,
        }
        dates_map = {"A": dates, "B": dates}
        medoid_ids = {0: "A"}
        cluster_members = {0: ["A", "B"]}
        result = build_prototypes(medoid_ids, cluster_members, series_map, dates_map)
        assert 0 in result
        assert result[0]["medoid_id"] == "A"
        assert len(result[0]["dates"]) == 31
        assert len(result[0]["medoid_values"]) == 31
        assert len(result[0]["p10"]) == 31
        assert len(result[0]["p90"]) == 31

    def test_small_cluster_no_envelope(self):
        dates = ["2024-01-01", "2024-01-02"]
        series_map = {"A": np.array([1.0, 2.0])}
        dates_map = {"A": dates}
        medoid_ids = {0: "A"}
        cluster_members = {0: ["A"]}
        result = build_prototypes(medoid_ids, cluster_members, series_map, dates_map)
        np.testing.assert_array_equal(result[0]["p10"], result[0]["medoid_values"])


# ---------------------------------------------------------------------------
# compute_feature_fingerprints
# ---------------------------------------------------------------------------

class TestFeatureFingerprints:
    def _make_series_data(self):
        from datetime import date, timedelta
        rng = np.random.RandomState(42)
        t = np.arange(365 * 3)
        start = date(2022, 1, 1)
        dates = [(start + timedelta(days=int(d))).isoformat() for d in t]
        series_map = {}
        dates_map = {}
        cluster_labels = {}

        for i in range(20):
            sid = f"S0_{i}"
            series_map[sid] = 10.0 + 3.0 * np.sin(2 * np.pi * t / 365) + rng.randn(len(t)) * 0.5
            dates_map[sid] = dates
            cluster_labels[sid] = 0

        for i in range(20):
            sid = f"S1_{i}"
            series_map[sid] = 2.0 + 0.3 * np.sin(2 * np.pi * t / 365) + rng.randn(len(t)) * 0.5
            dates_map[sid] = dates
            cluster_labels[sid] = 1

        return series_map, dates_map, cluster_labels

    def test_output_structure(self):
        series_map, dates_map, cluster_labels = self._make_series_data()
        normalized, raw, _ = compute_feature_fingerprints(series_map, dates_map, cluster_labels)
        assert set(normalized.keys()) == {0, 1}
        expected_features = {"mean", "std", "trend", "seasonality", "autocorr_365", "wet_dry_ratio"}
        assert set(normalized[0].keys()) == expected_features
        assert set(raw[0].keys()) == expected_features

    def test_normalized_range(self):
        series_map, dates_map, cluster_labels = self._make_series_data()
        normalized, _, _ = compute_feature_fingerprints(series_map, dates_map, cluster_labels)
        for cid in normalized:
            for feat, val in normalized[cid].items():
                assert 0.0 <= val <= 1.0, f"Cluster {cid}, {feat} = {val} not in [0,1]"

    def test_clusters_differ_on_mean(self):
        series_map, dates_map, cluster_labels = self._make_series_data()
        _, raw, _ = compute_feature_fingerprints(series_map, dates_map, cluster_labels)
        assert raw[0]["mean"] > raw[1]["mean"]

    def test_short_series_seasonality_nan(self):
        from datetime import date, timedelta
        series_map = {"A": np.array([1.0] * 100)}
        start = date(2024, 1, 1)
        dates_map = {"A": [(start + timedelta(days=i)).isoformat() for i in range(100)]}
        cluster_labels = {"A": 0}
        normalized, raw, _ = compute_feature_fingerprints(series_map, dates_map, cluster_labels)
        assert np.isnan(raw[0]["seasonality"])


# ---------------------------------------------------------------------------
# compute_cluster_shap
# ---------------------------------------------------------------------------

class TestClusterShap:
    def test_basic(self):
        pytest.importorskip("shap", reason="shap not installed")
        pytest.importorskip("sklearn", reason="sklearn not installed")
        rng = np.random.RandomState(42)
        n = 200
        features_df = {
            "mean": rng.randn(n),
            "std": rng.randn(n),
            "trend": rng.randn(n),
            "seasonality": rng.randn(n),
            "autocorr_365": rng.randn(n),
            "wet_dry_ratio": rng.randn(n),
        }
        labels = np.array([0 if features_df["mean"][i] < -0.3
                          else 2 if features_df["mean"][i] > 0.3
                          else 1 for i in range(n)])
        result = compute_cluster_shap(features_df, labels)
        assert "feature_importance" in result
        assert "shap_per_cluster" in result
        assert "proxy_accuracy" in result
        assert len(result["feature_importance"]) == 6
        assert max(result["feature_importance"], key=result["feature_importance"].get) == "mean"

    def test_single_cluster_returns_warning(self):
        features_df = {"mean": np.ones(50), "std": np.zeros(50)}
        labels = np.zeros(50, dtype=int)
        result = compute_cluster_shap(features_df, labels)
        assert result["warning"] is not None
        assert result["feature_importance"] == {}
