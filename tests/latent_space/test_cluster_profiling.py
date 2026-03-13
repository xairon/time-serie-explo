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
