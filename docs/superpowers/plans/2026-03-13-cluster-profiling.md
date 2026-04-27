# Cluster Profiling Panel — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 5-block cluster profiling panel (metadata distributions, concordance, temporal prototypes, feature fingerprints, SHAP) to the Latent Space page as a "Profiling" tab.

**Architecture:** Pure Python compute functions in `dashboard/utils/cluster_profiling.py` → Pydantic schemas → FastAPI GET endpoint with `asyncio.to_thread()` → React Query hook → 6 Plotly-based React components. Series data fetched via synchronous `postgres_connector` inside the thread, not the async session.

**Tech Stack:** numpy, scipy, scikit-learn, shap (backend); React 19, TanStack React Query, Plotly.js, Tailwind CSS (frontend)

**Spec:** `docs/superpowers/specs/2026-03-13-cluster-profiling-design.md`

---

## File Structure

### Create
| File | Responsibility |
|------|---------------|
| `dashboard/utils/cluster_profiling.py` | 6 pure compute functions (distributions, concordance, find_medoids, build_prototypes, fingerprints, SHAP) |
| `api/schemas/cluster_profiling.py` | Pydantic request/response models |
| `tests/latent_space/test_cluster_profiling.py` | Unit tests for all compute functions |
| `frontend/src/components/latent-space/ClusterProfiling.tsx` | Container component with skeleton loading |
| `frontend/src/components/latent-space/MetadataDistributions.tsx` | Plotly stacked bar chart |
| `frontend/src/components/latent-space/ConcordanceTable.tsx` | HTML table with color-coded cells |
| `frontend/src/components/latent-space/TemporalPrototypes.tsx` | Plotly subplots: medoid lines + P10/P90 envelopes |
| `frontend/src/components/latent-space/FeatureFingerprints.tsx` | Plotly Scatterpolar radar chart |
| `frontend/src/components/latent-space/ShapExplainability.tsx` | Plotly horizontal bar subplots |

### Modify
| File | Change |
|------|--------|
| `api/routers/latent_space.py` | Add `GET /profiling/{domain}` endpoint |
| `frontend/src/lib/api.ts` | Add `profiling()` to `latentSpace` namespace |
| `frontend/src/hooks/useLatentSpace.ts` | Add `useClusterProfiling` hook |
| `frontend/src/pages/LatentSpacePage.tsx` | Add Scatter/Profiling tab toggle |

---

## Chunk 1: Backend Compute Functions + Tests

### Task 1: Pydantic Schemas

**Files:**
- Create: `api/schemas/cluster_profiling.py`

- [ ] **Step 1: Create schema file**

```python
# api/schemas/cluster_profiling.py
"""Pydantic models for cluster profiling responses."""
from __future__ import annotations

from pydantic import BaseModel


class MetadataDistribution(BaseModel):
    key: str
    clusters: dict[str, dict[str, int]]  # cluster_id (str) → {value: count}


class ConcordanceMetric(BaseModel):
    key: str
    ari: float
    nmi: float
    cramers_v: float


class ClusterPrototype(BaseModel):
    cluster_id: int
    medoid_id: str
    dates: list[str]
    medoid_values: list[float]
    p10: list[float]
    p90: list[float]


class FeatureFingerprint(BaseModel):
    cluster_id: int
    features: dict[str, float]      # normalized [0,1]
    features_raw: dict[str, float]  # original values


class ShapExplanation(BaseModel):
    feature_importance: dict[str, float]
    shap_per_cluster: dict[str, dict[str, float]]  # cluster_id as string key
    proxy_accuracy: float
    warning: str | None = None


class ProfilingResponse(BaseModel):
    domain: str
    n_stations: int
    n_clusters: int
    distributions: list[MetadataDistribution]
    concordance: list[ConcordanceMetric]
    prototypes: list[ClusterPrototype]
    fingerprints: list[FeatureFingerprint]
    shap: ShapExplanation
    warnings: list[str] = []
```

- [ ] **Step 2: Verify schema imports work**

Run: `cd /home/ringuet/time-serie-explo && docker compose exec backend python -c "from api.schemas.cluster_profiling import ProfilingResponse; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add api/schemas/cluster_profiling.py
git commit -m "feat(profiling): add Pydantic schemas for cluster profiling"
```

---

### Task 2: compute_metadata_distributions + compute_concordance

**Files:**
- Create: `dashboard/utils/cluster_profiling.py`
- Create: `tests/latent_space/test_cluster_profiling.py`

- [ ] **Step 1: Write failing tests for distributions and concordance**

```python
# tests/latent_space/test_cluster_profiling.py
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
        # 3 clusters (0, 1, 2)
        assert len(result["milieu_eh"]) == 3
        # Cluster 0 all have "Poreux"
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
        """When cluster labels perfectly match metadata, ARI/NMI should be 1.0."""
        stations = _make_stations(30)
        # milieu_eh perfectly tracks cluster_id by construction
        result = compute_concordance(stations, ["milieu_eh"])
        assert result["milieu_eh"]["ari"] == pytest.approx(1.0)
        assert result["milieu_eh"]["nmi"] == pytest.approx(1.0)
        assert result["milieu_eh"]["cramers_v"] == pytest.approx(1.0)

    def test_random_agreement(self):
        """Random metadata should give low concordance."""
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
        """With only 1 cluster, metrics should be 0.0."""
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
        """Stations with cluster_id = -1 should be excluded."""
        stations = [
            {"id": "A", "cluster_id": 0, "metadata": {"k": "X"}},
            {"id": "B", "cluster_id": 1, "metadata": {"k": "Y"}},
            {"id": "C", "cluster_id": -1, "metadata": {"k": "Z"}},
        ] * 10
        result = compute_concordance(stations, ["k"])
        # Should only use stations with cluster_id >= 0
        assert "k" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/latent_space/test_cluster_profiling.py -v --tb=short 2>&1 | head -30`
Expected: ImportError or ModuleNotFoundError for `compute_metadata_distributions`

- [ ] **Step 3: Implement compute_metadata_distributions and compute_concordance**

```python
# dashboard/utils/cluster_profiling.py
"""Cluster profiling utilities: distributions, concordance, medoids, features, SHAP.

Pure Python module — NO framework imports.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# 1. Metadata distributions
# ---------------------------------------------------------------------------

def compute_metadata_distributions(
    stations: list[dict[str, Any]],
    meta_keys: list[str],
) -> dict[str, dict[str, dict[str, int]]]:
    """Count metadata value occurrences per cluster.

    Returns {key: {cluster_id_str: {value: count}}}.
    """
    result: dict[str, dict[str, dict[str, int]]] = {}
    for key in meta_keys:
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for s in stations:
            cid = s.get("cluster_id")
            val = s.get("metadata", {}).get(key)
            if cid is None or val is None or val == "":
                continue
            counts[str(cid)][str(val)] += 1
        result[key] = {k: dict(v) for k, v in counts.items()}
    return result


# ---------------------------------------------------------------------------
# 2. Concordance metrics (ARI, NMI, Cramér's V)
# ---------------------------------------------------------------------------

def compute_concordance(
    stations: list[dict[str, Any]],
    meta_keys: list[str],
) -> dict[str, dict[str, float]]:
    """Compute concordance between cluster assignments and metadata labels.

    Excludes noise stations (cluster_id < 0) and null metadata values.
    Returns {key: {ari, nmi, cramers_v}}.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from scipy.stats import chi2_contingency

    result: dict[str, dict[str, float]] = {}

    for key in meta_keys:
        cluster_labels = []
        meta_labels = []
        for s in stations:
            cid = s.get("cluster_id")
            val = s.get("metadata", {}).get(key)
            if cid is None or cid < 0 or val is None or val == "":
                continue
            cluster_labels.append(cid)
            meta_labels.append(str(val))

        # Need at least 2 distinct clusters for meaningful metrics
        unique_clusters = set(cluster_labels)
        if len(unique_clusters) <= 1:
            result[key] = {"ari": 0.0, "nmi": 0.0, "cramers_v": 0.0}
            continue

        ari = adjusted_rand_score(meta_labels, cluster_labels)
        nmi = normalized_mutual_info_score(meta_labels, cluster_labels)

        # Cramér's V via contingency table
        try:
            from collections import Counter
            # Build contingency matrix
            pairs = list(zip(meta_labels, cluster_labels))
            meta_vals = sorted(set(meta_labels))
            clust_vals = sorted(unique_clusters)
            contingency = np.zeros((len(meta_vals), len(clust_vals)), dtype=int)
            meta_idx = {v: i for i, v in enumerate(meta_vals)}
            clust_idx = {v: i for i, v in enumerate(clust_vals)}
            for m, c in pairs:
                contingency[meta_idx[m], clust_idx[c]] += 1

            chi2, _, _, _ = chi2_contingency(contingency)
            n = contingency.sum()
            r, c = contingency.shape
            denom = n * (min(r, c) - 1)
            cramers_v = float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0
        except Exception:
            cramers_v = 0.0

        result[key] = {
            "ari": float(ari),
            "nmi": float(nmi),
            "cramers_v": cramers_v,
        }

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/latent_space/test_cluster_profiling.py -v --tb=short 2>&1 | tail -20`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/utils/cluster_profiling.py tests/latent_space/test_cluster_profiling.py
git commit -m "feat(profiling): add metadata distributions and concordance functions with tests"
```

---

### Task 3: find_medoids + build_prototypes

**Files:**
- Modify: `dashboard/utils/cluster_profiling.py`
- Modify: `tests/latent_space/test_cluster_profiling.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/latent_space/test_cluster_profiling.py`:

```python
from dashboard.utils.cluster_profiling import find_medoids, build_prototypes

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
        """Clusters with < 3 stations should have p10==p90==medoid."""
        dates = ["2024-01-01", "2024-01-02"]
        series_map = {"A": np.array([1.0, 2.0])}
        dates_map = {"A": dates}
        medoid_ids = {0: "A"}
        cluster_members = {0: ["A"]}

        result = build_prototypes(medoid_ids, cluster_members, series_map, dates_map)
        np.testing.assert_array_equal(result[0]["p10"], result[0]["medoid_values"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/latent_space/test_cluster_profiling.py::TestFindMedoids -v --tb=short 2>&1 | tail -10`
Expected: ImportError for `find_medoids`

- [ ] **Step 3: Implement find_medoids and build_prototypes**

Add to `dashboard/utils/cluster_profiling.py`:

```python
# ---------------------------------------------------------------------------
# 3a. Find medoids (embedding space only — no series needed)
# ---------------------------------------------------------------------------

def find_medoids(
    embeddings_map: dict[str, np.ndarray],
    cluster_labels: dict[str, int],
) -> dict[int, str]:
    """Find the station closest to cluster centroid (L2) for each cluster.

    Returns {cluster_id: medoid_station_id}. Excludes noise (cluster_id < 0).
    """
    # Group station IDs by cluster
    clusters: dict[int, list[str]] = defaultdict(list)
    for sid, cid in cluster_labels.items():
        if cid >= 0:
            clusters[cid].append(sid)

    medoids: dict[int, str] = {}
    for cid, members in clusters.items():
        if len(members) == 1:
            medoids[cid] = members[0]
            continue
        # Stack embeddings for this cluster
        embs = np.stack([embeddings_map[sid] for sid in members])
        centroid = embs.mean(axis=0)
        dists = np.linalg.norm(embs - centroid, axis=1)
        medoids[cid] = members[int(np.argmin(dists))]
    return medoids


# ---------------------------------------------------------------------------
# 3b. Build temporal prototypes (medoid line + P10/P90 envelope)
# ---------------------------------------------------------------------------

def build_prototypes(
    medoid_ids: dict[int, str],
    cluster_members: dict[int, list[str]],
    series_map: dict[str, np.ndarray],
    dates_map: dict[str, list[str]],
    max_days: int = 1095,
) -> dict[int, dict[str, Any]]:
    """Build temporal prototypes: medoid series + P10/P90 envelope.

    series_map: {station_id: 1d array of values}
    dates_map: {station_id: list of date strings, same length as series}

    Truncates to last `max_days` days. If cluster has < 3 members with series,
    returns medoid only (p10 = p90 = medoid_values).
    """
    result: dict[int, dict[str, Any]] = {}

    for cid, med_id in medoid_ids.items():
        if med_id not in series_map or med_id not in dates_map:
            continue

        med_series = series_map[med_id]
        med_dates = dates_map[med_id]

        # Truncate to last max_days
        if len(med_series) > max_days:
            med_series = med_series[-max_days:]
            med_dates = med_dates[-max_days:]

        n_points = len(med_series)
        med_values = [float(v) if np.isfinite(v) else None for v in med_series]

        # Collect aligned series from cluster members for envelope
        members = cluster_members.get(cid, [])
        member_series = []
        for sid in members:
            if sid not in series_map or sid not in dates_map:
                continue
            s = series_map[sid]
            # Align to same length as medoid (truncate from end)
            if len(s) >= n_points:
                member_series.append(s[-n_points:])

        if len(member_series) >= 3:
            stacked = np.stack(member_series)
            p10 = [float(v) if np.isfinite(v) else None
                   for v in np.nanpercentile(stacked, 10, axis=0)]
            p90 = [float(v) if np.isfinite(v) else None
                   for v in np.nanpercentile(stacked, 90, axis=0)]
        else:
            p10 = med_values
            p90 = med_values

        result[cid] = {
            "medoid_id": med_id,
            "dates": list(med_dates[-n_points:]),
            "medoid_values": med_values,
            "p10": p10,
            "p90": p90,
        }

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/latent_space/test_cluster_profiling.py::TestFindMedoids tests/latent_space/test_cluster_profiling.py::TestBuildPrototypes -v --tb=short`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/utils/cluster_profiling.py tests/latent_space/test_cluster_profiling.py
git commit -m "feat(profiling): add find_medoids and build_prototypes functions with tests"
```

---

### Task 4: compute_feature_fingerprints

**Files:**
- Modify: `dashboard/utils/cluster_profiling.py`
- Modify: `tests/latent_space/test_cluster_profiling.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/latent_space/test_cluster_profiling.py`:

```python
from dashboard.utils.cluster_profiling import compute_feature_fingerprints

# ---------------------------------------------------------------------------
# compute_feature_fingerprints
# ---------------------------------------------------------------------------

class TestFeatureFingerprints:
    def _make_series_data(self):
        """Create synthetic daily series for 2 clusters."""
        from datetime import date, timedelta
        rng = np.random.RandomState(42)
        t = np.arange(365 * 3)  # 3 years daily
        start = date(2022, 1, 1)
        dates = [(start + timedelta(days=int(d))).isoformat() for d in t]
        series_map = {}
        dates_map = {}
        cluster_labels = {}

        # Cluster 0: high mean, strong seasonality
        for i in range(20):
            sid = f"S0_{i}"
            series_map[sid] = 10.0 + 3.0 * np.sin(2 * np.pi * t / 365) + rng.randn(len(t)) * 0.5
            dates_map[sid] = dates
            cluster_labels[sid] = 0

        # Cluster 1: low mean, weak seasonality
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
        normalized, _ = compute_feature_fingerprints(series_map, dates_map, cluster_labels)
        for cid in normalized:
            for feat, val in normalized[cid].items():
                assert 0.0 <= val <= 1.0, f"Cluster {cid}, {feat} = {val} not in [0,1]"

    def test_clusters_differ_on_mean(self):
        """Cluster 0 has higher mean than cluster 1 — raw values should reflect this."""
        series_map, dates_map, cluster_labels = self._make_series_data()
        _, raw = compute_feature_fingerprints(series_map, dates_map, cluster_labels)
        assert raw[0]["mean"] > raw[1]["mean"]

    def test_short_series_seasonality_nan(self):
        """Series shorter than 365 days should give NaN seasonality."""
        from datetime import date, timedelta
        series_map = {"A": np.array([1.0] * 100)}
        start = date(2024, 1, 1)
        dates_map = {"A": [(start + timedelta(days=i)).isoformat() for i in range(100)]}
        cluster_labels = {"A": 0}
        normalized, raw, _ = compute_feature_fingerprints(series_map, dates_map, cluster_labels)
        assert np.isnan(raw[0]["seasonality"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/latent_space/test_cluster_profiling.py::TestFeatureFingerprints::test_output_structure -v --tb=short 2>&1 | tail -10`
Expected: ImportError for `compute_feature_fingerprints`

- [ ] **Step 3: Implement compute_feature_fingerprints**

Add to `dashboard/utils/cluster_profiling.py`:

```python
# ---------------------------------------------------------------------------
# 4. Feature fingerprints (6 time-series features per station)
# ---------------------------------------------------------------------------

def _compute_station_features(
    values: np.ndarray,
    dates: list[str],
) -> dict[str, float]:
    """Compute 6 features for a single station's time series."""
    n = len(values)
    valid = values[np.isfinite(values)]
    if len(valid) < 30:
        return {k: float("nan") for k in
                ["mean", "std", "trend", "seasonality", "autocorr_365", "wet_dry_ratio"]}

    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))

    # Trend: slope of linear regression
    x = np.arange(n, dtype=float)
    mask = np.isfinite(values)
    if mask.sum() > 1:
        coeffs = np.polyfit(x[mask], values[mask], 1)
        trend = float(coeffs[0])
    else:
        trend = 0.0

    # Seasonality: FFT amplitude at frequency nearest 1/365
    if n >= 365:
        filled = np.where(np.isfinite(values), values, np.nanmean(values))
        fft_vals = np.fft.rfft(filled - np.mean(filled))
        freqs = np.fft.rfftfreq(n, d=1.0)  # d=1 day
        target_freq = 1.0 / 365.0
        idx = np.argmin(np.abs(freqs - target_freq))
        seasonality = float(2.0 * np.abs(fft_vals[idx]) / n)
    else:
        seasonality = float("nan")

    # Autocorrelation at lag 365
    if n > 365:
        x1 = values[:n - 365]
        x2 = values[365:]
        mask2 = np.isfinite(x1) & np.isfinite(x2)
        if mask2.sum() > 10:
            autocorr_365 = float(np.corrcoef(x1[mask2], x2[mask2])[0, 1])
        else:
            autocorr_365 = float("nan")
    else:
        autocorr_365 = float("nan")

    # Wet/dry ratio: mean(DJF) / mean(JJA)
    try:
        months = np.array([int(d.split("-")[1]) for d in dates])
        djf_mask = np.isin(months, [12, 1, 2]) & np.isfinite(values)
        jja_mask = np.isin(months, [6, 7, 8]) & np.isfinite(values)
        if djf_mask.sum() > 0 and jja_mask.sum() > 0:
            jja_mean = float(np.mean(values[jja_mask]))
            if abs(jja_mean) > 1e-10:
                wet_dry = float(np.clip(np.mean(values[djf_mask]) / jja_mean, 0.0, 5.0))
            else:
                wet_dry = float("nan")
        else:
            wet_dry = float("nan")
    except Exception:
        wet_dry = float("nan")

    return {
        "mean": mean,
        "std": std,
        "trend": trend,
        "seasonality": seasonality,
        "autocorr_365": autocorr_365,
        "wet_dry_ratio": wet_dry,
    }


def compute_feature_fingerprints(
    series_map: dict[str, np.ndarray],
    dates_map: dict[str, list[str]],
    cluster_labels: dict[str, int],
) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]], dict[str, dict[str, float]]]:
    """Compute 6 time-series features per station, aggregate per cluster.

    Returns (normalized, raw, per_station) where:
    - normalized: {cluster_id: {feature: value_in_0_1}} per-feature min-max scaled
    - raw: {cluster_id: {feature: median_value}} unscaled
    - per_station: {station_id: {feature: value}} for SHAP input
    """
    FEATURES = ["mean", "std", "trend", "seasonality", "autocorr_365", "wet_dry_ratio"]

    # Compute per-station features
    station_features: dict[str, dict[str, float]] = {}
    for sid in series_map:
        if sid in cluster_labels and cluster_labels[sid] >= 0:
            station_features[sid] = _compute_station_features(
                series_map[sid], dates_map.get(sid, [])
            )

    # Group by cluster and compute medians
    cluster_features: dict[int, list[dict[str, float]]] = defaultdict(list)
    for sid, feats in station_features.items():
        cluster_features[cluster_labels[sid]].append(feats)

    raw: dict[int, dict[str, float]] = {}
    for cid, feat_list in cluster_features.items():
        medians: dict[str, float] = {}
        for f in FEATURES:
            vals = [d[f] for d in feat_list if np.isfinite(d[f])]
            medians[f] = float(np.median(vals)) if vals else float("nan")
        raw[cid] = medians

    # Normalize per-feature to [0, 1] across clusters
    normalized: dict[int, dict[str, float]] = {}
    for cid in raw:
        normalized[cid] = {}

    for f in FEATURES:
        vals = [raw[cid][f] for cid in raw if np.isfinite(raw[cid][f])]
        if len(vals) < 2 or max(vals) == min(vals):
            for cid in raw:
                normalized[cid][f] = 0.5
        else:
            lo, hi = min(vals), max(vals)
            for cid in raw:
                v = raw[cid][f]
                normalized[cid][f] = float((v - lo) / (hi - lo)) if np.isfinite(v) else 0.5

    return normalized, raw, station_features
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/latent_space/test_cluster_profiling.py::TestFeatureFingerprints -v --tb=short`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/utils/cluster_profiling.py tests/latent_space/test_cluster_profiling.py
git commit -m "feat(profiling): add feature fingerprints computation with 6 time-series features"
```

---

### Task 5: compute_cluster_shap

**Files:**
- Modify: `dashboard/utils/cluster_profiling.py`
- Modify: `tests/latent_space/test_cluster_profiling.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/latent_space/test_cluster_profiling.py`:

```python
from dashboard.utils.cluster_profiling import compute_cluster_shap

# ---------------------------------------------------------------------------
# compute_cluster_shap
# ---------------------------------------------------------------------------

class TestClusterShap:
    def test_basic(self):
        pytest.importorskip("shap", reason="shap not installed")
        pytest.importorskip("sklearn", reason="sklearn not installed")
        rng = np.random.RandomState(42)
        # 200 samples, 6 features, 3 clusters
        n = 200
        features_df = {
            "mean": rng.randn(n),
            "std": rng.randn(n),
            "trend": rng.randn(n),
            "seasonality": rng.randn(n),
            "autocorr_365": rng.randn(n),
            "wet_dry_ratio": rng.randn(n),
        }
        # Make labels correlate with "mean" feature
        labels = np.array([0 if features_df["mean"][i] < -0.3
                          else 2 if features_df["mean"][i] > 0.3
                          else 1 for i in range(n)])
        result = compute_cluster_shap(features_df, labels)
        assert "feature_importance" in result
        assert "shap_per_cluster" in result
        assert "proxy_accuracy" in result
        assert len(result["feature_importance"]) == 6
        # "mean" should be most important since labels depend on it
        assert max(result["feature_importance"], key=result["feature_importance"].get) == "mean"

    def test_single_cluster_returns_warning(self):
        features_df = {"mean": np.ones(50), "std": np.zeros(50)}
        labels = np.zeros(50, dtype=int)
        result = compute_cluster_shap(features_df, labels)
        assert result["warning"] is not None
        assert result["feature_importance"] == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/latent_space/test_cluster_profiling.py::TestClusterShap::test_single_cluster_returns_warning -v --tb=short 2>&1 | tail -10`
Expected: ImportError for `compute_cluster_shap`

- [ ] **Step 3: Implement compute_cluster_shap**

Add to `dashboard/utils/cluster_profiling.py`:

```python
# ---------------------------------------------------------------------------
# 5. SHAP explainability (RF proxy → TreeExplainer)
# ---------------------------------------------------------------------------

def compute_cluster_shap(
    features_df: dict[str, np.ndarray],
    labels: np.ndarray,
) -> dict[str, Any]:
    """Train RF proxy on features → cluster labels, compute SHAP values.

    features_df: {feature_name: array of shape (n_samples,)}
    labels: array of shape (n_samples,) with cluster IDs

    Returns dict with feature_importance, shap_per_cluster, proxy_accuracy, warning.
    """
    unique_labels = sorted(set(labels))
    if len(unique_labels) <= 1:
        return {
            "feature_importance": {},
            "shap_per_cluster": {},
            "proxy_accuracy": 0.0,
            "warning": "Only 1 cluster — SHAP analysis not applicable.",
        }

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    import shap

    feature_names = list(features_df.keys())
    X = np.column_stack([features_df[f] for f in feature_names])

    # Replace NaN with column median for RF
    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            median_val = np.nanmedian(col)
            X[nan_mask, col_idx] = median_val if np.isfinite(median_val) else 0.0

    # Train RF
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf.fit(X, labels)

    # Cross-val accuracy (fallback to training accuracy if folds fail on tiny clusters)
    try:
        scores = cross_val_score(rf, X, labels, cv=min(5, len(unique_labels)), scoring="accuracy")
        accuracy = float(np.mean(scores))
    except ValueError:
        accuracy = float(rf.score(X, labels))

    if accuracy < 0.3:
        return {
            "feature_importance": {},
            "shap_per_cluster": {},
            "proxy_accuracy": accuracy,
            "warning": f"Proxy accuracy too low ({accuracy:.1%}) — SHAP values unreliable.",
        }

    # SHAP values
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)

    # Handle different shap_values return types:
    # - shap < 0.42: list of N arrays for multiclass, single array for binary
    # - shap >= 0.42: 3D array (n_samples, n_features, n_classes) for multiclass
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # shap >= 0.42: transpose to (n_classes, n_samples, n_features)
        stacked = np.transpose(shap_values, (2, 0, 1))
        global_importance = np.mean(np.abs(stacked), axis=(0, 1))
        feature_importance = {
            feature_names[i]: float(global_importance[i])
            for i in range(len(feature_names))
        }
        shap_per_cluster: dict[str, dict[str, float]] = {}
        classes = rf.classes_
        for cls_idx, cls_label in enumerate(classes):
            mask = labels == cls_label
            if mask.sum() == 0:
                continue
            mean_shap = np.mean(stacked[cls_idx][mask], axis=0)
            shap_per_cluster[str(cls_label)] = {
                feature_names[i]: float(mean_shap[i])
                for i in range(len(feature_names))
            }
    elif isinstance(shap_values, list):
        # shap < 0.42 multiclass: list of N arrays, each (n_samples, n_features)
        stacked = np.stack(shap_values)  # (n_classes, n_samples, n_features)
        global_importance = np.mean(np.abs(stacked), axis=(0, 1))
        feature_importance = {
            feature_names[i]: float(global_importance[i])
            for i in range(len(feature_names))
        }

        # Per-cluster signed values
        shap_per_cluster: dict[str, dict[str, float]] = {}
        classes = rf.classes_
        for cls_idx, cls_label in enumerate(classes):
            mask = labels == cls_label
            if mask.sum() == 0:
                continue
            mean_shap = np.mean(shap_values[cls_idx][mask], axis=0)
            shap_per_cluster[str(cls_label)] = {
                feature_names[i]: float(mean_shap[i])
                for i in range(len(feature_names))
            }
    else:
        # Binary: single array (n_samples, n_features)
        global_importance = np.mean(np.abs(shap_values), axis=0)
        feature_importance = {
            feature_names[i]: float(global_importance[i])
            for i in range(len(feature_names))
        }
        shap_per_cluster = {}
        for cls_label in unique_labels:
            mask = labels == cls_label
            if mask.sum() == 0:
                continue
            mean_shap = np.mean(shap_values[mask], axis=0)
            shap_per_cluster[str(cls_label)] = {
                feature_names[i]: float(mean_shap[i])
                for i in range(len(feature_names))
            }

    return {
        "feature_importance": feature_importance,
        "shap_per_cluster": shap_per_cluster,
        "proxy_accuracy": accuracy,
        "warning": None,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/latent_space/test_cluster_profiling.py::TestClusterShap -v --tb=short`
Expected: All PASS (or skipped if shap not installed locally — will pass in Docker)

- [ ] **Step 5: Commit**

```bash
git add dashboard/utils/cluster_profiling.py tests/latent_space/test_cluster_profiling.py
git commit -m "feat(profiling): add SHAP cluster explainability with RF proxy"
```

---

### Task 6: FastAPI Profiling Endpoint

**Files:**
- Modify: `api/routers/latent_space.py`

- [ ] **Step 1: Add profiling endpoint**

Add at the end of `api/routers/latent_space.py` (after the existing `similar` endpoint):

```python
@router.get("/profiling/{domain}", response_model=ProfilingResponse)
async def get_profiling(
    domain: str,
    hide_unclassified: bool = Query(False),
    session: AsyncSession = Depends(get_brgm_db),
) -> ProfilingResponse:
    """Compute comprehensive cluster profiling for a domain."""
    import numpy as np
    from dashboard.utils.latent_space import build_station_query, parse_pgvector, decode_eh_metadata
    from dashboard.utils.cluster_profiling import (
        compute_metadata_distributions,
        compute_concordance,
        find_medoids,
        build_prototypes,
        compute_feature_fingerprints,
        compute_cluster_shap,
    )

    if domain not in _VALID_DOMAINS:
        raise HTTPException(status_code=400, detail=f"Invalid domain: {domain}")

    # --- Phase 0: Load embeddings + metadata ---
    from api.schemas.latent_space import EmbeddingFilters
    sql, params = build_station_query(domain, EmbeddingFilters())
    result = await session.execute(sql, params)
    rows = result.fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail="No embeddings found")

    EH_KEYS = ["milieu_eh", "theme_eh", "etat_eh", "nature_eh"]
    META_KEYS = (EH_KEYS + ["departement"]) if domain == "piezo" else ["nom_cours_eau", "departement"]

    stations: list[dict] = []
    embeddings_map: dict[str, np.ndarray] = {}
    cluster_labels: dict[str, int] = {}

    for row in rows:
        raw = getattr(row, "embedding_raw", None)
        if raw is None:
            continue
        try:
            emb = parse_pgvector(raw)
        except Exception:
            continue

        sid = str(row.id)
        cid = int(row.cluster_id) if getattr(row, "cluster_id", None) is not None else -1

        if domain == "piezo":
            meta = decode_eh_metadata({
                "libelle_eh": getattr(row, "libelle_eh", None),
                "milieu_eh": getattr(row, "milieu_eh", None),
                "theme_eh": getattr(row, "theme_eh", None),
                "etat_eh": getattr(row, "etat_eh", None),
                "nature_eh": getattr(row, "nature_eh", None),
                "departement": getattr(row, "departement", None),
            })
        else:
            meta = {
                "nom_cours_eau": getattr(row, "nom_cours_eau", None),
                "departement": getattr(row, "departement", None),
            }

        # Apply hide_unclassified filter
        if hide_unclassified and domain == "piezo":
            if all(meta.get(k) in (None, "") for k in EH_KEYS):
                continue

        stations.append({"id": sid, "cluster_id": cid, "metadata": meta})
        embeddings_map[sid] = emb
        cluster_labels[sid] = cid

    if not stations:
        raise HTTPException(status_code=404, detail="No stations after filtering")

    n_clusters = len(set(cid for cid in cluster_labels.values() if cid >= 0))

    # --- Phase 1: compute medoids (no series needed) ---
    medoid_ids = find_medoids(embeddings_map, cluster_labels)

    # Group cluster members
    cluster_members: dict[int, list[str]] = defaultdict(list)
    for sid, cid in cluster_labels.items():
        if cid >= 0:
            cluster_members[cid].append(sid)

    # --- Phase 2: fetch series via synchronous connection ---
    def _blocking_profiling():
        import random
        from api.config import settings
        from dashboard.utils.postgres_connector import create_connection
        from sqlalchemy import text as sa_text

        engine = create_connection(
            host=settings.brgm_db_host,
            port=settings.brgm_db_port,
            database=settings.brgm_db_name,
            user=settings.brgm_db_user,
            password=settings.brgm_db_password,
        )

        # Determine table and column
        if domain == "piezo":
            table = "gold.hubeau_daily_chroniques"
            col = "niveau_nappe_eau"
            id_col = "code_bss"
            extra_filter = ""
        else:
            table = "gold.hydro_daily_chroniques"
            col = "resultat_obs_elab"
            id_col = "code_station"
            extra_filter = " AND grandeur_hydro_elab = 'QmnJ'"

        # Collect all station IDs we need series for
        all_sids = list(s["id"] for s in stations if cluster_labels.get(s["id"], -1) >= 0)

        # Fetch in batches of 500
        series_map: dict[str, np.ndarray] = {}
        dates_map: dict[str, list[str]] = {}

        for i in range(0, len(all_sids), 500):
            batch = all_sids[i:i + 500]
            placeholders = ", ".join(f":id_{j}" for j in range(len(batch)))
            query = sa_text(
                f"SELECT {id_col} AS station_id, date_mesure, {col} AS value "
                f"FROM {table} "
                f"WHERE {id_col} IN ({placeholders}){extra_filter} "
                f"AND date_mesure >= CURRENT_DATE - INTERVAL '5 years' "
                f"ORDER BY {id_col}, date_mesure"
            )
            params = {f"id_{j}": sid for j, sid in enumerate(batch)}

            with engine.connect() as conn:
                rows = conn.execute(query, params).fetchall()

            # Parse into per-station arrays
            current_sid = None
            current_dates: list[str] = []
            current_vals: list[float] = []
            for r in rows:
                sid = str(r.station_id)
                if sid != current_sid:
                    if current_sid and current_dates:
                        series_map[current_sid] = np.array(current_vals, dtype=float)
                        dates_map[current_sid] = current_dates
                    current_sid = sid
                    current_dates = []
                    current_vals = []
                current_dates.append(str(r.date_mesure))
                current_vals.append(float(r.value) if r.value is not None else float("nan"))
            if current_sid and current_dates:
                series_map[current_sid] = np.array(current_vals, dtype=float)
                dates_map[current_sid] = current_dates

        engine.dispose()

        # --- Compute all profiling blocks ---
        warnings: list[str] = []

        # 1. Metadata distributions
        distributions = compute_metadata_distributions(stations, META_KEYS)

        # 2. Concordance
        concordance = compute_concordance(stations, META_KEYS)

        # 3. Prototypes
        prototypes = build_prototypes(medoid_ids, cluster_members, series_map, dates_map)

        # 4. Feature fingerprints (also returns per-station features for SHAP)
        normalized_fp, raw_fp, per_station_features = compute_feature_fingerprints(
            series_map, dates_map, cluster_labels
        )

        # 5. SHAP — reuse per-station features from fingerprints
        feature_names = ["mean", "std", "trend", "seasonality", "autocorr_365", "wet_dry_ratio"]
        shap_features: dict[str, list[float]] = {f: [] for f in feature_names}
        shap_labels: list[int] = []
        for sid in all_sids:
            if sid in per_station_features:
                feats = per_station_features[sid]
                for f in feature_names:
                    shap_features[f].append(feats[f])
                shap_labels.append(cluster_labels[sid])

        shap_features_np = {f: np.array(v) for f, v in shap_features.items()}
        shap_result = compute_cluster_shap(shap_features_np, np.array(shap_labels))

        if shap_result.get("warning"):
            warnings.append(shap_result["warning"])

        return distributions, concordance, prototypes, normalized_fp, raw_fp, shap_result, warnings

    (distributions, concordance, prototypes, normalized_fp, raw_fp,
     shap_result, warnings) = await asyncio.to_thread(_blocking_profiling)

    # --- Build response ---
    from api.schemas.cluster_profiling import (
        ProfilingResponse, MetadataDistribution, ConcordanceMetric,
        ClusterPrototype, FeatureFingerprint, ShapExplanation,
    )

    return ProfilingResponse(
        domain=domain,
        n_stations=len(stations),
        n_clusters=n_clusters,
        distributions=[
            MetadataDistribution(key=k, clusters=v)
            for k, v in distributions.items()
        ],
        concordance=[
            ConcordanceMetric(key=k, **v)
            for k, v in concordance.items()
        ],
        prototypes=[
            ClusterPrototype(cluster_id=cid, **data)
            for cid, data in sorted(prototypes.items())
        ],
        fingerprints=[
            FeatureFingerprint(
                cluster_id=cid,
                features=normalized_fp.get(cid, {}),
                features_raw=raw_fp.get(cid, {}),
            )
            for cid in sorted(normalized_fp.keys())
        ],
        shap=ShapExplanation(**shap_result),
        warnings=warnings,
    )
```

Also add imports at the top of the file (alongside existing imports):

```python
from collections import Counter, defaultdict  # add defaultdict to existing Counter import
from api.schemas.cluster_profiling import ProfilingResponse
```

- [ ] **Step 2: Test endpoint in Docker**

Run: `cd /home/ringuet/time-serie-explo && docker compose up -d --build`
Then: `curl -s http://localhost:49513/api/v1/latent-space/profiling/piezo | python3 -m json.tool | head -30`
Expected: JSON response with `domain`, `n_stations`, `n_clusters`, `distributions`, etc.

- [ ] **Step 3: Commit**

```bash
git add api/routers/latent_space.py api/schemas/cluster_profiling.py
git commit -m "feat(profiling): add GET /profiling/{domain} endpoint"
```

---

## Chunk 2: Frontend Components

### Task 7: API Client + Hook

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/hooks/useLatentSpace.ts`

- [ ] **Step 1: Add profiling method to API client**

In `frontend/src/lib/api.ts`, add to the `latentSpace` object (after the `similar` method around line 267):

```typescript
profiling: (domain: string, hideUnclassified: boolean = false) =>
  fetchJson<Record<string, unknown>>(
    `/latent-space/profiling/${domain}?hide_unclassified=${hideUnclassified}`,
    { timeout: 60_000 },
  ),
```

- [ ] **Step 2: Add useClusterProfiling hook**

Append to `frontend/src/hooks/useLatentSpace.ts`:

```typescript
export function useClusterProfiling(domain: string, hideUnclassified: boolean) {
  return useQuery({
    queryKey: ['latent-space', 'profiling', domain, hideUnclassified],
    queryFn: () => api.latentSpace.profiling(domain, hideUnclassified),
    staleTime: 5 * 60 * 1000,
    gcTime: 30 * 60 * 1000,
    enabled: !!domain,
  })
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/api.ts frontend/src/hooks/useLatentSpace.ts
git commit -m "feat(profiling): add profiling API method and React Query hook"
```

---

### Task 8: ClusterProfiling Container + Tab Toggle

**Files:**
- Create: `frontend/src/components/latent-space/ClusterProfiling.tsx`
- Modify: `frontend/src/pages/LatentSpacePage.tsx`

- [ ] **Step 1: Create ClusterProfiling container component**

```typescript
// frontend/src/components/latent-space/ClusterProfiling.tsx
import { useClusterProfiling } from '@/hooks/useLatentSpace'
import { AlertTriangle } from 'lucide-react'
import { MetadataDistributions } from './MetadataDistributions'
import { ConcordanceTable } from './ConcordanceTable'
import { TemporalPrototypes } from './TemporalPrototypes'
import { FeatureFingerprints } from './FeatureFingerprints'
import { ShapExplainability } from './ShapExplainability'

interface ClusterProfilingProps {
  domain: 'piezo' | 'hydro'
  hideUnclassified: boolean
}

function SkeletonBlock({ label }: { label: string }) {
  return (
    <div className="bg-bg-card rounded-xl border border-white/5 p-6">
      <div className="flex items-center gap-3">
        <div className="w-5 h-5 border-2 border-accent-cyan border-t-transparent rounded-full animate-spin" />
        <span className="text-text-muted text-sm">{label}</span>
      </div>
    </div>
  )
}

export function ClusterProfiling({ domain, hideUnclassified }: ClusterProfilingProps) {
  const { data, isLoading, isError } = useClusterProfiling(domain, hideUnclassified)

  if (isLoading) {
    return (
      <div className="flex flex-col gap-4 overflow-y-auto pr-2">
        <SkeletonBlock label="Loading metadata distributions..." />
        <SkeletonBlock label="Loading concordance metrics..." />
        <SkeletonBlock label="Loading temporal prototypes..." />
        <SkeletonBlock label="Loading feature fingerprints..." />
        <SkeletonBlock label="Loading SHAP analysis..." />
      </div>
    )
  }

  if (isError || !data) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="bg-bg-card rounded-xl border border-white/5 p-8 flex flex-col items-center gap-4 max-w-md">
          <AlertTriangle className="w-10 h-10 text-accent-red" />
          <p className="text-text-primary text-center">Failed to load profiling data</p>
        </div>
      </div>
    )
  }

  const profiling = data as Record<string, unknown>
  const warnings = (profiling.warnings as string[]) ?? []

  return (
    <div className="flex flex-col gap-4 overflow-y-auto pr-2">
      {warnings.length > 0 && (
        <div className="flex items-center gap-2 bg-amber-500/10 text-amber-400 px-4 py-2 rounded-lg text-sm">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>{warnings.join(' | ')}</span>
        </div>
      )}

      <div className="text-text-muted text-xs">
        {profiling.n_stations as number} stations · {profiling.n_clusters as number} clusters
      </div>

      <MetadataDistributions
        distributions={(profiling.distributions as Record<string, unknown>[]) ?? []}
        domain={domain}
      />
      <ConcordanceTable
        concordance={(profiling.concordance as Record<string, unknown>[]) ?? []}
      />
      <TemporalPrototypes
        prototypes={(profiling.prototypes as Record<string, unknown>[]) ?? []}
      />
      <FeatureFingerprints
        fingerprints={(profiling.fingerprints as Record<string, unknown>[]) ?? []}
      />
      <ShapExplainability
        shap={(profiling.shap as Record<string, unknown>) ?? {}}
      />
    </div>
  )
}
```

- [ ] **Step 2: Add tab toggle to LatentSpacePage**

In `frontend/src/pages/LatentSpacePage.tsx`:

1. Add import: `import { ClusterProfiling } from '@/components/latent-space/ClusterProfiling'`

2. Add state (with existing state declarations, around line 34):
```typescript
const [activeTab, setActiveTab] = useState<'scatter' | 'profiling'>('scatter')
```

3. **CRITICAL: Restructure the page so the top bar (with tab toggle) renders BEFORE the isLoading/isError early returns.** Replace the entire component return with this structure:

```tsx
// Remove the existing isLoading and isError early returns (lines 252-282).
// Instead, restructure the main return to always show the top bar:

return (
  <div className="flex flex-col h-full gap-3 p-4 overflow-hidden">
    {/* Top bar: domain switch + tab toggle + stats — ALWAYS visible */}
    <div className="flex items-center gap-4 shrink-0">
      {/* Domain switch (existing code unchanged) */}
      <div className="flex rounded-lg overflow-hidden border border-white/10">
        {/* ... piezo/hydro buttons ... */}
      </div>

      {/* Tab toggle — NEW */}
      <div className="flex rounded-lg overflow-hidden border border-white/10">
        <button
          onClick={() => setActiveTab('scatter')}
          className={`px-4 py-2 text-sm transition-colors ${
            activeTab === 'scatter'
              ? 'bg-accent-cyan/20 text-accent-cyan'
              : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover'
          }`}
        >
          Scatter
        </button>
        <button
          onClick={() => setActiveTab('profiling')}
          className={`px-4 py-2 text-sm transition-colors ${
            activeTab === 'profiling'
              ? 'bg-accent-cyan/20 text-accent-cyan'
              : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover'
          }`}
        >
          Profiling
        </button>
      </div>

      {/* Stats (existing code) */}
      <span className="text-text-muted text-sm">...</span>
    </div>

    {/* Loading state — now inline, not early return */}
    {isLoading ? (
      <div className="flex items-center justify-center flex-1">
        <div className="flex flex-col items-center gap-3">
          <div className="w-10 h-10 border-2 border-accent-cyan border-t-transparent rounded-full animate-spin" />
          <span className="text-text-secondary text-sm">Loading embeddings...</span>
        </div>
      </div>
    ) : isError ? (
      <div className="flex items-center justify-center flex-1">
        {/* ... existing error UI ... */}
      </div>
    ) : (
      <div className="flex gap-4 flex-1 min-h-0">
        {/* Filter sidebar (existing) */}
        <div className="shrink-0 overflow-y-auto">
          <FilterPanel ... />
        </div>

        {/* Main content: scatter or profiling */}
        {activeTab === 'profiling' ? (
          <div className="flex-1 min-w-0">
            <ClusterProfiling domain={domain} hideUnclassified={hideUnclassified} />
          </div>
        ) : (
          <div className="flex-1 flex flex-col min-w-0 gap-2">
            {/* ... existing scatter + controls code unchanged ... */}
          </div>
        )}

        {/* Right sidebar (existing) */}
        <div className="shrink-0 flex flex-col gap-3 overflow-y-auto">
          {/* ... StationDetail + QualityMetrics ... */}
        </div>
      </div>
    )}
  </div>
)
```

The key change: remove the two `if (isLoading)` / `if (isError)` early returns (lines 252-282) and replace them with inline ternary rendering inside the main JSX tree. This ensures the top bar with domain switch + tab toggle is always visible.

- [ ] **Step 3: Verify it builds**

Run: `cd /home/ringuet/time-serie-explo/frontend && npm run build 2>&1 | tail -10`
Expected: Build succeeds (with stub components not yet created, may show import errors — address in next tasks)

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/latent-space/ClusterProfiling.tsx frontend/src/pages/LatentSpacePage.tsx
git commit -m "feat(profiling): add ClusterProfiling container and tab toggle to LatentSpacePage"
```

---

### Task 9: MetadataDistributions Component

**Files:**
- Create: `frontend/src/components/latent-space/MetadataDistributions.tsx`

- [ ] **Step 1: Create component**

```typescript
// frontend/src/components/latent-space/MetadataDistributions.tsx
import { useState, useMemo } from 'react'
import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'

interface MetadataDistributionsProps {
  distributions: Record<string, unknown>[]
  domain: 'piezo' | 'hydro'
}

const PIEZO_KEYS = ['milieu_eh', 'theme_eh', 'etat_eh', 'nature_eh', 'departement']
const HYDRO_KEYS = ['nom_cours_eau', 'departement']

// Qualitative color palette matching EmbeddingScatter
const COLORS = [
  '#06b6d4', '#8b5cf6', '#f59e0b', '#ef4444', '#10b981',
  '#ec4899', '#3b82f6', '#f97316', '#14b8a6', '#a855f7',
  '#eab308', '#6366f1', '#84cc16', '#e11d48', '#0ea5e9',
]

export function MetadataDistributions({ distributions, domain }: MetadataDistributionsProps) {
  const keys = domain === 'piezo' ? PIEZO_KEYS : HYDRO_KEYS
  const [selectedKey, setSelectedKey] = useState(keys[0])

  const dist = useMemo(() => {
    const d = distributions.find((d) => (d as { key: string }).key === selectedKey)
    return d ? (d as { key: string; clusters: Record<string, Record<string, number>> }).clusters : {}
  }, [distributions, selectedKey])

  // Sort clusters by total count descending
  const sortedClusters = useMemo(() => {
    return Object.entries(dist)
      .map(([cid, vals]) => ({
        cid,
        total: Object.values(vals).reduce((a, b) => a + b, 0),
        vals,
      }))
      .sort((a, b) => b.total - a.total)
  }, [dist])

  // Collect all unique values across clusters
  const allValues = useMemo(() => {
    const s = new Set<string>()
    for (const { vals } of sortedClusters) {
      for (const v of Object.keys(vals)) s.add(v)
    }
    return Array.from(s).sort()
  }, [sortedClusters])

  const traces = allValues.map((val, i) => ({
    type: 'bar' as const,
    orientation: 'h' as const,
    name: val.length > 25 ? val.slice(0, 22) + '...' : val,
    y: sortedClusters.map((c) => `Cluster ${c.cid}`),
    x: sortedClusters.map((c) => c.vals[val] ?? 0),
    marker: { color: COLORS[i % COLORS.length] },
    hovertemplate: `%{y}<br>${val}: %{x}<extra></extra>`,
  }))

  const selectClass =
    'bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-1.5 text-xs focus:outline-none focus:border-accent-cyan/50 transition-colors'

  return (
    <div className="bg-bg-card rounded-xl border border-white/5 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-text-primary text-sm font-medium">Metadata Distributions</h3>
        <select
          className={selectClass}
          value={selectedKey}
          onChange={(e) => setSelectedKey(e.target.value)}
        >
          {keys.map((k) => (
            <option key={k} value={k}>{k}</option>
          ))}
        </select>
      </div>
      <Plot
        data={traces}
        layout={{
          ...darkLayout,
          barmode: 'stack',
          margin: { l: 90, r: 20, t: 10, b: 30 },
          height: Math.max(200, sortedClusters.length * 35 + 50),
          yaxis: { ...darkLayout.yaxis, autorange: 'reversed' as const },
          legend: { ...darkLayout.legend, orientation: 'h', y: -0.15, font: { size: 10 } },
          showlegend: allValues.length <= 15,
        }}
        config={plotlyConfig}
        useResizeHandler
        className="w-full"
      />
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/latent-space/MetadataDistributions.tsx
git commit -m "feat(profiling): add MetadataDistributions stacked bar component"
```

---

### Task 10: ConcordanceTable Component

**Files:**
- Create: `frontend/src/components/latent-space/ConcordanceTable.tsx`

- [ ] **Step 1: Create component**

```typescript
// frontend/src/components/latent-space/ConcordanceTable.tsx

interface ConcordanceTableProps {
  concordance: Record<string, unknown>[]
}

function metricColor(value: number): string {
  if (value > 0.3) return 'bg-green-500/20 text-green-400'
  if (value > 0.1) return 'bg-amber-500/20 text-amber-400'
  return 'bg-red-500/20 text-red-400'
}

const TOOLTIPS: Record<string, string> = {
  ari: 'Adjusted Rand Index: agreement between two clusterings, adjusted for chance. 1.0 = perfect, 0.0 = random.',
  nmi: 'Normalized Mutual Information: shared information between clusterings. 1.0 = identical, 0.0 = independent.',
  cramers_v: "Cramér's V: association strength between categorical variables. 1.0 = perfect association, 0.0 = none.",
}

export function ConcordanceTable({ concordance }: ConcordanceTableProps) {
  const rows = concordance as { key: string; ari: number; nmi: number; cramers_v: number }[]

  if (rows.length === 0) return null

  return (
    <div className="bg-bg-card rounded-xl border border-white/5 p-4">
      <h3 className="text-text-primary text-sm font-medium mb-3">
        Concordance with Known Labels
      </h3>
      <table className="w-full text-xs">
        <thead>
          <tr className="text-text-muted border-b border-white/5">
            <th className="text-left py-2 pr-4">Variable</th>
            {['ARI', 'NMI', "Cramér's V"].map((label, i) => (
              <th key={label} className="text-center py-2 px-2" title={TOOLTIPS[['ari', 'nmi', 'cramers_v'][i]]}>
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.key} className="border-b border-white/5 last:border-0">
              <td className="py-2 pr-4 text-text-secondary">{row.key}</td>
              {(['ari', 'nmi', 'cramers_v'] as const).map((metric) => (
                <td key={metric} className="text-center py-2 px-2">
                  <span className={`inline-block px-2 py-0.5 rounded text-xs font-mono ${metricColor(row[metric])}`}>
                    {row[metric].toFixed(3)}
                  </span>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/latent-space/ConcordanceTable.tsx
git commit -m "feat(profiling): add ConcordanceTable component with color-coded metrics"
```

---

### Task 11: TemporalPrototypes Component

**Files:**
- Create: `frontend/src/components/latent-space/TemporalPrototypes.tsx`

- [ ] **Step 1: Create component**

```typescript
// frontend/src/components/latent-space/TemporalPrototypes.tsx
import Plot from 'react-plotly.js'
import type { Data, Layout } from 'plotly.js-dist-min'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'

interface TemporalPrototypesProps {
  prototypes: Record<string, unknown>[]
}

const COLORS = [
  '#06b6d4', '#8b5cf6', '#f59e0b', '#ef4444', '#10b981',
  '#ec4899', '#3b82f6', '#f97316', '#14b8a6', '#a855f7',
]

export function TemporalPrototypes({ prototypes }: TemporalPrototypesProps) {
  const protos = prototypes as {
    cluster_id: number
    medoid_id: string
    dates: string[]
    medoid_values: (number | null)[]
    p10: (number | null)[]
    p90: (number | null)[]
  }[]

  if (protos.length === 0) return null

  const cols = Math.min(4, protos.length)
  const rows = Math.ceil(protos.length / cols)

  const traces: Data[] = []
  const annotations: Partial<Layout['annotations']>[number][] = []

  protos.forEach((proto, idx) => {
    const row = Math.floor(idx / cols) + 1
    const col = (idx % cols) + 1
    const xaxis = idx === 0 ? 'x' : `x${idx + 1}`
    const yaxis = idx === 0 ? 'y' : `y${idx + 1}`
    const color = COLORS[idx % COLORS.length]

    // P10/P90 envelope (fill)
    traces.push({
      type: 'scatter',
      x: [...proto.dates, ...proto.dates.slice().reverse()],
      y: [...(proto.p90 as number[]), ...(proto.p10 as number[]).slice().reverse()],
      fill: 'toself',
      fillcolor: color + '26', // 15% opacity
      line: { color: 'transparent' },
      showlegend: false,
      hoverinfo: 'skip',
      xaxis: xaxis,
      yaxis: yaxis,
    } as Data)

    // Medoid line
    traces.push({
      type: 'scatter',
      x: proto.dates,
      y: proto.medoid_values as number[],
      mode: 'lines',
      line: { color, width: 1.5 },
      name: `Cluster ${proto.cluster_id}`,
      showlegend: false,
      hovertemplate: '%{x|%Y-%m-%d}<br>Value: %{y:.2f}<extra></extra>',
      xaxis: xaxis,
      yaxis: yaxis,
    } as Data)

    annotations.push({
      text: `Cluster ${proto.cluster_id} — ${proto.medoid_id}`,
      xref: `${xaxis} domain` as string,
      yref: `${yaxis} domain` as string,
      x: 0.5,
      y: 1.05,
      xanchor: 'center',
      yanchor: 'bottom',
      showarrow: false,
      font: { color: '#9ca3af', size: 10 },
    })
  })

  // Build subplot layout axes
  const layout: Partial<Layout> = {
    ...darkLayout,
    margin: { l: 50, r: 20, t: 30, b: 30 },
    height: rows * 200 + 50,
    grid: { rows, columns: cols, pattern: 'independent' },
    annotations: annotations as Layout['annotations'],
    showlegend: false,
  }

  // Configure each subplot axis
  protos.forEach((_, idx) => {
    const xKey = idx === 0 ? 'xaxis' : `xaxis${idx + 1}`
    const yKey = idx === 0 ? 'yaxis' : `yaxis${idx + 1}`
    ;(layout as Record<string, unknown>)[xKey] = {
      ...darkLayout.xaxis,
      tickfont: { size: 9 },
    }
    ;(layout as Record<string, unknown>)[yKey] = {
      ...darkLayout.yaxis,
      tickfont: { size: 9 },
    }
  })

  return (
    <div className="bg-bg-card rounded-xl border border-white/5 p-4">
      <h3 className="text-text-primary text-sm font-medium mb-3">Temporal Prototypes</h3>
      <Plot
        data={traces}
        layout={layout}
        config={plotlyConfig}
        useResizeHandler
        className="w-full"
      />
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/latent-space/TemporalPrototypes.tsx
git commit -m "feat(profiling): add TemporalPrototypes component with medoid lines and P10/P90 envelopes"
```

---

### Task 12: FeatureFingerprints Component

**Files:**
- Create: `frontend/src/components/latent-space/FeatureFingerprints.tsx`

- [ ] **Step 1: Create component**

```typescript
// frontend/src/components/latent-space/FeatureFingerprints.tsx
import Plot from 'react-plotly.js'
import type { Data } from 'plotly.js-dist-min'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'

interface FeatureFingerprintsProps {
  fingerprints: Record<string, unknown>[]
}

const COLORS = [
  '#06b6d4', '#8b5cf6', '#f59e0b', '#ef4444', '#10b981',
  '#ec4899', '#3b82f6', '#f97316', '#14b8a6', '#a855f7',
]

const FEATURE_LABELS: Record<string, string> = {
  mean: 'Mean',
  std: 'Std',
  trend: 'Trend',
  seasonality: 'Seasonality',
  autocorr_365: 'Autocorr 365d',
  wet_dry_ratio: 'Wet/Dry Ratio',
}

const FEATURE_ORDER = ['mean', 'std', 'trend', 'seasonality', 'autocorr_365', 'wet_dry_ratio']

export function FeatureFingerprints({ fingerprints }: FeatureFingerprintsProps) {
  const fps = fingerprints as {
    cluster_id: number
    features: Record<string, number>
    features_raw: Record<string, number>
  }[]

  if (fps.length === 0) return null

  const theta = FEATURE_ORDER.map((f) => FEATURE_LABELS[f] ?? f)
  // Close the polygon by repeating first point
  const thetaClosed = [...theta, theta[0]]

  const traces: Data[] = fps.map((fp, i) => {
    const r = FEATURE_ORDER.map((f) => fp.features[f] ?? 0)
    const rClosed = [...r, r[0]]
    const rawVals = FEATURE_ORDER.map((f) => fp.features_raw[f]?.toFixed(3) ?? 'N/A')
    const rawClosed = [...rawVals, rawVals[0]]

    return {
      type: 'scatterpolar',
      r: rClosed,
      theta: thetaClosed,
      fill: 'toself',
      fillcolor: COLORS[i % COLORS.length] + '1A', // 10% opacity
      line: { color: COLORS[i % COLORS.length], width: 2 },
      name: `Cluster ${fp.cluster_id}`,
      customdata: rawClosed,
      hovertemplate: '%{theta}: %{r:.2f}<br>Raw: %{customdata}<extra>Cluster ' + fp.cluster_id + '</extra>',
    } as Data
  })

  return (
    <div className="bg-bg-card rounded-xl border border-white/5 p-4">
      <h3 className="text-text-primary text-sm font-medium mb-3">Feature Fingerprints</h3>
      <Plot
        data={traces}
        layout={{
          ...darkLayout,
          margin: { l: 60, r: 60, t: 30, b: 30 },
          height: 400,
          polar: {
            bgcolor: 'transparent',
            radialaxis: {
              range: [0, 1],
              gridcolor: 'rgba(255,255,255,0.08)',
              tickfont: { size: 9 },
            },
            angularaxis: {
              gridcolor: 'rgba(255,255,255,0.08)',
              tickfont: { size: 10 },
            },
          },
          legend: {
            ...darkLayout.legend,
            orientation: 'h',
            y: -0.1,
            font: { size: 10 },
          },
          showlegend: true,
        }}
        config={plotlyConfig}
        useResizeHandler
        className="w-full"
      />
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/latent-space/FeatureFingerprints.tsx
git commit -m "feat(profiling): add FeatureFingerprints radar chart component"
```

---

### Task 13: ShapExplainability Component

**Files:**
- Create: `frontend/src/components/latent-space/ShapExplainability.tsx`

- [ ] **Step 1: Create component**

```typescript
// frontend/src/components/latent-space/ShapExplainability.tsx
import Plot from 'react-plotly.js'
import type { Data, Layout } from 'plotly.js-dist-min'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import { AlertTriangle } from 'lucide-react'

interface ShapExplainabilityProps {
  shap: Record<string, unknown>
}

function accuracyColor(acc: number): string {
  if (acc >= 0.8) return 'bg-green-500/20 text-green-400'
  if (acc >= 0.5) return 'bg-amber-500/20 text-amber-400'
  return 'bg-red-500/20 text-red-400'
}

export function ShapExplainability({ shap }: ShapExplainabilityProps) {
  const data = shap as {
    feature_importance: Record<string, number>
    shap_per_cluster: Record<string, Record<string, number>>
    proxy_accuracy: number
    warning: string | null
  }

  if (!data.feature_importance || Object.keys(data.feature_importance).length === 0) {
    if (data.warning) {
      return (
        <div className="bg-bg-card rounded-xl border border-white/5 p-4">
          <h3 className="text-text-primary text-sm font-medium mb-3">SHAP Explainability</h3>
          <div className="flex items-center gap-2 bg-amber-500/10 text-amber-400 px-3 py-2 rounded-lg text-xs">
            <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
            <span>{data.warning}</span>
          </div>
        </div>
      )
    }
    return null
  }

  // Sort features by global importance
  const sortedFeatures = Object.entries(data.feature_importance)
    .sort((a, b) => b[1] - a[1])
    .map(([f]) => f)

  const clusterIds = Object.keys(data.shap_per_cluster).sort(
    (a, b) => Number(a) - Number(b),
  )

  const cols = Math.min(4, clusterIds.length)
  const rows = Math.ceil(clusterIds.length / cols)

  const COLORS_POS = '#ef4444'  // red — pushes toward
  const COLORS_NEG = '#3b82f6'  // blue — pushes away

  const traces: Data[] = []

  clusterIds.forEach((cid, idx) => {
    const shapVals = data.shap_per_cluster[cid]
    const xaxis = idx === 0 ? 'x' : `x${idx + 1}`
    const yaxis = idx === 0 ? 'y' : `y${idx + 1}`

    traces.push({
      type: 'bar',
      orientation: 'h',
      y: sortedFeatures,
      x: sortedFeatures.map((f) => shapVals[f] ?? 0),
      marker: {
        color: sortedFeatures.map((f) => (shapVals[f] ?? 0) >= 0 ? COLORS_POS : COLORS_NEG),
      },
      showlegend: false,
      hovertemplate: '%{y}: %{x:.4f}<extra>Cluster ' + cid + '</extra>',
      xaxis,
      yaxis,
    } as Data)
  })

  const layout: Partial<Layout> = {
    ...darkLayout,
    margin: { l: 100, r: 20, t: 30, b: 30 },
    height: rows * 200 + 50,
    grid: { rows, columns: cols, pattern: 'independent' },
    annotations: clusterIds.map((cid, idx) => ({
      text: `Cluster ${cid}`,
      xref: `${idx === 0 ? 'x' : `x${idx + 1}`} domain` as string,
      yref: `${idx === 0 ? 'y' : `y${idx + 1}`} domain` as string,
      x: 0.5,
      y: 1.05,
      xanchor: 'center' as const,
      yanchor: 'bottom' as const,
      showarrow: false,
      font: { color: '#9ca3af', size: 10 },
    })) as Layout['annotations'],
    showlegend: false,
  }

  // Configure axes
  clusterIds.forEach((_, idx) => {
    const xKey = idx === 0 ? 'xaxis' : `xaxis${idx + 1}`
    const yKey = idx === 0 ? 'yaxis' : `yaxis${idx + 1}`
    ;(layout as Record<string, unknown>)[xKey] = {
      ...darkLayout.xaxis,
      zeroline: true,
      zerolinecolor: 'rgba(255,255,255,0.15)',
      tickfont: { size: 9 },
    }
    ;(layout as Record<string, unknown>)[yKey] = {
      ...darkLayout.yaxis,
      autorange: 'reversed',
      tickfont: { size: 9 },
    }
  })

  return (
    <div className="bg-bg-card rounded-xl border border-white/5 p-4">
      <div className="flex items-center gap-3 mb-3">
        <h3 className="text-text-primary text-sm font-medium">SHAP Explainability</h3>
        <span className={`px-2 py-0.5 rounded text-xs font-mono ${accuracyColor(data.proxy_accuracy)}`}>
          Proxy accuracy: {(data.proxy_accuracy * 100).toFixed(1)}%
        </span>
      </div>
      {data.warning && (
        <div className="flex items-center gap-2 bg-amber-500/10 text-amber-400 px-3 py-2 rounded-lg text-xs mb-3">
          <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
          <span>{data.warning}</span>
        </div>
      )}
      <Plot
        data={traces}
        layout={layout}
        config={plotlyConfig}
        useResizeHandler
        className="w-full"
      />
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/latent-space/ShapExplainability.tsx
git commit -m "feat(profiling): add ShapExplainability component with per-cluster SHAP bar charts"
```

---

### Task 14: Build, Deploy, and Verify

- [ ] **Step 1: Build frontend**

Run: `cd /home/ringuet/time-serie-explo/frontend && npm run build 2>&1 | tail -20`
Expected: Build succeeds without errors

- [ ] **Step 2: Rebuild Docker stack**

Run: `cd /home/ringuet/time-serie-explo && docker compose up -d --build 2>&1 | tail -15`
Expected: All containers healthy

- [ ] **Step 3: Test backend endpoint**

Run: `curl -s http://localhost:49513/api/v1/latent-space/profiling/piezo | python3 -m json.tool | head -50`
Expected: Valid JSON with all 5 profiling blocks

- [ ] **Step 4: Test hydro domain**

Run: `curl -s http://localhost:49513/api/v1/latent-space/profiling/hydro | python3 -m json.tool | head -30`
Expected: Valid JSON (different table, QmnJ filter)

- [ ] **Step 5: Run Python tests**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/latent_space/test_cluster_profiling.py -v --tb=short`
Expected: All tests pass

- [ ] **Step 6: Visual verification**

Open browser at `http://localhost:49513`, navigate to Latent Space page, click "Profiling" tab.
Verify: 5 blocks render with data, no console errors, loading skeletons show during fetch.

- [ ] **Step 7: Final commit**

```bash
git add -A
git commit -m "feat(profiling): complete cluster profiling panel with 5 analysis blocks"
```
