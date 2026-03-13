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
            if cid is None or cid < 0 or val is None or val == "":
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

        unique_clusters = set(cluster_labels)
        if len(unique_clusters) <= 1:
            result[key] = {"ari": 0.0, "nmi": 0.0, "cramers_v": 0.0}
            continue

        ari = adjusted_rand_score(meta_labels, cluster_labels)
        nmi = normalized_mutual_info_score(meta_labels, cluster_labels)

        # Cramér's V via contingency table
        try:
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
    clusters: dict[int, list[str]] = defaultdict(list)
    for sid, cid in cluster_labels.items():
        if cid >= 0:
            clusters[cid].append(sid)

    medoids: dict[int, str] = {}
    for cid, members in clusters.items():
        if len(members) == 1:
            medoids[cid] = members[0]
            continue
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
    Truncates to last max_days days. If cluster has < 3 members with series,
    returns medoid only (p10 = p90 = medoid_values).
    """
    result: dict[int, dict[str, Any]] = {}

    for cid, med_id in medoid_ids.items():
        if med_id not in series_map or med_id not in dates_map:
            continue

        med_series = series_map[med_id]
        med_dates = dates_map[med_id]

        if len(med_series) > max_days:
            med_series = med_series[-max_days:]
            med_dates = med_dates[-max_days:]

        n_points = len(med_series)
        med_values = [float(v) if np.isfinite(v) else None for v in med_series]

        members = cluster_members.get(cid, [])
        member_series = []
        for sid in members:
            if sid not in series_map or sid not in dates_map:
                continue
            s = series_map[sid]
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

    x = np.arange(n, dtype=float)
    mask = np.isfinite(values)
    if mask.sum() > 1:
        coeffs = np.polyfit(x[mask], values[mask], 1)
        trend = float(coeffs[0])
    else:
        trend = 0.0

    if n >= 365:
        filled = np.where(np.isfinite(values), values, np.nanmean(values))
        fft_vals = np.fft.rfft(filled - np.mean(filled))
        freqs = np.fft.rfftfreq(n, d=1.0)
        target_freq = 1.0 / 365.0
        idx = np.argmin(np.abs(freqs - target_freq))
        seasonality = float(2.0 * np.abs(fft_vals[idx]) / n)
    else:
        seasonality = float("nan")

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
    - normalized: {cluster_id: {feature: value_in_0_1}}
    - raw: {cluster_id: {feature: median_value}}
    - per_station: {station_id: {feature: value}} for SHAP input
    """
    FEATURES = ["mean", "std", "trend", "seasonality", "autocorr_365", "wet_dry_ratio"]

    station_features: dict[str, dict[str, float]] = {}
    for sid in series_map:
        if sid in cluster_labels and cluster_labels[sid] >= 0:
            station_features[sid] = _compute_station_features(
                series_map[sid], dates_map.get(sid, [])
            )

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

    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf.fit(X, labels)

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

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)

    # Handle different shap_values return types:
    # - shap < 0.42: list of N arrays for multiclass, single array for binary
    # - shap >= 0.42: 3D array (n_samples, n_features, n_classes) for multiclass
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
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
        stacked = np.stack(shap_values)
        global_importance = np.mean(np.abs(stacked), axis=(0, 1))
        feature_importance = {
            feature_names[i]: float(global_importance[i])
            for i in range(len(feature_names))
        }
        shap_per_cluster = {}
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
