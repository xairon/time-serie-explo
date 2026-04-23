"""Latent space utilities: query building, UMAP, clustering, similarity search.

Pure Python module — NO framework imports.
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np
from sqlalchemy import text


# ---------------------------------------------------------------------------
# SQL query builders
# ---------------------------------------------------------------------------

_VALID_DOMAINS = {"piezo", "hydro"}

# BDLISA code → label mappings (Sandre referential)
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

_ETAT_EH: dict[str, str] = {
    "1": "Libre seul",
    "2": "Libre et captif",
    "3": "Captif seul",
    "4": "Libre ou captif",
    "5": "Libre et captif affleurant",
    "6": "Captif sous couverture non aquifère",
    "X": "Indéterminé",
}

_NATURE_EH: dict[str, str] = {
    "0": "Entité hydrogéologique",
    "3": "Système aquifère",
    "4": "Domaine hydrogéologique",
    "5": "Système aquifère",
    "6": "Unité aquifère",
    "7": "Unité semi-perméable",
}


def _decode_eh(code: str | None, mapping: dict[str, str]) -> str | None:
    """Decode a BDLISA code to its label."""
    if code is None:
        return None
    return mapping.get(str(code), str(code))


def decode_eh_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Decode BDLISA codes to human-readable labels in a metadata dict."""
    out = dict(meta)
    out["milieu_eh"] = _decode_eh(meta.get("milieu_eh"), _MILIEU_EH)
    out["theme_eh"] = _decode_eh(meta.get("theme_eh"), _THEME_EH)
    out["etat_eh"] = _decode_eh(meta.get("etat_eh"), _ETAT_EH)
    out["nature_eh"] = _decode_eh(meta.get("nature_eh"), _NATURE_EH)
    return out


_SEASON_MONTHS: dict[str, list[int]] = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Get attribute from Pydantic model or dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def build_station_query(domain: str, filters, space: str = "multi") -> tuple[Any, dict]:
    """Build SQL query for station-level embeddings.

    Parameters
    ----------
    domain:
        "piezo" or "hydro".
    filters:
        EmbeddingFilters-like object (attributes: station_ids, libelle_eh,
        milieu_eh, theme_eh, etat_eh, nature_eh, departement, region,
        cluster_id).

    Returns
    -------
    (sqlalchemy.text, params_dict)
    """
    if domain not in _VALID_DOMAINS:
        raise ValueError(f"Invalid domain '{domain}'. Must be one of {_VALID_DOMAINS}.")

    params: dict[str, Any] = {}
    where_clauses: list[str] = []

    params["space"] = space
    where_clauses.append("e.space = :space")

    if domain == "piezo":
        select_cols = """
            e.code_bss            AS id,
            tme.libelle_eh,
            tme.milieu_eh,
            tme.theme_eh,
            tme.etat_eh,
            tme.nature_eh,
            s.code_departement    AS departement,
            s.nom_departement,
            s.altitude_station    AS altitude,
            e.cluster_id,
            e.n_windows,
            e.last_date,
            e.umap_2d_x,
            e.umap_2d_y,
            e.umap_3d_x,
            e.umap_3d_y,
            e.umap_3d_z,
            e.embedding::text     AS embedding_raw
        """
        from_clause = """
            ml.piezo_station_embeddings e
            JOIN gold.dim_piezo_stations s ON e.code_bss = s.code_bss
            LEFT JOIN (
                SELECT DISTINCT ON (m.code_bss)
                    m.code_bss,
                    m.libelle_eh,
                    m.milieu_eh,
                    m.theme_eh,
                    m.etat_eh,
                    m.nature_eh
                FROM gold.int_station_era5_mapping m
                ORDER BY m.code_bss
            ) tme ON tme.code_bss = e.code_bss
        """
        id_col = "e.code_bss"

        if _get(filters, 'station_ids'):
            where_clauses.append(f"{id_col} = ANY(:station_ids)")
            params["station_ids"] = list(_get(filters, 'station_ids'))
        if _get(filters, 'libelle_eh'):
            where_clauses.append("tme.libelle_eh ILIKE :libelle_eh")
            params["libelle_eh"] = f"%{_get(filters, 'libelle_eh')}%"
        if _get(filters, 'milieu_eh'):
            where_clauses.append("tme.milieu_eh = :milieu_eh")
            params["milieu_eh"] = _get(filters, 'milieu_eh')
        if _get(filters, 'theme_eh'):
            where_clauses.append("tme.theme_eh = :theme_eh")
            params["theme_eh"] = _get(filters, 'theme_eh')
        if _get(filters, 'etat_eh'):
            where_clauses.append("tme.etat_eh = :etat_eh")
            params["etat_eh"] = _get(filters, 'etat_eh')
        if _get(filters, 'nature_eh'):
            where_clauses.append("tme.nature_eh = :nature_eh")
            params["nature_eh"] = _get(filters, 'nature_eh')
        if _get(filters, 'departement'):
            where_clauses.append("s.code_departement = :departement")
            params["departement"] = _get(filters, 'departement')
        if _get(filters, 'cluster_id') is not None:
            where_clauses.append("e.cluster_id = :cluster_id")
            params["cluster_id"] = _get(filters, 'cluster_id')

    else:  # hydro
        select_cols = """
            e.code_station        AS id,
            s.nom_cours_eau,
            s.code_departement    AS departement,
            s.nom_departement,
            s.statut_station,
            e.cluster_id,
            e.n_windows,
            e.last_date,
            e.umap_2d_x,
            e.umap_2d_y,
            e.umap_3d_x,
            e.umap_3d_y,
            e.umap_3d_z,
            e.embedding::text     AS embedding_raw
        """
        from_clause = """
            ml.hydro_station_embeddings e
            JOIN gold.dim_hydro_stations s ON e.code_station = s.code_station
        """
        id_col = "e.code_station"

        if _get(filters, 'station_ids'):
            where_clauses.append(f"{id_col} = ANY(:station_ids)")
            params["station_ids"] = list(_get(filters, 'station_ids'))
        if _get(filters, 'departement'):
            where_clauses.append("s.code_departement = :departement")
            params["departement"] = _get(filters, 'departement')
        if _get(filters, 'cluster_id') is not None:
            where_clauses.append("e.cluster_id = :cluster_id")
            params["cluster_id"] = _get(filters, 'cluster_id')

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    sql = text(f"SELECT {select_cols} FROM {from_clause} {where_sql}")
    return sql, params


def build_window_query(
    domain: str,
    filters,
    year_min: int | None,
    year_max: int | None,
    season: str | None,
    space: str = "multi",
) -> tuple[Any, dict]:
    """Build SQL query for window-level embeddings.

    Parameters
    ----------
    domain:
        "piezo" or "hydro".
    filters:
        EmbeddingFilters-like object.
    year_min, year_max:
        Optional year range on window_start.
    season:
        Optional season code ("DJF", "MAM", "JJA", "SON").

    Returns
    -------
    (sqlalchemy.text, params_dict)
    """
    if domain not in _VALID_DOMAINS:
        raise ValueError(f"Invalid domain '{domain}'. Must be one of {_VALID_DOMAINS}.")

    params: dict[str, Any] = {}
    where_clauses: list[str] = []

    params["space"] = space
    where_clauses.append("w.space = :space")

    id_col = "code_bss" if domain == "piezo" else "code_station"

    select_cols = f"""
        w.{id_col}            AS id,
        w.window_start,
        w.window_end,
        w.embedding::text     AS embedding_raw
    """
    from_clause = f"ml.{domain}_window_embeddings w"

    if _get(filters, 'station_ids'):
        where_clauses.append(f"w.{id_col} = ANY(:station_ids)")
        params["station_ids"] = list(_get(filters, 'station_ids'))
    if year_min is not None:
        where_clauses.append("EXTRACT(YEAR FROM w.window_start) >= :year_min")
        params["year_min"] = year_min
    if year_max is not None:
        where_clauses.append("EXTRACT(YEAR FROM w.window_start) <= :year_max")
        params["year_max"] = year_max
    if season is not None:
        months = _SEASON_MONTHS.get(season)
        if months is None:
            raise ValueError(f"Invalid season '{season}'. Must be one of {list(_SEASON_MONTHS)}.")
        where_clauses.append("EXTRACT(MONTH FROM w.window_start) = ANY(:season_months)")
        params["season_months"] = months

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    sql = text(f"SELECT {select_cols} FROM {from_clause} {where_sql}")
    return sql, params


def build_similar_query(domain: str, station_id: str, k: int, space: str = "multi") -> tuple[Any, dict]:
    """Build SQL query for nearest-neighbour similarity search using pgvector.

    Uses cosine distance operator `<=>`.

    Parameters
    ----------
    domain:
        "piezo" or "hydro".
    station_id:
        Query station identifier.
    k:
        Number of neighbours to return (excluding the query station itself).

    Returns
    -------
    (sqlalchemy.text, params_dict)
    """
    if domain not in _VALID_DOMAINS:
        raise ValueError(f"Invalid domain '{domain}'. Must be one of {_VALID_DOMAINS}.")

    id_col = "code_bss" if domain == "piezo" else "code_station"
    table = f"ml.{domain}_station_embeddings"

    sql = text(f"""
        SELECT
            e.{id_col}                                         AS id,
            e.embedding <=> q.embedding                        AS distance,
            e.cluster_id
        FROM {table} e
        CROSS JOIN (
            SELECT embedding
            FROM {table}
            WHERE {id_col} = :station_id AND space = :space
            LIMIT 1
        ) q
        WHERE e.{id_col} != :station_id AND e.space = :space
        ORDER BY distance ASC
        LIMIT :k
    """)
    params: dict[str, Any] = {"station_id": station_id, "k": k, "space": space}
    return sql, params


# ---------------------------------------------------------------------------
# Embedding parsing
# ---------------------------------------------------------------------------


def parse_pgvector(raw: str) -> np.ndarray:
    """Convert pgvector string representation to a numpy array.

    Parameters
    ----------
    raw:
        String like "[0.1,0.2,0.3]".

    Returns
    -------
    numpy.ndarray of float32.
    """
    cleaned = raw.strip().lstrip("[").rstrip("]")
    values = [float(v) for v in cleaned.split(",")]
    return np.array(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------


def compute_umap(
    embeddings_matrix: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
) -> np.ndarray:
    """Reduce embeddings with UMAP.

    Parameters
    ----------
    embeddings_matrix:
        Shape (n_samples, n_dims).
    n_components, n_neighbors, min_dist, metric:
        Standard UMAP parameters.

    Returns
    -------
    numpy.ndarray of shape (n_samples, n_components).
    """
    import umap

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
    )
    return reducer.fit_transform(embeddings_matrix)


def compute_umap_quality(
    high_dim: np.ndarray,
    low_dim: np.ndarray,
    n_neighbors: int = 15,
) -> dict[str, float]:
    """Compute UMAP projection quality metrics.

    Returns trustworthiness (how well local neighborhoods are preserved)
    and continuity scores. Values in [0, 1], higher = better.
    """
    from sklearn.manifold import trustworthiness

    metrics: dict[str, float] = {}
    k = min(n_neighbors, high_dim.shape[0] - 1)
    if k < 2:
        return metrics
    try:
        tw = trustworthiness(high_dim, low_dim, n_neighbors=k, metric="cosine")
        metrics["trustworthiness"] = round(float(tw), 4)
    except Exception:
        pass
    return metrics


# ---------------------------------------------------------------------------
# Pipeline: PCA → VIZ + CLUSTERING (decoupled, cached independently)
# ---------------------------------------------------------------------------


def compute_pca(
    embeddings: np.ndarray,
    variance_threshold: float = 0.95,
    min_components: int = 15,
) -> dict[str, Any]:
    """PCA adaptive pre-reduction.

    Floor: max(min_components, n_auto) — never below min_components.
    Ceiling: min(n_99_percent, 100, N-1).

    Returns dict ready for Redis cache 'pca' section.
    """
    from sklearn.decomposition import PCA

    n_samples, n_dims = embeddings.shape
    max_components = min(100, n_dims, n_samples - 1)
    pca = PCA(n_components=max_components, random_state=42)
    pca.fit(embeddings)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_auto = int(np.searchsorted(cumvar, variance_threshold) + 1)
    n_components = max(min_components, n_auto)
    n_99 = int(np.searchsorted(cumvar, 0.99) + 1)
    n_components = min(n_components, n_99, max_components)

    reduced = pca.transform(embeddings)[:, :n_components]

    return {
        "reduced": [[round(float(v), 6) for v in row] for row in reduced],
        "n_components": n_components,
        "variance_threshold": variance_threshold,
        "variance_explained": round(float(cumvar[n_components - 1]), 4),
        "cumvar_curve": [round(float(v), 4) for v in cumvar[:max_components]],
        "embedding_dim": n_dims,
    }


def compute_viz(
    pca_reduced: np.ndarray,
    n_neighbors: int = 50,
    min_dist: float = 0.3,
) -> dict[str, Any]:
    """UMAP 2D visualization from PCA-reduced data.

    Returns dict ready for Redis cache 'viz' section.
    """
    from sklearn.manifold import trustworthiness

    coords = compute_umap(
        pca_reduced, n_components=2,
        n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine",
    )

    tw = 0.0
    k = min(n_neighbors, pca_reduced.shape[0] - 1)
    if k >= 2:
        try:
            tw = float(trustworthiness(pca_reduced, coords, n_neighbors=k, metric="cosine"))
        except Exception:
            pass

    return {
        "coords_2d": [[round(float(coords[i, 0]), 6), round(float(coords[i, 1]), 6)] for i in range(len(coords))],
        "params": {"n_neighbors": n_neighbors, "min_dist": min_dist},
        "trustworthiness": round(tw, 4),
    }


def _dbcv_score(labels: np.ndarray, X: np.ndarray) -> float:
    """DBCV score for HDBSCAN. Uses hdbscan validity_index if available."""
    n_clusters = len(set(labels.tolist()) - {-1})
    if n_clusters < 2:
        return -1.0

    try:
        from hdbscan.validity import validity_index
        return float(validity_index(X.astype(np.float64), labels))
    except (ImportError, Exception):
        pass

    # Composite fallback: silhouette with penalties for degenerate solutions
    from sklearn.metrics import silhouette_score
    mask = labels != -1
    if mask.sum() < 10:
        return -1.0
    sil = float(silhouette_score(X[mask], labels[mask]))
    noise_ratio = float((~mask).sum()) / len(labels)
    penalty = 0.0
    if n_clusters < 4:
        penalty += 0.3 * (4 - n_clusters) / 3
    if n_clusters > 30:
        penalty += 0.1 * min((n_clusters - 30) / 30, 1.0)
    if noise_ratio > 0.4:
        penalty += 0.2
    elif noise_ratio > 0.25:
        penalty += 0.1
    return sil - penalty


def _build_hdbscan_result(
    labels: np.ndarray, X: np.ndarray, mcs: int, ms: int,
) -> dict[str, Any]:
    from sklearn.metrics import silhouette_score
    n_clusters = len(set(labels.tolist()) - {-1})
    noise_ratio = float((labels == -1).sum()) / len(labels) if len(labels) > 0 else 0.0
    mask = labels != -1
    sil = float(silhouette_score(X[mask], labels[mask])) if n_clusters > 1 and mask.sum() > 10 else 0.0
    dbcv = _dbcv_score(labels, X)
    return {
        "params": {"min_cluster_size": int(mcs), "min_samples": int(ms)},
        "labels": labels.tolist(),
        "n_clusters": n_clusters,
        "dbcv": round(dbcv, 4),
        "noise_ratio": round(noise_ratio, 4),
        "silhouette": round(sil, 4),
    }


def _optuna_hdbscan(pca_reduced: np.ndarray, n_trials: int = 40) -> dict[str, Any]:
    """Optimize HDBSCAN params via Optuna, maximizing DBCV score."""
    import optuna
    from sklearn.cluster import HDBSCAN

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n = pca_reduced.shape[0]

    if n < 50:
        from hdbscan import HDBSCAN as HDBSCANOrig
        mcs = max(5, n // 5)
        labels = HDBSCANOrig(min_cluster_size=mcs, min_samples=3).fit_predict(pca_reduced)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        return {"min_cluster_size": mcs, "min_samples": 3, "n_clusters": n_clusters, "labels": labels.tolist()}

    best: dict[str, Any] = {}

    def objective(trial: optuna.Trial) -> float:
        nonlocal best
        mcs = trial.suggest_int("min_cluster_size", 10, min(200, n // 5))
        ms = trial.suggest_int("min_samples", 3, min(30, mcs))
        labels = HDBSCAN(min_cluster_size=mcs, min_samples=ms).fit_predict(pca_reduced)
        score = _dbcv_score(labels, pca_reduced)
        if not best or score > best.get("_score", -2.0):
            result = _build_hdbscan_result(labels, pca_reduced, mcs, ms)
            result["_score"] = score
            best = result
        return score

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Soft guidance: if < 5 clusters, retry with constrained range
    if best.get("n_clusters", 0) < 5:
        def retry(trial: optuna.Trial) -> float:
            nonlocal best
            mcs = trial.suggest_int("min_cluster_size", 5, 30)
            ms = trial.suggest_int("min_samples", 3, min(15, mcs))
            labels = HDBSCAN(min_cluster_size=mcs, min_samples=ms).fit_predict(pca_reduced)
            n_cl = len(set(labels.tolist()) - {-1})
            if n_cl < 3:
                return -1.0
            score = _dbcv_score(labels, pca_reduced)
            if not best or score > best.get("_score", -2.0):
                result = _build_hdbscan_result(labels, pca_reduced, mcs, ms)
                result["_score"] = score
                best = result
            return score

        study2 = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=43),
        )
        study2.optimize(retry, n_trials=20, show_progress_bar=False)

    # Fallback
    if not best or best.get("n_clusters", 0) < 2:
        labels = HDBSCAN(min_cluster_size=25, min_samples=5).fit_predict(pca_reduced)
        best = _build_hdbscan_result(labels, pca_reduced, 25, 5)

    best.pop("_score", None)
    return best


def compute_clustering_all(
    pca_reduced: np.ndarray,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
    n_optuna_trials: int = 40,
) -> dict[str, Any]:
    """Compute HDBSCAN + KMeans elbow, all on PCA-reduced data.

    If min_cluster_size and min_samples are provided, runs HDBSCAN with those
    explicit params (no Optuna). Otherwise runs Optuna optimization.

    KMeans always scans k=2..25.

    Returns dict ready for Redis cache 'clustering' section.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = pca_reduced.shape[0]

    # --- HDBSCAN ---
    if min_cluster_size is not None and min_samples is not None:
        from sklearn.cluster import HDBSCAN
        labels = HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples,
        ).fit_predict(pca_reduced)
        hdb_result = _build_hdbscan_result(labels, pca_reduced, min_cluster_size, min_samples)
    else:
        hdb_result = _optuna_hdbscan(pca_reduced, n_optuna_trials)

    # --- KMeans elbow (k=2..25) ---
    kmeans_elbow: list[dict[str, Any]] = []
    for k in range(2, 26):
        if k >= n:
            break
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbls = km.fit_predict(pca_reduced)
            sil = float(silhouette_score(pca_reduced, lbls))
            kmeans_elbow.append({
                "k": k,
                "labels": lbls.tolist(),
                "silhouette": round(sil, 4),
                "inertia": round(float(km.inertia_), 2),
            })
        except Exception:
            continue

    return {"hdbscan": hdb_result, "kmeans_elbow": kmeans_elbow}


# ---------------------------------------------------------------------------
# Stratified subsampling
# ---------------------------------------------------------------------------


def subsample_stratified(
    ids: list[str],
    embeddings: np.ndarray,
    metadata_list: list[dict],
    max_points: int,
    group_key: str = "station_id",
) -> tuple[list[str], np.ndarray, list[dict], bool, int]:
    """Stratified random subsample of embeddings.

    Groups are formed by the value of ``group_key`` in each metadata dict.
    Sampling is proportional across groups.

    Parameters
    ----------
    ids:
        List of point identifiers.
    embeddings:
        Shape (n_samples, n_dims).
    metadata_list:
        List of metadata dicts, one per point.
    max_points:
        Maximum number of points after subsampling.
    group_key:
        Key in metadata dicts used for stratification.

    Returns
    -------
    (ids_sub, embeddings_sub, metadata_sub, was_subsampled, original_count)
    """
    n = len(ids)
    if n <= max_points:
        return ids, embeddings, metadata_list, False, n

    # Build groups
    groups: dict[str, list[int]] = {}
    for i, meta in enumerate(metadata_list):
        key = meta.get(group_key, "__default__")
        if key not in groups:
            groups[key] = []
        groups[key].append(i)

    selected_indices: list[int] = []
    ratio = max_points / n

    for group_indices in groups.values():
        k = max(1, round(len(group_indices) * ratio))
        k = min(k, len(group_indices))
        sampled = random.sample(group_indices, k)
        selected_indices.extend(sampled)

    # Trim to max_points if rounding pushed us over
    if len(selected_indices) > max_points:
        selected_indices = random.sample(selected_indices, max_points)

    selected_indices.sort()
    ids_sub = [ids[i] for i in selected_indices]
    embeddings_sub = embeddings[selected_indices]
    metadata_sub = [metadata_list[i] for i in selected_indices]

    return ids_sub, embeddings_sub, metadata_sub, True, n


# ---------------------------------------------------------------------------
# Pre-computed clustering runs
# ---------------------------------------------------------------------------


async def list_clustering_runs(session, domain: str, space: str = "multi") -> list[dict]:
    """List available pre-computed clustering runs for a domain."""
    result = await session.execute(
        text("""
            SELECT id, domain, level, method, params, metrics,
                   n_clusters, n_stations, is_default, created_at
            FROM ml.clustering_runs
            WHERE domain = :domain AND level = 'stations' AND space = :space
            ORDER BY created_at DESC
            LIMIT 20
        """),
        {"domain": domain, "space": space},
    )
    rows = result.fetchall()
    return [
        {
            "id": r.id,
            "domain": r.domain,
            "level": r.level,
            "method": r.method,
            "params": r.params,
            "metrics": r.metrics,
            "n_clusters": r.n_clusters,
            "n_stations": r.n_stations,
            "is_default": r.is_default,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


async def load_clustering_run(session, run_id: int) -> dict:
    """Load a specific clustering run with all station labels and UMAP coords."""
    result = await session.execute(
        text("""
            SELECT id, domain, level, method, params, metrics,
                   n_clusters, n_stations, is_default, created_at
            FROM ml.clustering_runs
            WHERE id = :run_id
        """),
        {"run_id": run_id},
    )
    run_row = result.fetchone()
    if not run_row:
        return {}

    result = await session.execute(
        text("""
            SELECT station_id, cluster_id,
                   umap_2d_x, umap_2d_y,
                   umap_3d_x, umap_3d_y, umap_3d_z
            FROM ml.clustering_labels
            WHERE run_id = :run_id
            ORDER BY station_id
        """),
        {"run_id": run_id},
    )
    label_rows = result.fetchall()

    return {
        "id": run_row.id,
        "domain": run_row.domain,
        "method": run_row.method,
        "params": run_row.params,
        "metrics": run_row.metrics,
        "n_clusters": run_row.n_clusters,
        "n_stations": run_row.n_stations,
        "is_default": run_row.is_default,
        "created_at": run_row.created_at.isoformat() if run_row.created_at else None,
        "labels": [
            {
                "station_id": r.station_id,
                "cluster_id": r.cluster_id,
                "umap_2d": [r.umap_2d_x, r.umap_2d_y] if r.umap_2d_x is not None else None,
                "umap_3d": [r.umap_3d_x, r.umap_3d_y, r.umap_3d_z] if r.umap_3d_x is not None else None,
            }
            for r in label_rows
        ],
    }


async def load_station_windows(session, domain: str, station_id: str, space: str = "multi") -> list[dict]:
    """Load window embeddings for a single station, compute UMAP 2D on-the-fly."""
    import numpy as np

    id_col = "code_bss" if domain == "piezo" else "code_station"
    table = f"ml.{domain}_window_embeddings"

    result = await session.execute(
        text(f"""
            SELECT window_start, window_end, embedding::text AS embedding_raw
            FROM {table}
            WHERE {id_col} = :station_id AND space = :space
            ORDER BY window_start
        """),
        {"station_id": station_id, "space": space},
    )
    rows = result.fetchall()
    if not rows:
        return []

    embeddings = []
    windows = []
    for r in rows:
        emb = [float(x) for x in r.embedding_raw.strip("[]").split(",")]
        embeddings.append(emb)
        windows.append({
            "window_start": str(r.window_start),
            "window_end": str(r.window_end),
        })

    emb_array = np.array(embeddings, dtype=np.float32)
    if len(emb_array) >= 4:
        import umap
        nn = min(15, len(emb_array) - 1)
        coords = umap.UMAP(
            n_components=2, n_neighbors=nn,
            min_dist=0.05, metric="cosine", random_state=42,
        ).fit_transform(emb_array)
    else:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=min(2, emb_array.shape[0])).fit_transform(emb_array)

    for i, w in enumerate(windows):
        w["umap_x"] = float(coords[i, 0])
        w["umap_y"] = float(coords[i, 1])

    return windows
