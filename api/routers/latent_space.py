"""FastAPI router for latent space exploration (stations, UMAP, similarity)."""
from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter, defaultdict

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from api.cache import get_redis
from api.database import get_brgm_db
from api.schemas.cluster_profiling import ProfilingResponse
from api.schemas.latent_space import (
    AutoTuneRequest,
    EmbeddingFilters,
    RecomputeClusteringRequest,
    RecomputePCARequest,
    RecomputeVizRequest,
    SimilarResponse,
    SimilarStation,
    StationMetadata,
    StationPoint,
    StationsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/latent-space", tags=["latent-space"])

_VALID_DOMAINS = {"piezo", "hydro"}
_MAX_POINTS = 15_000
_TOP_LIBELLE = 12

# ---------------------------------------------------------------------------
# Redis cache for computed UMAP + clustering results (per domain/space)
# ---------------------------------------------------------------------------

_CACHE_PREFIX = "junon:latent-space:compute"

_cache_locks: dict[str, asyncio.Lock] = {}


def _get_cache_lock(domain: str, space: str) -> asyncio.Lock:
    key = f"{domain}:{space}"
    if key not in _cache_locks:
        _cache_locks[key] = asyncio.Lock()
    return _cache_locks[key]


def _cache_key(domain: str, space: str) -> str:
    return f"{_CACHE_PREFIX}:{domain}:{space}"


def _sanitize_for_json(obj):
    """Recursively convert Decimal/numpy types to JSON-safe Python types."""
    import decimal
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
    except ImportError:
        pass
    return obj


async def _save_to_cache(domain: str, space: str, data: dict) -> None:
    """Store compute result in Redis (no TTL — persistent until replaced)."""
    r = get_redis()
    if r is None:
        return
    data = _sanitize_for_json(data)
    try:
        import orjson
        payload = orjson.dumps(data)
    except ImportError:
        payload = json.dumps(data).encode()
    try:
        await r.set(_cache_key(domain, space), payload)
        logger.info("Cached latent-space compute for %s/%s (%d bytes)", domain, space, len(payload))
    except Exception as exc:
        logger.warning("Redis SET error for latent-space cache: %s", exc)


async def _load_from_cache(domain: str, space: str) -> dict | None:
    """Load cached compute result from Redis."""
    r = get_redis()
    if r is None:
        return None
    try:
        raw = await r.get(_cache_key(domain, space))
        if raw is None:
            return None
        import orjson
        return orjson.loads(raw)
    except ImportError:
        return json.loads(raw) if raw else None
    except Exception as exc:
        logger.warning("Redis GET error for latent-space cache: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_station_metadata_piezo(row) -> StationMetadata:
    from dashboard.utils.latent_space import decode_eh_metadata

    raw = {
        "libelle_eh": getattr(row, "libelle_eh", None),
        "milieu_eh": getattr(row, "milieu_eh", None),
        "theme_eh": getattr(row, "theme_eh", None),
        "etat_eh": getattr(row, "etat_eh", None),
        "nature_eh": getattr(row, "nature_eh", None),
        "departement": getattr(row, "departement", None),
        "nom_departement": getattr(row, "nom_departement", None),
        "altitude": getattr(row, "altitude", None),
    }
    decoded = decode_eh_metadata(raw)
    return StationMetadata(**decoded)


def _row_to_station_metadata_hydro(row) -> StationMetadata:
    return StationMetadata(
        nom_cours_eau=getattr(row, "nom_cours_eau", None),
        departement=getattr(row, "departement", None),
        nom_departement=getattr(row, "nom_departement", None),
        statut_station=getattr(row, "statut_station", None),
    )


def _apply_top_libelle(stations: list[StationPoint]) -> list[StationPoint]:
    """Replace rare libelle_eh values with 'Autre', keeping only top 12."""
    counts: Counter = Counter(
        s.metadata.libelle_eh for s in stations if s.metadata.libelle_eh
    )
    top_set = {label for label, _ in counts.most_common(_TOP_LIBELLE)}
    for s in stations:
        if s.metadata.libelle_eh and s.metadata.libelle_eh not in top_set:
            s.metadata.libelle_eh = "Autre"
    return stations


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/stations/{domain}", response_model=StationsResponse)
async def get_stations(
    domain: str,
    space: str = Query("multi"),
    session: AsyncSession = Depends(get_brgm_db),
) -> StationsResponse:
    """Return station-level UMAP coordinates and metadata from the DB."""
    if domain not in _VALID_DOMAINS:
        raise HTTPException(status_code=400, detail=f"Invalid domain '{domain}'. Must be one of {_VALID_DOMAINS}.")

    from dashboard.utils.latent_space import build_station_query

    filters = EmbeddingFilters()  # no filters — return all stations
    try:
        sql, params = build_station_query(domain, filters, space=space)
        result = await session.execute(sql, params)
        rows = result.fetchall()
    except Exception as exc:
        logger.exception("DB error in get_stations: %s", exc)
        raise HTTPException(status_code=500, detail="Database query failed") from exc

    stations: list[StationPoint] = []
    for row in rows:
        umap_2d = None
        x2, y2 = getattr(row, "umap_2d_x", None), getattr(row, "umap_2d_y", None)
        if x2 is not None and y2 is not None:
            umap_2d = [float(x2), float(y2)]

        umap_3d = None
        x3, y3, z3 = (
            getattr(row, "umap_3d_x", None),
            getattr(row, "umap_3d_y", None),
            getattr(row, "umap_3d_z", None),
        )
        if x3 is not None and y3 is not None and z3 is not None:
            umap_3d = [float(x3), float(y3), float(z3)]

        cluster_id = getattr(row, "cluster_id", None)
        n_windows = getattr(row, "n_windows", None)
        last_date = getattr(row, "last_date", None)

        meta = (
            _row_to_station_metadata_piezo(row)
            if domain == "piezo"
            else _row_to_station_metadata_hydro(row)
        )

        stations.append(
            StationPoint(
                id=str(row.id),
                umap_2d=umap_2d,
                umap_3d=umap_3d,
                cluster_id=int(cluster_id) if cluster_id is not None else None,
                n_windows=int(n_windows) if n_windows is not None else None,
                last_date=str(last_date) if last_date is not None else None,
                metadata=meta,
            )
        )

    if domain == "piezo":
        stations = _apply_top_libelle(stations)

    return StationsResponse(stations=stations)


async def _load_embeddings(domain: str, space: str, session):
    """Load embeddings + metadata from DB. Returns (ids, embeddings_matrix, metadata_list)."""
    from dashboard.utils.latent_space import build_station_query, decode_eh_metadata, parse_pgvector
    import numpy as np

    sql, params = build_station_query(domain, EmbeddingFilters(), space=space)
    result = await session.execute(sql, params)
    rows = result.fetchall()

    ids, raw_embs, metadata_list = [], [], []
    for row in rows:
        raw = getattr(row, "embedding_raw", None)
        if raw is None:
            continue
        try:
            emb = parse_pgvector(raw)
        except Exception:
            continue
        ids.append(str(row.id))
        raw_embs.append(emb)
        if domain == "piezo":
            meta = decode_eh_metadata({
                "libelle_eh": getattr(row, "libelle_eh", None),
                "milieu_eh": getattr(row, "milieu_eh", None),
                "theme_eh": getattr(row, "theme_eh", None),
                "etat_eh": getattr(row, "etat_eh", None),
                "nature_eh": getattr(row, "nature_eh", None),
                "departement": getattr(row, "departement", None),
                "nom_departement": getattr(row, "nom_departement", None),
                "altitude": getattr(row, "altitude", None),
                "station_id": str(row.id),
            })
        else:
            meta = {
                "nom_cours_eau": getattr(row, "nom_cours_eau", None),
                "departement": getattr(row, "departement", None),
                "nom_departement": getattr(row, "nom_departement", None),
                "statut_station": getattr(row, "statut_station", None),
                "station_id": str(row.id),
            }
        metadata_list.append(meta)

    if not ids:
        return [], None, []
    return ids, np.stack(raw_embs, axis=0), metadata_list


async def _full_pipeline(domain, space, session, variance_threshold=0.95, viz_params=None, clustering_params=None):
    """Run full PCA -> VIZ + CLUSTERING pipeline. Returns cache dict."""
    from dashboard.utils.latent_space import compute_pca, compute_viz, compute_clustering_all
    import numpy as np

    ids, embeddings_matrix, metadata_list = await _load_embeddings(domain, space, session)
    if embeddings_matrix is None or len(ids) < 20:
        raise HTTPException(status_code=400, detail=f"Not enough embeddings ({len(ids)})")

    vp = viz_params or {}
    cp = clustering_params or {}

    def _blocking():
        pca_result = compute_pca(embeddings_matrix, variance_threshold=variance_threshold)
        pca_reduced = np.array(pca_result["reduced"])
        viz_result = compute_viz(pca_reduced, n_neighbors=vp.get("n_neighbors", 50), min_dist=vp.get("min_dist", 0.3))
        clust_result = compute_clustering_all(pca_reduced, min_cluster_size=cp.get("min_cluster_size"), min_samples=cp.get("min_samples"))
        return pca_result, viz_result, clust_result

    pca_result, viz_result, clust_result = await asyncio.to_thread(_blocking)

    cache_data = {
        "pca": pca_result,
        "viz": viz_result,
        "clustering": clust_result,
        "station_ids": ids,
        "metadata": metadata_list,
    }
    await _save_to_cache(domain, space, cache_data)
    return cache_data


@router.get("/cached/{domain}")
async def get_cached(domain: str, space: str = Query("multi")):
    if domain not in _VALID_DOMAINS:
        raise HTTPException(status_code=400, detail=f"Invalid domain: {domain}")
    cached = await _load_from_cache(domain, space)
    if cached is None:
        raise HTTPException(status_code=404, detail="No cached result")
    # Strip pca.reduced from response (too large, frontend doesn't need it)
    if "pca" in cached and "reduced" in cached["pca"]:
        cached["pca"] = {k: v for k, v in cached["pca"].items() if k != "reduced"}
    return cached


@router.post("/recompute-pca")
async def recompute_pca(req: RecomputePCARequest, session: AsyncSession = Depends(get_brgm_db)):
    """Recompute PCA, cascade VIZ + CLUSTERING. Updates full cache."""
    # Load current cache for viz/clustering params (to preserve user's settings)
    current = await _load_from_cache(req.domain, req.space)
    viz_params = current.get("viz", {}).get("params", {}) if current else {}
    clustering_params = current.get("clustering", {}).get("hdbscan", {}).get("params", {}) if current else {}

    cache_data = await _full_pipeline(
        req.domain, req.space, session,
        variance_threshold=req.variance_threshold,
        viz_params=viz_params,
        clustering_params=clustering_params,
    )
    # Return without pca.reduced
    if "pca" in cache_data and "reduced" in cache_data["pca"]:
        resp = {**cache_data, "pca": {k: v for k, v in cache_data["pca"].items() if k != "reduced"}}
    else:
        resp = cache_data
    return resp


@router.post("/recompute-viz")
async def recompute_viz(req: RecomputeVizRequest):
    """Recompute UMAP 2D visualization only. Loads PCA from cache."""
    from dashboard.utils.latent_space import compute_viz
    import numpy as np

    async with _get_cache_lock(req.domain, req.space):
        cached = await _load_from_cache(req.domain, req.space)
        if cached is None or "pca" not in cached or "reduced" not in cached["pca"]:
            raise HTTPException(status_code=400, detail="No PCA data in cache. Run Recompute PCA first.")

        pca_reduced = np.array(cached["pca"]["reduced"])

        viz_result = await asyncio.to_thread(
            compute_viz, pca_reduced, n_neighbors=req.n_neighbors, min_dist=req.min_dist,
        )

        # Update viz section atomically
        cached["viz"] = viz_result
        await _save_to_cache(req.domain, req.space, cached)

    # Return without pca.reduced
    resp = {**cached, "pca": {k: v for k, v in cached["pca"].items() if k != "reduced"}}
    return resp


@router.post("/recompute-clustering")
async def recompute_clustering(req: RecomputeClusteringRequest):
    """Recompute HDBSCAN + KMeans elbow. Loads PCA from cache."""
    from dashboard.utils.latent_space import compute_clustering_all
    import numpy as np

    async with _get_cache_lock(req.domain, req.space):
        cached = await _load_from_cache(req.domain, req.space)
        if cached is None or "pca" not in cached or "reduced" not in cached["pca"]:
            raise HTTPException(status_code=400, detail="No PCA data in cache. Run Recompute PCA first.")

        pca_reduced = np.array(cached["pca"]["reduced"])

        clust_result = await asyncio.to_thread(
            compute_clustering_all, pca_reduced,
            min_cluster_size=req.min_cluster_size, min_samples=req.min_samples,
        )

        cached["clustering"] = clust_result
        await _save_to_cache(req.domain, req.space, cached)

    resp = {**cached, "pca": {k: v for k, v in cached["pca"].items() if k != "reduced"}}
    return resp


@router.post("/auto-tune")
async def auto_tune(req: AutoTuneRequest, session: AsyncSession = Depends(get_brgm_db)):
    """Full pipeline with defaults. Force-replaces entire cache."""
    cache_data = await _full_pipeline(req.domain, req.space, session)
    resp = {**cache_data, "pca": {k: v for k, v in cache_data["pca"].items() if k != "reduced"}}
    return resp


async def warmup_cache(session) -> None:
    """Warm cache for all 4 domain/space combos at startup."""
    combos = [("piezo", "uni"), ("piezo", "multi"), ("hydro", "uni"), ("hydro", "multi")]
    for domain, space in combos:
        cached = await _load_from_cache(domain, space)
        if cached is not None:
            logger.info("Cache already warm for %s/%s, skipping", domain, space)
            continue
        logger.info("Warming cache for %s/%s ...", domain, space)
        try:
            await _full_pipeline(domain, space, session)
            logger.info("Cache warmed for %s/%s", domain, space)
        except Exception:
            logger.exception("Failed to warm cache for %s/%s", domain, space)


@router.get("/similar/{domain}/{station_id}", response_model=SimilarResponse)
async def get_similar_stations(
    domain: str,
    station_id: str,
    k: int = Query(default=10, ge=1, le=100),
    space: str = Query("multi"),
    session: AsyncSession = Depends(get_brgm_db),
) -> SimilarResponse:
    """Return the k most similar stations to station_id using cosine distance."""
    if domain not in _VALID_DOMAINS:
        raise HTTPException(status_code=400, detail=f"Invalid domain '{domain}'. Must be one of {_VALID_DOMAINS}.")

    from dashboard.utils.latent_space import build_similar_query

    try:
        sql, params = build_similar_query(domain, station_id, k, space=space)
        result = await session.execute(sql, params)
        rows = result.fetchall()
    except Exception as exc:
        logger.exception("DB error in get_similar_stations: %s", exc)
        raise HTTPException(status_code=500, detail="Database query failed") from exc

    neighbors: list[SimilarStation] = []
    for row in rows:
        cluster_id = getattr(row, "cluster_id", None)
        neighbors.append(
            SimilarStation(
                id=str(row.id),
                distance=float(row.distance),
                cluster_id=int(cluster_id) if cluster_id is not None else None,
            )
        )

    return SimilarResponse(query_id=station_id, neighbors=neighbors)


@router.get("/station-windows/{domain}/{station_id}")
async def get_station_windows(
    domain: str,
    station_id: str,
    space: str = Query("multi"),
    session: AsyncSession = Depends(get_brgm_db),
):
    """Load window embeddings for a station with UMAP 2D coords."""
    if domain not in _VALID_DOMAINS:
        raise HTTPException(status_code=400, detail="Invalid domain")
    from dashboard.utils.latent_space import load_station_windows
    windows = await load_station_windows(session, domain, station_id, space)
    return {"station_id": station_id, "windows": windows}


@router.get("/profiling/{domain}", response_model=ProfilingResponse)
async def get_profiling(
    domain: str,
    hide_unclassified: bool = Query(False),
    space: str = Query("multi"),
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
    sql, params = build_station_query(domain, EmbeddingFilters(), space=space)
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
        # Note: gold tables use "date" as the date column (not "date_mesure")
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
                f"SELECT {id_col} AS station_id, date, {col} AS value "
                f"FROM {table} "
                f"WHERE {id_col} IN ({placeholders}){extra_filter} "
                f"AND date >= CURRENT_DATE - INTERVAL '5 years' "
                f"ORDER BY {id_col}, date"
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
                current_dates.append(str(r.date))
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
        MetadataDistribution, ConcordanceMetric,
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
            ClusterPrototype(cluster_id=cid, n_members=len(cluster_members.get(cid, [])), **data)
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
