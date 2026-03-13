"""FastAPI router for latent space exploration (stations, UMAP, similarity)."""
from __future__ import annotations

import asyncio
import logging
from collections import Counter, defaultdict

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_brgm_db
from api.schemas.cluster_profiling import ProfilingResponse
from api.schemas.latent_space import (
    ComputeRequest,
    ComputeResponse,
    ComputedPoint,
    EmbeddingFilters,
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
    session: AsyncSession = Depends(get_brgm_db),
) -> StationsResponse:
    """Return station-level UMAP coordinates and metadata from the DB."""
    if domain not in _VALID_DOMAINS:
        raise HTTPException(status_code=400, detail=f"Invalid domain '{domain}'. Must be one of {_VALID_DOMAINS}.")

    from dashboard.utils.latent_space import build_station_query

    filters = EmbeddingFilters()  # no filters — return all stations
    try:
        sql, params = build_station_query(domain, filters)
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
                metadata=meta,
            )
        )

    if domain == "piezo":
        stations = _apply_top_libelle(stations)

    return StationsResponse(stations=stations)


@router.post("/compute", response_model=ComputeResponse)
async def compute_latent_space(
    req: ComputeRequest,
    session: AsyncSession = Depends(get_brgm_db),
) -> ComputeResponse:
    """Compute on-demand UMAP + clustering for a given set of embeddings."""
    from dashboard.utils.latent_space import (
        build_station_query,
        build_window_query,
        compute_clustering,
        compute_umap,
        parse_pgvector,
        subsample_stratified,
    )
    import numpy as np

    domain = req.domain

    # --- Load embeddings from DB ---
    try:
        if req.embeddings_type == "stations":
            sql, params = build_station_query(domain, req.filters)
        else:
            sql, params = build_window_query(
                domain,
                req.filters,
                req.year_min,
                req.year_max,
                req.season,
            )
        result = await session.execute(sql, params)
        rows = result.fetchall()
    except Exception as exc:
        logger.exception("DB error in compute_latent_space: %s", exc)
        raise HTTPException(status_code=500, detail="Database query failed") from exc

    if not rows:
        return ComputeResponse(points=[], n_clusters=0)

    # --- Parse embeddings ---
    ids: list[str] = []
    raw_embeddings: list[np.ndarray] = []
    metadata_list: list[dict] = []
    window_starts: list[str | None] = []
    window_ends: list[str | None] = []

    for row in rows:
        raw = getattr(row, "embedding_raw", None)
        if raw is None:
            continue
        try:
            emb = parse_pgvector(raw)
        except Exception:
            continue
        ids.append(str(row.id))
        raw_embeddings.append(emb)
        window_starts.append(str(getattr(row, "window_start", None) or ""))
        window_ends.append(str(getattr(row, "window_end", None) or ""))
        if domain == "piezo":
            from dashboard.utils.latent_space import decode_eh_metadata
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
        return ComputeResponse(points=[], n_clusters=0)

    embeddings_matrix = np.stack(raw_embeddings, axis=0)

    # --- Subsample if needed ---
    subsampled = False
    subsampled_from: int | None = None
    ids, embeddings_matrix, metadata_list, subsampled, original_count = subsample_stratified(
        ids, embeddings_matrix, metadata_list, max_points=_MAX_POINTS
    )
    if subsampled:
        subsampled_from = original_count
        # Align window lists
        # Re-derive from remaining ids (we need index mapping)

    # Recompute window_starts/window_ends aligned to (potentially) subsampled ids
    # Build id->window maps before subsample disrupted the alignment
    # Since subsample_stratified returns indices in sorted order, we need to track
    # the index mapping. We work around this by rebuilding from metadata, which
    # always travels with the points.
    # (window_starts/ends are not in metadata — rebuild by id lookup)
    id_to_ws: dict[str, str] = {}
    id_to_we: dict[str, str] = {}
    for orig_id, ws, we in zip(
        [str(r.id) for r in rows if getattr(r, "embedding_raw", None) is not None],
        window_starts,
        window_ends,
    ):
        id_to_ws[orig_id] = ws
        id_to_we[orig_id] = we

    # --- UMAP + clustering (blocking → offload to thread) ---
    umap_params = req.umap
    clustering_params = req.clustering

    def _blocking_compute():
        from dashboard.utils.latent_space import compute_umap_quality

        coords = compute_umap(
            embeddings_matrix,
            n_components=umap_params.n_components,
            n_neighbors=umap_params.n_neighbors,
            min_dist=umap_params.min_dist,
            metric=umap_params.metric,
        )
        labels, metrics = compute_clustering(
            embeddings_matrix,
            method=clustering_params.method,
            params=clustering_params,
            n_umap_dims=clustering_params.n_umap_dims,
        )
        # UMAP visualization quality
        umap_viz_quality = compute_umap_quality(
            embeddings_matrix, coords, n_neighbors=umap_params.n_neighbors,
        )
        umap_viz_quality["input_dim"] = int(embeddings_matrix.shape[1])
        umap_viz_quality["output_dim"] = umap_params.n_components
        umap_viz_quality["n_neighbors"] = umap_params.n_neighbors
        umap_viz_quality["min_dist"] = umap_params.min_dist
        metrics["umap_visualization"] = umap_viz_quality
        metrics["n_points"] = len(ids)

        return coords, labels, metrics

    try:
        coords, labels, metrics = await asyncio.to_thread(_blocking_compute)
    except Exception as exc:
        logger.exception("UMAP/clustering failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Computation failed: {exc}") from exc

    # --- Build response ---
    n_clusters = len(set(labels) - {-1})
    points: list[ComputedPoint] = []

    for i, point_id in enumerate(ids):
        meta_dict = metadata_list[i]
        meta = StationMetadata(
            libelle_eh=meta_dict.get("libelle_eh"),
            milieu_eh=meta_dict.get("milieu_eh"),
            theme_eh=meta_dict.get("theme_eh"),
            etat_eh=meta_dict.get("etat_eh"),
            nature_eh=meta_dict.get("nature_eh"),
            departement=meta_dict.get("departement"),
            nom_departement=meta_dict.get("nom_departement"),
            altitude=meta_dict.get("altitude"),
            nom_cours_eau=meta_dict.get("nom_cours_eau"),
            statut_station=meta_dict.get("statut_station"),
        )
        ws = id_to_ws.get(point_id, "") or None
        we = id_to_we.get(point_id, "") or None
        points.append(
            ComputedPoint(
                id=point_id,
                coords=[float(v) for v in coords[i]],
                cluster_label=int(labels[i]),
                window_start=ws,
                window_end=we,
                metadata=meta,
            )
        )

    return ComputeResponse(
        points=points,
        n_clusters=n_clusters,
        subsampled=subsampled,
        subsampled_from=subsampled_from if subsampled else None,
        metrics=metrics if metrics else None,
    )


@router.get("/similar/{domain}/{station_id}", response_model=SimilarResponse)
async def get_similar_stations(
    domain: str,
    station_id: str,
    k: int = Query(default=10, ge=1, le=100),
    session: AsyncSession = Depends(get_brgm_db),
) -> SimilarResponse:
    """Return the k most similar stations to station_id using cosine distance."""
    if domain not in _VALID_DOMAINS:
        raise HTTPException(status_code=400, detail=f"Invalid domain '{domain}'. Must be one of {_VALID_DOMAINS}.")

    from dashboard.utils.latent_space import build_similar_query

    try:
        sql, params = build_similar_query(domain, station_id, k)
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
