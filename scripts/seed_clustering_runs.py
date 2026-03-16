"""Seed ml.clustering_runs with initial clustering configs.

Run inside junon-backend container:
    docker exec junon-backend python3 scripts/seed_clustering_runs.py
"""

import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import psycopg2


def load_embeddings(conn, domain: str):
    """Load station embeddings from ml.{domain}_station_embeddings."""
    id_col = "code_bss" if domain == "piezo" else "code_station"
    cur = conn.cursor()
    cur.execute(f"SELECT {id_col}, embedding::text FROM ml.{domain}_station_embeddings")
    rows = cur.fetchall()
    ids = [r[0] for r in rows]
    embs = np.array(
        [[float(x) for x in r[1].strip("[]").split(",")] for r in rows],
        dtype=np.float32,
    )
    return ids, embs


def run_clustering(embs, min_cluster_size, min_samples, umap_dims, umap_nn, umap_md):
    """Run UMAP pre-reduction + HDBSCAN, return labels + metrics."""
    import umap as umap_lib
    import hdbscan
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

    logger.info(
        f"  UMAP {embs.shape[1]}d → {umap_dims}d (nn={umap_nn}, md={umap_md})"
    )
    reduced = umap_lib.UMAP(
        n_components=umap_dims, n_neighbors=umap_nn, min_dist=umap_md,
        metric="cosine", random_state=42,
    ).fit_transform(embs)

    logger.info(
        f"  HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples})"
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean",
    )
    labels = clusterer.fit_predict(reduced)

    mask = labels >= 0
    n_clusters = len(set(labels[mask])) if mask.any() else 0
    n_noise = int((labels == -1).sum())

    sil = float(silhouette_score(reduced[mask], labels[mask])) if n_clusters >= 2 else -1.0
    db = float(davies_bouldin_score(reduced[mask], labels[mask])) if n_clusters >= 2 else -1.0
    ch = float(calinski_harabasz_score(reduced[mask], labels[mask])) if n_clusters >= 2 else -1.0
    dbcv = float(getattr(clusterer, "relative_validity_", 0.0))

    metrics = {
        "silhouette": round(sil, 4),
        "davies_bouldin": round(db, 4),
        "calinski_harabasz": round(ch, 2),
        "dbcv": round(dbcv, 4),
        "noise_ratio": round(n_noise / len(labels), 4) if len(labels) > 0 else 0,
    }
    params = {
        "umap_n_components": umap_dims,
        "umap_n_neighbors": umap_nn,
        "umap_min_dist": umap_md,
        "hdbscan_min_cluster_size": min_cluster_size,
        "hdbscan_min_samples": min_samples,
        "tuned": False,
    }
    logger.info(f"  → {n_clusters} clusters, {n_noise} noise, sil={sil:.4f}, dbcv={dbcv:.4f}")
    return labels, n_clusters, n_noise, metrics, params


def compute_umap_viz(embs):
    """Compute UMAP 2D + 3D for visualization."""
    import umap as umap_lib
    logger.info("  Computing UMAP 2D...")
    umap_2d = umap_lib.UMAP(
        n_components=2, n_neighbors=30, min_dist=0.05,
        metric="cosine", random_state=42,
    ).fit_transform(embs)
    logger.info("  Computing UMAP 3D...")
    umap_3d = umap_lib.UMAP(
        n_components=3, n_neighbors=30, min_dist=0.05,
        metric="cosine", random_state=42,
    ).fit_transform(embs)
    return umap_2d, umap_3d


def save_run(conn, domain, labels, ids, umap_2d, umap_3d, n_clusters, metrics, params, is_default):
    """Save clustering run to ml.clustering_runs + ml.clustering_labels."""
    cur = conn.cursor()

    if is_default:
        cur.execute(
            "UPDATE ml.clustering_runs SET is_default = FALSE "
            "WHERE domain = %s AND level = 'stations' AND is_default = TRUE",
            (domain,),
        )

    cur.execute(
        """
        INSERT INTO ml.clustering_runs
            (domain, level, method, params, metrics, n_clusters, n_stations, is_default)
        VALUES (%s, 'stations', 'hdbscan', %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (domain, json.dumps(params), json.dumps(metrics), n_clusters, len(ids), is_default),
    )
    run_id = cur.fetchone()[0]

    for i, (sid, label) in enumerate(zip(ids, labels)):
        cur.execute(
            """
            INSERT INTO ml.clustering_labels
                (run_id, station_id, cluster_id, umap_2d_x, umap_2d_y,
                 umap_3d_x, umap_3d_y, umap_3d_z)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (run_id, sid, int(label),
             float(umap_2d[i, 0]), float(umap_2d[i, 1]),
             float(umap_3d[i, 0]), float(umap_3d[i, 1]), float(umap_3d[i, 2])),
        )

    conn.commit()
    logger.info(f"  Saved run {run_id}: {domain} {n_clusters} clusters (default={is_default})")
    return run_id


CONFIGS = [
    # Config 1: Fixed defaults (good for hydro) — marked as default
    {"min_cluster_size": 10, "min_samples": 5, "umap_dims": 10, "umap_nn": 15, "umap_md": 0.0, "is_default": True},
    # Config 2: Wider clusters
    {"min_cluster_size": 25, "min_samples": 10, "umap_dims": 15, "umap_nn": 20, "umap_md": 0.1, "is_default": False},
]


def seed_domain(conn, domain):
    """Seed clustering configs for a domain."""
    logger.info(f"=== Seeding {domain} ===")
    ids, embs = load_embeddings(conn, domain)
    logger.info(f"Loaded {len(ids)} stations, embedding dim={embs.shape[1]}")

    # Shared UMAP 2D/3D visualization coords
    umap_2d, umap_3d = compute_umap_viz(embs)

    for i, cfg in enumerate(CONFIGS):
        logger.info(f"Config {i+1}: {cfg}")
        labels, n_clust, _, metrics, params = run_clustering(
            embs,
            min_cluster_size=cfg["min_cluster_size"],
            min_samples=cfg["min_samples"],
            umap_dims=cfg["umap_dims"],
            umap_nn=cfg["umap_nn"],
            umap_md=cfg["umap_md"],
        )
        save_run(conn, domain, labels, ids, umap_2d, umap_3d,
                 n_clust, metrics, params, is_default=cfg["is_default"])


def main():
    import os
    conn = psycopg2.connect(
        host=os.environ.get("BRGM_DB_HOST", "brgm-postgres"),
        port=int(os.environ.get("BRGM_DB_PORT", 5432)),
        dbname=os.environ.get("BRGM_DB_NAME", "postgres"),
        user=os.environ.get("BRGM_DB_USER", "postgres"),
        password=os.environ.get("BRGM_DB_PASSWORD", "postgres"),
    )
    try:
        seed_domain(conn, "piezo")
        seed_domain(conn, "hydro")
    finally:
        conn.close()
    logger.info("Done!")


if __name__ == "__main__":
    main()
