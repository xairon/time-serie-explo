"""Observatory WFS proxy router — fetches SANDRE WFS layers with Redis caching."""
from __future__ import annotations

import gzip
import logging
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from starlette.responses import Response

from api.config import settings
from dashboard.utils.cache import get_cached

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/observatory/wfs", tags=["observatory-wfs"])

WFS_TTL = 86400  # 24h — reference data, rarely changes

WFS_LAYERS = {
    "region-hydro": {
        "base_url": "https://services.sandre.eaufrance.fr/geo/zonage",
        "typename": "RegionHydro",
    },
    "secteur-hydro": {
        "base_url": "https://services.sandre.eaufrance.fr/geo/zonage",
        "typename": "SecteurHydro",
    },
    "sous-secteur-hydro": {
        "base_url": "https://services.sandre.eaufrance.fr/geo/zonage",
        "typename": "SousSecteurHydro",
    },
    "zone-hydro": {
        "base_url": "https://services.sandre.eaufrance.fr/geo/zonage",
        "typename": "ZoneHydro",
    },
    "cours-eau-1": {
        "base_url": "https://services.sandre.eaufrance.fr/geo/zonage",
        "typename": "CoursEau1",
    },
    "cours-eau-2": {
        "base_url": "https://services.sandre.eaufrance.fr/geo/zonage",
        "typename": "CoursEau2",
    },
    "plan-eau": {
        "base_url": "https://services.sandre.eaufrance.fr/geo/zonage",
        "typename": "PlanEau_FXX",
    },
    "masse-eau-riv": {
        "base_url": "https://services.sandre.eaufrance.fr/geo/MasseDEau_VRAP2022",
        "typename": "MasseDEauRiviere_VRAP2022_FXX",
    },
}


def _fetch_wfs_raw(layer_id: str, bbox: Optional[str] = None) -> bytes:
    """Fetch WFS layer as raw bytes (sync httpx)."""
    layer = WFS_LAYERS[layer_id]
    params = {
        "SERVICE": "WFS",
        "VERSION": "2.0.0",
        "REQUEST": "GetFeature",
        "TYPENAMES": layer["typename"],
        "OUTPUTFORMAT": "application/json; subtype=geojson",
        "SRSNAME": "EPSG:4326",
    }
    if bbox:
        params["BBOX"] = bbox

    resp = httpx.get(layer["base_url"], params=params, timeout=120.0)
    if resp.status_code != 200:
        logger.error("WFS error for %s: %s %s", layer_id, resp.status_code, resp.text[:200])
        raise HTTPException(status_code=502, detail=f"WFS service error for {layer_id}")
    return resp.content


@router.get("/{layer_id}")
def get_wfs_layer(
    request: Request,
    layer_id: str,
    bbox: Optional[str] = Query(None, description="Bounding box: min_lon,min_lat,max_lon,max_lat"),
):
    if layer_id not in WFS_LAYERS:
        raise HTTPException(status_code=404, detail=f"Unknown layer: {layer_id}")

    accepts_gzip = "gzip" in request.headers.get("accept-encoding", "")

    # For WFS we bypass the JSON-based get_cached and use Redis directly
    # because we store compressed bytes, not JSON dicts.
    import hashlib
    import json
    import redis as sync_redis

    cache_params = {"layer_id": layer_id, "bbox": bbox}
    raw_key = json.dumps(cache_params, sort_keys=True, default=str)
    h = hashlib.sha256(raw_key.encode()).hexdigest()[:32]
    key = f"junon:obs_wfs_{layer_id}:{h}"

    try:
        pool = sync_redis.ConnectionPool.from_url(settings.redis_url, decode_responses=False)
        r = sync_redis.Redis(connection_pool=pool)
        cached_val = r.get(key)
        if cached_val:
            if accepts_gzip:
                return Response(
                    content=cached_val, media_type="application/json",
                    headers={"Content-Encoding": "gzip", "Vary": "Accept-Encoding"},
                )
            return Response(content=gzip.decompress(cached_val), media_type="application/json")
    except Exception:
        r = None

    raw = _fetch_wfs_raw(layer_id, bbox)
    compressed = gzip.compress(raw, compresslevel=6)

    if r is not None:
        try:
            r.setex(key, WFS_TTL, compressed)
            logger.info("WFS cached %s (%d raw -> %d gz)", layer_id, len(raw), len(compressed))
        except Exception as e:
            logger.debug("Redis WFS cache error: %s", e)

    if accepts_gzip:
        return Response(
            content=compressed, media_type="application/json",
            headers={"Content-Encoding": "gzip", "Vary": "Accept-Encoding"},
        )
    return Response(content=raw, media_type="application/json")
